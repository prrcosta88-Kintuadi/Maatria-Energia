import json
import os
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import duckdb
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ── Neon PostgreSQL ───────────────────────────────────────────────────────────
try:
    import db_neon
except Exception:
    db_neon = None  # type: ignore

def _NEON_OK() -> bool:
    """Verifica conexão Neon em tempo de execução (não no import)."""
    import os as _os
    import streamlit as _st
    _url = _os.getenv("DATABASE_URL", "")
    if not _url:
        _st.session_state["_neon_error"] = "DATABASE_URL vazia"
        return False
    try:
        import psycopg2 as _pg
        _c = _pg.connect(_url)
        _c.close()
        return True
    except Exception as _e:
        _st.session_state["_neon_error"] = str(_e)
        return False


def _core_cache_token() -> str:
    """Token de cache baseado no max(data) do PLD — muda quando novos dados chegam."""
    if not _NEON_OK():
        return "neon:offline"
    try:
        row = db_neon.fetchone(
            "SELECT MAX(mes_referencia * 100 + dia)::text FROM pld_historical"
        )
        return f"neon:{row[0] if row else 'empty'}"
    except Exception:
        return "neon:error"


def _core_file_diagnostics() -> list[str]:
    if not _NEON_OK():
        return ["DATABASE_URL não configurada ou psycopg2 não instalado."]
    msgs = []
    for table in ["geracao_tipo_hora","curva_carga","pld_historical","ear_diario_subsistema","despacho_gfom"]:
        row = db_neon.fetchone(f"SELECT COUNT(*) FROM {table}")
        msgs.append(f"{table}: {row[0]:,} linhas" if row else f"{table}: erro")
    return msgs


def _configure_duckdb_low_memory(con: duckdb.DuckDBPyConnection) -> None:
    # Ajustes recomendados pelo próprio DuckDB para ambientes com RAM restrita (ex.: Render free)
    try:
        con.execute("SET threads=1")
    except Exception:
        pass
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass
    try:
        con.execute("SET memory_limit='320MB'")
    except Exception:
        pass


def _decode_parquet_row(row: Any, columns: list[str]) -> Dict[str, Any]:
    if not row:
        return {}

    col_idx = {c.lower(): i for i, c in enumerate(columns)}
    for candidate in ["core_json", "core", "payload", "json"]:
        i = col_idx.get(candidate)
        if i is None:
            continue
        val = row[i]
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip():
            try:
                loaded = json.loads(val)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass

    rec = {col: row[i] for i, col in enumerate(columns)}
    if all(k in rec for k in ["timestamp", "hydrology", "prices"]):
        return rec
    return {}


@st.cache_data
def _load_section_cached(_token: str, section: str) -> Dict[str, Any]:
    """
    Carrega UMA seção do parquet correspondente e retorna o dict.
    Cache por seção — cada uma ocupa memória independentemente.
    Descarta o JSON bruto assim que o dict é construído.
    """
    p = _section_path(section)
    if p is None:
        return {}
    try:
        con = duckdb.connect()
        try:
            _configure_duckdb_low_memory(con)
            try:
                row = con.execute("SELECT section_json FROM read_parquet(?) LIMIT 1", [str(p)]).fetchone()
            except Exception:
                quoted = str(p).replace("'", "''")
                row = con.execute(f"SELECT section_json FROM read_parquet('{quoted}') LIMIT 1").fetchone()
        finally:
            con.close()
        if not (row and row[0]):
            return {}
        val = row[0]
        loaded = json.loads(val) if isinstance(val, str) else val
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _get_section(section: str) -> Dict[str, Any]:
    """Retorna dados de uma seção, usando cache por seção."""
    p = _section_path(section)
    token = f"{p}:{int(p.stat().st_mtime)}" if p else f"{section}:missing"
    return _load_section_cached(token, section)


@st.cache_data(show_spinner=False)
def _build_hourly_df_cached(_token: str) -> pd.DataFrame:
    """
    Constrói o DataFrame horário lendo o Neon PostgreSQL via queries leves.
    Cada query retorna apenas as colunas necessárias — RAM < 40MB no Render.
    """
    if not _NEON_OK():
        return pd.DataFrame()

    df = pd.DataFrame()

    def _join(s: pd.Series) -> None:
        nonlocal df
        if s.empty:
            return
        df = df.join(s, how="outer") if not df.empty else s.to_frame()

    def _ts(s: pd.Series) -> pd.Series:
        s.index = pd.to_datetime(s.index, errors="coerce")
        if getattr(s.index, "tz", None):
            s.index = s.index.tz_localize(None)
        return s.dropna()

    # ── economic: CMO por submercado ─────────────────────────────────────────
    cmo_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, id_subsistema, val_cmo AS valor "
        "FROM cmo WHERE din_instante IS NOT NULL AND val_cmo IS NOT NULL "
        "ORDER BY din_instante"
    )
    if not cmo_df.empty:
        cmo_df["instante"] = pd.to_datetime(cmo_df["instante"], errors="coerce")
        cmo_df = cmo_df.dropna(subset=["instante"])
        cmo_best = "SUDESTE" if "SUDESTE" in cmo_df["id_subsistema"].values else cmo_df["id_subsistema"].iloc[0]
        sub = cmo_df[cmo_df["id_subsistema"] == cmo_best]
        if not sub.empty:
            _join(_ts(pd.to_numeric(sub.set_index("instante")["valor"], errors="coerce").rename("cmo_dominante")))
    del cmo_df

    # ── PLD (CCEE) ────────────────────────────────────────────────────────────
    pld_df = db_neon.fetchdf(
        "SELECT "
        "  MAKE_DATE("
        "    CAST(SUBSTR(CAST(mes_referencia AS TEXT), 1, 4) AS INTEGER), "
        "    CAST(SUBSTR(CAST(mes_referencia AS TEXT), 5, 2) AS INTEGER), "
        "    dia"
        "  ) + (hora * INTERVAL '1 hour') AS ts, "
        "  submercado, "
        "  AVG(pld_hora) AS pld_hora "
        "FROM pld_historical "
        "WHERE mes_referencia IS NOT NULL AND pld_hora IS NOT NULL "
        "GROUP BY mes_referencia, dia, hora, submercado "
        "ORDER BY ts, submercado"
    )
    if not pld_df.empty:
        pld_df["ts"] = pd.to_datetime(pld_df["ts"], errors="coerce")
        pld_df = pld_df.dropna(subset=["ts", "submercado"])
        pld_df["pld_hora"] = pd.to_numeric(pld_df["pld_hora"], errors="coerce")

        # Série por submercado
        SUB_COL = {"SUDESTE": "pld_se", "NORDESTE": "pld_ne",
                   "SUL": "pld_s", "NORTE": "pld_n",
                   "SE": "pld_se", "NE": "pld_ne",
                   "S": "pld_s", "N": "pld_n"}
        for sub in pld_df["submercado"].unique():
            col_name = SUB_COL.get(sub.upper().strip(), f"pld_{sub.lower()[:3]}")
            sub_s = pld_df[pld_df["submercado"] == sub].set_index("ts")["pld_hora"].rename(col_name)
            _join(_ts(pd.to_numeric(sub_s, errors="coerce")))

        # Média SIN como série principal "pld"
        sin_s = pld_df.groupby("ts")["pld_hora"].mean().rename("pld")
        _join(_ts(sin_s))
    del pld_df

    # ── Geração por fonte ────────────────────────────────────────────────────
    gen_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, tipo_geracao, "
        "  SUM(val_geracao_mw) AS valor "
        "FROM geracao_tipo_hora "
        "WHERE din_instante IS NOT NULL AND val_geracao_mw IS NOT NULL "
        "GROUP BY din_instante, tipo_geracao ORDER BY din_instante"
    )
    if not gen_df.empty:
        gen_df["instante"] = pd.to_datetime(gen_df["instante"], errors="coerce")
        gen_df = gen_df.dropna(subset=["instante"])
        # tipo_geracao já normalizado: solar | wind | hydro | thermal | nuclear | other
        for tipo in ("solar", "wind", "hydro", "thermal", "nuclear"):
            sub = gen_df[gen_df["tipo_geracao"] == tipo]
            if not sub.empty:
                s = sub.groupby("instante")["valor"].sum().rename(tipo)
                _join(_ts(pd.to_numeric(s, errors="coerce")))
    del gen_df

    # ── Carga SIN ────────────────────────────────────────────────────────────
    carga_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, "
        "  UPPER(TRIM(id_subsistema)) AS id_subsistema, "
        "  SUM(val_cargaenergiahomwmed) AS valor "
        "FROM curva_carga "
        "WHERE din_instante IS NOT NULL AND val_cargaenergiahomwmed IS NOT NULL "
        "GROUP BY din_instante, id_subsistema ORDER BY din_instante"
    )
    if not carga_df.empty:
        carga_df["instante"] = pd.to_datetime(carga_df["instante"], errors="coerce")
        carga_df = carga_df.dropna(subset=["instante", "id_subsistema"])
        carga_df["valor"] = pd.to_numeric(carga_df["valor"], errors="coerce")

        _SUB_MAP = {"SE": "se", "NE": "ne", "S": "s", "N": "n",
                    "SUDESTE": "se", "NORDESTE": "ne", "SUL": "s", "NORTE": "n"}
        carga_df["sub_key"] = carga_df["id_subsistema"].map(_SUB_MAP)
        carga_df = carga_df.dropna(subset=["sub_key"])

        # Série por submercado: load_se, load_ne, load_s, load_n
        for sub_key, grp in carga_df.groupby("sub_key"):
            col = f"load_{sub_key}"
            s = grp.groupby("instante")["valor"].sum().rename(col)
            _join(_ts(pd.to_numeric(s, errors="coerce")))

        # Série SIN total
        sin_s = carga_df.groupby("instante")["valor"].sum().rename("load")
        _join(_ts(pd.to_numeric(sin_s, errors="coerce")))
    del carga_df

    # ── EAR / ENA ────────────────────────────────────────────────────────────
    ear_df = db_neon.fetchdf(
        "SELECT ear_data AS instante, "
        "  SUM(ear_verif_subsistema_mwmes) AS ear, SUM(ear_max_subsistema) AS earmaxp "
        "FROM ear_diario_subsistema WHERE ear_data IS NOT NULL "
        "GROUP BY ear_data ORDER BY ear_data"
    )
    if not ear_df.empty:
        ear_df["instante"] = pd.to_datetime(ear_df["instante"], errors="coerce").dt.normalize()
        ear_df = ear_df.dropna(subset=["instante"])
        ear_df["ear_pct"] = ear_df["ear"] / ear_df["earmaxp"].replace(0, np.nan) * 100
        # Série diária → expandir para cada hora do dia via resample + ffill
        ear_s = pd.to_numeric(ear_df.set_index("instante")["ear_pct"], errors="coerce").rename("ear_pct")
        ear_s = ear_s.resample("h").ffill()   # cria 24 entradas por dia com mesmo valor
        _join(_ts(ear_s))
    del ear_df

    # ── ENA diário ───────────────────────────────────────────────────────────
    # Busca histórico completo de ENA + percentil 90 dos últimos 365 dias
    # O P90 serve como denominador estável para ENA_norm, independente do
    # período selecionado na tela pelo usuário.
    ena_df = db_neon.fetchdf(
        "SELECT ena_data AS instante, "
        "  SUM(ena_bruta_regiao_mwmed) AS ena_bruta, "
        "  SUM(ena_armazenavel_regiao_mwmed) AS ena_arm "
        "FROM ena_diario_subsistema WHERE ena_data IS NOT NULL "
        "GROUP BY ena_data ORDER BY ena_data"
    )
    # Denominador para ENA_norm: percentil 90 dos últimos 365 dias
    # Janela de 365 dias captura um ciclo hidrológico completo (cheia + seca),
    # tornando a normalização estável e comparável ao longo do tempo.
    _ena_norm_denom = np.nan
    if not ena_df.empty:
        _ena_ref = pd.to_numeric(ena_df["ena_arm"], errors="coerce").dropna()
        _cutoff  = pd.Timestamp.now().normalize() - pd.Timedelta(days=365)
        _ena_ref_365 = _ena_ref[pd.to_datetime(ena_df["instante"], errors="coerce") >= _cutoff]
        if not _ena_ref_365.empty:
            _ena_norm_denom = float(_ena_ref_365.quantile(0.90))
        elif not _ena_ref.empty:
            _ena_norm_denom = float(_ena_ref.quantile(0.90))   # fallback: série toda

    if not ena_df.empty:
        ena_df["instante"] = pd.to_datetime(ena_df["instante"], errors="coerce").dt.normalize()
        ena_df = ena_df.dropna(subset=["instante"]).set_index("instante")
        # Série diária → expandir para cada hora do dia via resample + ffill
        for _col, _name in [("ena_bruta", "ena_bruta"), ("ena_arm", "ena_arm")]:
            ena_s = pd.to_numeric(ena_df[_col].rename(_name), errors="coerce")
            ena_s = ena_s.resample("h").ffill()
            _join(_ts(ena_s))
    del ena_df

    # ── Disponibilidade sincronizada por tipo ────────────────────────────────
    disp_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, tipo_geracao, "
        "  SUM(val_disp_sincronizada) AS disp_sinc "
        "FROM disponibilidade_tipo_hora "
        "WHERE din_instante IS NOT NULL "
        "GROUP BY din_instante, tipo_geracao ORDER BY din_instante"
    )
    if not disp_df.empty:
        disp_df["instante"] = pd.to_datetime(disp_df["instante"], errors="coerce")
        disp_df = disp_df.dropna(subset=["instante"])
        for _tipo, _col in (("hydro","disp_hydro"),("thermal","disp_thermal"),("nuclear","disp_nuclear"),("solar","disp_solar"),("wind","disp_wind")):
            sub = disp_df[disp_df["tipo_geracao"] == _tipo]
            if not sub.empty:
                _join(_ts(pd.to_numeric(sub.groupby("instante")["disp_sinc"].sum().rename(_col), errors="coerce")))
        _join(_ts(pd.to_numeric(disp_df.groupby("instante")["disp_sinc"].sum().rename("disp_total"), errors="coerce")))
    del disp_df

    # ── Restrição renovável — curtailment solar + eólico ────────────────────
    restr_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, fonte, "
        "  SUM(val_geracao) AS gerado, "
        "  SUM(val_geracaolimitada) AS limitado, "
        "  SUM(val_disponibilidade) AS disponivel, "
        "  SUM(val_geracaoreferencia) AS referencia "
        "FROM restricao_renovavel "
        "WHERE din_instante IS NOT NULL "
        "GROUP BY din_instante, fonte ORDER BY din_instante"
    )
    if not restr_df.empty:
        restr_df["instante"] = pd.to_datetime(restr_df["instante"], errors="coerce")
        restr_df = restr_df.dropna(subset=["instante"])
        # curtailment = limitado - gerado (positivo = energia cortada)
        _curtail_total = pd.Series(0.0, index=restr_df["instante"].unique())
        for _fonte, _sfx in (("solar","solar"), ("wind","wind")):
            _sub = restr_df[restr_df["fonte"] == _fonte].set_index("instante")
            if not _sub.empty:
                _curtail = (_sub["limitado"] - _sub["gerado"]).clip(lower=0).rename(f"curtail_{_sfx}")
                _avail   = _sub["disponivel"].rename(f"avail_{_sfx}")
                _join(_ts(pd.to_numeric(_curtail, errors="coerce")))
                _join(_ts(pd.to_numeric(_avail, errors="coerce")))
                _curtail_total = _curtail_total.add(_curtail.reindex(_curtail_total.index).fillna(0), fill_value=0)
        _join(_ts(pd.to_numeric(_curtail_total.rename("curtail_total"), errors="coerce")))
        # avail_ren = avail_solar + avail_wind (total disponível renovável)
        _avail_ren = restr_df.groupby("instante")["disponivel"].sum().rename("avail_ren")
        _join(_ts(pd.to_numeric(_avail_ren, errors="coerce")))
    del restr_df

    # ── Despacho GFOM ────────────────────────────────────────────────────────
    gfom_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, "
        "  val_verifgeracao AS gfom_ger, "
        "  val_verifconstrainedoff AS constrained_off, "
        "  val_verifinflexibilidade AS thermal_inflex_gfom, "
        "  val_verifordemmerito AS thermal_merit, "
        "  val_verifgfom AS gfom "
        "FROM despacho_gfom WHERE din_instante IS NOT NULL ORDER BY din_instante"
    )
    if not gfom_df.empty:
        gfom_df["instante"] = pd.to_datetime(gfom_df["instante"], errors="coerce")
        gfom_df = gfom_df.dropna(subset=["instante"]).set_index("instante")
        for _col in ("gfom_ger","constrained_off","thermal_inflex_gfom","thermal_merit","gfom"):
            if _col in gfom_df.columns:
                _join(_ts(pd.to_numeric(gfom_df[_col].rename(_col), errors="coerce")))
    del gfom_df

    # ── CVU médio semanal térmico ────────────────────────────────────────────
    cvu_df = db_neon.fetchdf(
        "SELECT dat_fimsemana AS instante, AVG(val_cvu) AS cvu_semana "
        "FROM cvu_usina_termica WHERE val_cvu > 0 "
        "GROUP BY dat_fimsemana ORDER BY dat_fimsemana"
    )
    if not cvu_df.empty:
        cvu_df["instante"] = pd.to_datetime(cvu_df["instante"], errors="coerce")
        cvu_df = cvu_df.dropna(subset=["instante"])
        _join(_ts(pd.to_numeric(cvu_df.set_index("instante")["cvu_semana"].rename("cvu_semana"), errors="coerce")))
    del cvu_df

    # ── Intercâmbio ──────────────────────────────────────────────────────────
    itc_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, "
        "  id_subsistema_origem AS orig, id_subsistema_destino AS dest, "
        "  SUM(val_intercambiomwmed) AS mw "
        "FROM intercambio WHERE din_instante IS NOT NULL "
        "GROUP BY din_instante, id_subsistema_origem, id_subsistema_destino "
        "ORDER BY din_instante"
    )
    if not itc_df.empty:
        itc_df["instante"] = pd.to_datetime(itc_df["instante"], errors="coerce")
        itc_df = itc_df.dropna(subset=["instante"])
        itc_df["par"] = itc_df["orig"].str.upper() + "_" + itc_df["dest"].str.upper()
        for _par, _grp in itc_df.groupby("par"):
            _join(_ts(pd.to_numeric(_grp.set_index("instante")["mw"].rename(f"itc_{_par.lower()}"), errors="coerce")))
        _join(_ts(pd.to_numeric(itc_df.groupby("instante")["mw"].sum().rename("itc_total"), errors="coerce")))
    del itc_df

    # ── Colunas derivadas ─────────────────────────────────────────────────────
    if not df.empty:
        df = df.sort_index()
        z = pd.Series(0.0, index=df.index)

        def _col(name):
            return pd.to_numeric(df[name], errors="coerce") if name in df.columns else z

        load_s   = _col("load")
        solar_s  = _col("solar")
        wind_s   = _col("wind")
        hydro_s  = _col("hydro")
        thermal_s = _col("thermal")
        nuclear_s = _col("nuclear")

        df["net_load"]     = load_s - solar_s.fillna(0) - wind_s.fillna(0)
        df["carga_total"]  = load_s
        df["renov_total"]  = solar_s.fillna(0) + wind_s.fillna(0)
        df["geracao_total"]= solar_s.fillna(0) + wind_s.fillna(0) + hydro_s.fillna(0) + thermal_s.fillna(0) + nuclear_s.fillna(0)

        # SIN_cost ponderado por submercado
        _sub_pairs = [("se", "se"), ("ne", "ne"), ("s", "s"), ("n", "n")]
        _sin_cost = pd.Series(0.0, index=df.index)
        _any_sub = False
        for _sk, _pk in _sub_pairs:
            _lc, _pc = f"load_{_sk}", f"pld_{_pk}"
            if _lc in df.columns and _pc in df.columns:
                _sin_cost += _col(_lc).fillna(0) * _col(_pc).fillna(0)
                _any_sub = True
        if not _any_sub:
            _sin_cost = _col("load").fillna(0) * _col("pld").fillna(0)
        df["sin_cost"]      = _sin_cost.where(_sin_cost > 0, np.nan)
        df["SIN_cost_R$/h"] = df["sin_cost"]
        df["pld_ponderado"] = (df["sin_cost"] / load_s.replace(0, np.nan)).where(load_s > 0, np.nan)

        # CVU semanal interpolado para horário
        if "cvu_semana" in df.columns:
            df["cvu_semana"] = df["cvu_semana"].ffill().bfill()

        # Custo real e por mérito térmico
        _thermal_merit = _col("thermal_merit")
        _cvu           = _col("cvu_semana")
        df["thermal_merit_cost"]          = _thermal_merit.fillna(0) * _cvu.fillna(0)
        df["thermal_real_cost"]           = thermal_s.fillna(0) * _cvu.fillna(0)

        # thermal_inflex: usar gfom se disponível, senão disp_thermal como proxy
        _thermal_inflex = _col("thermal_inflex_gfom") if "thermal_inflex_gfom" in df.columns else pd.Series(0.0, index=df.index)
        df["thermal_inflex"] = _thermal_inflex
        df["Thermal_inflex_ratio"] = (_thermal_inflex / thermal_s.replace(0, np.nan)).clip(lower=0, upper=1)

        # Disponibilidade hidro sincronizada
        _disp_hydro = _col("disp_hydro") if "disp_hydro" in df.columns else hydro_s
        df["disp_sync_uhe"]   = _disp_hydro
        df["Hydro_preserved"] = (_disp_hydro - hydro_s.fillna(0)).clip(lower=0)
        df["Water_value_R$/h"]= df["Hydro_preserved"] * _col("cmo_dominante").fillna(0)

        # Curtailment total
        _curtail_s  = _col("curtail_solar").fillna(0)
        _curtail_w  = _col("curtail_wind").fillna(0)
        _curtail    = (_curtail_s + _curtail_w)
        df["curtail_total"]     = _curtail.where(_curtail > 0, np.nan)
        df["curtailment_loss"]  = _curtail * _col("pld").fillna(0)
        df["Curtailment_loss_R$/h"] = df["curtailment_loss"]

        # CVaR implícito = (PLD - CMO).clip(0)
        _pld_s = _col("pld")
        _cmo_s = _col("cmo_dominante")
        df["CVaR_implicit"] = (_pld_s - _cmo_s).clip(lower=0)
        df["cvar_implicit"] = df["CVaR_implicit"]
        _pld_teto = _pld_s.groupby(_pld_s.index.year).transform("max")
        _pld_no_teto = _pld_s < _pld_teto * 0.95
        df.loc[~_pld_no_teto, "cvar_implicit"] = np.nan

        # Risk Gap = CVaR - CVU
        df["risk_gap"] = (df["CVaR_implicit"] - _cvu).where(_cvu.notna(), np.nan)

        # Geração necessária e hidro required
        _mandatory = solar_s.fillna(0) + wind_s.fillna(0) + nuclear_s.fillna(0) + _thermal_inflex.fillna(0)
        _req_hydro = (load_s - _mandatory).clip(lower=0)
        _hydro_gap = hydro_s.fillna(0) - _req_hydro
        _tol = 1e-6
        df["system_state"] = np.where(
            _hydro_gap > _tol, "Hydro Preservation",
            np.where(_hydro_gap.abs() <= _tol, "Hydro Necessary", "Hydro Deficit")
        )

        # Transferências econômicas
        df["t_prudencia"]     = np.where(_cmo_s > _pld_s, df["Hydro_preserved"] * (_cmo_s - _pld_s), 0.0)
        df["t_hidro"]         = df["Water_value_R$/h"]
        df["t_eletric"]       = df["thermal_merit_cost"]
        df["t_sistemica"]     = df["geracao_total"] * (_cmo_s - _pld_s)
        df["t_total"]         = df["t_eletric"] + df["t_hidro"] + df["t_prudencia"] + df["t_sistemica"]
        df["infra_marginal_rent"] = df["sin_cost"] - df["t_total"]

        # GFOM %
        if "gfom" in df.columns and "gfom_ger" in df.columns:
            df["gfom_pct"] = (_col("gfom") / _col("gfom_ger").replace(0, np.nan) * 100).clip(lower=0, upper=100)

        # IPR = renov_disponível / load
        # ISR = renov_disponível / carga_líquida
        _avail_ren = _col("avail_ren") if "avail_ren" in df.columns else df["renov_total"]
        _avail_ren = _avail_ren.fillna(df["renov_total"])
        df["ipr"] = (_avail_ren / load_s.replace(0, np.nan)).clip(lower=0)
        _net_load_s = df["net_load"].replace(0, np.nan)
        df["isr"] = (_avail_ren / _net_load_s.abs().replace(0, np.nan)).clip(lower=0)

        # EAR % (já pode estar no df como ear_pct — garantir nome consistente)
        if "ear_pct" not in df.columns and "ear" in df.columns and "earmaxp" in df.columns:
            df["ear_pct"] = (df["ear"] / df["earmaxp"].replace(0, np.nan) * 100).clip(lower=0, upper=100)

        # ── Normalizações para Coerência Operativa (tabs[3]) ─────────────────
        # EAR_norm: EAR% / 100, clampado em [0, 1]
        if "ear_pct" in df.columns:
            df["EAR_norm"] = (df["ear_pct"] / 100.0).clip(lower=0, upper=1)
        else:
            df["EAR_norm"] = np.nan

        # ENA_norm: ENA armazenável / P90 dos últimos 365 dias (denominador estável)
        # Janela de 365 dias captura cheia + seca sem depender do período selecionado.
        # P90 (e não máximo) evita que valores extremos de cheia inflacionem o
        # denominador e comprimam artificialmente os scores em anos secos.
        if "ena_arm" in df.columns:
            # _ena_norm_denom calculado acima na query de ENA (P90 / 365 dias)
            _denom = _ena_norm_denom if (pd.notna(_ena_norm_denom) and _ena_norm_denom > 0) else np.nan
            if pd.notna(_denom):
                df["ENA_norm"] = (df["ena_arm"] / _denom).clip(lower=0, upper=1)
            else:
                # fallback: max da série presente no df (comportamento anterior)
                _fallback = df["ena_arm"].max()
                df["ENA_norm"] = (df["ena_arm"] / _fallback).clip(lower=0, upper=1)                     if pd.notna(_fallback) and _fallback > 0 else np.nan
        else:
            df["ENA_norm"] = np.nan

        # Load_norm: carga / capacidade sincronizada total
        _disp_total = _col("disp_total") if "disp_total" in df.columns else pd.Series(np.nan, index=df.index)
        if _disp_total.notna().any():
            df["Load_norm"] = (load_s / _disp_total.replace(0, np.nan)).clip(lower=0, upper=2)
        else:
            df["Load_norm"] = np.nan

    return _ensure_hourly(df)


def _build_hourly_df(_unused: Any = None) -> pd.DataFrame:
    return _build_hourly_df_cached(_core_cache_token())


def _get_section(section: str) -> Dict[str, Any]:
    """Compatibilidade — retorna dict vazio; dados vêm do Neon agora."""
    return {}



def _series_from_hourly(d: Dict[str, Any], name: str) -> pd.Series:
    if not isinstance(d, dict) or not d:
        return pd.Series(dtype=float, name=name)
    s = pd.Series(d, name=name)
    s.index = pd.to_datetime(s.index, errors="coerce")
    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    except Exception:
        pass
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()


def _series_from_operacao(records, value_key: str, name: str) -> pd.Series:
    if not isinstance(records, list) or not records:
        return pd.Series(dtype=float, name=name)
    df = pd.DataFrame(records)
    if "instante" not in df.columns or value_key not in df.columns:
        return pd.Series(dtype=float, name=name)
    df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
    df[value_key] = pd.to_numeric(df[value_key], errors="coerce")
    df = df.dropna(subset=["instante", value_key])
    s = df.groupby("instante")[value_key].sum().sort_index().rename(name)
    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    except Exception:
        pass
    return s



def _fmt_ptbr(value: Any, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        s = f"{float(value):,.{decimals}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def _fmt_money_compact(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000:
        return f"R$ {_fmt_ptbr(v/1_000_000, 2)} MM"
    if av >= 1_000:
        return f"R$ {_fmt_ptbr(v/1_000, 2)} k"
    return f"R$ {_fmt_ptbr(v, 2)}"


def _send_feedback(sender_email: str, message: str) -> tuple[bool, str]:
    """Envia email de feedback para maatriaenergia@gmail.com via SMTP Gmail.

    Requer variáveis de ambiente no Render:
        GMAIL_USER  — endereço Gmail remetente (ex: maatriaenergia@gmail.com)
        GMAIL_PASS  — senha de app de 16 dígitos gerada em
                      myaccount.google.com > Segurança > Senhas de app
    """
    gmail_user = os.getenv("GMAIL_USER", "")
    gmail_pass = os.getenv("GMAIL_PASS", "")
    if not gmail_user or not gmail_pass:
        return False, "Credenciais de email não configuradas no servidor."

    dest = "maatriaenergia@gmail.com"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[MAÁTria] Feedback de {sender_email or 'anônimo'}"
        msg["From"]    = gmail_user
        msg["To"]      = dest
        msg["Reply-To"] = sender_email if sender_email else gmail_user

        body_txt = (
            f"Remetente: {sender_email or 'não informado'}\n\n"
            f"Mensagem:\n{message}"
        )
        body_html = f"""
        <html><body style="font-family:sans-serif;color:#222">
          <h3 style="color:#c8a44d">Novo feedback — MAÁTria Energia</h3>
          <p><strong>Remetente:</strong> {sender_email or '<em>não informado</em>'}</p>
          <hr style="border-color:#c8a44d"/>
          <p style="white-space:pre-wrap">{message}</p>
        </body></html>
        """
        msg.attach(MIMEText(body_txt, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html",  "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, dest, msg.as_string())
        return True, "Mensagem enviada com sucesso!"
    except smtplib.SMTPAuthenticationError:
        return False, "Falha de autenticação. Verifique GMAIL_USER e GMAIL_PASS no Render."
    except Exception as e:
        return False, f"Erro ao enviar: {e}"


def _prepare_logo(path: Path) -> Optional[Path]:
    """Recorta bordas escuras do PNG para reduzir fundo/preenchimento visual."""
    if not path.exists():
        return None
    try:
        from PIL import Image

        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        # pixels não quase-pretos e não transparentes
        mask = (arr[:, :, 3] > 5) & ((arr[:, :, 0] > 20) | (arr[:, :, 1] > 20) | (arr[:, :, 2] > 20))
        if not mask.any():
            return path
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cropped = img.crop((max(0, x0 - 4), max(0, y0 - 4), min(img.width, x1 + 5), min(img.height, y1 + 5)))
        out = Path("data") / "emblema_maatria_trimmed.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out)
        return out
    except Exception:
        return path


def _kpi_card(label: str, value: str, border_color: str):
    st.markdown(
        f"""
        <div style='background:#131722;border:1px solid #2a2f3a;border-top:3px solid {border_color};
                    border-radius:12px;padding:10px 12px;height:95px;'>
          <div style='font-size:12px;color:#9ba3af;'>{label}</div>
          <div style='font-size:23px;color:#f3f4f6;font-weight:700;line-height:1.2'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _system_text(row: pd.Series) -> str:
    s = row.get("system_state")
    if isinstance(s, str) and s:
        return f"Regime {s} com carga líquida de {_fmt_ptbr(row.get('net_load', np.nan), 0)} MWmed."
    return "Sem dados suficientes para diagnóstico automático da hora selecionada."


def _plot_df(dff: pd.DataFrame) -> pd.DataFrame:
    out = dff.copy().reset_index()
    first_col = out.columns[0]
    if first_col != "instante":
        out = out.rename(columns={first_col: "instante"})
    return out


def _ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida qualquer série semihorária em base horária (média de :00 e :30)."""
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if out.empty:
        return out
    out["hora_ref"] = out.index.floor("h")
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in out.columns if c not in num_cols + ["hora_ref"]]
    agg_map = {c: "mean" for c in num_cols}
    agg_map.update({c: "last" for c in other_cols})
    out = out.groupby("hora_ref", as_index=True).agg(agg_map).sort_index()
    out.index.name = "instante"
    return out


def _latest_operational_dates(core: Dict[str, Any]) -> Dict[str, Optional[date]]:
    oper = core.get("operacao", {}) if isinstance(core, dict) else {}

    def _max_date_from_records(records: Any) -> Optional[date]:
        if not isinstance(records, list) or not records:
            return None
        df = pd.DataFrame(records)
        if "instante" not in df.columns:
            return None
        ts = pd.to_datetime(df["instante"], errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            return None
        return ts.max().date()

    load_records = ((oper.get("load", {}) or {}).get("sin", {}) or {}).get("serie", [])
    load_day = _max_date_from_records(load_records)

    generation = oper.get("generation", {}) or {}
    gen_days = []
    for payload in generation.values():
        day = _max_date_from_records((payload or {}).get("serie", []))
        if day is not None:
            gen_days.append(day)
    generation_day = max(gen_days) if gen_days else None

    return {"load": load_day, "generation": generation_day}




def _latest_operational_dates(core: Dict[str, Any]) -> Dict[str, Optional[date]]:
    oper = core.get("operacao", {}) if isinstance(core, dict) else {}

    def _max_date_from_records(records: Any) -> Optional[date]:
        if not isinstance(records, list) or not records:
            return None
        df = pd.DataFrame(records)
        if "instante" not in df.columns:
            return None
        ts = pd.to_datetime(df["instante"], errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            return None
        return ts.max().date()

    load_records = ((oper.get("load", {}) or {}).get("sin", {}) or {}).get("serie", [])
    load_day = _max_date_from_records(load_records)

    generation = oper.get("generation", {}) or {}
    gen_days = []
    for payload in generation.values():
        day = _max_date_from_records((payload or {}).get("serie", []))
        if day is not None:
            gen_days.append(day)
    generation_day = max(gen_days) if gen_days else None

    return {"load": load_day, "generation": generation_day}




def main():
    st.set_page_config(page_title="MAÁTria Energia", layout="wide", initial_sidebar_state="collapsed")

    st.markdown(
        """
        <style>
          .stApp { background-color:#0b0f14; color:#f3f4f6; }
          [data-testid="stSidebar"] { display:none !important; }
          .block-container { padding-top: 50px; padding-bottom: 0; margin-bottom: 0; }
          .fixed-header { position: fixed; top: 0; left:0; right:0; z-index:999; background:#0b0f14; }
          .full-bleed-line { height:0.1px; background:#c8a44d; width:100vw; margin-left:calc(50% - 50vw); }
          .tabs-layer { background: linear-gradient(180deg, #0b1222 0%, #070d1a 100%); padding:0.01rem 0.01rem 0.01rem 0.01rem; }
          label { color:#ffffff !important; font-weight:700 !important; }
          .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: thin; }
          .stTabs [data-baseweb="tab"] { color:#e5e7eb; border-radius:6px; padding:0.25rem 0.45rem; font-size:0.78rem; white-space:nowrap; }
          .stTabs [aria-selected="true"] { background:#152238 !important; color:#f8fafc !important; border:1px solid #c8a44d !important; }
          div[data-testid="stFormSubmitButton"] > button {
            background:#d4af37 !important; color:#111827 !important; font-weight:800 !important; border:0px solid #b38f2b !important;
          }
          div[data-testid="stFormSubmitButton"] > button:hover { background:#e3bf4c !important; color:#000 !important; }
          .cards-row { margin-bottom: 5px; }
                  /* NOVO: Reduzir espaçamento entre elementos */
          .element-container {
              margin-bottom: 1 !important;
          }
          .stMarkdown {
              margin-bottom: 1 !important;
          }
          hr {
              margin-top: 15px !important;
              margin-bottom: 6px !important;
          }
          div[data-testid="stVerticalBlock"] {
              gap: 0.75rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not _NEON_OK():
        import os as _os
        _db_url = _os.getenv("DATABASE_URL", "")
        _err = st.session_state.get("_neon_error", "sem detalhes")
        st.error("❌ Banco de dados Neon não configurado ou inacessível.")
        st.code(f"DATABASE_URL: {'definida' if _db_url else 'NÃO DEFINIDA'} | Erro: {_err}")
        return

    with st.spinner(
        "⏳ Carregando dados do banco… O tempo médio de carregamento é de **2 a 3 minutos** "
        "na primeira abertura. Por favor, aguarde."
    ):
        df = _build_hourly_df()

    if df.empty:
        st.warning("Sem séries horárias suficientes no core para renderizar o painel.")
        return

    min_d, max_d = df.index.min().date(), df.index.max().date()
    default_day = date.today() - pd.Timedelta(days=1)
    if default_day < min_d or default_day > max_d:
        default_day = max_d

    if "date_start" not in st.session_state:
        st.session_state["date_start"] = default_day
    if "date_end" not in st.session_state:
        st.session_state["date_end"] = default_day

    st.markdown("<div class='fixed-header'>", unsafe_allow_html=True)
    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    colc1, colc2, colc3 = st.columns([1.18, 1, 0.82])
    with colc2:
        logo = _prepare_logo(Path("streamlit/img/emblema_maatria.png"))
        if logo and logo.exists():
            st.image(str(logo), width=200)
        else:
            st.markdown("## MAÁTria Energia")

    with colc3:
        st.markdown(
            "<p style='color:#c8a44d;font-size:0.78rem;margin-bottom:4px;"
            "letter-spacing:0.04em;text-transform:uppercase;font-weight:600'>"
            "✉ Contato & Sugestões</p>",
            unsafe_allow_html=True,
        )
        with st.popover("Enviar mensagem", use_container_width=True):
            st.markdown(
                "<span style='color:#c8a44d;font-weight:600'>MAÁTria Energia</span> — "
                "sua mensagem chega diretamente à equipe.",
                unsafe_allow_html=True,
            )
            fb_email = st.text_input(
                "Seu e-mail (opcional)",
                placeholder="voce@exemplo.com",
                key="fb_email",
            )
            fb_msg = st.text_area(
                "Mensagem",
                placeholder="Sugestões, dúvidas, erros encontrados…",
                height=130,
                key="fb_msg",
            )
            if st.button("Enviar ✉", type="primary", key="fb_send"):
                if not fb_msg.strip():
                    st.warning("Escreva uma mensagem antes de enviar.")
                else:
                    with st.spinner("Enviando…"):
                        ok, info = _send_feedback(fb_email.strip(), fb_msg.strip())
                    if ok:
                        st.success(info)
                    else:
                        st.error(info)

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    analyze_clicked = False
    form_col, _ = st.columns([0.4, 0.6])
    with form_col:
        with st.form("period_form", clear_on_submit=False):
            # CSS específico para este formulário
            st.markdown("""
            <style>
            div[data-testid="stForm"] .stDateInput label {
                font-size: 0.7rem !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stForm"] .stDateInput input {
                font-size: 0.75rem !important;
                padding: 0.2rem 0.5rem !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button {
                font-size: 0.7rem !important;
                padding: 0.2rem 0.5rem !important;
            }
                        /* NOVO: Reduzir margem inferior do formulário */
            div[data-testid="stForm"] {
                margin-bottom: 0 !important;
                padding-bottom: 0.25 !important;
            }
            /* NOVO: Reduzir margem do container do formulário */
            .element-container:has(div[data-testid="stForm"]) {
                margin-bottom: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 0.8])
            with c1:
                dt_start = st.date_input("DE", value=st.session_state["date_start"], min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
            with c2:
                dt_end = st.date_input("ATÉ", value=st.session_state["date_end"], min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
            with c3:
                st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
                analyze_clicked = st.form_submit_button("ANALISAR", use_container_width=True)

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    tabs = st.tabs([
        "📸 Fotografia Operativa",
        "💰 Decomposição Econômica",
        "⚡ Curtailment & Restrições",
        "🧠 Coerência Operativa",
        "🔋 Simulação BESS",
        "📊 Matriz Horária do SIN",
        "📘 Metodologia & Glossário",
    ])

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    if analyze_clicked:
        if dt_start > dt_end:
            st.error("Período inválido: DE deve ser menor ou igual a ATÉ.")
        else:
            st.session_state["date_start"] = dt_start
            st.session_state["date_end"] = dt_end
            st.rerun()

    selected_start = st.session_state.get("date_start", default_day)
    selected_end = st.session_state.get("date_end", default_day)

    dff = df[(df.index.date >= selected_start) & (df.index.date <= selected_end)].copy()
    if dff.empty:
        st.warning("Não há dados para o período selecionado.")
        return

    latest_operational = {"load": None, "generation": None}
    if _NEON_OK():
        try:
            r = db_neon.fetchdf(
                "SELECT 'carga' AS tipo, MAX(din_instante)::date::text AS mx FROM curva_carga "
                "UNION ALL "
                "SELECT 'geracao', MAX(din_instante)::date::text FROM geracao_tipo_hora"
            )
            for _, row in r.iterrows():
                if row["mx"]:
                    latest_operational["load" if row["tipo"]=="carga" else "generation"] = pd.Timestamp(row["mx"]).date()
        except Exception:
            pass
    available_reference_days = [d for d in [latest_operational.get("load"), latest_operational.get("generation")] if d is not None]
    photo_day = max(available_reference_days) if available_reference_days else max_d
    if photo_day < selected_start or photo_day > selected_end:
        photo_day = selected_end
    dff_photo = dff[dff.index.date == photo_day].copy()
    if dff_photo.empty:
        dff_photo = dff.copy()

    dff = _ensure_hourly(dff)
    dff_photo = _ensure_hourly(dff_photo)

    current = dff.mean(numeric_only=True)
    current_state = dff["system_state"].dropna().iloc[-1] if "system_state" in dff.columns and not dff["system_state"].dropna().empty else "-"

    # Totais do período selecionado (soma hora a hora)
    total_sin_cost = pd.to_numeric(dff.get("sin_cost", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_prud = pd.to_numeric(dff.get("t_prudencia", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_agua = pd.to_numeric(dff.get("t_hidro", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_sistemica = pd.to_numeric(dff.get("t_sistemica", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_eletrica = pd.to_numeric(dff.get("t_eletric", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_infra_marginal = pd.to_numeric(dff.get("infra_marginal_rent", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_curt_loss = pd.to_numeric(dff.get("curtailment_loss", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_gfom = pd.to_numeric(dff.get("gfom_pct", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_isr = pd.to_numeric(dff.get("isr", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_ipr = pd.to_numeric(dff.get("ipr", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)

    kpis = [
        ("PLD médio", f"R$ {_fmt_ptbr(current.get('pld', np.nan),2)}", "#22c55e"),
        ("CMO dominante", f"R$ {_fmt_ptbr(current.get('cmo_dominante', np.nan),2)}", "#3b82f6"),
        ("Custo Total SIN", _fmt_money_compact(total_sin_cost), "#f59e0b"),
        ("Custo Prudência", _fmt_money_compact(total_prud), "#ef4444"),
        ("Custo Hídrico", _fmt_money_compact(total_agua), "#14b8a6"),
        ("Custo Elétrico", _fmt_money_compact(total_eletrica), "#FFEE00"),
        ("Custo Sistêmico", _fmt_money_compact(total_sistemica), "#004918"),
        ("Custo Infra-marginal", _fmt_money_compact(total_infra_marginal), "#3D0049"),
        ("Curtailment", f"{_fmt_ptbr(current.get('curtail_total', np.nan),2)} MWmed", "#a78bfa"),
        ("Valor (R$) Curtailment", _fmt_money_compact(total_curt_loss), "#eab308"),
        ("GFOM", _fmt_ptbr(total_gfom,2), "#38bdf8"),
        ("ISR", _fmt_ptbr(total_isr,2), "#f97316"),
        ("IPR", _fmt_ptbr(total_ipr,2), "#84cc16"),
        ("Risk Gap", _fmt_ptbr(current.get("risk_gap", np.nan),2), "#fb7185"),
        ("CVaR Implícito", f"R$ {_fmt_ptbr(current.get('cvar_implicit', np.nan),2)}", "#60a5fa"),
    ]

    for base in (0, 5, 10):
        cols = st.columns(5)
        for i in range(5):
            idx = base + i
            if idx < len(kpis):
                lab, val, color = kpis[idx]
                with cols[i]:
                    _kpi_card(lab, val, color)
        st.markdown("<div class='cards-row'></div>", unsafe_allow_html=True)

    st.info(f"Estado Operativo do SIN: **{current_state}** | Período: **{selected_start}** até **{selected_end}**")

    with st.expander("Ver tabela dos cards (hora a hora)", expanded=False):
        card_cols = [
            c
            for c in [
                "pld",
                "cmo_dominante",
                "sin_cost",
                "t_prudencia",
                "t_hidro",
                "t_eletric",
                "t_sistemica",
                "infra_marginal_rent",
                "curtailment_loss",
                "curtail_total"
                "gfom_pct",
                "isr",
                "ipr",
                "risk_gap",
                "cvar_implicit",
            ]
            if c in dff.columns
        ]
        st.dataframe(_plot_df(dff[card_cols]), width="stretch", height=320)

    with tabs[0]:
        _ref_dates = [d for d in (latest_operational.get("load"), latest_operational.get("generation")) if d]
        _last_date = max(_ref_dates).strftime("%d/%m/%Y") if _ref_dates else "N/D"
        st.caption(f"Dados extraídos até o dia **{_last_date}**.")
        st.caption("Montagem: séries horárias observadas de geração por fonte + carga e carga líquida (`Carga - (Solar + Eólica)`).")
        st.write(_system_text(current))

        # Período de um dia → fotografia (dff_photo); múltiplos dias → série completa (dff)
        _single_day = (selected_start == selected_end)
        _graf_df = dff_photo if _single_day else dff

        fig = go.Figure()
        labels = {
            "hydro": "Hidro", "thermal": "Térmica", "nuclear": "Nuclear", "solar": "Solar", "wind": "Eólica"
        }
        for src in ["hydro", "thermal", "nuclear", "solar", "wind"]:
            if src in _graf_df.columns:
                fig.add_bar(x=_graf_df.index, y=_graf_df[src], name=labels[src])
        if "carga_total" in _graf_df.columns:
            fig.add_scatter(x=_graf_df.index, y=_graf_df["carga_total"], name="Carga Total", mode="lines")
        if "net_load" in _graf_df.columns:
            fig.add_scatter(x=_graf_df.index, y=_graf_df["net_load"], name="Carga Líquida", mode="lines")
        _title = (
            f"Fotografia — {selected_start.strftime('%d/%m/%Y')}"
            if _single_day
            else f"Geração por fonte — {selected_start.strftime('%d/%m/%Y')} a {selected_end.strftime('%d/%m/%Y')}"
        )
        fig.update_layout(
            template="plotly_dark",
            barmode="stack",
            height=420,
            title=dict(text=_title, font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Ver dados do gráfico (hora a hora)"):
            plot_cols = [c for c in ["carga_total", "net_load", "solar", "wind", "hydro", "thermal", "nuclear"] if c in _graf_df.columns]
            st.dataframe(_plot_df(_graf_df[plot_cols]), width="stretch", height=280)

        # ── Gráfico PLD e CMO por submercado ────────────────────────────────────
        st.markdown("#### PLD e CMO por Submercado")
        _fig_pld = go.Figure()
        _SUB_COLORS = {"se": "#f59e0b", "ne": "#34d399", "s": "#60a5fa", "n": "#f472b6"}
        _SUB_LABELS = {"se": "SE", "ne": "NE", "s": "S", "n": "N"}
        _pld_added = False

        # PLD por submercado (já está no dff)
        for _sk, _color in _SUB_COLORS.items():
            _pld_col = f"pld_{_sk}"
            if _pld_col in dff.columns:
                _s = pd.to_numeric(dff[_pld_col], errors="coerce").dropna()
                if not _s.empty:
                    _fig_pld.add_trace(go.Scatter(
                        x=_s.index, y=_s.values,
                        name=f"PLD {_SUB_LABELS[_sk]}",
                        line=dict(color=_color, width=1.8, dash="solid"),
                        mode="lines",
                    ))
                    _pld_added = True

        # CMO por subsistema — query com filtro de período + normalização de id_subsistema
        # O campo id_subsistema pode conter: SE/SUDESTE/SECO, NE/NORDESTE, S/SUL, N/NORTE
        _cmo_pivot = pd.DataFrame()
        _cmo_added = False
        _cmo_err = None
        try:
            _ts_min = dff.index.min()
            _ts_max = dff.index.max()
            _cmo_raw = db_neon.fetchdf(
                "SELECT din_instante AS instante, "
                "  UPPER(TRIM(id_subsistema)) AS sub, "
                "  val_cmo "
                "FROM cmo "
                "WHERE din_instante IS NOT NULL "
                "  AND val_cmo IS NOT NULL "
                f" AND din_instante >= '{_ts_min}' "
                f" AND din_instante <= '{_ts_max}' "
                "ORDER BY din_instante"
            )
            if not _cmo_raw.empty:
                _cmo_raw["instante"] = pd.to_datetime(_cmo_raw["instante"], errors="coerce")
                _cmo_raw["val_cmo"]  = pd.to_numeric(_cmo_raw["val_cmo"], errors="coerce")
                _cmo_raw = _cmo_raw.dropna(subset=["instante", "val_cmo"])

                # Normalizar id_subsistema para chave curta (se/ne/s/n)
                _ID_NORM = {
                    "SE": "se", "SUDESTE": "se", "SECO": "se",
                    "NE": "ne", "NORDESTE": "ne",
                    "S":  "s",  "SUL": "s",
                    "N":  "n",  "NORTE": "n",
                }
                _cmo_raw["sk"] = _cmo_raw["sub"].map(_ID_NORM)
                _cmo_raw = _cmo_raw.dropna(subset=["sk"])

                # Pivotar: instante × subsistema → val_cmo
                _cmo_pivot = _cmo_raw.pivot_table(
                    index="instante", columns="sk", values="val_cmo", aggfunc="mean"
                )

                for _sk, _color in _SUB_COLORS.items():
                    if _sk in _cmo_pivot.columns:
                        _s = _cmo_pivot[_sk].dropna()
                        if not _s.empty:
                            _fig_pld.add_trace(go.Scatter(
                                x=_s.index, y=_s.values,
                                name=f"CMO {_SUB_LABELS[_sk]}",
                                line=dict(color=_color, width=1.2, dash="dash"),
                                mode="lines",
                            ))
                            _cmo_added = True
        except Exception as _e:
            _cmo_err = str(_e)

        if _pld_added or _cmo_added:
            _fig_pld.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,20,0.95)",
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                            font=dict(size=11)),
                yaxis=dict(title="R$/MWh", gridcolor="#1e293b"),
                xaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(_fig_pld, width="stretch")
            if _cmo_err:
                st.caption(f"⚠️ CMO não carregado: {_cmo_err}")
            with st.expander("Ver dados PLD/CMO (hora a hora)"):
                # Montar tabela com índice Timestamp comum
                _pld_cols = [col for col in ["pld_se","pld_ne","pld_s","pld_n","pld"] if col in dff.columns]
                _tbl = dff[_pld_cols].copy() if _pld_cols else pd.DataFrame(index=dff.index)
                _tbl.index = pd.to_datetime(_tbl.index, errors="coerce")
                _tbl = _tbl.rename(columns={"pld_se":"PLD SE","pld_ne":"PLD NE",
                                             "pld_s":"PLD S","pld_n":"PLD N","pld":"PLD SIN"})
                if not _cmo_pivot.empty:
                    _cmo_tbl = _cmo_pivot.copy()
                    _cmo_tbl.index = pd.to_datetime(_cmo_tbl.index, errors="coerce")
                    _cmo_tbl = _cmo_tbl.rename(columns={"se":"CMO SE","ne":"CMO NE",
                                                          "s":"CMO S","n":"CMO N"})
                    _tbl = _tbl.join(_cmo_tbl, how="outer")
                _tbl = _tbl[~_tbl.index.isna()].sort_index(ascending=False)
                _tbl.index.name = "instante"
                if not _tbl.empty:
                    st.dataframe(_tbl, width="stretch", height=260)

    with tabs[1]:
        pdf = _plot_df(dff)
        decomp_cols = [c for c in ["t_hidro", "t_sistemica", "t_prudencia", "t_eletric"] if c in pdf.columns]
        if decomp_cols:
            st.caption("Montagem: decomposição econômica horária `T_total = T_hidro + T_sistêmica + T_prudência`.")
            label_map = {
                "t_hidro": "Custo Hídrico", "t_sistemica": "Custo Sistêmico", "t_prudencia": "Custo Prudencial", "t_eletric": "Custo elétrico"
            }
            fig = go.Figure()
            fig = px.bar(
                pdf,
                x="instante",
                y=decomp_cols,
                template="plotly_dark",
                barmode="stack"
            )
            # adicionar linha
            fig.add_scatter(
                x=pdf["instante"],
                y=pdf["sin_cost"],
                mode="lines",
                name="Custo Total SIN",
                line=dict(width=3)
            ) 
            fig.update_layout(title="Decomposição horária empilhada (R$/h)")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver dados do gráfico (hora a hora)"):
                st.dataframe(pdf[["instante"] + decomp_cols], width="stretch", height=280)

        if {"thermal", "thermal_prudential_dispatch"}.issubset(dff.columns):
            g2 = _plot_df(dff[["thermal", "thermal_prudential_dispatch"]])
            thermal_labels = {
            "thermal": "Geração Térmica Total", "thermal_prudential_dispatch": "Geração Térmica Prudencial"
            }
            fig2 = px.line(g2, x="instante", y=["thermal", "thermal_prudential_dispatch"], template="plotly_dark", labels=thermal_labels)
            fig2.update_layout(title="Despacho térmico total vs despacho prudencial (MWmed)")
            st.plotly_chart(fig2, width="stretch")
            st.caption("A segunda curva mostra a parcela térmica associada à prudência operativa.")
            with st.expander("Ver dados do gráfico térmico (hora a hora)"):
                st.dataframe(g2, width="stretch", height=260)

        # ===============================
        # HEATMAP – INFRA MARGINAL RENT
        # ===============================


        if "infra_marginal_rent" in df.columns:

            st.subheader("Mapa estrutural — Renda Infra-Marginal do SIN")
            st.caption(
                "Mapa calculado sobre toda a base histórica disponível. "
                "Os dados são diários, mas o eixo Y marca as mudanças de mês."
            )

            heat_df = df.copy()
            
            # FILTRO DE OUTLIERS - remover valores menores que -200 milhões
            outliers_removidos = (heat_df["infra_marginal_rent"] < -400_000_000).sum()
            heat_df = heat_df[heat_df["infra_marginal_rent"] >= -400_000_000]
            
            if outliers_removidos > 0:
                st.info(f"🔍 Foram removidos {outliers_removidos} registros com valores inferiores a -200 milhões R$/h (outliers).")

            heat_df["data"] = heat_df.index.date
            heat_df["hora"] = heat_df.index.hour

            pivot = heat_df.pivot_table(
                index="data",
                columns="hora",
                values="infra_marginal_rent",
                aggfunc="mean"
            )

            # verificar se há dados após o filtro
            if pivot.empty:
                st.warning("⚠️ Não há dados disponíveis após a remoção dos outliers.")
            else:
                # converter índice para datetime para manipular meses
                pivot.index = pd.to_datetime(pivot.index)

                # localizar início de cada mês
                month_starts = pivot.index.to_series().groupby(
                    [pivot.index.year, pivot.index.month]
                ).first()

                y_ticks = pd.to_datetime(month_starts.values)
                y_labels = [d.strftime("%m-%Y") for d in y_ticks]

                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale="RdBu_r",
                        colorbar=dict(title="R$/h"),
                        xgap=2,   # espaço entre horas
                        ygap=0    # pequeno espaço entre dias
                    )
                )

                fig_heat.update_layout(
                    template="plotly_dark",
                    height=5000,
                    xaxis=dict(
                        title="Hora do dia",
                        tickmode="linear",
                        dtick=1
                    ),
                    yaxis=dict(
                        title="Data",
                        tickmode="array",
                        tickvals=y_ticks,
                        ticktext=y_labels
                    )
                )

                st.plotly_chart(fig_heat, width="stretch")

                with st.expander("Ver dados do mapa de calor (base completa)"):
                    st.dataframe(pivot, width="stretch", height=300)

    with tabs[2]:
        cdf = _plot_df(dff)
        cols = [c for c in ["curtail_solar", "curtail_wind", "curtail_total"] if c in cdf.columns]
        
        if cols:
            st.caption("Montagem: curtailment horário por fonte (solar/eólica) e total agregado.")
            
            _curtail_plot_cols = [c for c in ["curtail_solar", "curtail_wind"] if c in cdf.columns]
            
            if _curtail_plot_cols:
                # Verificar se é período de um único dia ou múltiplos dias
                is_single_day = (selected_start == selected_end)
                
                if is_single_day:
                    # Para um único dia: gráfico de barras normal
                    fig = px.bar(
                        cdf,
                        x="instante",
                        y=_curtail_plot_cols,
                        template="plotly_dark",
                        barmode="stack",
                        labels={"curtail_solar": "Solar", "curtail_wind": "Eólica", "value": "MW"},
                    )
                    fig.update_layout(
                        title=f"Curtailment horário - {selected_start.strftime('%d/%m/%Y')}",
                        xaxis_title="Hora do dia",
                        yaxis_title="MW",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    # Para múltiplos dias: duas visualizações complementares
                    
                    # 1. Heatmap para ver padrões diários/horários
                    st.subheader("Mapa de calor - Curtailment por hora do dia")
                    
                    # Preparar dados para heatmap
                    heat_curtail = dff[["curtail_total"]].copy() if "curtail_total" in dff.columns else dff[["curtail_solar", "curtail_wind"]].sum(axis=1).to_frame("curtail_total")
                    heat_curtail["data"] = heat_curtail.index.date
                    heat_curtail["hora"] = heat_curtail.index.hour
                    
                    pivot_curtail = heat_curtail.pivot_table(
                        index="data",
                        columns="hora",
                        values="curtail_total",
                        aggfunc="mean"
                    )
                    
                    fig_heat = go.Figure(
                        data=go.Heatmap(
                            z=pivot_curtail.values,
                            x=pivot_curtail.columns,
                            y=pivot_curtail.index,
                            colorscale="YlOrRd",
                            colorbar=dict(title="MW"),
                            xgap=1,
                            ygap=1,
                        )
                    )
                    
                    fig_heat.update_layout(
                        template="plotly_dark",
                        height=400,
                        xaxis=dict(title="Hora do dia", dtick=2),
                        yaxis=dict(title="Data"),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    st.caption("🔥 Células mais claras indicam maior curtailment. O heatmap revela padrões sazonais e horários.")
                    
                    # 2. Gráfico de área empilhada (melhor que barras para séries longas)
                    st.subheader("Série temporal - Curtailment por fonte")
                    
                    # Amostragem para não sobrecarregar se período for muito longo
                    plot_df = cdf.copy()
                    n_points = len(plot_df)
                    
                    if n_points > 1000:  # Se mais de ~40 dias
                        st.info(f"📊 Período extenso ({n_points} horas) - exibindo amostragem a cada 6h para melhor visualização")
                        # Amostrar a cada 6 horas
                        plot_df = plot_df.iloc[::6].copy()
                    
                    fig_area = px.area(
                        plot_df,
                        x="instante",
                        y=_curtail_plot_cols,
                        template="plotly_dark",
                        labels={
                            "curtail_solar": "Solar", 
                            "curtail_wind": "Eólica", 
                            "value": "MW",
                            "instante": "Data/Hora"
                        },
                        title=f"Curtailment - {selected_start.strftime('%d/%m/%Y')} a {selected_end.strftime('%d/%m/%Y')}",
                    )
                    fig_area.update_layout(
                        height=450,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_area, use_container_width=True)
                    
                    # Estatísticas do período
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_curtail = dff["curtail_total"].sum() / 1000  # em MWh
                        st.metric("Curtailment Total", f"{total_curtail:,.0f} MWh")
                    with col2:
                        media_curtail = dff["curtail_total"].mean()
                        st.metric("Média Horária", f"{media_curtail:.1f} MW")
                    with col3:
                        max_curtail = dff["curtail_total"].max()
                        st.metric("Pico Máximo", f"{max_curtail:.0f} MW")
                    with col4:
                        horas_com_curtail = (dff["curtail_total"] > 1).sum()
                        st.metric("Horas c/ Curtailment", f"{horas_com_curtail}")
                
                # Expander com dados hora a hora (sempre disponível)
                with st.expander("Ver dados detalhados (hora a hora)"):
                    display_cols = ["instante"] + cols
                    
                    # Formatar DataFrame para exibição
                    df_display = cdf[display_cols].copy()
                    if not df_display.empty:
                        # Arredondar valores numéricos
                        for col in cols:
                            if col in df_display.columns:
                                df_display[col] = df_display[col].round(2)
                        
                        # Ordenar por instante (mais recente primeiro)
                        df_display = df_display.sort_values("instante", ascending=False)
                        
                        st.dataframe(df_display, width="stretch", height=280)
                        
                        # Botão para download dos dados
                        csv = df_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"curtailment_{selected_start}_{selected_end}.csv",
                            mime="text/csv",
                        )
        else:
            st.info("ℹ️ Sem dados de curtailment no período selecionado.")
        
        st.caption("Distribuição por tipo de restrição disponível no painel horário do core quando fornecido pelo ONS.")

    with tabs[3]:
        # Ler os valores de normalizações diretamente do dff (df filtrado)
        # EAR_norm, ENA_norm e Load_norm são calculados hora a hora no df
        # e já expandidos corretamente para 24h/dia via resample+ffill
        def _last_valid(col: str) -> float:
            if col in dff.columns:
                s = pd.to_numeric(dff[col], errors="coerce").dropna()
                return float(s.iloc[-1]) if not s.empty else np.nan
            return np.nan

        _ear_norm   = _last_valid("EAR_norm")
        _ena_norm   = _last_valid("ENA_norm")
        _load_norm  = _last_valid("Load_norm")

        metrics = {
            "Risk Gap":      current.get("risk_gap", np.nan),
            "CVaR":          current.get("cvar_implicit", np.nan),
            "EAR_norm":      _ear_norm,
            "ENA_norm":      _ena_norm,
            "Load pressure": _load_norm,
        }

        def _safe(v: float, default: float = 0.0) -> float:
            return default if pd.isna(v) else float(v)

        metrics_norm = {
            "risk": np.tanh(_safe(metrics["Risk Gap"]) / 300),
            "cvar": _safe(metrics["CVaR"]) / 100,
            "ear":  1 - _safe(metrics["EAR_norm"], 0.5),   # 0.5 se ausente (neutro)
            "ena":  1 - _safe(metrics["ENA_norm"], 0.5),
            "load": abs(_safe(metrics["Load pressure"], 1.0) - 1),
        }

        score_vals = [abs(v) for v in metrics_norm.values() if pd.notna(v)]
        coherence = 100 * (1 - np.mean(score_vals)) if score_vals else np.nan
        coherence = float(np.clip(coherence, 0, 100)) if pd.notna(coherence) else np.nan
        color = "🟢" if pd.notna(coherence) and coherence >= 70 else (
                "🟡" if pd.notna(coherence) and coherence >= 40 else "🔴")

        st.metric("Métrica de Coerência do SIN", "-" if pd.isna(coherence) else f"{coherence:.1f}")
        st.markdown(f"Classificação: {color}")

        # Mostrar detalhamento das métricas com labels legíveis
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Métricas base**")
            st.json({
                "EAR_norm":      round(_ear_norm, 4)  if pd.notna(_ear_norm)  else None,
                "ENA_norm":      round(_ena_norm, 4)  if pd.notna(_ena_norm)  else None,
                "Load pressure": round(_load_norm, 4) if pd.notna(_load_norm) else None,
                "CVaR (R$/MWh)": round(float(metrics["CVaR"]), 2) if pd.notna(metrics["CVaR"]) else None,
                "Risk Gap":      round(float(metrics["Risk Gap"]), 2) if pd.notna(metrics["Risk Gap"]) else None,
            })
        with col_b:
            st.markdown("**Scores normalizados (0 = ideal)**")
            st.json({k: round(float(v), 4) if pd.notna(v) else None
                     for k, v in metrics_norm.items()})

        # Série temporal de EAR_norm e ENA_norm no período selecionado
        norm_cols = [c for c in ["EAR_norm", "ENA_norm", "Load_norm"] if c in dff.columns]
        if norm_cols:
            _norm_df = dff[norm_cols].dropna(how="all").reset_index().rename(columns={"index": "instante"})
            if not _norm_df.empty:
                st.caption("Evolução das normalizações no período selecionado")
                fig_norm = px.line(
                    _norm_df, x="instante", y=norm_cols,
                    template="plotly_dark",
                    labels={"value": "Valor normalizado (0–1)", "variable": "Série"},
                )
                fig_norm.update_layout(yaxis_range=[0, 1.1])
                st.plotly_chart(fig_norm, use_container_width=True)

    with tabs[4]:

        sim_shift = st.slider(
            "Percentual de deslocamento do curtailment solar para 19h–23h",
            0, 100, 0
        )
        sim = dff.copy()
        if sim.empty:
            st.warning("Sem dados suficientes para simulação.")
            st.stop()
        frac = sim_shift / 100.0
        solar_curt = sim.get("curtail_solar", pd.Series(0, index=sim.index)).fillna(0)
        # energia total deslocada
        energy_shift = solar_curt * frac
        night_hours = [19, 20, 21, 22, 23]
        night_mask = sim.index.hour.isin(night_hours)
        if night_mask.sum() == 0:
            st.warning("Não há horas noturnas no período selecionado.")
            st.stop()
        # distribuição uniforme
        per_hour = energy_shift.sum() / night_mask.sum()
        thermal = sim.get("thermal", pd.Series(0, index=sim.index)).fillna(0)
        hydro = sim.get("hydro", pd.Series(0, index=sim.index)).fillna(0)
        thermal_after = thermal.copy()
        hydro_after = hydro.copy()
        for ts in sim.index[night_mask]:
            remove_thermal = min(thermal_after.loc[ts], per_hour)
            thermal_after.loc[ts] -= remove_thermal
            remainder = per_hour - remove_thermal
            if remainder > 0:
                remove_hydro = min(hydro_after.loc[ts], remainder)
                hydro_after.loc[ts] -= remove_hydro
        # --------------------------
        # cálculo da geração BESS
        # --------------------------
        bess = (thermal - thermal_after) + (hydro - hydro_after)
        # dataframe antes
        before = pd.DataFrame({
            "hydro": hydro,
            "thermal": thermal,
            "nuclear": sim.get("nuclear", 0),
            "solar": sim.get("solar", 0),
            "wind": sim.get("wind", 0)
        })
        # dataframe depois
        after = pd.DataFrame({
            "hydro": hydro_after,
            "thermal": thermal_after,
            "nuclear": sim.get("nuclear", 0),
            "solar": sim.get("solar", 0),
            "wind": sim.get("wind", 0),
            "bess": bess
        })
        labels = {
            "hydro": "Hidro",
            "thermal": "Térmica",
            "nuclear": "Nuclear",
            "solar": "Solar",
            "wind": "Eólica",
            "bess": "BESS"
        }
        # --------------------------
        # gráfico antes
        # --------------------------
        st.subheader("Situação observada")
        fig_before = go.Figure()
        for src in ["hydro", "thermal", "nuclear", "solar", "wind"]:
            if src in before.columns:
                fig_before.add_bar(
                    x=before.index,
                    y=before[src],
                    name=labels[src]
                )
        fig_before.update_layout(
            template="plotly_dark",
            barmode="stack",
            height=400
        )
        st.plotly_chart(
            fig_before,
            width="stretch",
            key="bess_generation_before"
        )
        # --------------------------
        # gráfico depois
        # --------------------------
        st.subheader("Situação simulada (curtailment redistribuído)")
        fig_after = go.Figure()
        for src in ["hydro", "thermal", "nuclear", "solar", "wind", "bess"]:
            if src in after.columns:
                fig_after.add_bar(
                    x=after.index,
                    y=after[src],
                    name=labels[src]
                )
        fig_after.update_layout(
            template="plotly_dark",
            barmode="stack",
            height=400
        )
        st.plotly_chart(
            fig_after,
            width="stretch",
            key="bess_generation_after"
        )

    with tabs[5]:
        matrix_cols = [
            c for c in ["pld","cmo_dominante","load","net_load","hydro","thermal","nuclear","solar","wind","gfom_pct","curtail_total","ear","ena","risk_gap","system_state"] if c in dff.columns
        ]
        m = dff[matrix_cols].copy()
        if not m.empty:
            m["interpretacao"] = m.apply(_system_text, axis=1)
            st.dataframe(m, width="stretch", height=420)
            st.download_button("Exportar CSV", data=m.reset_index().to_csv(index=False).encode("utf-8"), file_name="matriz_horaria_sin.csv", mime="text/csv")

    with tabs[6]:

        st.markdown("### 📘 Metodologia & Glossário")
        st.caption("Guia conceitual da plataforma: como os indicadores são calculados e como interpretar a operação do SIN.")

        with st.expander("🎯 1) Propósito da Plataforma", expanded=False):
            st.markdown("""
A plataforma **não tem como objetivo prever o PLD**.

O foco é analisar a **coerência entre condições físicas do sistema e os sinais econômicos observados**, permitindo interpretar a operação do SIN em base **hora a hora**.

A análise cruza informações de:

- hidrologia
- disponibilidade de geração
- despacho térmico
- penetração de renováveis
- curtailment
- preços marginais (PLD e CMO)

**Fotografia Operativa do SIN**

é um diagnóstico instantâneo da condição **física, energética e econômica** do sistema.
""")
            st.info("Interprete esta aba como um guia de leitura do sistema elétrico, não como um modelo de previsão de preços.")

        with st.expander("⚙️ 2) Conceitos Fundamentais do SIN", expanded=False):
            st.markdown("""
**Carga (Demanda)**  
Energia total consumida pelo sistema em uma determinada hora.

**Geração**  
Energia efetivamente produzida pelas diferentes fontes do sistema.

**Carga Líquida**  
Parcela da demanda que precisa ser atendida por fontes **flexíveis** (hidrelétricas e térmicas).

**Fórmula**

`Carga Líquida = Carga − (Solar + Eólica)`

**Valor da Água**

Representa o custo de oportunidade de utilizar água armazenada nos reservatórios agora em vez de preservá-la para uso futuro.

Como aproximação operacional, utilizamos o **CMO (Custo Marginal de Operação)** como proxy desse valor.
""")

        with st.expander("📊 3) Métricas Principais", expanded=False):
            st.markdown("""
### GFOM (Geração Fora da Ordem de Mérito)

Indica a parcela da geração térmica despachada **fora da lógica econômica do mérito de custo**.

**Fórmula**

`GFOM = Térmica_GFOM / Térmica_Total`

**Interpretação típica**

- `< 5%` → despacho majoritariamente econômico  
- `5–15%` → despacho misto  
- `> 15%` → presença relevante de decisão operativa

---

### Curtailment

Energia renovável **disponível mas não utilizada pelo sistema**.

Principais causas:

- restrições de transmissão
- estabilidade elétrica
- saturação de geração
- inflexibilidade de usinas térmicas ou nucleares

**Leitura econômica**

energia de baixo custo que deixa de ser utilizada.

---

### IPR — Índice de Pressão Renovável

Mede o peso das renováveis sobre a demanda.

`IPR = Renovável Disponível / Carga`

---

### ISR — Índice de Saturação Renovável

Avalia a pressão renovável sobre a parcela flexível da demanda.

`ISR = Renovável Disponível / Carga Líquida`

Quando:

`ISR > 1`

há risco de **saturação estrutural de geração renovável**.

---

### EAR — Energia Armazenada

Estoque de energia contido nos reservatórios hidráulicos.

Representa a **segurança energética futura do sistema**.

---

### ENA — Energia Natural Afluente

Energia hidrológica que entra naturalmente no sistema por meio das vazões.

Interpretação:

- **ENA alta** → tendência de alívio hidrológico  
- **ENA baixa** → aumento do risco de escassez
""")

            st.warning("Interprete sempre IPR e ISR junto com o curtailment para distinguir excesso renovável de restrições elétricas.")

        with st.expander("💰 4) Decomposição Econômica do Sistema", expanded=False):
            st.markdown("""
A plataforma separa o custo horário do sistema em componentes econômicos.

### Estrutura central

`T_total = T_elétrico + T_hidro + T_prudência + T_sistêmica`

onde:

**T_elétrico — Custo Estrutural de Geração**

Representa o custo mínimo necessário para atender a carga considerando o despacho por mérito econômico.

Principal componente:

`Térmica por mérito × CVU médio`

---

**T_hidro — Custo Hidrológico**

Valor econômico associado ao uso da água armazenada.

A água funciona como um **ativo energético armazenável**, cujo valor é aproximado pelo **CMO**.

---

**T_prudência — Custo de Decisão Operativa**

Representa o custo adicional associado a decisões conservadoras do operador, como:

- preservação de reservatórios
- despacho térmico preventivo
- restrições operativas

---

**T_sistêmica — Ajuste Estrutural do Sistema**

Captura diferenças entre o valor econômico do mercado e o custo físico da geração.

Pode assumir valores positivos ou negativos dependendo da condição estrutural do sistema.
""")

        with st.expander("🛡️ 5) CVaR e Aversão ao Risco", expanded=False):
            st.markdown("""
O planejamento da operação considera cenários hidrológicos adversos.

Para isso são utilizados mecanismos de **aversão ao risco**, como o **CVaR (Conditional Value at Risk)**.

Exemplo de parametrização usada no setor:

`(15%, 40%)`

significando:

- análise dos **15% piores cenários hidrológicos**
- com **peso de 40% na decisão operativa**

Maior aversão ao risco tende a produzir:

- maior preservação hídrica
- maior despacho térmico
- maior pressão sobre o PLD

### CVaR Implícito Observado

A plataforma estima uma aproximação do valor implícito da aversão ao risco:

`CVaR_implícito = max(PLD − CMO, 0)`

Quando o **PLD atinge o teto regulatório**, o valor implícito não é observável diretamente.
""")

        with st.expander("📉 6) Risk Aversion Gap", expanded=False):
            st.markdown("""
O **Risk Aversion Gap** compara o nível de aversão ao risco observado com o custo médio da geração térmica.

**Definição**

`Risk Gap = CVaR_implícito − CVU_médio`

**Interpretação**

- **positivo** → operação conservadora  
- **próximo de zero** → operação neutra  
- **negativo** → sistema em regime de abundância energética
""")

        with st.expander("💧 7) Teste de Necessidade Hidráulica", expanded=False):
            st.markdown("""
Esse teste avalia quanto da geração hidráulica é **estruturalmente necessária** para atender a demanda.

### Passo 1 — Geração mandatória

`Renováveis + Nuclear + Térmica Inflexível`

### Passo 2 — Hidro necessária

`Hidro_necessária = Carga − Geração_mandatória`

### Passo 3 — Comparação com a geração observada

- `Hidro observada > Hidro necessária`  
→ sistema com forte presença hidráulica

- `Hidro observada < Hidro necessária`  
→ maior dependência térmica
""")

        with st.expander("🧾 8) Custo Econômico do SIN (R$/h)", expanded=False):
            st.markdown("""
A exposição econômica total do sistema pode ser aproximada por:

`Custo SIN = Carga × PLD`

Esse valor representa o **valor econômico da energia liquidada no mercado** naquela hora.

Ele não corresponde diretamente ao custo físico de geração.
""")

        with st.expander("📊 9) Infra-Marginal Rent", expanded=False):
            st.markdown("""
Chamamos de **Infra-Marginal Rent** a diferença entre:

`Valor econômico da energia`  
e  
`Custo físico estimado de produção`.

Na plataforma:

`Infra_marginal = Custo SIN − T_total`

### Interpretação

**Infra-marginal positivo**

O valor pago pelo mercado é maior que o custo físico de geração.

Isso ocorre tipicamente quando:

- o PLD é definido por usinas marginais caras
- grande parte da geração vem de fontes mais baratas (hidro, renováveis)

Nesse caso, **geradores recebem renda infra-marginal**.

---

**Infra-marginal negativo**

O valor econômico do mercado é inferior ao custo físico estimado.

Isso ocorre em situações como:

- excesso estrutural de oferta
- saturação renovável
- PLD muito baixo

Nesse caso, o sistema está operando com **sinais de preço comprimidos**.
""")

        with st.expander("🌡️ 10) Heatmap de Infra-Marginal ao Longo do Tempo", expanded=False):
            st.markdown("""
O mapa de calor apresentado no dashboard mostra a evolução histórica da diferença entre:

`Custo SIN` e `T_total`.

Cada célula representa **uma hora de operação do sistema**.

Esse gráfico permite identificar **regimes estruturais de mercado**, como:

**Regime de escassez**

PLD elevado e forte renda infra-marginal.

**Regime de abundância**

PLD baixo e compressão de receitas.

**Transições estruturais**

Mudanças de comportamento ao longo das estações hidrológicas ou da expansão renovável.

Por utilizar toda a base histórica disponível, o heatmap permite visualizar **padrões operativos recorrentes do SIN**.
""")

        with st.expander("🏷️ 11) Coerência Operativa do SIN", expanded=False):
            st.markdown("""
## Coerência Operativa do SIN

A aba **Coerência Operativa** avalia se o comportamento econômico observado no mercado
(PLD, CVaR, Risk Gap) é **internamente consistente com as condições físicas do sistema**
(armazenamento, afluência e pressão de carga). Um sistema coerente é aquele em que
os preços refletem fielmente o estado físico — sem distorções regulatórias, anomalias
de despacho ou ruídos de formação de preços.

---

### Normalizações — o que são e por que existem

Cada variável do SIN opera em uma escala diferente (MW, MWmês, R$/MWh, %).
Para combiná-las em um único indicador é necessário **normalizar** cada série para
o intervalo **[0, 1]**, onde 0 representa o extremo mais favorável e 1 o mais crítico.

| Variável | Fórmula de normalização | Interpretação do valor |
|---|---|---|
| **EAR_norm** | EAR% ÷ 100 | 0 = reservatório vazio · 1 = cheio |
| **ENA_norm** | ENA_arm ÷ max(ENA_arm histórico) | 0 = afluência mínima · 1 = máxima |
| **Load_norm** | Carga / Capacidade sincronizada | 0 = sistema folgado · 1 = no limite |
| **CVaR** | (PLD − CMO).clip(0) em R$/MWh | divergência entre preço e custo marginal |
| **Risk Gap** | CVaR − CVU semanal | prêmio de risco além do custo variável |

> **Nota sobre EAR_norm e ENA_norm:** por serem dados diários (publicados uma vez por dia
> pelo ONS), o mesmo valor é atribuído a todas as 24 horas do dia correspondente.
> Isso é metodologicamente correto — o armazenamento e a afluência são grandezas
> de fluxo diário, não horário.

---

### Scores normalizados — como funcionam

Cada variável é convertida em um **score de desvio** (quanto aquela dimensão contribui
para a incoerência do sistema). O score ideal é **0** (sem desvio); scores próximos de
**1** indicam pressão máxima naquela dimensão.

| Score | Fórmula | Lógica |
|---|---|---|
| **ear** | 1 − EAR_norm | Reservatórios baixos → maior risco |
| **ena** | 1 − ENA_norm | Afluência baixa → menor folga hídrica |
| **load** | abs(Load_norm − 1) | Carga muito próxima da capacidade → risco operativo |
| **cvar** | CVaR ÷ 100 | Divergência PLD/CMO como fração de R$100/MWh |
| **risk** | tanh(Risk Gap ÷ 300) | Prêmio de risco saturado em ±1 (300 R$/MWh como referência) |

A função `tanh` no score de risco evita que valores extremos de Risk Gap dominem
o indicador — ela comprime a escala de forma suave, preservando o sinal direcional
mas limitando o peso de outliers.

---

### Métrica de Coerência do SIN — score final

O score final é calculado como:

```
Coerência (%) = 100 × (1 − média dos scores)
```

**Interpretação:**

| Faixa | Sinal | Significado |
|---|---|---|
| 🟢 ≥ 70 | Coerente | Preços consistentes com o estado físico |
| 🟡 40–70 | Atenção | Alguma divergência entre sinal físico e econômico |
| 🔴 < 40 | Incoerente | Pressão severa ou descolamento significativo |

**Exemplo prático:** um sistema com EAR baixo (EAR_norm = 0,30), afluência reduzida
(ENA_norm = 0,40) e PLD muito acima do CMO (CVaR alto) terá scores individuais
elevados em todas as dimensões, resultando em Coerência < 40 — sinal de stress
sistêmico consistente, onde o preço alto reflete genuinamente as condições físicas.

Por outro lado, se o PLD estiver alto mas EAR e ENA estiverem em níveis confortáveis,
a Coerência também será baixa — mas por incoerência: o preço não encontra respaldo
nas condições físicas do reservatório, o que pode indicar distorção regulatória,
restrições de transmissão ou inflexibilidade térmica excessiva.

---

### Limitações

- **ENA_norm** usa o P90 dos últimos 365 dias como denominador, calculado sobre o
  histórico completo do Neon — não sobre o período selecionado na tela. Em bases
  com menos de 365 dias de história, o P90 recai sobre toda a série disponível.
- O score de CVaR usa R$100/MWh como unidade de referência implícita.
  Em cenários de PLD próximo ao teto regulatório, o CVaR pode ser estruturalmente
  alto sem refletir incoerência real.
- Dados ausentes (NaN) recebem score neutro de **0,5**, para não penalizar nem
  beneficiar a métrica na ausência de informação.
""")

        with st.expander("🧭 12) Como Interpretar o Dashboard", expanded=False):
            st.markdown("""
### Roteiro sugerido de leitura

1️⃣ selecione o período de análise  
2️⃣ observe os indicadores principais  
3️⃣ verifique o score de coerência operativa  
4️⃣ analise a decomposição econômica  
5️⃣ identifique causas de curtailment  
6️⃣ observe o heatmap histórico  
7️⃣ utilize a simulação BESS

### Objetivo final

Avaliar se o comportamento observado do **PLD** está **coerente com as condições físicas do SIN**.
""")

            st.success("Combine sempre sinais físicos (carga, geração, reservatórios) com sinais econômicos (PLD, CMO e custos).")

if __name__ == "__main__":

    main()