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
        # tipo_geracao já vem normalizado: solar/wind/hydro/thermal/nuclear/other
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
        ear_df["instante"] = pd.to_datetime(ear_df["instante"], errors="coerce")
        ear_df = ear_df.dropna(subset=["instante"])
        ear_df["ear_pct"] = ear_df["ear"] / ear_df["earmaxp"].replace(0, np.nan) * 100
        _join(_ts(pd.to_numeric(ear_df.set_index("instante")["ear_pct"], errors="coerce").rename("ear")))
    del ear_df

    # ── ENA diário ───────────────────────────────────────────────────────────
    ena_df = db_neon.fetchdf(
        "SELECT ena_data AS instante, "
        "  SUM(ena_bruta_regiao_mwmed) AS ena_bruta, "
        "  SUM(ena_armazenavel_regiao_mwmed) AS ena_arm "
        "FROM ena_diario_subsistema WHERE ena_data IS NOT NULL "
        "GROUP BY ena_data ORDER BY ena_data"
    )
    if not ena_df.empty:
        ena_df["instante"] = pd.to_datetime(ena_df["instante"], errors="coerce")
        ena_df = ena_df.dropna(subset=["instante"]).set_index("instante")
        _join(_ts(pd.to_numeric(ena_df["ena_bruta"].rename("ena_bruta"), errors="coerce")))
        _join(_ts(pd.to_numeric(ena_df["ena_arm"].rename("ena_arm"), errors="coerce")))
    del ena_df

    # ── Disponibilidade por tipo ─────────────────────────────────────────────
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
        for tipo in ("hydro", "thermal", "nuclear", "solar", "wind"):
            sub = disp_df[disp_df["tipo_geracao"] == tipo]
            if not sub.empty:
                s = sub.groupby("instante")["disp_sinc"].sum().rename(f"disp_{tipo}")
                _join(_ts(pd.to_numeric(s, errors="coerce")))
        disp_total = disp_df.groupby("instante")["disp_sinc"].sum().rename("disp_total")
        _join(_ts(pd.to_numeric(disp_total, errors="coerce")))
    del disp_df

    # ── Restrição renovável (curtailment) ────────────────────────────────────
    restr_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, fonte, "
        "  SUM(val_geracao) AS gerado, SUM(val_geracaolimitada) AS limitado, "
        "  SUM(val_disponibilidade) AS disponivel "
        "FROM restricao_renovavel "
        "WHERE din_instante IS NOT NULL "
        "GROUP BY din_instante, fonte ORDER BY din_instante"
    )
    if not restr_df.empty:
        restr_df["instante"] = pd.to_datetime(restr_df["instante"], errors="coerce")
        restr_df = restr_df.dropna(subset=["instante"])
        for fonte, sfx in (("solar", "solar"), ("wind", "wind")):
            sub = restr_df[restr_df["fonte"] == fonte].set_index("instante")
            if not sub.empty:
                curtail = (sub["limitado"] - sub["gerado"]).clip(lower=0).rename(f"curtail_{sfx}")
                fator = (sub["gerado"] / sub["disponivel"].replace(0, np.nan) * 100).rename(f"fator_cap_{sfx}")
                _join(_ts(pd.to_numeric(curtail, errors="coerce")))
                _join(_ts(pd.to_numeric(fator, errors="coerce")))
    del restr_df

    # ── Despacho GFOM ────────────────────────────────────────────────────────
    gfom_df = db_neon.fetchdf(
        "SELECT din_instante AS instante, "
        "  val_verifgeracao AS gfom_ger, "
        "  val_verifconstrainedoff AS constrained_off, "
        "  val_verifinflexibilidade AS thermal_inflex, "
        "  val_verifordemmerito AS thermal_merit, "
        "  val_verifgfom AS gfom "
        "FROM despacho_gfom WHERE din_instante IS NOT NULL ORDER BY din_instante"
    )
    if not gfom_df.empty:
        gfom_df["instante"] = pd.to_datetime(gfom_df["instante"], errors="coerce")
        gfom_df = gfom_df.dropna(subset=["instante"]).set_index("instante")
        for col in ("gfom_ger", "constrained_off", "thermal_inflex", "thermal_merit", "gfom"):
            if col in gfom_df.columns:
                _join(_ts(pd.to_numeric(gfom_df[col].rename(col), errors="coerce")))
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
        s = cvu_df.set_index("instante")["cvu_semana"].rename("cvu_semana")
        _join(_ts(pd.to_numeric(s, errors="coerce")))
    del cvu_df

    # ── Intercâmbio inter-subsistema ─────────────────────────────────────────
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
        for par, grp in itc_df.groupby("par"):
            s = grp.set_index("instante")["mw"].rename(f"itc_{par.lower()}")
            _join(_ts(pd.to_numeric(s, errors="coerce")))
        itc_total = itc_df.groupby("instante")["mw"].sum().rename("itc_total")
        _join(_ts(pd.to_numeric(itc_total, errors="coerce")))
    del itc_df

    # ── Colunas derivadas ─────────────────────────────────────────────────────
    if not df.empty:
        df = df.sort_index()
        z = pd.Series(0.0, index=df.index)

        def _col(name):
            return pd.to_numeric(df[name], errors="coerce") if name in df.columns else z

        load_s  = _col("load")
        solar_s = _col("solar")
        wind_s  = _col("wind")

        df["net_load"]    = load_s - solar_s.fillna(0) - wind_s.fillna(0)
        df["carga_total"] = load_s

        # SIN_cost ponderado por submercado:
        # custo_total = load_se*pld_se + load_ne*pld_ne + load_s*pld_s + load_n*pld_n
        _sub_pairs = [("se", "se"), ("ne", "ne"), ("s", "s"), ("n", "n")]
        _sin_cost = pd.Series(0.0, index=df.index)
        _any_sub = False
        for _sk, _pk in _sub_pairs:
            _lc, _pc = f"load_{_sk}", f"pld_{_pk}"
            if _lc in df.columns and _pc in df.columns:
                _sin_cost += _col(_lc).fillna(0) * _col(_pc).fillna(0)
                _any_sub = True
        if not _any_sub and "load" in df.columns and "pld" in df.columns:
            # Fallback: SIN total * PLD médio
            _sin_cost = _col("load").fillna(0) * _col("pld").fillna(0)
        df["SIN_cost_R$/h"] = _sin_cost.where(_sin_cost > 0, np.nan)

        # Custo médio ponderado R$/MWh (custo_total / carga_total)
        df["pld_ponderado"] = (df["SIN_cost_R$/h"] / load_s.replace(0, np.nan)).where(load_s > 0, np.nan)

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
          .block-container { padding-top: 40px; }
          .fixed-header { position: fixed; top: 0; left:0; right:0; z-index:999; background:#0b0f14; }
          .full-bleed-line { height:0.1px; background:#c8a44d; width:100vw; margin-left:calc(50% - 50vw); }
          .tabs-layer { background: linear-gradient(180deg, #0b1222 0%, #070d1a 100%); padding:0.01rem 0.01rem 0.01rem 0.01rem; }
          label { color:#ffffff !important; font-weight:700 !important; }
          .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: thin; }
          .stTabs [data-baseweb="tab"] { color:#e5e7eb; border-radius:6px; padding:0.25rem 0.45rem; font-size:0.78rem; white-space:nowrap; }
          .stTabs [aria-selected="true"] { background:#152238 !important; color:#f8fafc !important; border:1px solid #c8a44d !important; }
          div[data-testid="stFormSubmitButton"] > button {
            background:#d4af37 !important; color:#111827 !important; font-weight:800 !important; border:1px solid #b38f2b !important;
          }
          div[data-testid="stFormSubmitButton"] > button:hover { background:#e3bf4c !important; color:#000 !important; }
          .cards-row { margin-bottom: 5px; }
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

    colc1, colc2, colc3 = st.columns([1, 1, 1])
    with colc2:
        logo = _prepare_logo(Path("streamlit/img/emblema_maatria.png"))
        if logo and logo.exists():
            st.image(str(logo), width=200)
        else:
            st.markdown("## MAÁTria Energia")

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
                margin-bottom: 2px !important;
            }
            div[data-testid="stForm"] .stDateInput input {
                font-size: 0.75rem !important;
                padding: 0.2rem 0.5rem !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button {
                font-size: 0.7rem !important;
                padding: 0.2rem 0.5rem !important;
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
        fig = go.Figure()
        labels = {
            "hydro": "Hidro", "thermal": "Térmica", "nuclear": "Nuclear", "solar": "Solar", "wind": "Eólica"
        }
        for src in ["hydro", "thermal", "nuclear", "solar", "wind"]:
            if src in dff_photo.columns:
                fig.add_bar(x=dff_photo.index, y=dff_photo[src], name=labels[src])
        if "carga_total" in dff_photo.columns:
            fig.add_scatter(x=dff_photo.index, y=dff_photo["carga_total"], name="Carga Total", mode="lines")
        if "net_load" in dff_photo.columns:
            fig.add_scatter(x=dff_photo.index, y=dff_photo["net_load"], name="Carga Líquida", mode="lines")
        fig.update_layout(template="plotly_dark", barmode="stack", height=420)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Ver dados do gráfico (hora a hora)"):
            plot_cols = [c for c in ["carga_total", "net_load", "solar", "wind", "hydro", "thermal", "nuclear"] if c in dff_photo.columns]
            st.dataframe(_plot_df(dff_photo[plot_cols]), width="stretch", height=280)

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

            heat_df["data"] = heat_df.index.date
            heat_df["hora"] = heat_df.index.hour

            pivot = heat_df.pivot_table(
                index="data",
                columns="hora",
                values="infra_marginal_rent",
                aggfunc="mean"
            )

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
            fig = go.Figure()
            fig = px.bar(
                pdf,
                x="instante",
                y=["curtail_solar", "curtail_wind"],
                template="plotly_dark",
                barmode="stack"
            )
            #fig = px.bar(cdf, x="instante", y=cols, template="plotly_dark", barmode="group")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver dados do gráfico (hora a hora)"):
                st.dataframe(cdf[["instante"] + cols], width="stretch", height=280)
        st.caption("Distribuição por tipo de restrição disponível no painel horário do core quando fornecido pelo ONS.")

    with tabs[3]:
        metrics = {
            "Risk Gap": current.get("risk_gap", np.nan),
            "CVaR": current.get("cvar_implicit", np.nan),
            "EAR_norm": np.nan,
            "ENA_norm": np.nan,
            "Load pressure": np.nan,
        }
        metrics_norm = {
            "risk": np.tanh(metrics["Risk Gap"]/300),
            "cvar": metrics["CVaR"]/100,
            "ear": 1-metrics["EAR_norm"],
            "ena": 1-metrics["ENA_norm"],
            "load": abs(metrics["Load pressure"]-1),
        }

        norm = {}
        if norm and not dff.empty:
            tkey = dff.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            metrics["EAR_norm"] = (norm.get("EAR_norm") or {}).get(tkey, np.nan)
            metrics["ENA_norm"] = (norm.get("ENA_norm") or {}).get(tkey, np.nan)
            metrics["Load pressure"] = (norm.get("Load_norm") or {}).get(tkey, np.nan)
        score_vals = [abs(v) for v in metrics_norm.values() if pd.notna(v)]
        coherence = 100*(1-np.mean(score_vals))
        coherence = np.clip(coherence,0,100)
        color = "🟢" if coherence >= 70 else ("🟡" if coherence >= 40 else "🔴")
        st.metric("Métrica de Coerência do SIN", "-" if pd.isna(coherence) else f"{coherence:.1f}")
        st.markdown(f"Classificação: {color}")
        st.json(metrics)

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

        with st.expander("🏷️ 11) Classificação Operativa do SIN", expanded=False):
            st.markdown("""
A plataforma classifica o estado do sistema em regimes operativos.

**Escassez Hidrológica**

Baixo armazenamento e maior dependência térmica.

---

**Preservação Hídrica**

Estratégia deliberada de poupar reservatórios.

---

**Saturação Renovável**

Excesso instantâneo de geração renovável.

---

**Stress Operativo**

Sinais simultâneos de risco físico e pressão econômica.

---

**Equilíbrio Estrutural**

Operação estável sem pressões relevantes.
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