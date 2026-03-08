"""
Core analysis utilities for Kintuadi Energy.

CORE = visão sistêmica do SIN
- ONS (CSV) como fonte física primária
- CCEE como fonte econômica

VERSÃO REVISADA: Análise térmica com dupla perspectiva (sistema vs gerador)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import unicodedata

import pandas as pd
import numpy as np

try:
    import duckdb
except Exception:
    duckdb = None

# =====================================================================
# CONSTANTES REGULATÓRIAS - ANEEL/CCEE 2025
# =====================================================================
PLD_PISO = 57.31  # R$/MWh
PLD_TETO_ESTRUTURAL = 785.27  # R$/MWh (média semanal)
PLD_TETO_HORARIO = 1611.04  # R$/MWh (máximo horário)

# =====================================================================
# Utilities
# =====================================================================



_DUCKDB_PATH = os.path.join("data", "kintuadi.duckdb")

def _duckdb_connect() -> Optional[Any]:
    if duckdb is None:
        return None
    if not os.path.exists(_DUCKDB_PATH):
        return None
    try:
        return duckdb.connect(_DUCKDB_PATH, read_only=True)
    except Exception:
        return None

def _duckdb_table_exists(con: Any, table_name: str) -> bool:
    try:
        q = "SELECT 1 FROM information_schema.tables WHERE lower(table_name)=lower(?) LIMIT 1"
        return con.execute(q, [table_name]).fetchone() is not None
    except Exception:
        return False

def _duckdb_num_expr(col: str) -> str:
    """
    Conversão numérica tolerante a formatos PT-BR e EN:
    - '1.234,56' -> 1234.56
    - '1234.56'   -> 1234.56
    - '0,00E+00'  -> 0.0
    """
    raw = f"TRIM(CAST({col} AS VARCHAR))"
    normalized = (
        f"CASE WHEN INSTR({raw}, ',') > 0 "
        f"THEN REPLACE(REPLACE({raw}, '.', ''), ',', '.') "
        f"ELSE {raw} END"
    )
    return f"TRY_CAST({normalized} AS DOUBLE)"

def _duckdb_date_expr(col: str) -> str:
    return (
        f"COALESCE("
        f"TRY_CAST({col} AS TIMESTAMP), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d/%m/%Y %H:%M:%S'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d/%m/%Y %H:%M'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d/%m/%Y'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d-%m-%Y %H:%M:%S'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d-%m-%y %H:%M:%S'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d-%m-%Y'), "
        f"TRY_STRPTIME(CAST({col} AS VARCHAR), '%d-%m-%y')"
        f")"
    )




def _duckdb_fetchdf(sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
    con = _duckdb_connect()
    if con is None:
        return pd.DataFrame()
    try:
        return con.execute(sql, params or []).fetchdf()
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def _safe_get(dct: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur = dct
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _extract_sources(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "ons": raw.get("sources", {}).get("ons", {}),
        "ccee": raw.get("sources", {}).get("ccee", {}),
    }

def _find_ons_csv(ons: Dict[str, Any], dataset_prefix: str) -> Optional[str]:
    """Modo DuckDB-only: leitura direta de CSV desabilitada no core_analysis."""
    return None


def _find_ons_csv_all(ons: Dict[str, Any], dataset_prefix: str) -> List[str]:
    """Modo DuckDB-only: leitura direta de CSV desabilitada no core_analysis."""
    return []

def _extract_ccee_records(obj: Any) -> List[Dict[str, Any]]:
    if not obj:
        return []
    if isinstance(obj, dict):
        return obj.get("records", []) or obj.get("data", []) or []
    if isinstance(obj, list):
        return obj
    return []


def _status_from_records(records: List[Dict[str, Any]]) -> str:
    return "disponível" if records else "indisponível"


# =====================================================================
# Hidrologia
# =====================================================================

def _hydrology_status(ear: Optional[float]) -> Dict[str, Any]:
    if ear is None:
        return {"classe": "dados ausentes", "descricao": "EAR não disponível."}

    if ear < 40:
        c = "crítico"
    elif ear < 55:
        c = "alerta"
    elif ear < 70:
        c = "atenção"
    elif ear < 85:
        c = "confortável"
    else:
        c = "abundante"

    return {
        "classe": c,
        "descricao": "Classificação baseada no EAR médio dos subsistemas.",
    }


def _compute_hydrology_from_csv(ons: Dict[str, Any]) -> Dict[str, Any]:
    """Hidrologia em modo DuckDB-only."""
    ear_medio = ena_media = tendencia = None
    try:
        ear_by_sub, ena_by_sub = _load_ear_ena_monthly_by_submercado(ons)
        if ear_by_sub:
            df_ear = pd.concat(ear_by_sub, axis=1)
            ear_mensal = df_ear.mean(axis=1, skipna=True).dropna().sort_index()
            if not ear_mensal.empty:
                ear_medio = float(ear_mensal.mean())
                recent = float(ear_mensal.tail(3).mean())
                past = float(ear_mensal.tail(12).mean())
                tendencia = float(recent - past) if past else None
        if ena_by_sub:
            df_ena = pd.concat(ena_by_sub, axis=1)
            ena_mensal = df_ena.mean(axis=1, skipna=True).dropna().sort_index()
            if not ena_mensal.empty:
                ena_media = float(ena_mensal.mean())
    except Exception:
        pass

    return {
        "ear_medio": ear_medio,
        "ena_media": ena_media,
        "tendencia": tendencia,
        "classificacao": _hydrology_status(ear_medio),
    }


# =====================================================================
# Operação horária (fonte única: Open Data histórico)
# =====================================================================

def _normalize_power_to_mw(series: pd.Series) -> pd.Series:
    """
    Normaliza potência para MW quando os dados parecem estar em Watts.
    Heurística: medianas muito altas (>1e6) são tratadas como W.
    """
    if series.empty:
        return series

    med = series.dropna().abs().median() if not series.dropna().empty else 0
    if med > 1_000_000:
        return series / 1_000_000.0
    return series


def _extract_open_data_historical_operation(ons: Dict[str, Any]) -> Dict[str, Any]:
    """Consolida operação histórica via DuckDB (geracao_usina_horaria + curva_carga)."""
    generation: Dict[str, Dict[str, Any]] = {}
    load: Dict[str, Dict[str, Any]] = {}

    con = _duckdb_connect()
    if con is None:
        return {"generation": generation, "load": load, "status": "indisponível"}

    try:
        if _duckdb_table_exists(con, "geracao_usina_horaria"):
            q = f"""
                SELECT
                    {_duckdb_date_expr('din_instante')} AS din_instante,
                    UPPER(TRIM(CAST(nom_tipousina AS VARCHAR))) AS fonte,
                    {_duckdb_num_expr('val_geracao')} AS val_geracao
                FROM geracao_usina_horaria
                WHERE din_instante IS NOT NULL
            """
            g = con.execute(q).fetchdf()
            if not g.empty:
                g = g.dropna(subset=["din_instante", "fonte", "val_geracao"])
                g["val_geracao"] = _normalize_power_to_mw(pd.to_numeric(g["val_geracao"], errors="coerce"))
                g = g.dropna(subset=["val_geracao"])
                for fonte, grp in g.groupby("fonte"):
                    s = grp.groupby("din_instante")["val_geracao"].sum().sort_index()
                    generation[fonte.lower()] = {
                        "media": float(s.mean()),
                        "max": float(s.max()),
                        "min": float(s.min()),
                        "rampa_max": float(s.diff().abs().max()) if len(s) > 1 else 0.0,
                        "serie": [{"instante": i.strftime('%Y-%m-%d %H:%M:%S'), "geracao": float(v)} for i, v in s.items()],
                    }

        if _duckdb_table_exists(con, "curva_carga"):
            q = f"""
                SELECT
                    {_duckdb_date_expr('din_instante')} AS din_instante,
                    TRIM(CAST(id_subsistema AS VARCHAR)) AS id_subsistema,
                    {_duckdb_num_expr('val_cargaenergiahomwmed')} AS carga
                FROM curva_carga
                WHERE din_instante IS NOT NULL
            """
            c = con.execute(q).fetchdf()
            if not c.empty:
                c = c.dropna(subset=["din_instante", "id_subsistema", "carga"])
                c["submercado"] = c["id_subsistema"].map(_normalize_submercado_name)
                c = c.dropna(subset=["submercado"])
                for sm, grp in c.groupby("submercado"):
                    s = grp.groupby("din_instante")["carga"].sum().sort_index()
                    load[sm.lower()] = {
                        "media": float(s.mean()),
                        "max": float(s.max()),
                        "min": float(s.min()),
                        "rampa_max": float(s.diff().abs().max()) if len(s) > 1 else 0.0,
                        "serie": [{"instante": i.strftime('%Y-%m-%d %H:%M:%S'), "carga": float(v)} for i, v in s.items()],
                    }
                s_sin = c.groupby("din_instante")["carga"].sum().sort_index()
                load["sin"] = {
                    "media": float(s_sin.mean()),
                    "max": float(s_sin.max()),
                    "min": float(s_sin.min()),
                    "rampa_max": float(s_sin.diff().abs().max()) if len(s_sin) > 1 else 0.0,
                    "serie": [{"instante": i.strftime('%Y-%m-%d %H:%M:%S'), "carga": float(v)} for i, v in s_sin.items()],
                }
    except Exception:
        pass
    finally:
        con.close()

    return {"generation": generation, "load": load, "status": "disponível" if (generation or load) else "indisponível"}



# =====================================================================
# CURTAILMENT RENOVÁVEL (ONS)
# =====================================================================

def _compute_curtailment_from_csv(
    ons: Dict[str, Any],
    dataset_name: str,
    col_estimada: str,
    col_verificada: str,
    col_flag_invalido: Optional[str] = None,
) -> Dict[str, Any]:
    table_name = re.sub(r"[^a-z0-9_]", "", dataset_name.lower())

    def _finalize(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"status": "indisponível"}

        df = df.dropna(subset=["din_instante", "curtailment_abs"]) 
        df = df[df["curtailment_abs"] >= 0]
        if df.empty:
            return {"status": "indisponível"}

        serie = df.groupby("din_instante")["curtailment_abs"].sum().sort_index()
        total_curtail = float(df["curtailment_abs"].sum())

        disponivel_total = float(pd.to_numeric(df.get("disponivel", pd.Series(dtype=float)), errors="coerce").dropna().sum()) if "disponivel" in df.columns else 0.0
        verificada_total = float(pd.to_numeric(df.get("verificada", pd.Series(dtype=float)), errors="coerce").dropna().sum()) if "verificada" in df.columns else 0.0

        # Breakdown por código da restrição (quando disponível)
        restricao_por_codigo = {}
        if "cod_razaorestricao" in df.columns:
            dfr = df.copy()
            dfr["cod_razaorestricao"] = dfr["cod_razaorestricao"].astype(str).str.strip().str.upper()
            dfr = dfr[dfr["cod_razaorestricao"].notna() & (dfr["cod_razaorestricao"] != "")]
            if not dfr.empty:
                grp = dfr.groupby("cod_razaorestricao", as_index=False)["curtailment_abs"].sum()
                restricao_por_codigo = {str(r["cod_razaorestricao"]): float(r["curtailment_abs"]) for _, r in grp.iterrows()}

        return {
            "status": "disponível",
            "curtailment_total_mwh": total_curtail,
            "curtailment_total_mwmes": total_curtail,
            "geracao_disponivel_total_mwh": disponivel_total,
            "geracao_realizada_total_mwh": verificada_total,
            "geracao_verificada_total_mwmes": verificada_total,
            "curtailment_pct_total": float(total_curtail / disponivel_total) if disponivel_total > 0 else None,
            "curtailment_medio_hora": float(serie.mean()) if not serie.empty else 0,
            "curtailment_max_hora": float(serie.max()) if not serie.empty else 0,
            "restricao_por_codigo_mwmes": restricao_por_codigo,
            "serie": serie.reset_index().rename(columns={"din_instante": "instante", "curtailment_abs": "valor"}).to_dict("records"),
        }

    # Prioridade: DuckDB
    con = _duckdb_connect()
    if con is not None and _duckdb_table_exists(con, table_name):
        try:
            cols_info = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            cols = {str(c[1]).lower(): str(c[1]) for c in cols_info}

            has_limitada = "val_geracaolimitada" in cols
            has_geracao = "val_geracao" in cols
            has_disp = "val_disponibilidade" in cols
            has_ref_final = "val_geracaoreferenciafinal" in cols
            has_ref = "val_geracaoreferencia" in cols
            has_est = col_estimada.lower() in cols
            has_ver = col_verificada.lower() in cols
            has_cod = "cod_razaorestricao" in cols

            flag_filter = ""
            if col_flag_invalido and col_flag_invalido.lower() in cols:
                flag_filter = f" AND COALESCE(TRY_CAST({cols[col_flag_invalido.lower()]} AS BOOLEAN), FALSE) = FALSE"

            # Curto-circuito: layout novo (tm) prioriza val_geracaolimitada como curtailment
            if has_limitada:
                curta_expr = _duckdb_num_expr(cols["val_geracaolimitada"])
                if has_ref:
                    disp_expr = _duckdb_num_expr(cols["val_geracaoreferencia"])
                elif has_ref_final:
                    disp_expr = _duckdb_num_expr(cols["val_geracaoreferenciafinal"])
                elif has_disp:
                    disp_expr = _duckdb_num_expr(cols["val_disponibilidade"])
                elif has_geracao:
                    disp_expr = f"COALESCE({_duckdb_num_expr(cols['val_geracao'])},0) + COALESCE({_duckdb_num_expr(cols['val_geracaolimitada'])},0)"
                else:
                    disp_expr = "NULL"
                ver_expr = _duckdb_num_expr(cols["val_geracao"]) if has_geracao else ( _duckdb_num_expr(cols[col_verificada.lower()]) if has_ver else "NULL")
            else:
                # Layout antigo detail_tm
                if not (has_est and has_ver):
                    return {"status": "indisponível"}
                curta_expr = f"GREATEST(COALESCE({_duckdb_num_expr(cols[col_estimada.lower()])},0) - COALESCE({_duckdb_num_expr(cols[col_verificada.lower()])},0), 0)"
                disp_expr = _duckdb_num_expr(cols[col_estimada.lower()])
                ver_expr = _duckdb_num_expr(cols[col_verificada.lower()])

            cod_select = f", TRIM(CAST({cols['cod_razaorestricao']} AS VARCHAR)) AS cod_razaorestricao" if has_cod else ""

            q = f"""
                SELECT
                    {_duckdb_date_expr(cols.get('din_instante','din_instante'))} AS din_instante,
                    {curta_expr} AS curtailment_abs,
                    {disp_expr} AS disponivel,
                    {ver_expr} AS verificada
                    {cod_select}
                FROM {table_name}
                WHERE {_duckdb_date_expr(cols.get('din_instante','din_instante'))} IS NOT NULL
                {flag_filter}
            """
            df = con.execute(q).fetchdf()
            return _finalize(df)
        except Exception:
            pass
        finally:
            con.close()

    return {"status": "indisponível"}


def _compute_renewable_curtailment(ons: Dict[str, Any]) -> Dict[str, Any]:

    solar = _compute_curtailment_from_csv(
        ons,
        dataset_name="Restricao_fotovoltaica",
        col_estimada="val_geracaoestimada",
        col_verificada="val_geracaoverificada",
        col_flag_invalido="flg_dadoirradianciainvalido"
    )

    eolica = _compute_curtailment_from_csv(
        ons,
        dataset_name="Restricao_eolica",
        col_estimada="val_geracaoestimada",
        col_verificada="val_geracaoverificada",
        col_flag_invalido="flg_dadoventoinvalido"
    )

    total = 0
    disponivel_total = 0
    if solar.get("curtailment_total_mwh"):
        total += solar["curtailment_total_mwh"]
    if solar.get("geracao_disponivel_total_mwh"):
        disponivel_total += solar["geracao_disponivel_total_mwh"]
    if eolica.get("curtailment_total_mwh"):
        total += eolica["curtailment_total_mwh"]
    if eolica.get("geracao_disponivel_total_mwh"):
        disponivel_total += eolica["geracao_disponivel_total_mwh"]

    return {
        "solar": solar,
        "eolica": eolica,
        "total_mwh": total,
        "curtailment_pct_total": float(total / disponivel_total) if disponivel_total > 0 else None,
    }
# =====================================================================
# INDICE DE SATURAÇÃO RENOVÁVEL
# =====================================================================

def _compute_isr(
    geracao_solar: pd.Series,
    geracao_eolica: pd.Series,
    carga_liquida: pd.Series
) -> Optional[float]:

    if geracao_solar.empty or geracao_eolica.empty or carga_liquida.empty:
        return None

    renovavel_total = geracao_solar.mean() + geracao_eolica.mean()
    carga_media = carga_liquida.mean()

    if carga_media <= 0:
        return None

    return renovavel_total / carga_media


def _normalize_br_numeric_series(series: pd.Series) -> pd.Series:
    """Converte números em formato brasileiro/ambíguo para float."""
    if series.empty:
        return pd.Series(dtype=float)

    raw = series.astype(str).str.strip()

    has_comma = raw.str.contains(",", regex=False)
    many_dots = raw.str.count(r"\.") > 1

    parsed = raw.copy()

    # Padrão BR: 1.234,56 -> 1234.56
    parsed.loc[has_comma] = (
        parsed.loc[has_comma]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    # Valores com múltiplos pontos: 25.000.000.000 -> 25000000000
    parsed.loc[~has_comma & many_dots] = parsed.loc[~has_comma & many_dots].str.replace(".", "", regex=False)

    # Notação científica (ex.: 0E-8) já é interpretada por to_numeric e vira 0.0
    parsed = parsed.str.replace(" ", "", regex=False)

    return pd.to_numeric(parsed, errors="coerce")




def _parse_date_series(series: pd.Series) -> pd.Series:
    """Parse robusto para datas priorizando padrão brasileiro (dd/mm/aaaa)."""
    if series is None or series.empty:
        return pd.Series(dtype="datetime64[ns]")

    raw = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    # ISO/ano-primeiro: mantém parsing padrão para evitar inversões.
    mask_iso = raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    if mask_iso.any():
        out.loc[mask_iso] = pd.to_datetime(raw.loc[mask_iso], errors="coerce")

    # Fontes ONS/CCEE com dia primeiro (dd/mm/aaaa [HH:MM[:SS]]).
    mask_br = ~mask_iso
    if mask_br.any():
        out.loc[mask_br] = pd.to_datetime(raw.loc[mask_br], errors="coerce", dayfirst=True)

    # Fallback final para casos residuais.
    rem = out.isna()
    if rem.any():
        out.loc[rem] = pd.to_datetime(raw.loc[rem], errors="coerce")

    return out

def _to_series(records: List[Dict[str, Any]], value_key: str) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)

    try:
        df = pd.DataFrame(records)
        if "instante" not in df.columns or value_key not in df.columns:
            return pd.Series(dtype=float)

        df["instante"] = _parse_date_series(df["instante"])
        df[value_key] = _normalize_br_numeric_series(df[value_key])
        df = df.dropna(subset=["instante", value_key]).sort_values("instante")
        if df.empty:
            return pd.Series(dtype=float)
        return _ensure_tz_naive_index(df.set_index("instante")[value_key])
    except Exception:
        return pd.Series(dtype=float)


def _safe_corr(a: pd.Series, b: pd.Series, min_points: int = 24) -> Optional[float]:
    try:
        a = _ensure_tz_naive_index(a)
        b = _ensure_tz_naive_index(b)
        if a.empty or b.empty:
            return None
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(df) < min_points:
            return None
        # Evita RuntimeWarning do numpy quando uma das séries tem desvio zero.
        if df["a"].nunique(dropna=True) <= 1 or df["b"].nunique(dropna=True) <= 1:
            return None
        if float(np.nanstd(df["a"].values)) == 0.0 or float(np.nanstd(df["b"].values)) == 0.0:
            return None
        corr = df["a"].corr(df["b"])
        if pd.isna(corr):
            return None
        return float(corr)
    except Exception:
        return None


def _ensure_tz_naive_index(series: pd.Series) -> pd.Series:
    """Padroniza índice temporal para datetime naive sem alterar o horário."""
    
    s = series.copy()

    try:
        idx = pd.to_datetime(s.index, errors="coerce")
        s = s[~idx.isna()]
        idx = idx[~idx.isna()]

        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_localize(None)

        s.index = idx
    except Exception:
        pass

    try:
        s = s[~s.index.duplicated(keep="last")]
    except Exception:
        pass

    return s.sort_index()


def _dataset_file(ds: Dict[str, Any]) -> Optional[str]:
    file = ds.get("file")
    if isinstance(file, str):
        file = file.replace("\\", os.sep)
    return file


def _normalize_submercado_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().upper()
    v = v.replace("/", "").replace("-", "").replace(" ", "")

    mapping = {
        "1": "SUDESTE",
        "2": "SUL",
        "3": "NORDESTE",
        "4": "NORTE",
        "N": "NORTE",
        "NE": "NORDESTE",
        "SE": "SUDESTE",
        "SECO": "SUDESTE",
        "SUDESTECENTROOESTE": "SUDESTE",
        "SUDESTECENTRO-OESTE": "SUDESTE",
        "S": "SUL",
        "NORTE": "NORTE",
        "NORDESTE": "NORDESTE",
        "SUDESTE": "SUDESTE",
        "SUL": "SUL",
    }
    return mapping.get(v)




def _normalize_text_key(value: Any) -> str:
    txt = str(value or "").strip().lower()
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    return txt


def _load_gfom_hourly(ons: Dict[str, Any]) -> pd.DataFrame:
    if duckdb is None:
        return pd.DataFrame(columns=["ger", "gfom"])
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "despacho_gfom"):
        if con is not None:
            con.close()
        return pd.DataFrame(columns=["ger", "gfom"])
    try:
        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                SUM({_duckdb_num_expr('val_verifgeracao')}) AS ger,
                SUM({_duckdb_num_expr('val_verifgfom')}) AS gfom
            FROM despacho_gfom
            GROUP BY 1
            HAVING din_instante IS NOT NULL
            ORDER BY 1
        """
        df = con.execute(q).fetchdf()
        if df.empty:
            return pd.DataFrame(columns=["ger", "gfom"])
        return df.set_index("din_instante")[["ger", "gfom"]]
    except Exception:
        return pd.DataFrame(columns=["ger", "gfom"])
    finally:
        con.close()


def _load_gfom_hourly_by_submarket(ons: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    if duckdb is None:
        return {}
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "despacho_gfom"):
        if con is not None:
            con.close()
        return {}
    try:
        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS submercado_raw,
                SUM({_duckdb_num_expr('val_verifgeracao')}) AS ger,
                SUM({_duckdb_num_expr('val_verifgfom')}) AS gfom
            FROM despacho_gfom
            GROUP BY 1,2
            HAVING din_instante IS NOT NULL
        """
        df = con.execute(q).fetchdf()
        if df.empty:
            return {}
        df["submercado"] = df["submercado_raw"].map(_normalize_submercado_name)
        df = df.dropna(subset=["submercado"])
        out: Dict[str, pd.DataFrame] = {}
        for sm, grp in df.groupby("submercado"):
            out[str(sm)] = grp.groupby("din_instante")[["ger", "gfom"]].sum().sort_index()
        return out
    except Exception:
        return {}
    finally:
        con.close()


def _load_cmo_hourly_by_submarket() -> Dict[str, pd.Series]:
    """Consolida CMO semi-horário em horário (média dos minutos 00 e 30) por submercado."""
    if duckdb is None:
        return {}
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "cmo"):
        if con is not None:
            con.close()
        return {}
    try:
        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS sub_raw,
                {_duckdb_num_expr('val_cmo')} AS val_cmo
            FROM cmo
            WHERE {_duckdb_date_expr('din_instante')} IS NOT NULL
              AND {_duckdb_num_expr('val_cmo')} IS NOT NULL
        """
        df = con.execute(q).fetchdf()
        if df.empty:
            return {}
        df["submercado"] = df["sub_raw"].map(_normalize_submercado_name)
        df = df.dropna(subset=["submercado", "din_instante", "val_cmo"])
        if df.empty:
            return {}
        df["hora"] = pd.to_datetime(df["din_instante"], errors="coerce").dt.floor("h")
        df = df.dropna(subset=["hora"])
        out: Dict[str, pd.Series] = {}
        for sm, grp in df.groupby("submercado"):
            s = grp.groupby("hora")["val_cmo"].mean().sort_index()
            if not s.empty:
                out[str(sm)] = _ensure_tz_naive_index(s.astype(float))
        return out
    except Exception:
        return {}
    finally:
        con.close()


def _load_capacidade_instalada_ativa_por_fonte(ons: Dict[str, Any]) -> Dict[str, float]:
    """
    Soma de val_potenciaefetiva (MW) por tipo de usina considerando apenas usinas ativas
    (dat_desativacao nula/vazia) na tabela capacidade_instalada.
    """
    if duckdb is None:
        return {}
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "capacidade_instalada"):
        if con is not None:
            con.close()
        return {}
    try:
        q = f"""
            SELECT
                UPPER(TRIM(CAST(nom_tipousina AS VARCHAR))) AS tipousina,
                SUM({_duckdb_num_expr('val_potenciaefetiva')}) AS potencia_mw
            FROM capacidade_instalada
            WHERE COALESCE(TRIM(CAST(dat_desativacao AS VARCHAR)), '') = ''
            GROUP BY 1
            HAVING potencia_mw IS NOT NULL
        """
        df = con.execute(q).fetchdf()
        if df.empty:
            return {}
        out = {}
        for _, r in df.iterrows():
            t = str(r['tipousina']).upper()
            if 'HIDRO' in t:
                k = 'HIDROELETRICA'
            elif 'TERM' in t:
                k = 'TERMICA'
            elif 'EOL' in t:
                k = 'EOLIELETRICA'
            elif 'FOTOV' in t or 'SOLAR' in t:
                k = 'FOTOVOLTAICA'
            elif 'NUCL' in t:
                k = 'NUCLEAR'
            else:
                continue
            out[k] = float(out.get(k, 0.0) + (r['potencia_mw'] if pd.notna(r['potencia_mw']) else 0.0))
        return out
    except Exception:
        return {}
    finally:
        con.close()


def _load_disponibilidade_horaria() -> Dict[str, pd.Series]:
    """Capacidade sincronizada horária por tipo de usina (UHE, UTE, UTN) e total."""
    
    out = {
        "uhe": pd.Series(dtype=float),
        "ute": pd.Series(dtype=float),
        "utn": pd.Series(dtype=float),
        "total": pd.Series(dtype=float),
    }

    if duckdb is None:
        return out

    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "disponibilidade_usina"):
        if con is not None:
            con.close()
        return out

    try:

        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                UPPER(TRIM(CAST(id_tipousina AS VARCHAR))) AS tipousina,
                {_duckdb_num_expr('val_dispsincronizada')} AS disp_sync
            FROM disponibilidade_usina
            WHERE {_duckdb_date_expr('din_instante')} IS NOT NULL
              AND UPPER(TRIM(CAST(id_tipousina AS VARCHAR))) IN ('UHE','UTE','UTN')
        """

        df = con.execute(q).fetchdf()

        if df.empty:
            return out

        df["din_instante"] = pd.to_datetime(df["din_instante"])

        # separa tipos
        uhe = df[df["tipousina"] == "UHE"]
        ute = df[df["tipousina"] == "UTE"]
        utn = df[df["tipousina"] == "UTN"]

        if not uhe.empty:
            s = uhe.groupby("din_instante")["disp_sync"].sum()
            out["uhe"] = _ensure_tz_naive_index(_normalize_power_to_mw(s).astype(float).sort_index())

        if not ute.empty:
            s = ute.groupby("din_instante")["disp_sync"].sum()
            out["ute"] = _ensure_tz_naive_index(_normalize_power_to_mw(s).astype(float).sort_index())

        if not utn.empty:
            s = utn.groupby("din_instante")["disp_sync"].sum()
            out["utn"] = _ensure_tz_naive_index(_normalize_power_to_mw(s).astype(float).sort_index())

        # TOTAL = soma de tudo
        s_total = df.groupby("din_instante")["disp_sync"].sum()
        out["total"] = _ensure_tz_naive_index(_normalize_power_to_mw(s_total).astype(float).sort_index())

        return out

    except Exception:
        return out

    finally:
        con.close()


def _load_gfom_components_hourly() -> pd.DataFrame:
    """Componentes horários de despacho térmico (GFOM e decomposição)."""
    cols_out = [
        "ger",
        "ordem_merito",
        "inflex_pura",
        "inflex_total",
        "inflex_embut_merito",
        "ordem_demerito_acima_inflex",
        "razao_eletrica",
        "constrained_off",
        "gfom",
    ]
    if duckdb is None:
        return pd.DataFrame(columns=cols_out)
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "despacho_gfom"):
        if con is not None:
            con.close()
        return pd.DataFrame(columns=cols_out)
    try:
        info = con.execute("PRAGMA table_info('despacho_gfom')").fetchall()
        cols = {str(c[1]).lower(): str(c[1]) for c in info}

        def num_or_zero(name: str) -> str:
            c = cols.get(name)
            return _duckdb_num_expr(c) if c else "0"

        q = f"""
            SELECT
                {_duckdb_date_expr(cols.get('din_instante', 'din_instante'))} AS din_instante,
                SUM({num_or_zero('val_verifgeracao')}) AS ger,
                SUM({num_or_zero('val_verifordemmerito')}) AS ordem_merito,
                SUM({num_or_zero('val_verifinflexpura')}) AS inflex_pura,
                SUM({num_or_zero('val_verifinflexibilidade')}) AS inflex_total,
                SUM({num_or_zero('val_verifinflexembutmerito')}) AS inflex_embut_merito,
                SUM({num_or_zero('val_verifordemdemeritoacimadainflex')}) AS ordem_demerito_acima_inflex,
                SUM({num_or_zero('val_verifrazaoeletrica')}) AS razao_eletrica,
                SUM({num_or_zero('val_verifconstrainedoff')}) AS constrained_off,
                SUM({num_or_zero('val_verifgfom')}) AS gfom
            FROM despacho_gfom
            GROUP BY 1
            HAVING din_instante IS NOT NULL
            ORDER BY 1
        """
        df = con.execute(q).fetchdf()
        if df.empty:
            return pd.DataFrame(columns=cols_out)
        out = df.set_index("din_instante")
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].sort_index()
        return out[cols_out]
    except Exception:
        return pd.DataFrame(columns=cols_out)
    finally:
        con.close()


def _series_to_hourly_dict(series: pd.Series, ndigits: int = 6) -> Dict[str, float]:
    if not isinstance(series, pd.Series) or series.empty:
        return {}
    s = _ensure_tz_naive_index(pd.to_numeric(series, errors="coerce")).dropna().sort_index()
    return {pd.Timestamp(i).strftime("%Y-%m-%d %H:%M:%S"): round(float(v), ndigits) for i, v in s.items()}


def _load_carga_sin_horaria_duckdb() -> pd.Series:
    if duckdb is None:
        return pd.Series(dtype=float)
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "curva_carga"):
        if con is not None:
            con.close()
        return pd.Series(dtype=float)
    try:
        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                UPPER(TRIM(CAST(id_subsistema AS VARCHAR))) AS id_subsistema,
                {_duckdb_num_expr('val_cargaenergiahomwmed')} AS carga
            FROM curva_carga
            WHERE {_duckdb_date_expr('din_instante')} IS NOT NULL
        """
        df = con.execute(q).fetchdf().dropna(subset=["din_instante", "id_subsistema", "carga"])
        if df.empty:
            return pd.Series(dtype=float)
        df = df[df["id_subsistema"].isin(["N", "NE", "SE", "S"]) ]
        s = df.groupby("din_instante")["carga"].sum().sort_index()
        return _ensure_tz_naive_index(s.astype(float))
    except Exception:
        return pd.Series(dtype=float)
    finally:
        con.close()


def _load_geracao_tipos_horaria_duckdb() -> Dict[str, pd.Series]:
    out = {
        "solar": pd.Series(dtype=float),
        "eolica": pd.Series(dtype=float),
        "termica": pd.Series(dtype=float),
        "hidro": pd.Series(dtype=float),
        "nuclear": pd.Series(dtype=float),
    }
    if duckdb is None:
        return out
    con = _duckdb_connect()
    if con is None or not _duckdb_table_exists(con, "geracao_usina_horaria"):
        if con is not None:
            con.close()
        return out
    try:
        q = f"""
            SELECT
                {_duckdb_date_expr('din_instante')} AS din_instante,
                UPPER(TRIM(CAST(nom_tipousina AS VARCHAR))) AS tipousina,
                {_duckdb_num_expr('val_geracao')} AS val_geracao
            FROM geracao_usina_horaria
            WHERE {_duckdb_date_expr('din_instante')} IS NOT NULL
        """
        df = con.execute(q).fetchdf().dropna(subset=["din_instante", "tipousina", "val_geracao"])
        if df.empty:
            return out
        df["k"] = df["tipousina"].apply(_normalize_text_key)
        solar = df[df["k"].str.contains("fotov|solar", regex=True)]
        eolica = df[df["k"].str.contains("eol", regex=True)]
        termica = df[df["k"].str.contains("term", regex=True)]
        hidro = df[df["k"].str.contains("hidro|hidraul", regex=True)]
        nuclear = df[df["k"].str.contains("nuclear", regex=True)]
        if not solar.empty:
            out["solar"] = _ensure_tz_naive_index(solar.groupby("din_instante")["val_geracao"].sum().sort_index().astype(float))
        if not eolica.empty:
            out["eolica"] = _ensure_tz_naive_index(eolica.groupby("din_instante")["val_geracao"].sum().sort_index().astype(float))
        if not termica.empty:
            out["termica"] = _ensure_tz_naive_index(termica.groupby("din_instante")["val_geracao"].sum().sort_index().astype(float))
        if not hidro.empty:
            out["hidro"] = _ensure_tz_naive_index(hidro.groupby("din_instante")["val_geracao"].sum().sort_index().astype(float))
        if not nuclear.empty:
            out["nuclear"] = _ensure_tz_naive_index(nuclear.groupby("din_instante")["val_geracao"].sum().sort_index().astype(float))
        return out
    except Exception:
        return out
    finally:
        con.close()


def _load_renovavel_disponivel_restricao_horaria() -> pd.DataFrame:
    cols = [
        "instante",
        "renov_disponivel",
        "solar_disponivel",
        "eolica_disponivel",
        "cod_razaorestricao_solar",
        "dsc_restricao_solar",
        "cod_razaorestricao_eolica",
        "dsc_restricao_eolica",
        "val_geracaolimitada_solar",
        "val_geracaolimitada_eolica",
    ]
    if duckdb is None:
        return pd.DataFrame(columns=cols)

    con = _duckdb_connect()
    if con is None:
        return pd.DataFrame(columns=cols)

    def _load_one(table: str, prefix: str) -> pd.DataFrame:
        if not _duckdb_table_exists(con, table):
            return pd.DataFrame()
        info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        c = {str(i[1]).lower(): str(i[1]) for i in info}
        if "din_instante" not in c or "val_geracaoreferencia" not in c:
            return pd.DataFrame()
        lim_col = c.get("val_geracaolimitada")
        cod_col = c.get("cod_razaorestricao")
        dsc_col = c.get("dsc_restricao")
        q = f"""
            SELECT
                {_duckdb_date_expr(c['din_instante'])} AS instante,
                {_duckdb_num_expr(c['val_geracaoreferencia'])} AS disponivel,
                {_duckdb_num_expr(lim_col) if lim_col else 'NULL'} AS limitada,
                TRIM(CAST({cod_col} AS VARCHAR)) AS cod,
                TRIM(CAST({dsc_col} AS VARCHAR)) AS dsc
            FROM {table}
            WHERE {_duckdb_date_expr(c['din_instante'])} IS NOT NULL
        """
        df = con.execute(q).fetchdf().dropna(subset=["instante", "disponivel"])
        if df.empty:
            return pd.DataFrame()
        df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
        df = df.dropna(subset=["instante"]) 
        agg = df.groupby("instante", as_index=False).agg({"disponivel": "sum", "limitada": "sum"})
        top = (
            df.assign(limitada=pd.to_numeric(df["limitada"], errors="coerce").fillna(0))
            .sort_values(["instante", "limitada"], ascending=[True, False])
            .drop_duplicates(subset=["instante"])[["instante", "cod", "dsc"]]
        )
        out = agg.merge(top, on="instante", how="left")
        out = out.rename(columns={
            "disponivel": f"{prefix}_disponivel",
            "limitada": f"val_geracaolimitada_{prefix}",
            "cod": f"cod_razaorestricao_{prefix}",
            "dsc": f"dsc_restricao_{prefix}",
        })
        return out

    try:
        solar = _load_one("restricao_fotovoltaica", "solar")
        eolica = _load_one("restricao_eolica", "eolica")
        if solar.empty and eolica.empty:
            return pd.DataFrame(columns=cols)
        if solar.empty:
            out = eolica.copy()
        elif eolica.empty:
            out = solar.copy()
        else:
            out = solar.merge(eolica, on="instante", how="outer")

        for c in ["solar_disponivel", "eolica_disponivel", "val_geracaolimitada_solar", "val_geracaolimitada_eolica"]:
            if c not in out.columns:
                out[c] = np.nan

        out["renov_disponivel"] = out[["solar_disponivel", "eolica_disponivel"]].fillna(0).sum(axis=1)
        for c in [
            "cod_razaorestricao_solar", "dsc_restricao_solar",
            "cod_razaorestricao_eolica", "dsc_restricao_eolica",
        ]:
            if c not in out.columns:
                out[c] = None
        return out[cols].sort_values("instante")
    except Exception:
        return pd.DataFrame(columns=cols)
    finally:
        con.close()


def _load_ear_ena_monthly_by_submercado(ons: Dict[str, Any]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Consolida EAR e ENA mensais por submercado apenas via DuckDB."""
    ear_by_sub: Dict[str, pd.Series] = {}
    ena_by_sub: Dict[str, pd.Series] = {}

    con = _duckdb_connect()
    if con is None:
        return ear_by_sub, ena_by_sub

    try:
        if _duckdb_table_exists(con, "ear_diario_subsistema"):
            q_ear = f"""
                SELECT
                    DATE_TRUNC('month', {_duckdb_date_expr('ear_data')}) AS mes,
                    UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS submercado_raw,
                    SUM({_duckdb_num_expr('ear_verif_subsistema_mwmes')}) / NULLIF(COUNT(DISTINCT DATE_TRUNC('day', {_duckdb_date_expr('ear_data')})), 0) AS valor
                FROM ear_diario_subsistema
                GROUP BY 1,2
                HAVING mes IS NOT NULL AND valor IS NOT NULL
            """
            dfe = con.execute(q_ear).fetchdf()
            if not dfe.empty:
                dfe['submercado'] = dfe['submercado_raw'].map(_normalize_submercado_name)
                dfe = dfe.dropna(subset=['submercado'])
                for sm, grp in dfe.groupby('submercado'):
                    idx = pd.to_datetime(grp['mes']) + pd.offsets.MonthEnd(0)
                    ear_by_sub[str(sm)] = pd.Series(grp['valor'].values, index=idx).sort_index()

        if _duckdb_table_exists(con, "ena_diario_subsistema"):
            q_ena = f"""
                SELECT
                    DATE_TRUNC('month', {_duckdb_date_expr('ena_data')}) AS mes,
                    UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS submercado_raw,
                    SUM({_duckdb_num_expr('ena_armazenavel_regiao_mwmed')}) / NULLIF(COUNT(DISTINCT DATE_TRUNC('day', {_duckdb_date_expr('ena_data')})), 0) AS valor
                FROM ena_diario_subsistema
                GROUP BY 1,2
                HAVING mes IS NOT NULL AND valor IS NOT NULL
            """
            dfn = con.execute(q_ena).fetchdf()
            if not dfn.empty:
                dfn['submercado'] = dfn['submercado_raw'].map(_normalize_submercado_name)
                dfn = dfn.dropna(subset=['submercado'])
                for sm, grp in dfn.groupby('submercado'):
                    idx = pd.to_datetime(grp['mes']) + pd.offsets.MonthEnd(0)
                    ena_by_sub[str(sm)] = pd.Series(grp['valor'].values, index=idx).sort_index()
    except Exception:
        pass
    finally:
        con.close()

    return ear_by_sub, ena_by_sub


def _compute_effective_availability_margin(
    ons: Dict[str, Any],
    carga_sin_series: pd.Series
) -> Dict[str, Any]:
    if carga_sin_series.empty or carga_sin_series.mean() <= 0:
        return {"status": "indisponível"}

    disponibilidade_h = _load_disponibilidade_horaria()

    if not isinstance(disponibilidade_h, dict) or not disponibilidade_h:
        return {"status": "indisponível"}

    disp_total = disponibilidade_h.get("total", pd.Series(dtype=float))

    if not isinstance(disp_total, pd.Series) or disp_total.empty:
        return {"status": "indisponível"}

    cap_disp_media = None
    if isinstance(disp_total, pd.Series) and not disp_total.empty:
        cap_disp_media = float(disp_total.mean())
    carga_media = float(carga_sin_series.mean())
    if cap_disp_media is None or carga_media <= 0:
        return {"status": "indisponível"}

    margem = (cap_disp_media - carga_media) / carga_media
    return {
        "status": "disponível",
        "capacidade_disponivel_efetiva_media": cap_disp_media,
        "carga_media": carga_media,
        "margem_estrutural_oferta": float(margem),
        "coluna_origem": "duckdb:disponibilidade_usina",
    }


def _compute_termica_share_from_gfom(ons: Dict[str, Any]) -> Optional[float]:
    """Dependência térmica efetiva via DuckDB (soma de val_verifgeracao no despacho GFOM)."""
    gf = _load_gfom_hourly(ons)
    if gf.empty or "ger" not in gf.columns:
        return None
    try:
        return float(pd.to_numeric(gf["ger"], errors="coerce").dropna().sum())
    except Exception:
        return None


def _compute_advanced_cross_metrics(
    ons: Dict[str, Any],
    operacao: Dict[str, Any],
    pld_series: pd.Series,
    pld_series_by_submercado: Optional[Dict[str, pd.Series]],
    ear_medio: Optional[float],
    ena_media: Optional[float],
    pld_medio: Optional[float],
    curtailment: Dict[str, Any],
) -> Dict[str, Any]:
    pld_series = _ensure_tz_naive_index(pld_series)
    pld_series_by_submercado = {
        _normalize_submercado_name(k) or str(k): _ensure_tz_naive_index(v)
        for k, v in (pld_series_by_submercado or {}).items()
        if isinstance(v, pd.Series)
    }

    generation = operacao.get("generation", {})
    load = operacao.get("load", {})
    step_errors: Dict[str, str] = {}

    carga_sin = _ensure_tz_naive_index(_to_series(load.get("sin", {}).get("serie", []), "carga"))
    carga_sin_db = _load_carga_sin_horaria_duckdb()

    if isinstance(carga_sin_db, pd.Series) and not carga_sin_db.empty:
        carga_sin = carga_sin_db

    solar_key = next((k for k in generation.keys() if any(t in _normalize_text_key(k) for t in ["solar", "fotov"])) , None)
    eolica_key = next((k for k in generation.keys() if any(t in _normalize_text_key(k) for t in ["eolica", "eolie", "eoli"])) , None)
    termica_key = next((k for k in generation.keys() if "termica" in _normalize_text_key(k)), None)
    hidro_key = next((k for k in generation.keys() if any(t in _normalize_text_key(k) for t in ["hidro", "hidraul"])) , None)
    nuclear_key = next((k for k in generation.keys() if "nuclear" in _normalize_text_key(k)), None)

    solar = _ensure_tz_naive_index(_to_series(generation.get(solar_key, {}).get("serie", []), "geracao")) if solar_key else pd.Series(dtype=float)
    eolica = _ensure_tz_naive_index(_to_series(generation.get(eolica_key, {}).get("serie", []), "geracao")) if eolica_key else pd.Series(dtype=float)
    termica = _ensure_tz_naive_index(_to_series(generation.get(termica_key, {}).get("serie", []), "geracao")) if termica_key else pd.Series(dtype=float)
    hidro = _ensure_tz_naive_index(_to_series(generation.get(hidro_key, {}).get("serie", []), "geracao")) if hidro_key else pd.Series(dtype=float)
    nuclear = _ensure_tz_naive_index(_to_series(generation.get(nuclear_key, {}).get("serie", []), "geracao")) if nuclear_key else pd.Series(dtype=float)

    tipos_db = _load_geracao_tipos_horaria_duckdb()
    if not tipos_db.get("solar", pd.Series(dtype=float)).empty:
        solar = _ensure_tz_naive_index(tipos_db["solar"])
    if not tipos_db.get("eolica", pd.Series(dtype=float)).empty:
        eolica = _ensure_tz_naive_index(tipos_db["eolica"])
    if not tipos_db.get("termica", pd.Series(dtype=float)).empty:
        termica = _ensure_tz_naive_index(tipos_db["termica"])
    if not tipos_db.get("hidro", pd.Series(dtype=float)).empty:
        hidro = _ensure_tz_naive_index(tipos_db["hidro"])
    if not tipos_db.get("nuclear", pd.Series(dtype=float)).empty:
        nuclear = _ensure_tz_naive_index(tipos_db["nuclear"])

    gen_parts = []

    for s in [solar, eolica, termica, hidro, nuclear]:
        if isinstance(s, pd.Series) and not s.empty:
            gen_parts.append(s)

    if gen_parts:
        df_sum = pd.concat(gen_parts, axis=1)
        geracao_total = _ensure_tz_naive_index(
            df_sum.sum(axis=1, min_count=1)
        )
    else:
        geracao_total = pd.Series(dtype=float)

    carga_liquida = pd.Series(dtype=float)
    horas_renovavel_gt_carga_liquida = None
    if not carga_sin.empty:
        renovaveis = solar.add(eolica, fill_value=0)
        carga_liquida = carga_sin.sub(renovaveis, fill_value=np.nan)
        aligned = pd.DataFrame({"renov": renovaveis, "carga_liquida": carga_liquida}).dropna()
        if not aligned.empty:
            horas_renovavel_gt_carga_liquida = int((aligned["renov"] > aligned["carga_liquida"]).sum())

    # IPR e ISR (horário)
    ipr_medio = None
    isr_medio = None
    ipr_horario = pd.Series(dtype=float)
    isr_horario = pd.Series(dtype=float)

    restr_h = _load_renovavel_disponivel_restricao_horaria()
    if isinstance(restr_h, pd.DataFrame) and not restr_h.empty:
        renov_disponivel_h = pd.Series(dtype=float)

    if isinstance(restr_h, pd.DataFrame) and not restr_h.empty:
        renov_disponivel_h = _ensure_tz_naive_index(
            pd.Series(
                restr_h["renov_disponivel"].values,
                index=pd.to_datetime(restr_h["instante"])
            )
        )

    if not carga_sin.empty and not renov_disponivel_h.empty:
        df_ipr = pd.DataFrame({"renov_disp": renov_disponivel_h, "carga": carga_sin}).dropna()
        df_ipr = df_ipr[df_ipr["carga"] > 0]
        if not df_ipr.empty:
            ipr_horario = _ensure_tz_naive_index(df_ipr["renov_disp"] / df_ipr["carga"])
            ipr_medio = float(ipr_horario.iloc[-1]) if not ipr_horario.empty else None

    if not carga_liquida.empty and not renov_disponivel_h.empty:
        df_isr = pd.DataFrame({"renov_disp": renov_disponivel_h, "carga_liquida": carga_liquida}).dropna()
        df_isr = df_isr[df_isr["carga_liquida"] > 0]
        if not df_isr.empty:
            isr_horario = _ensure_tz_naive_index(df_isr["renov_disp"] / df_isr["carga_liquida"])
            isr_medio = float(isr_horario.iloc[-1]) if not isr_horario.empty else None

    dependencia_termica_pct = None
    if not termica.empty and not geracao_total.empty:
        df_term = pd.DataFrame({"termica": termica, "total": geracao_total}).dropna()
        df_term = df_term[df_term["total"] > 0]
        if not df_term.empty:
            dependencia_termica_pct = float((df_term["termica"].sum() / df_term["total"].sum()) * 100)

    # Fallback para datasets GFOM quando Energia Agora não estiver disponível.
    if dependencia_termica_pct is None:
        total_termica_gfom = _compute_termica_share_from_gfom(ons)
        if total_termica_gfom is not None and not geracao_total.empty and geracao_total.sum() > 0:
            dependencia_termica_pct = float((total_termica_gfom / float(geracao_total.sum())) * 100)

    margem_oferta = _compute_effective_availability_margin(ons, carga_sin)
    capacidade_instalada_ativa_por_fonte = _load_capacidade_instalada_ativa_por_fonte(ons)

    # Capacidade disponível real / margem operativa real / stress operacional
    capacidade_disp_h = _load_disponibilidade_horaria()
    if isinstance(capacidade_disp_h, dict):
        capacidade_disp_h = capacidade_disp_h.get("total", pd.Series(dtype=float))
    margem_operativa_media_mensal = None
    margem_operativa_p5_mensal = None
    stress_operacional_medio = None
    stress_operacional_horario = None
    tendencia_estrutural_mensal = None
    if not capacidade_disp_h.empty and not carga_sin.empty:
        df_cap = pd.DataFrame({"cap": capacidade_disp_h, "carga": carga_sin}).dropna()
        df_cap = df_cap[df_cap["carga"] > 0]
        if not df_cap.empty:
            df_cap["margem"] = (df_cap["cap"] - df_cap["carga"]) / df_cap["carga"]
            df_cap["stress"] = df_cap["carga"] / df_cap["cap"].replace(0, np.nan)
            stress_operacional_horario = (
                df_cap["stress"].dropna().reset_index().rename(columns={"index": "instante", "stress": "valor"}).to_dict("records")
            )
            if not df_cap["stress"].dropna().empty:
                stress_operacional_medio = float(df_cap["stress"].dropna().mean())

            mensal = df_cap.resample("ME").agg({"margem": ["mean", lambda x: x.quantile(0.05)]})
            mensal.columns = ["margem_media", "margem_p5"]
            mensal = mensal.dropna(how="all")
            if not mensal.empty:
                margem_operativa_media_mensal = {
                    i.strftime("%Y-%m"): float(v) for i, v in mensal["margem_media"].dropna().items()
                }
                margem_operativa_p5_mensal = {
                    i.strftime("%Y-%m"): float(v) for i, v in mensal["margem_p5"].dropna().items()
                }
                tendencia_estrutural_mensal = "alta" if mensal["margem_media"].iloc[-1] > mensal["margem_media"].iloc[0] else "baixa"

    corr_pld_carga_liquida = None
    rolling_corr_90d = None
    try:
        pld_series = _ensure_tz_naive_index(pld_series)
        carga_liquida = _ensure_tz_naive_index(carga_liquida)
        corr_pld_carga_liquida = _safe_corr(pld_series, carga_liquida, min_points=24)
        if not pld_series.empty and not carga_liquida.empty:
            df_rl = pd.DataFrame({"pld": pld_series, "carga_liquida": carga_liquida}).dropna().sort_index()
            if len(df_rl) >= 24:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rolling = df_rl["pld"].rolling(window=90 * 24, min_periods=24).corr(df_rl["carga_liquida"])
                if not rolling.dropna().empty:
                    rolling_corr_90d = float(rolling.dropna().iloc[-1])
    except Exception as e:
        step_errors["pld_vs_carga_liquida"] = str(e)

    corr_pld_ear_mensal = None
    pld_vs_ear_mensal_por_submercado: Dict[str, Optional[float]] = {}
    pld_vs_ena_mensal_por_submercado: Dict[str, Optional[float]] = {}

    ear_media_mensal = None
    ena_media_mensal = None
    ear_media_mensal_por_submercado = {}
    ena_media_mensal_por_submercado = {}
    ear_media_diaria = None
    ena_media_diaria = None
    ear_diaria_por_submercado = {}
    ena_diaria_por_submercado = {}
    ear_percentual_diaria_sin = None
    ear_percentual_diaria_por_submercado = {}
    ena_bruta_mwmed_diaria_sin = None
    ena_bruta_mwmed_diaria_por_submercado = {}
    ena_bruta_percentualmlt_diaria_sin = None
    ena_bruta_percentualmlt_diaria_por_submercado = {}
    ena_armazenavel_percentualmlt_diaria_sin = None
    ena_armazenavel_percentualmlt_diaria_por_submercado = {}
    ear_low_thr = None
    ear_high_thr = None
    matriz_cenario_mensal: List[Dict[str, Any]] = []
    matriz_cenario_diaria: List[Dict[str, Any]] = []
    try:
        ear_by_sub, ena_by_sub = _load_ear_ena_monthly_by_submercado(ons)
        for sm, pld_sm in pld_series_by_submercado.items():
            pld_m_sm = _ensure_tz_naive_index(pld_sm).resample("ME").mean()
            ear_sm = ear_by_sub.get(sm)
            ena_sm = ena_by_sub.get(sm)
            pld_vs_ear_mensal_por_submercado[sm] = _safe_corr(pld_m_sm, ear_sm, min_points=3)
            pld_vs_ena_mensal_por_submercado[sm] = _safe_corr(pld_m_sm, ena_sm, min_points=3)

        if ear_by_sub:
            ear_media_mensal_por_submercado = {
                sm: {i.strftime("%Y-%m"): float(v) for i, v in ser.dropna().sort_index().items()}
                for sm, ser in ear_by_sub.items()
            }
            ear_media_mensal = {
                i.strftime("%Y-%m"): float(v)
                for i, v in pd.concat(ear_by_sub, axis=1).mean(axis=1, skipna=True).dropna().sort_index().items()
            }
        if ena_by_sub:
            ena_media_mensal_por_submercado = {
                sm: {i.strftime("%Y-%m"): float(v) for i, v in ser.dropna().sort_index().items()}
                for sm, ser in ena_by_sub.items()
            }
            ena_media_mensal = {
                i.strftime("%Y-%m"): float(v)
                for i, v in pd.concat(ena_by_sub, axis=1).mean(axis=1, skipna=True).dropna().sort_index().items()
            }

        if ear_media_mensal and not pld_series.empty:
            pld_m = _ensure_tz_naive_index(pld_series).resample("ME").mean()
            ear_m = pd.Series({pd.to_datetime(k) + pd.offsets.MonthEnd(0): v for k, v in ear_media_mensal.items()})
            corr_pld_ear_mensal = _safe_corr(pld_m, ear_m, min_points=3)

        # Séries diárias (prioridade DuckDB; fallback CSV somente se necessário)
        con = _duckdb_connect()
        if con is not None:
            try:
                if _duckdb_table_exists(con, "ear_diario_subsistema"):
                    q_ear_d = f"""
                        SELECT DATE_TRUNC('day', {_duckdb_date_expr('ear_data')}) AS dia,
                               UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS sub_raw,
                               SUM({_duckdb_num_expr('ear_verif_subsistema_mwmes')}) AS ear_val,
                               AVG({_duckdb_num_expr('ear_verif_subsistema_percentual')}) AS ear_pct
                        FROM ear_diario_subsistema
                        GROUP BY 1,2
                        HAVING dia IS NOT NULL
                        ORDER BY 1
                    """
                    dfe = con.execute(q_ear_d).fetchdf()
                    if not dfe.empty:
                        dfe["submercado"] = dfe["sub_raw"].map(_normalize_submercado_name)
                        dfe = dfe.dropna(subset=["submercado"])
                        ear_diaria_por_submercado = {}
                        for sm, grp in dfe.groupby("submercado"):
                            ear_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ear_val"])
                                if pd.notna(v)
                            }
                            ear_percentual_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ear_pct"])
                                if pd.notna(v)
                            }
                        dfe_sin = dfe.groupby("dia", as_index=False)["ear_val"].sum()
                        ear_media_diaria = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfe_sin["dia"], dfe_sin["ear_val"])
                            if pd.notna(v)
                        }
                        dfe_pct = dfe.groupby("dia", as_index=False)["ear_pct"].mean()
                        ear_percentual_diaria_sin = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfe_pct["dia"], dfe_pct["ear_pct"])
                            if pd.notna(v)
                        }

                if _duckdb_table_exists(con, "ena_diario_subsistema"):
                    q_ena_d = f"""
                        SELECT DATE_TRUNC('day', {_duckdb_date_expr('ena_data')}) AS dia,
                               UPPER(TRIM(CAST(COALESCE(nom_subsistema, id_subsistema) AS VARCHAR))) AS sub_raw,
                               SUM({_duckdb_num_expr('ena_armazenavel_regiao_mwmed')}) AS ena_val,
                               AVG({_duckdb_num_expr('ena_armazenavel_regiao_percentualmlt')}) AS ena_arm_pct,
                               SUM({_duckdb_num_expr('ena_bruta_regiao_mwmed')}) AS ena_bruta_mwmed,
                               AVG({_duckdb_num_expr('ena_bruta_regiao_percentualmlt')}) AS ena_bruta_pct
                        FROM ena_diario_subsistema
                        GROUP BY 1,2
                        HAVING dia IS NOT NULL
                        ORDER BY 1
                    """
                    dfn = con.execute(q_ena_d).fetchdf()
                    if not dfn.empty:
                        dfn["submercado"] = dfn["sub_raw"].map(_normalize_submercado_name)
                        dfn = dfn.dropna(subset=["submercado"])
                        ena_diaria_por_submercado = {}
                        for sm, grp in dfn.groupby("submercado"):
                            ena_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ena_val"])
                                if pd.notna(v)
                            }
                            ena_bruta_mwmed_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ena_bruta_mwmed"])
                                if pd.notna(v)
                            }
                            ena_bruta_percentualmlt_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ena_bruta_pct"])
                                if pd.notna(v)
                            }
                            ena_armazenavel_percentualmlt_diaria_por_submercado[str(sm)] = {
                                pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                                for i, v in zip(grp["dia"], grp["ena_arm_pct"])
                                if pd.notna(v)
                            }
                        dfn_sin = dfn.groupby("dia", as_index=False)["ena_val"].sum()
                        ena_media_diaria = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfn_sin["dia"], dfn_sin["ena_val"])
                            if pd.notna(v)
                        }
                        dfn_bruta_sin = dfn.groupby("dia", as_index=False)["ena_bruta_mwmed"].sum()
                        ena_bruta_mwmed_diaria_sin = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfn_bruta_sin["dia"], dfn_bruta_sin["ena_bruta_mwmed"])
                            if pd.notna(v)
                        }
                        dfn_bruta_pct = dfn.groupby("dia", as_index=False)["ena_bruta_pct"].mean()
                        ena_bruta_percentualmlt_diaria_sin = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfn_bruta_pct["dia"], dfn_bruta_pct["ena_bruta_pct"])
                            if pd.notna(v)
                        }
                        dfn_arm_pct = dfn.groupby("dia", as_index=False)["ena_arm_pct"].mean()
                        ena_armazenavel_percentualmlt_diaria_sin = {
                            pd.Timestamp(i).strftime("%Y-%m-%d"): float(v)
                            for i, v in zip(dfn_arm_pct["dia"], dfn_arm_pct["ena_arm_pct"])
                            if pd.notna(v)
                        }
            except Exception:
                pass
            finally:
                con.close()

        pld_m_global = _ensure_tz_naive_index(pld_series).resample("ME").mean()
        carga_liquida_m = _ensure_tz_naive_index(carga_liquida).resample("ME").mean() if not carga_liquida.empty else pd.Series(dtype=float)
        termica_pct_m = (pd.DataFrame({"termica": termica, "total": geracao_total}).dropna().query("total > 0").eval("(termica/total)*100").resample("ME").mean() if (not termica.empty and not geracao_total.empty) else pd.Series(dtype=float))

        ear_vals = pd.Series(list((ear_media_mensal or {}).values()), dtype=float).dropna()
        ear_low_thr = float(ear_vals.quantile(0.25)) if not ear_vals.empty else None
        ear_high_thr = float(ear_vals.quantile(0.75)) if not ear_vals.empty else None

        idx = pld_m_global.index
        for extra in [
            (pd.to_datetime(list((ear_media_mensal or {}).keys()), format="%Y-%m", errors="coerce") + pd.offsets.MonthEnd(0)) if ear_media_mensal else pd.DatetimeIndex([]),
            (pd.to_datetime(list((ena_media_mensal or {}).keys()), format="%Y-%m", errors="coerce") + pd.offsets.MonthEnd(0)) if ena_media_mensal else pd.DatetimeIndex([]),
            carga_liquida_m.index if not carga_liquida_m.empty else pd.DatetimeIndex([]),
            termica_pct_m.index if not termica_pct_m.empty else pd.DatetimeIndex([]),
        ]:
            idx = idx.union(extra)

        seen_months = set()
        for month in sorted([i for i in idx if not pd.isna(i)]):
            if pd.Timestamp(month) < pd.Timestamp("2021-01-01"):
                continue
            month_label = month.strftime("%Y-%m")
            if month_label in seen_months:
                continue
            seen_months.add(month_label)
            pld_v = float(pld_m_global.get(month)) if month in pld_m_global.index and pd.notna(pld_m_global.get(month)) else None
            ear_v = ear_media_mensal.get(month.strftime("%Y-%m")) if ear_media_mensal else None
            ena_v = ena_media_mensal.get(month.strftime("%Y-%m")) if ena_media_mensal else None
            carga_v = float(carga_liquida_m.get(month)) if month in carga_liquida_m.index and pd.notna(carga_liquida_m.get(month)) else None
            term_v = float(termica_pct_m.get(month)) if month in termica_pct_m.index and pd.notna(termica_pct_m.get(month)) else None

            if pld_v is None and carga_v is None and term_v is None:
                # Evita meses antigos apenas hidrológicos (sem PLD/carga/operação)
                continue

            if pld_v is None or ear_v is None:
                cenario = "dados_insuficientes"
            elif pld_v >= PLD_TETO_ESTRUTURAL * 0.8 and ear_low_thr is not None and ear_v < ear_low_thr:
                cenario = "estresse_hidrico"
            elif pld_v <= PLD_PISO * 1.2 and ear_high_thr is not None and ear_v > ear_high_thr:
                cenario = "abundancia_hidrica"
            elif term_v is not None and term_v > 25 and pld_medio is not None and pld_v > pld_medio:
                cenario = "pressao_termica"
            else:
                cenario = "equilibrio_operacional"

            matriz_cenario_mensal.append({
                "mes": month_label,
                "pld_medio": pld_v,
                "ear_medio": ear_v,
                "ena_media": ena_v,
                "carga_liquida_media": carga_v,
                "percentual_termica_medio": term_v,
                "cenario": cenario,
            })
    except Exception as e:
        step_errors["ear_ena_vs_pld_por_submercado"] = str(e)

    percentual_termica_h = pd.Series(dtype=float)
    if not termica.empty and not geracao_total.empty:
        df_pctt = pd.DataFrame({"termica": termica, "total": geracao_total}).dropna()
        df_pctt = df_pctt[df_pctt["total"] > 0]
        if not df_pctt.empty:
            percentual_termica_h = (df_pctt["termica"] / df_pctt["total"]) * 100

    corr_pld_pct_termica = _safe_corr(pld_series, percentual_termica_h, min_points=24)

    sigma_pld_intradiario = None
    sigma_carga_intradiario = None
    sigma_eolica_intradiario = None
    if not pld_series.empty:
        s = pld_series.groupby(pld_series.index.floor("D")).std().dropna()
        if not s.empty:
            sigma_pld_intradiario = float(s.mean())
    if not carga_sin.empty:
        s = carga_sin.groupby(carga_sin.index.floor("D")).std().dropna()
        if not s.empty:
            sigma_carga_intradiario = float(s.mean())
    if not eolica.empty:
        s = eolica.groupby(eolica.index.floor("D")).std().dropna()
        if not s.empty:
            sigma_eolica_intradiario = float(s.mean())

    amplificacao_numerica = None
    if sigma_pld_intradiario is not None:
        base = np.nanmean([v for v in [sigma_carga_intradiario, sigma_eolica_intradiario] if v is not None])
        if not np.isnan(base) and base > 0:
            amplificacao_numerica = bool(sigma_pld_intradiario > (2 * base))

    corr_curtail_ear = None
    classificacao_curtail_ear = "indisponível"
    if ear_medio is not None:
        if ear_high_thr is not None and ear_medio > ear_high_thr and curtailment.get("total_mwh", 0) > 0:
            classificacao_curtail_ear = "estrutural"
        elif ear_low_thr is not None and ear_medio < ear_low_thr and curtailment.get("total_mwh", 0) > 0:
            classificacao_curtail_ear = "restricao_local"
        else:
            classificacao_curtail_ear = "indeterminado"

    intercambio_classificacao = "indisponível"
    intercambio_saturado = None
    try:
        intercambio_series = pd.Series(dtype=float)
        limite_series = pd.Series(dtype=float)

        con = _duckdb_connect()
        if con is not None:
            try:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall() if "intercambio" in str(r[0]).lower()]
                for t in tables:
                    info = con.execute(f"PRAGMA table_info('{t}')").fetchall()
                    cols = {c[1].lower(): c[1] for c in info}
                    ts_col = cols.get("instante") or cols.get("din_instante")
                    interc_col = cols.get("intercambio") or cols.get("val_intercambiomwmed")
                    lim_col = cols.get("limite")
                    if not ts_col or (not interc_col and not lim_col):
                        continue
                    if interc_col:
                        qi = f"SELECT {_duckdb_date_expr(ts_col)} AS ts, {_duckdb_num_expr(interc_col)} AS v FROM {t}"
                        dfi = con.execute(qi).fetchdf().dropna(subset=["ts", "v"])
                        if not dfi.empty:
                            s_i = dfi.groupby("ts")["v"].sum().sort_index()
                            intercambio_series = s_i if intercambio_series.empty else intercambio_series.add(s_i, fill_value=0)
                    if lim_col:
                        ql = f"SELECT {_duckdb_date_expr(ts_col)} AS ts, {_duckdb_num_expr(lim_col)} AS v FROM {t}"
                        dfl = con.execute(ql).fetchdf().dropna(subset=["ts", "v"])
                        if not dfl.empty:
                            s_l = dfl.groupby("ts")["v"].sum().sort_index()
                            limite_series = s_l if limite_series.empty else limite_series.add(s_l, fill_value=0)
            except Exception:
                pass
            finally:
                con.close()

        if not intercambio_series.empty and not limite_series.empty and curtailment.get("total_mwh", 0) > 0:
            df_x = pd.DataFrame({"interc": intercambio_series.abs(), "lim": limite_series.abs()}).dropna()
            if not df_x.empty:
                sat = (df_x["interc"] >= (0.95 * df_x["lim"]))
                intercambio_saturado = bool(sat.any())
                intercambio_classificacao = "transmissao" if sat.any() else "estrutural"
    except Exception:
        intercambio_classificacao = "indisponível"

    regime_abundancia = None
    if (
        dependencia_termica_pct is not None and ear_medio is not None and pld_medio is not None
    ):
        regime_abundancia = bool(dependencia_termica_pct < 15 and (ear_high_thr is not None and ear_medio > ear_high_thr) and pld_medio <= PLD_PISO * 1.15)

    # GFOM x PLD
    gfom_h = _load_gfom_hourly(ons)
    gfom_pct = None
    gfom_pld_corr = None
    gfom_pld_cenario = "indisponível"
    gfom_alto_pld_baixo = None
    gfom_alto_pld_alto = None
    if not gfom_h.empty:
        try:
            gfom_h = gfom_h.copy()
            gfom_h.index = pd.to_datetime(gfom_h.index, errors="coerce", utc=True).tz_localize(None)
            gfom_h = gfom_h[~gfom_h.index.isna()]
            gfom_h = gfom_h.groupby(level=0).sum().sort_index()

            total_ger = float(gfom_h["ger"].sum()) if "ger" in gfom_h else 0
            total_gfom = float(gfom_h["gfom"].sum()) if "gfom" in gfom_h else 0
            if total_ger == 0:
                gfom_pct = "não houve despacho de térmica"
            elif total_gfom == 0:
                gfom_pct = "não houve despacho de térmica fora de mérito"
            else:
                gfom_pct = round(float((total_gfom / total_ger) * 100), 4)

            if not pld_series.empty:
                gfom_pct_h = (gfom_h["gfom"] / gfom_h["ger"].replace(0, np.nan)) * 100
                gfom_pct_h = _ensure_tz_naive_index(gfom_pct_h.replace([np.inf, -np.inf], np.nan))
                gfom_pld_corr = _safe_corr(pld_series, gfom_pct_h, min_points=24)

                df_gp = pd.DataFrame({"pld": pld_series, "gfom_pct": gfom_pct_h}).dropna()
                if not df_gp.empty:
                    pld_low = df_gp["pld"].quantile(0.2)
                    pld_high = df_gp["pld"].quantile(0.8)
                    gfom_high = df_gp["gfom_pct"].quantile(0.8)
                    gfom_low = df_gp["gfom_pct"].quantile(0.2)
                    gfom_alto_pld_baixo = int(((df_gp["pld"] <= pld_low) & (df_gp["gfom_pct"] >= gfom_high)).sum())
                    gfom_alto_pld_alto = int(((df_gp["pld"] >= pld_high) & (df_gp["gfom_pct"] <= gfom_low)).sum())
                    if gfom_alto_pld_baixo > gfom_alto_pld_alto:
                        gfom_pld_cenario = "A: PLD baixo + GFOM alto"
                    else:
                        gfom_pld_cenario = "B: PLD alto + GFOM baixo"
        except Exception as e:
            step_errors["gfom_vs_pld"] = str(e)

    # GFOM x PLD por submercado (PLD horário pode diferir por submercado)
    gfom_vs_pld_por_submercado: Dict[str, Any] = {}
    gfom_sub = _load_gfom_hourly_by_submarket(ons)
    sm_suffix = {
        "NORTE": "n",
        "NORDESTE": "ne",
        "SUL": "s",
        "SUDESTE": "se",
    }
    for sm, gfdf in gfom_sub.items():
        try:
            pld_sm = pld_series_by_submercado.get(sm)
            if pld_sm is None or pld_sm.empty:
                continue
            gfdf = gfdf.copy()
            gfdf.index = pd.to_datetime(gfdf.index, errors="coerce", utc=True).tz_localize(None)
            gfdf = gfdf[~gfdf.index.isna()].groupby(level=0).sum().sort_index()
            gfom_pct_sm = (gfdf["gfom"] / gfdf["ger"].replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
            gfom_pct_sm = _ensure_tz_naive_index(gfom_pct_sm)
            key_pct = f"gfom_{sm_suffix.get(sm, sm.lower())}_pct"
            gfom_vs_pld_por_submercado[sm] = {
                key_pct: round(float((gfdf["gfom"].sum() / gfdf["ger"].sum()) * 100), 4) if gfdf["ger"].sum() > 0 else None,
                "corr": _safe_corr(pld_sm, gfom_pct_sm, min_points=24),
            }
        except Exception as e:
            gfom_vs_pld_por_submercado[sm] = {"erro": str(e)}

    cmo_horario_por_submercado: Dict[str, Dict[str, float]] = {}
    try:
        cmo_by_sm = _load_cmo_hourly_by_submarket()
        for sm, ser in cmo_by_sm.items():
            cmo_horario_por_submercado[sm] = {
                pd.Timestamp(i).strftime("%Y-%m-%d %H:%M:%S"): float(v)
                for i, v in ser.dropna().sort_index().items()
            }
    except Exception as e:
        step_errors["cmo_horario_por_submercado"] = str(e)

    # Curtailement estrutural vs elétrico (nova abordagem)
    curtailment_class_nova = "indisponível"
    if curtailment.get("total_mwh", 0) > 0:
        if intercambio_saturado:
            curtailment_class_nova = "eletrico"
        elif ipr_medio is not None and ipr_medio > 1 and ear_medio is not None and ear_medio > 60 and pld_medio is not None and pld_medio <= PLD_PISO * 1.1:
            curtailment_class_nova = "estrutural"
        else:
            curtailment_class_nova = "operacional"

    # Matriz diária para uso no dashboard (cards c1..c7 por período)
    try:
        pld_h = _ensure_tz_naive_index(pld_series)
        pld_d = pld_h.groupby(pld_h.index.floor("D")).mean() if not pld_h.empty else pd.Series(dtype=float)

        gfom_pct_d = pd.Series(dtype=float)
        if not gfom_h.empty:
            gtmp = gfom_h.copy()
            gtmp.index = pd.to_datetime(gtmp.index, errors="coerce", utc=True).tz_localize(None)
            gtmp = gtmp[~gtmp.index.isna()].groupby(level=0).sum().sort_index()
            if not gtmp.empty:
                gtmp_d = gtmp.groupby(gtmp.index.floor("D")).sum()
                gfom_pct_d = (gtmp_d["gfom"] / gtmp_d["ger"].replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)

        gfom_corr_d = pd.Series(dtype=float)
        if not pld_h.empty and not gfom_h.empty:
            gtmp = gfom_h.copy()
            gtmp.index = pd.to_datetime(gtmp.index, errors="coerce", utc=True).tz_localize(None)
            gtmp = gtmp[~gtmp.index.isna()].groupby(level=0).sum().sort_index()
            gfom_pct_h_local = (gtmp["gfom"] / gtmp["ger"].replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
            df_gp_h = pd.DataFrame({"pld": pld_h, "gfom_pct": gfom_pct_h_local}).dropna()
            if not df_gp_h.empty:
                vals = {}
                for d, grp in df_gp_h.groupby(df_gp_h.index.floor("D")):
                    vals[d] = _safe_corr(grp["pld"], grp["gfom_pct"], min_points=12)
                gfom_corr_d = pd.Series(vals)

        stress_d = pd.Series(dtype=float)
        if not capacidade_disp_h.empty and not carga_sin.empty:
            df_capd = pd.DataFrame({"cap": capacidade_disp_h, "carga": carga_sin}).dropna()
            df_capd = df_capd[df_capd["cap"] > 0]
            if not df_capd.empty:
                stress_d = (df_capd["carga"] / df_capd["cap"]).groupby(df_capd.index.floor("D")).mean()

        ipr_d = ipr_horario.groupby(ipr_horario.index.floor("D")).mean() if not ipr_horario.empty else pd.Series(dtype=float)
        isr_d = isr_horario.groupby(isr_horario.index.floor("D")).mean() if not isr_horario.empty else pd.Series(dtype=float)

        term_dep_d = pd.Series(dtype=float)
        if not termica.empty and not geracao_total.empty:
            dft = pd.DataFrame({"term": termica, "tot": geracao_total}).dropna()
            dft = dft[dft["tot"] > 0]
            if not dft.empty:
                term_dep_d = ((dft["term"] / dft["tot"]) * 100).groupby(dft.index.floor("D")).mean()

        curtailment_d = pd.Series(dtype=float)
        try:
            sol_rec = ((curtailment.get("solar") or {}).get("serie") or [])
            eol_rec = ((curtailment.get("eolica") or {}).get("serie") or [])
            s_sol = _to_series(sol_rec, "valor") if sol_rec else pd.Series(dtype=float)
            s_eol = _to_series(eol_rec, "valor") if eol_rec else pd.Series(dtype=float)
            if not s_sol.empty or not s_eol.empty:
                curtailment_d = s_sol.add(s_eol, fill_value=0).groupby(lambda x: pd.Timestamp(x).floor("D")).sum()
        except Exception:
            curtailment_d = pd.Series(dtype=float)

        ear_daily = pd.Series({pd.to_datetime(k): v for k, v in (ear_media_diaria or {}).items()}) if ear_media_diaria else pd.Series(dtype=float)

        idx_days = pd.DatetimeIndex([])
        for ser in [pld_d, gfom_pct_d, gfom_corr_d, stress_d, ipr_d, isr_d, term_dep_d, curtailment_d, ear_daily]:
            if not ser.empty:
                idx_days = idx_days.union(pd.DatetimeIndex(ser.index))

        for d in sorted(idx_days):
            pldv = pld_d.get(d)
            gpv = gfom_pct_d.get(d)
            gcv = gfom_corr_d.get(d)
            stv = stress_d.get(d)
            ipv = ipr_d.get(d)
            isv = isr_d.get(d)
            tdv = term_dep_d.get(d)
            ctv = curtailment_d.get(d)
            earv = ear_daily.get(d)

            if all((x is None or pd.isna(x)) for x in [pldv, gpv, gcv, stv, ipv, isv]) and (earv is not None and not pd.isna(earv)):
                # Evita dias sem sinais operacionais/econômicos (apenas EAR histórico)
                continue

            if ctv is None or pd.isna(ctv) or ctv <= 0:
                curt_state = "inexistente"
            elif intercambio_saturado:
                curt_state = "eletrico"
            elif (ipv is not None and not pd.isna(ipv) and ipv > 1) and (earv is not None and not pd.isna(earv) and earv > 60) and (pldv is not None and not pd.isna(pldv) and pldv <= PLD_PISO * 1.1):
                curt_state = "estrutural"
            else:
                curt_state = "operacional"

            abund = None
            if tdv is not None and not pd.isna(tdv) and earv is not None and not pd.isna(earv) and pldv is not None and not pd.isna(pldv):
                abund = bool(tdv < 15 and (ear_high_thr is not None and earv > ear_high_thr) and pldv <= PLD_PISO * 1.15)

            matriz_cenario_diaria.append({
                "dia": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "gfom_pct": None if pd.isna(gpv) else round(float(gpv), 4),
                "gfom_vs_pld_corr": None if pd.isna(gcv) else float(gcv),
                "curtailment_estado": curt_state,
                "stress_operacional_medio": None if pd.isna(stv) else float(stv),
                "ipr_medio": None if pd.isna(ipv) else float(ipv),
                "isr_medio": None if pd.isna(isv) else float(isv),
                "regime_abundancia": abund,
                "pld_medio_dia": None if pd.isna(pldv) else float(pldv),
                "ear_medio_dia": None if pd.isna(earv) else float(earv),
            })
    except Exception as e:
        step_errors["matriz_cenario_diaria"] = str(e)

    # Mudança de regime histórica (trimestral)
    mudanca_regime_trimestral = {}
    try:
        capacidade_disp_h = _ensure_tz_naive_index(capacidade_disp_h)
        carga_sin = _ensure_tz_naive_index(carga_sin)
        if not pld_series.empty and not capacidade_disp_h.empty and not carga_sin.empty:
            df_reg = pd.DataFrame({"pld": pld_series, "cap": capacidade_disp_h, "carga": carga_sin}).dropna()
            if not df_reg.empty:
                df_reg["stress"] = df_reg["carga"] / df_reg["cap"].replace(0, np.nan)
                g = df_reg.groupby(df_reg.index.to_period("Q")).agg({"pld": "mean", "stress": "mean"}).dropna()
                for q, row in g.iterrows():
                    if row["stress"] < 0.8 and row["pld"] > pld_series.quantile(0.7):
                        reg = "desalinhamento_estrutural"
                    elif row["stress"] > 1:
                        reg = "estresse_operacional"
                    else:
                        reg = "equilibrio"
                    mudanca_regime_trimestral[str(q)] = reg
    except Exception as e:
        step_errors["mudanca_regime_historica_trimestral"] = str(e)

    painel_horario_renovavel = []
    try:
        pld_h = _ensure_tz_naive_index(pld_series)
        gfom_hx = _load_gfom_hourly(ons)
        gfom_pct_hor = pd.Series(dtype=object)
        if not gfom_hx.empty:
            gx = gfom_hx.copy()
            gx.index = pd.to_datetime(gx.index, errors="coerce")
            gx = gx[~gx.index.isna()].groupby(level=0).sum().sort_index()
            ger_s = gx.get("ger", pd.Series(dtype=float)).astype(float)
            gf_s = gx.get("gfom", pd.Series(dtype=float)).astype(float)
            pct = ((gf_s / ger_s.replace(0, np.nan)) * 100).round(4)
            pct = pct.replace([np.inf, -np.inf], np.nan)
            gfom_pct_hor = pct.astype(object)
            gfom_pct_hor.loc[ger_s.fillna(0) == 0] = "não houve despacho de térmica"
            gfom_pct_hor.loc[(ger_s.fillna(0) > 0) & (gf_s.fillna(0) == 0)] = "não houve térmica fora de mérito"

        ear_d_series = pd.Series({pd.to_datetime(k): v for k, v in (ear_media_diaria or {}).items()}) if ear_media_diaria else pd.Series(dtype=float)
        ena_d_series = pd.Series({pd.to_datetime(k): v for k, v in (ena_media_diaria or {}).items()}) if ena_media_diaria else pd.Series(dtype=float)

        restr_h_df = restr_h.copy() if isinstance(restr_h, pd.DataFrame) else pd.DataFrame()
        if not restr_h_df.empty:
            restr_h_df["instante"] = pd.to_datetime(restr_h_df["instante"], errors="coerce")
            restr_h_df = restr_h_df.dropna(subset=["instante"]).sort_values("instante")

        idxh = pd.DatetimeIndex([])
        for ser in [ipr_horario, isr_horario, pld_h, gfom_pct_hor if isinstance(gfom_pct_hor, pd.Series) else pd.Series(dtype=float), carga_liquida]:
            if isinstance(ser, pd.Series) and not ser.empty:
                idxh = idxh.union(pd.DatetimeIndex(ser.index))
        if not restr_h_df.empty:
            idxh = idxh.union(pd.DatetimeIndex(restr_h_df["instante"]))

        for t in sorted(idxh):
            rec = {
                "instante": pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
                "ipr": None if (ipr_horario.empty or pd.isna(ipr_horario.get(t))) else float(ipr_horario.get(t)),
                "isr": None if (isr_horario.empty or pd.isna(isr_horario.get(t))) else float(isr_horario.get(t)),
                "pld": None if (pld_h.empty or pd.isna(pld_h.get(t))) else float(pld_h.get(t)),
                "gfom_pct": None if (not isinstance(gfom_pct_hor, pd.Series) or gfom_pct_hor.empty or pd.isna(gfom_pct_hor.get(t))) else gfom_pct_hor.get(t),
                "ear": None if (ear_d_series.empty or pd.isna(ear_d_series.get(pd.Timestamp(t).floor("D")))) else float(ear_d_series.get(pd.Timestamp(t).floor("D"))),
                "ena": None if (ena_d_series.empty or pd.isna(ena_d_series.get(pd.Timestamp(t).floor("D")))) else float(ena_d_series.get(pd.Timestamp(t).floor("D"))),
                "cod_razaorestricao_solar": "sem restrição",
                "dsc_restricao_solar": "sem restrição",
                "cod_razaorestricao_eolica": "sem restrição",
                "dsc_restricao_eolica": "sem restrição",
                "val_geracaolimitada_solar": None,
                "val_geracaolimitada_eolica": None,
            }
            if not restr_h_df.empty:
                row = restr_h_df[restr_h_df["instante"] == t]
                if not row.empty:
                    rr = row.iloc[0]
                    rec["cod_razaorestricao_solar"] = rr.get("cod_razaorestricao_solar") or "sem restrição"
                    rec["dsc_restricao_solar"] = rr.get("dsc_restricao_solar") or "sem restrição"
                    rec["cod_razaorestricao_eolica"] = rr.get("cod_razaorestricao_eolica") or "sem restrição"
                    rec["dsc_restricao_eolica"] = rr.get("dsc_restricao_eolica") or "sem restrição"
                    rec["val_geracaolimitada_solar"] = None if pd.isna(rr.get("val_geracaolimitada_solar")) else float(rr.get("val_geracaolimitada_solar"))
                    rec["val_geracaolimitada_eolica"] = None if pd.isna(rr.get("val_geracaolimitada_eolica")) else float(rr.get("val_geracaolimitada_eolica"))

            if rec["ipr"] is None and rec["isr"] is None and rec["pld"] is None and rec["gfom_pct"] is None and rec["cod_razaorestricao_solar"] == "sem restrição" and rec["cod_razaorestricao_eolica"] == "sem restrição":
                continue
            painel_horario_renovavel.append(rec)
    except Exception as e:
        step_errors["painel_horario_renovavel"] = str(e)

    gfom_horario = []
    despacho_termico_horario = []
    try:
        if isinstance(gfom_h, pd.DataFrame) and not gfom_h.empty:
            gfh = gfom_h.copy()
            gfh.index = pd.to_datetime(gfh.index, errors="coerce", utc=True).tz_localize(None)
            gfh = gfh[~gfh.index.isna()].groupby(level=0).sum().sort_index()
            gfom_horario = [
                {"instante": i.strftime("%Y-%m-%d %H:%M:%S"), "valor": float(v)}
                for i, v in gfh.get("gfom", pd.Series(dtype=float)).dropna().items()
            ]
            despacho_termico_horario = [
                {"instante": i.strftime("%Y-%m-%d %H:%M:%S"), "valor": float(v)}
                for i, v in gfh.get("ger", pd.Series(dtype=float)).dropna().items()
            ]
    except Exception as e:
        step_errors["gfom_horario_series"] = str(e)

    # -----------------------------------------------------------------
    # Decomposição Econômica Operacional Horária (determinística)
    # -----------------------------------------------------------------
    economic: Dict[str, Any] = {}
    try:
        # Séries-base
        pld_h = _ensure_tz_naive_index(pld_series)
        if not pld_series_by_submercado:
            pld_series_by_submercado = {}

        # CMO horário por submercado (média de :00 e :30)
        cmo_by_sm = _load_cmo_hourly_by_submarket()

        # Escolha determinística do submercado dominante para CVaR implícito
        dom_order = ["SUDESTE", "SUL", "NORDESTE", "NORTE"]
        dominant_sm = None
        for sm in dom_order:
            if sm in pld_series_by_submercado and sm in cmo_by_sm and not pld_series_by_submercado[sm].empty and not cmo_by_sm[sm].empty:
                dominant_sm = sm
                break
        if dominant_sm is None:
            commons = [sm for sm in pld_series_by_submercado.keys() if sm in cmo_by_sm]
            if commons:
                dominant_sm = commons[0]

        if dominant_sm is not None:
            pld_dom = _ensure_tz_naive_index(pld_series_by_submercado.get(dominant_sm, pd.Series(dtype=float)))
            cmo_dom = _ensure_tz_naive_index(cmo_by_sm.get(dominant_sm, pd.Series(dtype=float)))
            if not pld_dom.empty:
                pld_h = pld_dom
        else:
            cmo_dom = pd.Series(dtype=float)

        # GFOM componentes
        gfom_comp = _load_gfom_components_hourly()
        thermal_inflex = _ensure_tz_naive_index(gfom_comp.get("inflex_pura", pd.Series(dtype=float))) if not gfom_comp.empty else pd.Series(dtype=float)
        thermal_merit = _ensure_tz_naive_index(gfom_comp.get("ordem_merito", pd.Series(dtype=float))) if not gfom_comp.empty else pd.Series(dtype=float)

        # Capacidade sincronizada total
        disp = _load_disponibilidade_horaria()

        disp_sync_uhe = disp.get("uhe", pd.Series(dtype=float))
        disp_sync_ute = disp.get("ute", pd.Series(dtype=float))
        disp_sync_utn = disp.get("utn", pd.Series(dtype=float))
        disp_sync_total = disp.get("total", pd.Series(dtype=float))

        # Curtailment/renovável disponível horários
        curtailed_h = pd.Series(dtype=float)
        avail_ren_h = pd.Series(dtype=float)
        if isinstance(restr_h, pd.DataFrame) and not restr_h.empty:
            rr = restr_h.copy()
            rr["instante"] = pd.to_datetime(rr["instante"], errors="coerce")
            rr = rr.dropna(subset=["instante"])
            curtailed_h = _ensure_tz_naive_index(
                pd.Series(
                    pd.to_numeric(rr.get("val_geracaolimitada_solar"), errors="coerce").fillna(0).values
                    + pd.to_numeric(rr.get("val_geracaolimitada_eolica"), errors="coerce").fillna(0).values,
                    index=rr["instante"],
                )
            )
            avail_ren_h = _ensure_tz_naive_index(pd.Series(pd.to_numeric(rr.get("renov_disponivel"), errors="coerce").values, index=rr["instante"]))

        # CVU semanal -> horário
        cvu_weekly = _load_cvu_weekly_series(ons)

        # Índice comum horário para computar todos os termos de forma vetorizada
        idx = pd.DatetimeIndex([])
        for ser in [pld_h, cmo_dom, carga_sin, solar, eolica, termica, hidro, nuclear, thermal_inflex, thermal_merit, disp_sync_total, curtailed_h, avail_ren_h]:
            if isinstance(ser, pd.Series) and not ser.empty:
                idx = idx.union(pd.DatetimeIndex(ser.index))

        if not idx.empty:
            idx = idx.sort_values().unique()
            df_e = pd.DataFrame(index=idx)
            df_e.index = pd.to_datetime(df_e.index)
            year = df_e.index.year
            df_e["pld"] = pd.to_numeric(pld_h.reindex(idx), errors="coerce")
            df_e["ano"] = df_e.index.year
            df_e["pld_teto"] = df_e["pld"].groupby(year).transform("max")
            df_e["pld_no_teto"] = df_e["pld"] >= df_e["pld_teto"] * 0.95
            df_e["cmo"] = pd.to_numeric(cmo_dom.reindex(idx), errors="coerce")
            df_e["load"] = pd.to_numeric(carga_sin.reindex(idx), errors="coerce")
            df_e["solar"] = pd.to_numeric(solar.reindex(idx), errors="coerce")
            df_e["wind"] = pd.to_numeric(eolica.reindex(idx), errors="coerce")
            df_e["thermal_total"] = pd.to_numeric(termica.reindex(idx), errors="coerce")
            df_e["hydro"] = pd.to_numeric(hidro.reindex(idx), errors="coerce")
            df_e["nuclear"] = pd.to_numeric(nuclear.reindex(idx), errors="coerce")
            df_e["thermal_inflex"] = pd.to_numeric(thermal_inflex.reindex(idx), errors="coerce")
            df_e["thermal_merit"] = pd.to_numeric(thermal_merit.reindex(idx), errors="coerce")
            df_e["disp_sync_total"] = pd.to_numeric(disp_sync_total.reindex(idx), errors="coerce")
            df_e["disp_sync_uhe"] = pd.to_numeric(disp_sync_uhe.reindex(idx), errors="coerce")
            df_e["disp_sync_ute"] = pd.to_numeric(disp_sync_ute.reindex(idx), errors="coerce")
            df_e["disp_sync_utn"] = pd.to_numeric(disp_sync_utn.reindex(idx), errors="coerce")
            df_e["geracao_total"] = pd.to_numeric(geracao_total.reindex(idx), errors="coerce")
            df_e["curtailed"] = pd.to_numeric(curtailed_h.reindex(idx), errors="coerce")
            df_e["avail_ren"] = pd.to_numeric(avail_ren_h.reindex(idx), errors="coerce")

            if not cvu_weekly.empty:
                cvu_ref = _ensure_tz_naive_index(cvu_weekly.sort_index())
                df_e["cvu_semana"] = cvu_ref.reindex(idx, method="ffill")
                if df_e["cvu_semana"].isna().all():
                    df_e["cvu_semana"] = cvu_ref.reindex(idx, method="bfill")
            else:
                df_e["cvu_semana"] = np.nan

            # EAR/ENA diários -> horários
            ear_pct_d = pd.Series({pd.to_datetime(k): v for k, v in (ear_percentual_diaria_sin or {}).items()}) if ear_percentual_diaria_sin else pd.Series(dtype=float)
            ena_arm_pct_d = pd.Series({pd.to_datetime(k): v for k, v in (ena_armazenavel_percentualmlt_diaria_sin or {}).items()}) if ena_armazenavel_percentualmlt_diaria_sin else pd.Series(dtype=float)
            if not ear_pct_d.empty:
                df_e["ear_pct"] = ear_pct_d.reindex(pd.DatetimeIndex(df_e.index.floor("D"))).values
            else:
                df_e["ear_pct"] = np.nan
            if not ena_arm_pct_d.empty:
                df_e["ena_arm_pct"] = ena_arm_pct_d.reindex(pd.DatetimeIndex(df_e.index.floor("D"))).values
            else:
                df_e["ena_arm_pct"] = np.nan

            # Step 1 — Normalizações
            df_e["EAR_norm"] = (df_e["ear_pct"] / 100.0).clip(lower=0, upper=1)
            df_e["ENA_norm"] = (df_e["ena_arm_pct"] / 100.0).clip(lower=0, upper=1)
            df_e["Load_norm"] = df_e["load"] / df_e["disp_sync_total"].replace(0, np.nan)
            df_e["Curtailment_norm"] = df_e["curtailed"] / df_e["avail_ren"].replace(0, np.nan)
            df_e["Thermal_inflex_ratio"] = (df_e["thermal_inflex"] / df_e["thermal_total"].replace(0, np.nan)).clip(lower=0, upper=1)

            # Step 2-10
            
            df_e["SIN_cost_R$/h"] = df_e["load"] * df_e["pld"]
            df_e["thermal_real_cost"] = df_e["thermal_total"] * df_e["cvu_semana"]
            df_e["thermal_merit_cost"] = df_e["thermal_merit"] * df_e["cvu_semana"]
            df_e["thermal_prudential_dispatch"] = df_e["thermal_total"] - df_e["thermal_merit"]

            df_e["mandatory_generation"] = df_e[["wind", "solar", "nuclear", "thermal_inflex"]].sum(axis=1, min_count=1)
            df_e["required_hydro"] = (df_e["load"] - df_e["mandatory_generation"]).clip(lower=0)
            df_e["Hydro_gap"] = df_e["hydro"] - df_e["required_hydro"]
            tol = 1e-6
            df_e["system_state"] = np.where(
                df_e["Hydro_gap"] > tol,
                "Hydro Preservation",
                np.where(df_e["Hydro_gap"].abs() <= tol, "Hydro Necessary", "Hydro Deficit"),
            )

            df_e["Hydro_preserved"] = (df_e["disp_sync_uhe"] - df_e["hydro"]).clip(lower=0)
            df_e["Water_value_R$/h"] = df_e["Hydro_preserved"] * df_e["cmo"]
            df_e["Curtailment_loss_R$/h"] = df_e["curtailed"] * df_e["pld"]
            df_e["avoidable_curtailment"] = df_e["curtailed"] * (1 - df_e["Thermal_inflex_ratio"].fillna(1))
            df_e["CVaR_implicit"] = (df_e["pld"] - df_e["cmo"]).clip(lower=0)
            df_e.loc[df_e["pld_no_teto"], "CVaR_implicit"] = np.nan
            df_e["Risk_Aversion_Gap"] = df_e["CVaR_implicit"] - df_e["cvu_semana"]
            df_e["T_prudencia"] = np.where(
                df_e["cmo"] > df_e["pld"],
                df_e["Hydro_preserved"] * (df_e["cmo"] - df_e["pld"]),
                0
            )
            df_e["T_eletric"] = df_e["thermal_merit_cost"]
            df_e["T_hidro"] = df_e["Water_value_R$/h"]
            df_e["T_sistemica"] = df_e["geracao_total"] * (df_e["cmo"] - df_e["pld"])
            df_e["T_total"] = (df_e["T_eletric"] + df_e["T_hidro"] + df_e["T_prudencia"] + df_e["T_sistemica"])
            df_e["infra_marginal_rent"] = df_e["SIN_cost_R$/h"] - (df_e["T_eletric"] + df_e["T_hidro"] + df_e["T_prudencia"] + df_e["T_sistemica"])

            economic = {
                "dominant_submarket": dominant_sm,
                "normalization_hourly": {
                    "EAR_norm": _series_to_hourly_dict(df_e["EAR_norm"]),
                    "ENA_norm": _series_to_hourly_dict(df_e["ENA_norm"]),
                    "Load_norm": _series_to_hourly_dict(df_e["Load_norm"]),
                    "Curtailment_norm": _series_to_hourly_dict(df_e["Curtailment_norm"]),
                    "Thermal_inflex_ratio": _series_to_hourly_dict(df_e["Thermal_inflex_ratio"]),
                },
                "sin_cost_hourly": _series_to_hourly_dict(df_e["SIN_cost_R$/h"], ndigits=4),
                "infra_marginal_rent_hourly": _series_to_hourly_dict(df_e["infra_marginal_rent"], ndigits=4),
                "disp_sync_uhe_hourly": _series_to_hourly_dict(df_e["disp_sync_uhe"], ndigits=4),
                "geracao_total_hourly": _series_to_hourly_dict(df_e["geracao_total"], ndigits=4),
                "pld_hourly": _series_to_hourly_dict(df_e["pld"], ndigits=4),
                "cmo_hourly": _series_to_hourly_dict(df_e["cmo"], ndigits=4),
                "thermal_real_cost_hourly": _series_to_hourly_dict(df_e["thermal_real_cost"], ndigits=4),
                "thermal_merit_cost_hourly": _series_to_hourly_dict(df_e["thermal_merit_cost"], ndigits=4),
                "thermal_prudential_dispatch_hourly": _series_to_hourly_dict(df_e["thermal_prudential_dispatch"], ndigits=4),
                "required_hydro_hourly": _series_to_hourly_dict(df_e["required_hydro"], ndigits=4),
                "mandatory_generation_hourly": _series_to_hourly_dict(df_e["mandatory_generation"], ndigits=4),
                "hydro_gap_hourly": _series_to_hourly_dict(df_e["Hydro_gap"], ndigits=4),
                "system_state_hourly": {pd.Timestamp(i).strftime("%Y-%m-%d %H:%M:%S"): str(v) for i, v in df_e["system_state"].items()},
                "water_value_hourly": _series_to_hourly_dict(df_e["Water_value_R$/h"], ndigits=4),
                "curtailment_loss_hourly": _series_to_hourly_dict(df_e["Curtailment_loss_R$/h"], ndigits=4),
                "avoidable_curtailment_hourly": _series_to_hourly_dict(df_e["avoidable_curtailment"], ndigits=4),
                "CVaR_implicit_hourly": _series_to_hourly_dict(df_e["CVaR_implicit"], ndigits=4),
                "Risk_Aversion_Gap_hourly": _series_to_hourly_dict(df_e["Risk_Aversion_Gap"], ndigits=4),
                "T_prudencia_hourly": _series_to_hourly_dict(df_e["T_prudencia"], ndigits=4),
                "T_sistemica_hourly": _series_to_hourly_dict(df_e["T_sistemica"], ndigits=4),
                "T_total_hourly": _series_to_hourly_dict(df_e["T_total"], ndigits=4),
                "T_eletric_hourly": _series_to_hourly_dict(df_e["T_eletric"], ndigits=4),
                "T_hidro_hourly": _series_to_hourly_dict(df_e["T_hidro"], ndigits=4),
            }
    except Exception as e:
        step_errors["economic_decomposition"] = str(e)

    return {
        "status": "parcial" if step_errors else "disponível",
        "diagnostico_etapas": step_errors,
        "margem_estrutural_oferta": margem_oferta,
        "dependencia_termica_efetiva_pct": dependencia_termica_pct,
        "regime_abundancia": regime_abundancia,
        "ena_media": ena_media,
        "ear_media_mensal": ear_media_mensal,
        "ena_media_mensal": ena_media_mensal,
        "ear_media_mensal_por_submercado": ear_media_mensal_por_submercado,
        "ena_media_mensal_por_submercado": ena_media_mensal_por_submercado,
        "ear_diaria_sin": ear_media_diaria,
        "ena_diaria_sin": ena_media_diaria,
        "ear_diaria_por_submercado": ear_diaria_por_submercado,
        "ena_diaria_por_submercado": ena_diaria_por_submercado,
        "ear_percentual_diaria_sin": ear_percentual_diaria_sin,
        "ear_percentual_diaria_por_submercado": ear_percentual_diaria_por_submercado,
        "ena_bruta_mwmed_diaria_sin": ena_bruta_mwmed_diaria_sin,
        "ena_bruta_mwmed_diaria_por_submercado": ena_bruta_mwmed_diaria_por_submercado,
        "ena_bruta_percentualmlt_diaria_sin": ena_bruta_percentualmlt_diaria_sin,
        "ena_bruta_percentualmlt_diaria_por_submercado": ena_bruta_percentualmlt_diaria_por_submercado,
        "ena_armazenavel_percentualmlt_diaria_sin": ena_armazenavel_percentualmlt_diaria_sin,
        "ena_armazenavel_percentualmlt_diaria_por_submercado": ena_armazenavel_percentualmlt_diaria_por_submercado,
        "ear_media_diaria": ear_media_diaria,
        "ena_media_diaria": ena_media_diaria,
        "matriz_cenario_mensal": matriz_cenario_mensal,
        "matriz_cenario_diaria": matriz_cenario_diaria,
        "horas_renovavel_gt_carga_liquida": horas_renovavel_gt_carga_liquida,
        "curtailment_percentual_total": curtailment.get("curtailment_pct_total"),
        "correlacoes": {
            "curtailment_vs_ear": corr_curtail_ear,
            "pld_vs_carga_liquida": corr_pld_carga_liquida,
            "pld_vs_carga_liquida_rolling_90d": rolling_corr_90d,
            "pld_vs_ear_mensal": corr_pld_ear_mensal,
            "pld_vs_percentual_termica": corr_pld_pct_termica,
            "pld_vs_ear_mensal_por_submercado": pld_vs_ear_mensal_por_submercado,
            "pld_vs_ena_mensal_por_submercado": pld_vs_ena_mensal_por_submercado,
        },
        "classificacoes": {
            "curtailment_x_ear": classificacao_curtail_ear,
            "curtailment_x_intercambio": intercambio_classificacao,
            "curtailment_estrutural_vs_eletrico": curtailment_class_nova,
        },
        "metodologia": {
            "gfom_vs_pld": "GFOM% horário = (val_verifgfom / val_verifgeracao)*100; correlação de Pearson com PLD horário após alinhamento temporal.",
            "margem_operativa_real": "margem = (capacidade_disponivel_real - carga)/carga; margem média mensal = média da margem horária no mês; margem p5 = percentil 5% da margem horária no mês.",
            "curtailment": "usa val_geracaolimitada (TM) como limitação renovável; razões por fonte (solar/eólica) via cod_razaorestricao e dsc_restricao.",
            "ipr_isr": "IPR horário = geracao_renovavel_disponivel/carga; ISR horário = geracao_renovavel_disponivel/carga_liquida.",
            "regime_abundancia": "True quando dependência térmica <15%, EAR acima do quartil superior da série e PLD <= 1.15*piso regulatório.",
            "mudanca_regime_trimestral": "desalinhamento_estrutural: stress<0.8 e PLD no quantil alto; estresse_operacional: stress>1; senão equilíbrio.",
        },
        "volatilidade_intradiaria": {
            "sigma_pld": sigma_pld_intradiario,
            "sigma_carga": sigma_carga_intradiario,
            "sigma_eolica": sigma_eolica_intradiario,
            "hipotese_amplificacao_numerica": amplificacao_numerica,
        },
        "aderencia_fisico_economica": {
            "pld_vs_carga_liquida": corr_pld_carga_liquida,
            "pld_vs_ear_mensal": corr_pld_ear_mensal,
            "pld_vs_percentual_termica": corr_pld_pct_termica,
            "pld_vs_ear_mensal_por_submercado": pld_vs_ear_mensal_por_submercado,
            "pld_vs_ena_mensal_por_submercado": pld_vs_ena_mensal_por_submercado,
            "gfom_pct": gfom_pct,
            "gfom_horario_mwmed": gfom_horario,
            "despacho_termico_horario_mwmed": despacho_termico_horario,
            "gfom_vs_pld_corr": gfom_pld_corr,
            "gfom_vs_pld_cenario": gfom_pld_cenario,
            "horas_cenario_A": gfom_alto_pld_baixo,
            "horas_cenario_B": gfom_alto_pld_alto,
            "gfom_vs_pld_por_submercado": gfom_vs_pld_por_submercado,
            "cmo_horario_por_submercado": cmo_horario_por_submercado,
        },
        "capacidade_operativa_real": {
            "capacidade_disponivel_real_media_mw": float(capacidade_disp_h.mean()) if not capacidade_disp_h.empty else None,
            "margem_operativa_media_mensal": margem_operativa_media_mensal,
            "margem_operativa_p5_mensal": margem_operativa_p5_mensal,
            "stress_operacional_medio": stress_operacional_medio,
            "stress_operacional_horario": stress_operacional_horario,
            "capacidade_instalada_ativa_por_fonte_mw": capacidade_instalada_ativa_por_fonte,
            "tendencia_estrutural_mensal": tendencia_estrutural_mensal,
        },
        "indices_renovaveis": {
            "ipr_horario": {i.strftime("%Y-%m-%d %H:%M:%S"): float(v) for i, v in ipr_horario.dropna().items()} if not ipr_horario.empty else {},
            "isr_horario": {i.strftime("%Y-%m-%d %H:%M:%S"): float(v) for i, v in isr_horario.dropna().items()} if not isr_horario.empty else {},
            "ipr_ultimo": ipr_medio,
            "isr_ultimo": isr_medio,
        },
        "painel_horario_renovavel": painel_horario_renovavel,
        "economic": economic,
        "mudanca_regime_historica_trimestral": mudanca_regime_trimestral,
    }


def _classificar_curtailment(
    curtailment_total: float,
    ear_medio: Optional[float],
    pld_medio: Optional[float]
) -> str:

    if curtailment_total <= 0:
        return "inexistente"

    if ear_medio and ear_medio > 70 and pld_medio and pld_medio <= PLD_PISO * 1.05:
        return "excesso_estrutural"

    if ear_medio and ear_medio < 50:
        return "seguranca_operacional"

    return "restricao_rede"



# =====================================================================
# ANÁLISE TÉRMICA REVISADA (V5) - COM DUPLA PERSPECTIVA
# =====================================================================

def calcular_razao_cvu_pld(pld_medio: Optional[float], cvu_medio: Optional[float]) -> Optional[float]:
    """
    Calcula a razão CVU/PLD (indicador fundamental).
    
    Retorna:
    - < 0.8: CVU significativamente menor que PLD
    - 0.8-0.95: CVU próximo do PLD
    - 0.95-1.0: CVU muito próximo do PLD
    - 1.0-1.5: CVU maior que PLD
    - > 1.5: CVU muito maior que PLD (folga estrutural)
    """
    if pld_medio is None or cvu_medio is None or pld_medio <= 0:
        return None
    
    return cvu_medio / pld_medio


def calcular_margem_seguranca_sistema(pld_medio: Optional[float], cvu_medio: Optional[float]) -> Optional[float]:
    """
    Calcula margem de segurança do SISTEMA.
    
    Margem = ((PLD - CVU) / PLD) × 100%  se PLD > CVU
           = 0%                          se PLD <= CVU
    
    Interpretação (perspectiva do sistema):
    - > 20%: Margem adequada
    - 10-20%: Margem reduzida
    - 5-10%: Margem crítica
    - < 5%: Margem insuficiente
    - = 0%: CVU >= PLD (risco de custos)
    """
    if pld_medio is None or cvu_medio is None or pld_medio <= 0:
        return None
    
    if pld_medio > cvu_medio:
        return ((pld_medio - cvu_medio) / pld_medio) * 100
    else:
        return 0.0


def calcular_margem_vs_teto(cvu_medio: Optional[float]) -> Optional[float]:
    """
    Calcula margem de segurança em relação ao teto estrutural.
    
    Margem = ((Teto estrutural - CVU) / Teto estrutural) × 100%
    
    Interpretação:
    - > 5%: Margem adequada
    - 1-5%: Margem reduzida
    - < 1%: Margem crítica
    - <= 0%: Teto comprometido
    """
    if cvu_medio is None:
        return None
    
    return ((PLD_TETO_ESTRUTURAL - cvu_medio) / PLD_TETO_ESTRUTURAL) * 100


def calcular_viabilidade_termica(pld_medio: Optional[float], cvu_medio: Optional[float]) -> Dict[str, Any]:
    """
    Analisa viabilidade das térmicas (perspectiva do GERADOR).
    
    Retorna:
    - spread absoluto (R$/MWh)
    - viabilidade econômica (booleana)
    - classificação da perspectiva do gerador
    """
    if pld_medio is None or cvu_medio is None:
        return {
            "spread_absoluto": None,
            "viabilidade_economica": None,
            "perspectiva_gerador": "indisponível"
        }
    
    spread = pld_medio - cvu_medio
    
    if spread > 0:
        return {
            "spread_absoluto": spread,
            "viabilidade_economica": True,
            "perspectiva_gerador": "competitiva",
            "descricao": "Despacho economicamente viável para térmicas"
        }
    else:
        return {
            "spread_absoluto": spread,
            "viabilidade_economica": False,
            "perspectiva_gerador": "estrutural",
            "descricao": "Despacho por necessidade do sistema (EAR baixo ou restrição)"
        }


def calcular_dependencia_termica_efetiva(
    razao_cvu_pld: Optional[float], 
    ear_medio: Optional[float]
) -> Optional[float]:
    """
    Calcula dependência térmica EFETIVA considerando contexto hídrico.
    
    Fórmula revisada: Dependência = max(0, (razao_cvu_pld - 0.8)) × (1 - EAR_normalizado)
    
    Onde:
    - razao_cvu_pld - 0.8: penaliza apenas quando CVU > 80% do PLD
    - 1 - EAR_normalizado: inverso da condição hídrica
    
    Interpretação:
    - Baixa (< 0.1): Sistema com folga
    - Moderada (0.1-0.3): Atenção
    - Alta (0.3-0.5): Dependência significativa
    - Crítica (> 0.5): Sistema altamente dependente
    """
    if razao_cvu_pld is None or ear_medio is None:
        return None
    
    # Só considera dependência se CVU > 80% do PLD
    excesso_sobre_limiar = max(0, razao_cvu_pld - 0.8)
    
    # Normaliza EAR (0-1)
    ear_norm = max(0, min(1, ear_medio / 100))
    
    # Dependência = excesso de custo × (1 - folga hídrica)
    dependencia = excesso_sobre_limiar * (1 - ear_norm)
    
    return dependencia


def calcular_indicadores_termicos_revisados(
    pld_medio: Optional[float], 
    cvu_medio: Optional[float], 
    ear_medio: Optional[float]
) -> Dict[str, Any]:
    """
    Calcula indicadores térmicos com DUPLA PERSPECTIVA.
    
    Versão V5: Correção do conceito - CVU alto vs PLD baixo = FOLGA, não risco.
    """
    
    # 1. CÁLCULOS FUNDAMENTAIS
    razao_cvu_pld = calcular_razao_cvu_pld(pld_medio, cvu_medio)
    percentual_cvu_pld = razao_cvu_pld * 100 if razao_cvu_pld is not None else None
    
    margem_seguranca = calcular_margem_seguranca_sistema(pld_medio, cvu_medio)
    margem_vs_teto = calcular_margem_vs_teto(cvu_medio)
    dependencia_efetiva = calcular_dependencia_termica_efetiva(razao_cvu_pld, ear_medio)
    
    # Análise de viabilidade do gerador
    analise_gerador = calcular_viabilidade_termica(pld_medio, cvu_medio)
    
    # 2. ANÁLISE DO SISTEMA (PERSPECTIVA DA MODICIDADE TARIFÁRIA)
    
    # Cenário 1: CVU muito maior que PLD → FOLGA ESTRUTURAL
    if percentual_cvu_pld is not None and percentual_cvu_pld > 150:
        classificacao_sistema = "folga_estrutural"
        risco_sistêmico = "muito_baixo"
        descricao_sistema = (
            f"Sistema operando com folga ampla. "
            f"CVU (💰 {cvu_medio:.1f}) muito acima do PLD (💰 {pld_medio:.1f}) "
            f"indica térmicas fora do despacho econômico."
        )
        recomendacao_sistema = "Operação normal. Modicidade tarifária preservada."
    
    # Cenário 2: CVU entre 100-150% do PLD → RISCO DE CUSTOS
    elif percentual_cvu_pld and percentual_cvu_pld >= 100:
        classificacao_sistema = "risco_custo"
        risco_sistêmico = "alto" if ear_medio and ear_medio < 50 else "moderado"
        descricao_sistema = (
            f"Sistema pode requerer despacho térmico com prejuízo econômico. "
            f"CVU (R$ {cvu_medio:.1f}) ≥ PLD (R$ {pld_medio:.1f})."
        )
        if ear_medio and ear_medio < 50:
            recomendacao_sistema = (
                "Despacho térmico necessário por escassez hídrica. "
                "Monitorar impactos tarifários."
            )
        else:
            recomendacao_sistema = (
                "Avaliar necessidade real de despacho térmico. "
                "Considerar alternativas operacionais."
            )
    
    # Cenário 3: CVU entre 95-100% do PLD → PRESSÃO MODERADA
    elif percentual_cvu_pld and percentual_cvu_pld >= 95:
        classificacao_sistema = "pressão_moderada"
        risco_sistêmico = "moderado"
        descricao_sistema = (
            f"CVU (R$ {cvu_medio:.1f}) muito próximo do PLD (R$ {pld_medio:.1f}). "
            f"Térmicas próximas da competitividade econômica."
        )
        recomendacao_sistema = (
            "Acompanhar evolução da relação PLD-CVU. "
            "Preparar planos de contingência se necessário."
        )
    
    # Cenário 4: CVU entre 80-95% do PLD → ATENÇÃO
    elif percentual_cvu_pld and percentual_cvu_pld >= 80:
        classificacao_sistema = "atenção"
        risco_sistêmico = "baixo"
        descricao_sistema = (
            f"CVU (R$ {cvu_medio:.1f}) representa {percentual_cvu_pld:.0f}% do PLD. "
            f"Margem de segurança adequada."
        )
        recomendacao_sistema = "Monitoramento rotineiro. Sistema operando normalmente."
    
    # Cenário 5: CVU < 80% do PLD → FOLGA OPERACIONAL
    elif percentual_cvu_pld and percentual_cvu_pld < 80:
        classificacao_sistema = "folga_operacional"
        risco_sistêmico = "muito_baixo"
        descricao_sistema = (
            f"CVU (R$ {cvu_medio:.1f}) significativamente abaixo do PLD (R$ {pld_medio:.1f}). "
            f"Sistema com ampla folga em relação às térmicas."
        )
        recomendacao_sistema = "Operação confortável. Otimização de custos garantida."
    
    # Cenário 6: Dados insuficientes
    else:
        classificacao_sistema = "indisponível"
        risco_sistêmico = "indeterminado"
        descricao_sistema = "Dados insuficientes para análise térmica."
        recomendacao_sistema = "Aguardar disponibilidade de dados."
    
    # 3. CONTEXTUALIZAÇÃO HIDROLÓGICA
    if ear_medio is not None:
        if ear_medio > 70:
            contexto_hidrologico = "abundante"
            impacto_hidrologico = "mitigante"
        elif ear_medio > 55:
            contexto_hidrologico = "confortável"
            impacto_hidrologico = "neutro"
        elif ear_medio > 40:
            contexto_hidrologico = "atenção"
            impacto_hidrologico = "agravante"
        else:
            contexto_hidrologico = "crítico"
            impacto_hidrologico = "fortemente_agravante"
    else:
        contexto_hidrologico = "indisponível"
        impacto_hidrologico = "indeterminado"
    
    return {
        # =============================================
        # INDICADORES QUANTITATIVOS
        # =============================================
        "indicadores_quantitativos": {
            "razao_cvu_pld": razao_cvu_pld,
            "percentual_cvu_pld": percentual_cvu_pld,
            "spread_absoluto": analise_gerador["spread_absoluto"],
            "margem_seguranca_sistema": margem_seguranca,
            "margem_vs_teto": margem_vs_teto,
            "dependencia_termica_efetiva": dependencia_efetiva,
        },
        
        # =============================================
        # ANÁLISE DO SISTEMA (MODICIDADE TARIFÁRIA)
        # =============================================
        "analise_sistema": {
            "classificacao": classificacao_sistema,
            "risco_sistêmico": risco_sistêmico,
            "descricao": descricao_sistema,
            "recomendacao": recomendacao_sistema,
            "interpretacao": f"CVU representa {percentual_cvu_pld:.0f}% do PLD" if percentual_cvu_pld else "N/A"
        },
        
        # =============================================
        # ANÁLISE DO GERADOR TÉRMICO
        # =============================================
        "analise_gerador": analise_gerador,
        
        # =============================================
        # CONTEXTO HIDROLÓGICO
        # =============================================
        "contexto_hidrologico": {
            "ear_medio": ear_medio,
            "classificacao_hidrologica": contexto_hidrologico,
            "impacto_pressao_termica": impacto_hidrologico,
            "dependencia_efetiva": dependencia_efetiva,
            "interpretacao": (
                f"EAR {ear_medio:.1f}% ({contexto_hidrologico}) "
                f"{'agravando' if impacto_hidrologico in ['agravante', 'fortemente_agravante'] else 'mitigando'} "
                f"pressão térmica" if ear_medio is not None else "N/A"
            )
        },
        
        # =============================================
        # DADOS DE REFERÊNCIA
        # =============================================
        "dados_referencia": {
            "pld_medio": pld_medio,
            "cvu_medio": cvu_medio,
            "teto_estrutural": PLD_TETO_ESTRUTURAL,
            "limite_folga_estrutural": 150,  # % acima do qual é folga estrutural
            "limite_pressao": 95,  # % acima do qual há pressão
            "limite_risco": 100,   # % acima do qual há risco de custos
        },
        
        # =============================================
        # METADADOS DA ANÁLISE
        # =============================================
        "metadados": {
            "versao_analise": "termica_v5_dupla_perspectiva",
            "data_calculo": datetime.now().isoformat(),
            "perspectivas_incluidas": ["sistema_modicidade", "gerador_viabilidade"],
            "explicacao": (
                "Análise térmica revisada com dupla perspectiva: "
                "1) Sistema (modicidade tarifária) e "
                "2) Gerador (viabilidade econômica). "
                "CVU alto vs PLD baixo = FOLGA ESTRUTURAL, não risco."
            )
        }
    }


def _load_cvu_weekly_series(ons: Dict[str, Any]) -> pd.Series:
    """Retorna série semanal média de CVU por dat_fimsemana."""
    con = _duckdb_connect()
    if con is not None:
        try:
            if _duckdb_table_exists(con, "cvu_usina_termica"):
                q = f"""
                    SELECT
                        {_duckdb_date_expr('dat_fimsemana')} AS dat_fimsemana,
                        AVG({_duckdb_num_expr('val_cvu')}) AS val_cvu
                    FROM cvu_usina_termica
                    GROUP BY 1
                    HAVING dat_fimsemana IS NOT NULL AND val_cvu > 0
                    ORDER BY 1
                """
                df = con.execute(q).fetchdf()
                if not df.empty:
                    s = pd.Series(df['val_cvu'].values, index=pd.to_datetime(df['dat_fimsemana']))
                    return s.sort_index().astype(float)
        except Exception:
            pass
        finally:
            con.close()

    files = _find_ons_csv_all(ons, "CVU_Usina_Termica")
    if not files:
        return pd.Series(dtype=float)

    frames: List[pd.DataFrame] = []
    for cvu_file in files:
        try:
            df = pd.read_csv(cvu_file, sep=None, engine="python")
            needed = {"dat_iniciosemana", "dat_fimsemana", "val_cvu"}
            if not needed.issubset(df.columns):
                continue
            df = df[["dat_iniciosemana", "dat_fimsemana", "val_cvu"]].copy()
            df["dat_iniciosemana"] = _parse_date_series(df["dat_iniciosemana"])
            df["dat_fimsemana"] = _parse_date_series(df["dat_fimsemana"])
            df["val_cvu"] = _normalize_br_numeric_series(df["val_cvu"])
            df = df.dropna(subset=["dat_iniciosemana", "dat_fimsemana", "val_cvu"])
            df = df[df["val_cvu"] > 0]
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.Series(dtype=float)

    all_df = pd.concat(frames, ignore_index=True)
    weekly = (
        all_df.groupby(["dat_iniciosemana", "dat_fimsemana"], as_index=False)["val_cvu"]
        .mean()
        .sort_values("dat_fimsemana")
    )
    if weekly.empty:
        return pd.Series(dtype=float)
    return weekly.set_index("dat_fimsemana")["val_cvu"].astype(float)


def _expand_cvu_weekly_to_daily(ons: Dict[str, Any]) -> pd.Series:
    """Expande CVU semanal para valor diário no intervalo dat_iniciosemana..dat_fimsemana."""
    con = _duckdb_connect()
    if con is not None:
        try:
            if _duckdb_table_exists(con, "cvu_usina_termica"):
                q = f"""
                    SELECT
                        {_duckdb_date_expr('dat_iniciosemana')} AS dat_iniciosemana,
                        {_duckdb_date_expr('dat_fimsemana')} AS dat_fimsemana,
                        AVG({_duckdb_num_expr('val_cvu')}) AS val_cvu
                    FROM cvu_usina_termica
                    GROUP BY 1,2
                    HAVING dat_iniciosemana IS NOT NULL AND dat_fimsemana IS NOT NULL AND val_cvu > 0
                """
                wk = con.execute(q).fetchdf()
                if not wk.empty:
                    daily_vals: Dict[pd.Timestamp, float] = {}
                    for _, r in wk.iterrows():
                        start = pd.Timestamp(r["dat_iniciosemana"]).floor("D")
                        end = pd.Timestamp(r["dat_fimsemana"]).floor("D")
                        if end < start:
                            start, end = end, start
                        for d in pd.date_range(start, end, freq="D"):
                            daily_vals[d] = float(r["val_cvu"])
                    if daily_vals:
                        return pd.Series(daily_vals).sort_index()
        except Exception:
            pass
        finally:
            con.close()

    files = _find_ons_csv_all(ons, "CVU_Usina_Termica")
    if not files:
        return pd.Series(dtype=float)

    frames: List[pd.DataFrame] = []
    for cvu_file in files:
        try:
            df = pd.read_csv(cvu_file, sep=None, engine="python")
            needed = {"dat_iniciosemana", "dat_fimsemana", "val_cvu"}
            if not needed.issubset(df.columns):
                continue
            df = df[["dat_iniciosemana", "dat_fimsemana", "val_cvu"]].copy()
            df["dat_iniciosemana"] = _parse_date_series(df["dat_iniciosemana"])
            df["dat_fimsemana"] = _parse_date_series(df["dat_fimsemana"])
            df["val_cvu"] = _normalize_br_numeric_series(df["val_cvu"])
            df = df.dropna(subset=["dat_iniciosemana", "dat_fimsemana", "val_cvu"])
            df = df[df["val_cvu"] > 0]
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.Series(dtype=float)

    wk = (
        pd.concat(frames, ignore_index=True)
        .groupby(["dat_iniciosemana", "dat_fimsemana"], as_index=False)["val_cvu"]
        .mean()
    )
    daily_vals: Dict[pd.Timestamp, float] = {}
    for _, r in wk.iterrows():
        start = pd.Timestamp(r["dat_iniciosemana"]).floor("D")
        end = pd.Timestamp(r["dat_fimsemana"]).floor("D")
        if end < start:
            start, end = end, start
        for d in pd.date_range(start, end, freq="D"):
            daily_vals[d] = float(r["val_cvu"])

    if not daily_vals:
        return pd.Series(dtype=float)
    return pd.Series(daily_vals).sort_index()


def _compute_cvu_from_csv(ons: Dict[str, Any]) -> Optional[float]:
    s = _load_cvu_weekly_series(ons)
    if s.empty:
        return None
    try:
        return float(s.iloc[-1])
    except Exception:
        return None


# =====================================================================
# NOVAS FUNÇÕES PARA ANÁLISE DE PLD
# =====================================================================

def _calcular_volatilidade_normalizada(pld_series: pd.Series) -> Optional[float]:
    """Calcula volatilidade normalizada considerando limites regulatórios."""
    if pld_series.empty:
        return None
    
    desvio_padrao = pld_series.std()
    banda_total = PLD_TETO_ESTRUTURAL - PLD_PISO
    
    if banda_total > 0:
        return (desvio_padrao / banda_total) * 100  # Em percentual
    return None


def _calcular_posicao_relativa_pld(pld_medio: Optional[float]) -> Optional[float]:
    """Calcula posição relativa do PLD médio na banda regulatória."""
    if pld_medio is None:
        return None
    
    banda_total = PLD_TETO_ESTRUTURAL - PLD_PISO
    if banda_total > 0:
        posicao = ((pld_medio - PLD_PISO) / banda_total) * 100
        return max(0, min(100, posicao))  # Clip entre 0-100%
    return None


def _classificar_volatilidade_pld(volatilidade_norm: Optional[float]) -> str:
    """Classifica a volatilidade do PLD considerando a banda regulatória."""
    if volatilidade_norm is None:
        return "indisponível"
    
    if volatilidade_norm < 10:
        return "baixa"
    elif volatilidade_norm < 25:
        return "moderada"
    elif volatilidade_norm < 40:
        return "alta"
    else:
        return "extrema"


def _classificar_nivel_pld(pld_medio: Optional[float]) -> str:
    """Classifica o nível do PLD médio."""
    if pld_medio is None:
        return "indisponível"
    
    posicao_relativa = _calcular_posicao_relativa_pld(pld_medio)
    if posicao_relativa is None:
        return "indisponível"
    
    if posicao_relativa < 33:
        return "baixo"
    elif posicao_relativa < 66:
        return "moderado"
    else:
        return "elevado"


def _analisar_tendencia_pld(pld_series: pd.Series) -> Dict[str, Any]:
    """Analisa tendência do PLD nas últimas 24h."""
    if pld_series.empty or len(pld_series) < 24:
        return {"tendencia": None, "descricao": "Dados insuficientes"}
    
    # Últimas 24 horas
    ultimas_24h = pld_series.tail(24)
    if len(ultimas_24h) < 12:
        return {"tendencia": None, "descricao": "Dados insuficientes"}
    
    # Calcular tendência linear
    try:
        x = range(len(ultimas_24h))
        y = ultimas_24h.values
        coeficiente = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        
        if coeficiente > 5:
            tendencia = "alta"
            descricao = "Tendência de alta forte (> R$ 5/h)"
        elif coeficiente > 1:
            tendencia = "leve alta"
            descricao = "Tendência de leve alta"
        elif coeficiente < -5:
            tendencia = "baixa"
            descricao = "Tendência de baixa forte"
        elif coeficiente < -1:
            tendencia = "leve baixa"
            descricao = "Tendência de leve baixa"
        else:
            tendencia = "estável"
            descricao = "Preços estáveis"
            
        return {
            "tendencia": tendencia,
            "coeficiente": float(coeficiente),
            "descricao": descricao
        }
    except Exception:
        return {"tendencia": None, "descricao": "Erro no cálculo"}


# =====================================================================
# Core builder
# =====================================================================

def _core_log(stage: str, message: str, **context: Any) -> None:
    ts = datetime.now().isoformat()
    ctx = " | ".join(f"{k}={v}" for k, v in context.items())
    line = f"[{ts}] [build_core_analysis] [{stage}] {message}"
    if ctx:
        line = f"{line} | {ctx}"

    # stdout imediato (útil no terminal/powershell)
    print(line, flush=True)

    # persistência em arquivo para diagnóstico quando stdout não aparece
    try:
        log_path = os.environ.get("KINTUADI_CORE_LOG_PATH", os.path.join("data", "core_analysis_debug.log"))
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
    except Exception:
        # logging não pode quebrar o pipeline principal
        pass


def _try_load_fresh_core_cache(output_dir: str = "data") -> Optional[Dict[str, Any]]:
    """Retorna core já persistido quando está sincronizado com o DuckDB (atalho de performance)."""
    try:
        final_path = os.path.join(output_dir, "core_analysis_latest.json")
        if not os.path.exists(final_path) or not os.path.exists(_DUCKDB_PATH):
            return None

        with open(final_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return None

        core_mtime = os.path.getmtime(final_path)
        db_mtime = os.path.getmtime(_DUCKDB_PATH)

        # Se o DB não mudou desde a geração do core, reaproveita.
        if core_mtime >= db_mtime:
            return cached
    except Exception:
        return None
    return None


def build_core_analysis(raw_data: Dict[str, Any], output_dir: str = "data", force_rebuild: bool = False) -> Dict[str, Any]:
    _core_log("START", "Entrou no build_core_analysis", output_dir=output_dir)
    if duckdb is None or not os.path.exists(_DUCKDB_PATH):
        raise RuntimeError("DuckDB obrigatório para build_core_analysis no modo atual.")

    if not force_rebuild:
        cached = _try_load_fresh_core_cache(output_dir=output_dir)
        if isinstance(cached, dict):
            _core_log("CACHE", "core_analysis_latest.json reaproveitado (DB inalterado)")
            return cached

    sources = _extract_sources(raw_data)
    ons = sources["ons"]
    ccee = sources["ccee"]

    # ---------------- Hidrologia ----------------
    _core_log("HIDRO", "Iniciando cálculo de hidrologia")
    hydrology = _compute_hydrology_from_csv(ons)
    _core_log("HIDRO", "Hidrologia calculada", ear_medio=hydrology.get("ear_medio"), ena_media=hydrology.get("ena_media"))

    # ---------------- Operação ONS ----------------
    # Fonte única de geração/carga: Open Data histórico (geracao_usina_horaria + curva_carga)
    _core_log("OPERACAO", "Extraindo operação histórica Open Data (fonte única)")
    operacao = _extract_open_data_historical_operation(ons)
    _core_log("OPERACAO", "Operação consolidada", status=operacao.get("status"))

    # ---------------- Preços (PLD horário CCEE) ----------------
    pld_medio = pld_std = pld_min = pld_max = None
    pld_por_submercado = {}
    pld_serie_7d = {}
    pld_series_full = pd.Series(dtype=float)
    pld_series_by_submercado: Dict[str, pd.Series] = {}

    # --------------------------------------------
    # 🔎 Consolidar PLD histórico via DuckDB
    # --------------------------------------------
    df_pld = _duckdb_fetchdf("""
        SELECT
            data AS timestamp,
            submercado,
            pld AS pld_hora,
            ano,
            mes,
            hora,
            dia,
            mes_referencia,
            periodo_comercializacao
        FROM pld_historical
        WHERE data IS NOT NULL AND pld IS NOT NULL
        ORDER BY data
    """)
    _core_log("PLD", "Registros PLD consolidados (duckdb)", total_registros=len(df_pld), dataframe_vazio=df_pld.empty)

    ccee_structured = {"metadata": {}, "data": []}

    if not df_pld.empty:

        # 1️⃣ Normalizar colunas
        df_pld.columns = [c.lower() for c in df_pld.columns]

        if "pld" in df_pld.columns:
            df_pld.rename(columns={"pld": "pld_hora"}, inplace=True)

        # 2️⃣ Criar timestamp
        if "timestamp" not in df_pld.columns:
            if all(col in df_pld.columns for col in ["mes_referencia", "dia", "hora"]):
                df_pld["mes_referencia"] = df_pld["mes_referencia"].astype(str)
                df_pld["timestamp"] = pd.to_datetime(
                    df_pld["mes_referencia"].str[:4] + "-" +
                    df_pld["mes_referencia"].str[4:6] + "-" +
                    df_pld["dia"].astype(str) + " " +
                    df_pld["hora"].astype(str) + ":00",
                    errors="coerce"
                )

        # 3️⃣ Tipos numéricos
        df_pld["pld_hora"] = pd.to_numeric(df_pld["pld_hora"], errors="coerce")

        # 4️⃣ Limpeza
        df_pld = df_pld.dropna(subset=["timestamp", "pld_hora"])


        # Preenche granularidade CCEE quando ausente
        if "dia" not in df_pld.columns or df_pld["dia"].isna().all():
            df_pld["dia"] = pd.to_datetime(df_pld["timestamp"], errors="coerce").dt.day
        if "mes_referencia" not in df_pld.columns or df_pld["mes_referencia"].isna().all():
            df_pld["mes_referencia"] = pd.to_datetime(df_pld["timestamp"], errors="coerce").dt.strftime("%Y%m")

        if not df_pld.empty:

            df_pld["timestamp"] = df_pld["timestamp"].dt.tz_localize(
                "America/Sao_Paulo",
                ambiguous="NaT",
                nonexistent="shift_forward"
            )

            df_pld = df_pld.sort_values("timestamp").reset_index(drop=True)

            # ===============================
            # 📈 Estatísticas (prices)
            # ===============================
            pld_medio = df_pld["pld_hora"].mean()
            pld_std = df_pld["pld_hora"].std()
            pld_min = df_pld["pld_hora"].min()
            pld_max = df_pld["pld_hora"].max()

            pld_series_full = _ensure_tz_naive_index(df_pld.set_index("timestamp")["pld_hora"])

            # Submercados
            if "submercado" in df_pld.columns:
                for sub, grp in df_pld.groupby("submercado"):
                    pld_por_submercado[sub] = grp["pld_hora"].mean()
                    sm = _normalize_submercado_name(sub) or str(sub)
                    pld_series_by_submercado[sm] = _ensure_tz_naive_index(
                        grp.sort_values("timestamp").set_index("timestamp")["pld_hora"]
                    )

            # Últimos 7 dias
            last_ts = df_pld["timestamp"].max()
            cutoff = last_ts - pd.Timedelta(days=7)

            df_7d = df_pld[df_pld["timestamp"] >= cutoff]

            pld_serie_7d = {}

            if "submercado" in df_7d.columns:

                for sub, grp in df_7d.groupby("submercado"):

                    grp = grp.sort_values("timestamp")

                    pld_serie_7d[sub] = {
                        ts.isoformat(): float(v)
                        for ts, v in zip(grp["timestamp"], grp["pld_hora"])
                    }

            # ===============================
            # 📦 Estrutura CCEE consolidada
            # ===============================
            required_cols = [
                "mes_referencia",
                "submercado",
                "periodo_comercializacao",
                "dia",
                "hora",
                "pld_hora"
            ]

            for col in required_cols:
                if col not in df_pld.columns:
                    df_pld[col] = None

            records_out = df_pld[required_cols].to_dict(orient="records")

            for i, row in enumerate(records_out, start=1):
                row["_id"] = i
                row["_dataset"] = "pld_horario"

            ccee_structured = {
                "metadata": {
                    "source": "CCEE",
                    "dataset": "PLD_HORARIO",
                    "status": "success",
                    "records_processed": len(records_out),
                    "collection_time": datetime.now().isoformat()
                },
                "data": records_out
            }

    # ---------------- Curtailment Renovável ----------------
    _core_log("CURTAILMENT", "Iniciando cálculo de curtailment renovável")
    curtailment = _compute_renewable_curtailment(ons)
    _core_log("CURTAILMENT", "Curtailment calculado", total_mwh=curtailment.get("total_mwh"))

    classificacao_curtailment = _classificar_curtailment(
        curtailment_total=curtailment.get("total_mwh", 0),
        ear_medio=hydrology.get("ear_medio"),
        pld_medio=pld_medio
    )
 
    # ---------------- Séries para MCP econômico ----------------
    pld_series = pd.Series(dtype=float)
    carga_sin_series = pd.Series(dtype=float)
    geracao_hidro_sin_series = pd.Series(dtype=float)

    if not df_pld.empty:
        pld_series = _ensure_tz_naive_index(
            df_pld
            .sort_values("timestamp")
            .set_index("timestamp")["pld_hora"]
        )

    oper = operacao.get("generation", {})
    load = operacao.get("load", {})

    # Carga SIN
    if "sin" in load:
        carga_sin_series = pd.Series(
            [x["carga"] for x in load["sin"]["serie"]],
            index=[x["instante"] for x in load["sin"]["serie"]],
        )

    # Geração hidráulica SIN
    if "sin_hidraulica" in oper:
        geracao_hidro_sin_series = pd.Series(
            [x["geracao"] for x in oper["sin_hidraulica"]["serie"]],
            index=[x["instante"] for x in oper["sin_hidraulica"]["serie"]],
        )

    # ---------------- Despacho térmico ----------------
    cvu_semanal = _load_cvu_weekly_series(ons)
    cvu_diario = _expand_cvu_weekly_to_daily(ons)
    cvu_medio = _compute_cvu_from_csv(ons)
    
    # Calcular indicadores térmicos REVISADOS (v5)
    _core_log("TERMICA", "Calculando indicadores térmicos revisados")
    indicadores_termicos = calcular_indicadores_termicos_revisados(
        pld_medio=pld_medio,
        cvu_medio=cvu_medio,
        ear_medio=hydrology.get("ear_medio")
    )

    # ---------------- Análises de PLD (NOVAS) ----------------
    # Calcular volatilidade normalizada
    volatilidade_norm = _calcular_volatilidade_normalizada(pld_series_full)
    classificacao_vol = _classificar_volatilidade_pld(volatilidade_norm)
    
    # Calcular posição relativa
    posicao_relativa = _calcular_posicao_relativa_pld(pld_medio)
    classificacao_nivel = _classificar_nivel_pld(pld_medio)
    
    # Análise de tendência
    tendencia_pld = _analisar_tendencia_pld(pld_series_full)

    # ---------------- Métricas avançadas solicitadas ----------------
    _core_log("ADVANCED", "Calculando métricas avançadas")
    try:
        metricas_avancadas = _compute_advanced_cross_metrics(
            ons=ons,
            operacao=operacao,
            pld_series=pld_series_full,
            pld_series_by_submercado=pld_series_by_submercado,
            ear_medio=hydrology.get("ear_medio"),
            ena_media=hydrology.get("ena_media"),
            pld_medio=pld_medio,
            curtailment=curtailment,
        )
    except Exception as e:
        _core_log("ADVANCED", "Erro ao calcular métricas avançadas", erro=str(e))
        metricas_avancadas = {
            "status": "erro",
            "erro": str(e),
            "correlacoes": {},
            "classificacoes": {},
        }
    
    # ---------------- Alerts (ATUALIZADOS com nova lógica) ----------------
    alerts: List[str] = []

    # Alertas hídricos
    if hydrology["classificacao"]["classe"] in {"crítico", "alerta"}:
        alerts.append("Estresse hídrico relevante.")

    # Alertas de PLD
    if pld_medio and posicao_relativa and posicao_relativa > 66:
        alerts.append(f"PLD médio elevado ({pld_medio:.2f} R$/MWh, {posicao_relativa:.0f}% da banda).")

    # Alertas térmicos REVISADOS (usando nova lógica)
    analise_sistema = indicadores_termicos.get("analise_sistema", {})
    classificacao_sistema = analise_sistema.get("classificacao")
    risco_sistêmico = analise_sistema.get("risco_sistêmico")
    
    if risco_sistêmico == "alto":
        percentual_cvu_pld = indicadores_termicos.get("indicadores_quantitativos", {}).get("percentual_cvu_pld")
        if percentual_cvu_pld:
            alerts.append(f"Risco térmico alto: CVU em {percentual_cvu_pld:.0f}% do PLD (despacho com prejuízo possível).")
    
    # Alertas de margem vs teto
    margem_vs_teto = indicadores_termicos.get("indicadores_quantitativos", {}).get("margem_vs_teto")
    if margem_vs_teto is not None:
        if margem_vs_teto < 1:
            alerts.append(f"Margem vs teto crítica ({margem_vs_teto:.1f}%). CVU próximo do teto estrutural.")
        elif margem_vs_teto < 5:
            alerts.append(f"Margem vs teto reduzida ({margem_vs_teto:.1f}%).")
    
    # Alertas de volatilidade extrema
    if classificacao_vol == "extrema":
        alerts.append(f"Volatilidade extrema do PLD ({volatilidade_norm:.0f}% da banda).")

    # ---------------- Construir estrutura CORE ----------------
    core = {
        "timestamp": datetime.now().isoformat(),
        "hydrology": hydrology,
        "renewables": {
            "curtailment": curtailment,
            "classificacao": classificacao_curtailment,
        },
        "ccee": ccee_structured,
        "prices": {
            "pld_medio": pld_medio,
            "pld_min": pld_min,
            "pld_max": pld_max,
            "pld_std": pld_std,
            "pld_volatilidade_norm": volatilidade_norm,
            "pld_posicao_relativa": posicao_relativa,
            "pld_classificacao_vol": classificacao_vol,
            "pld_classificacao_nivel": classificacao_nivel,
            "pld_tendencia": tendencia_pld,
            "limites_regulatorios": {
                "piso": PLD_PISO,
                "teto_estrutural": PLD_TETO_ESTRUTURAL,
                "teto_horario": PLD_TETO_HORARIO
            },
            "por_submercado": pld_por_submercado,
            "pld_horario_7d": pld_serie_7d,
        },
        # ESTRUTURA REVISADA: Análise térmica com dupla perspectiva
        "thermal_analysis": {**indicadores_termicos, "cvu_semanal": {d.strftime("%Y-%m-%d"): float(v) for d, v in cvu_semanal.items()} if not cvu_semanal.empty else {}, "cvu_diario": {d.strftime("%Y-%m-%d"): float(v) for d, v in cvu_diario.items()} if not cvu_diario.empty else {}},
        "advanced_metrics": metricas_avancadas,
        "economic": metricas_avancadas.get("economic", {}) if isinstance(metricas_avancadas, dict) else {},
        "operacao": operacao,
        "alerts": alerts,
        "metadata": {
            "analysis_version": "core-6.0",  # Atualizada para v6 com correção conceitual
            "sources": ["ONS (Open Data histórico)", "CCEE"],
            "limites_aneel_2025": True,
            "analise_termica_versao": "v5_dupla_perspectiva",
            "correcao_conceitual": True,  # Sinaliza que CVU alto vs PLD baixo = FOLGA
            "perspectivas_incluidas": ["sistema_modicidade", "gerador_viabilidade"],
            "generated_at": datetime.now().isoformat(),
            "duckdb_path": _DUCKDB_PATH,
            "duckdb_mtime": datetime.fromtimestamp(os.path.getmtime(_DUCKDB_PATH)).isoformat() if os.path.exists(_DUCKDB_PATH) else None,
        },
    }

    # ---------------- Persist ----------------
    _core_log("PERSIST", "Iniciando persistência do core")

    # 1) Garantir diretório
    os.makedirs(output_dir, exist_ok=True)

    # 2) Salvar em arquivo temporário primeiro
    temp_path = os.path.join(output_dir, "core_analysis_temp.json")
    final_path = os.path.join(output_dir, "core_analysis_latest.json")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(core, f, indent=2, ensure_ascii=False, default=str)

        _core_log("PERSIST", "Arquivo temporário salvo com sucesso", temp_path=temp_path)

        # 3) Limpar versões antigas (exceto temp recém-gerado)
        removidos = []
        for filename in os.listdir(output_dir):
            if (
                filename.startswith("core_analysis_")
                and filename.endswith(".json")
                and filename != "core_analysis_temp.json"
            ):
                target = os.path.join(output_dir, filename)
                os.remove(target)
                removidos.append(filename)

        _core_log("PERSIST", "Arquivos anteriores removidos", removidos=len(removidos))

        # 4) Promover temp para definitivo de forma atômica quando possível
        os.replace(temp_path, final_path)

    except Exception as e:
        _core_log("PERSIST", "Falha ao salvar core_analysis_latest.json", final_path=final_path, erro=str(e))
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise

    _core_log(
        "PERSIST",
        "core_analysis_latest.json salvo com sucesso",
        path=final_path,
        tamanho_bytes=os.path.getsize(final_path),
    )
    return core