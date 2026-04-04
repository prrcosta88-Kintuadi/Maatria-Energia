from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PLOT_BG = "#0b0f14"
CARD_BG = "#111827"
GRID = "#1f2937"
TEXT = "#e5e7eb"
ACCENT = "#c8a44d"
GREEN = "#34d399"
RED = "#f87171"
BLUE = "#60a5fa"


@dataclass
class BESSProject:
    """Validated BESS project definition."""

    project_id: str = "BESS-001"
    cliente: str = ""
    submercado: str = "SE/CO"
    data_inicio: str = "2026-01-01"
    horizonte_anos: int = 20
    taxa_desconto: float = 10.0
    inflacao: float = 4.5
    tipo_projeto: str = "industrial"

    potencia_mw: float = 10.0
    energia_mwh: float = 40.0
    duracao_h: float = 4.0
    eficiencia_rt: float = 0.87
    ciclos_dia_max: int = 2
    degradacao_anual: float = 2.5
    vida_util: int = 15
    soc_min: float = 10.0
    soc_max: float = 90.0
    c_rate_max: float = 0.5

    usar_pld_modelado: bool = True
    cenario_pld: str = "Base"
    pld_fixo: float = 200.0
    spread_tarifario: float = 1.3
    pld_submercado: str = ""  # Submercado de referência para PLD (override do campo submercado)

    modo: str = "arbitragem"
    carga_inicio: int = 1
    carga_fim: int = 6
    descarga_inicio: int = 17
    descarga_fim: int = 21
    limite_demanda_kw: float = 7000.0
    prioridade: str = "receita"
    usar_otimizacao_lp: bool = True  # Usa LP para despacho ótimo

    capex_bess_kwh: float = 1200.0
    capex_pcs_kw: float = 250.0
    capex_bop_kw: float = 120.0
    capex_engenharia: float = 500000.0
    capex_instalacao: float = 800000.0

    opex_fixo: float = 150000.0
    opex_variavel: float = 5.0
    seguro_pct: float = 0.5
    opex_om: float = 50000.0

    demanda_contratada: float = 10000.0
    custo_energia_base: float = 320.0
    custo_demanda_base: float = 85.0

    # ── Exposição à liquidação do mercado pelo PLD ───────────────────────────
    # O projeto pode optar por consumir ACIMA do contratado no PLD baixo (para
    # carregar a BESS) e ABAIXO no PLD alto (descarregando). A diferença é
    # liquidada no CCEE ao PLD do instante.
    # carga_media_mw         : carga média consumida da rede (MW)
    # contrato_mw            : contrato de energia firmado (MW flat)
    # flexibilidade_mw       : banda de exposição disponível ao mercado (MW)
    #                          Positivo = pode comprar mais / vender menos
    # encargo_tusd_rs_mwh    : custo adicional sobre energia ciclada (TUSD, ESS, etc.)
    carga_media_mw: float = 0.0          # 0 = não considera liquidação de mercado
    contrato_mw: float = 0.0
    flexibilidade_mw: float = 0.0
    encargo_tusd_rs_mwh: float = 0.0

    n_simulacoes: int = 1000
    horizonte_horas: int = 8760
    usar_montecarlo: bool = True
    seed: int = 42
    vol_pld: float = 0.25
    vol_carga: float = 0.10

    load_profile: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def capex_total(self) -> float:
        return (
            self.capex_bess_kwh * self.energia_mwh * 1000
            + self.capex_pcs_kw * self.potencia_mw * 1000
            + self.capex_bop_kw * self.potencia_mw * 1000
            + self.capex_engenharia
            + self.capex_instalacao
        )

    @property
    def opex_anual(self) -> float:
        return self.opex_fixo + self.opex_om + (self.seguro_pct / 100.0) * self.capex_total

    @property
    def energia_util_mwh(self) -> float:
        return self.energia_mwh * (self.soc_max - self.soc_min) / 100.0


@dataclass
class FinanceInputs:
    """Indicative lender-style assumptions used for the BESS dashboard."""

    gearing_max_pct: float = 70.0
    debt_cost_pct: float = 12.0
    debt_tenor_years: int = 10
    grace_years: int = 1
    target_dscr: float = 1.35
    mmra_months: int = 6
    augmentation_factor_pct: float = 50.0


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "sim", "s"}:
        return True
    if text in {"0", "false", "f", "no", "n", "nao", "não"}:
        return False
    return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return float(default)
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    text = text.replace("R$", "").replace("%", "").replace(" ", "").replace("\u00a0", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    return int(round(_as_float(value, float(default))))


def _safe_datetime(value: Any, default: str) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(default)
    return ts.strftime("%Y-%m-%d")


def _fmt_currency(value: Optional[float], scale: float = 1.0, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"R${value / scale:,.2f}{suffix}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.2f}%"


def _load_workbook_sheets(file_path: str) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(file_path)
    return {sheet: xl.parse(sheet, header=None) for sheet in xl.sheet_names}


def parse_excel(file_path: str) -> BESSProject:
    """Read the BESS template and return a validated project object."""

    sheets = _load_workbook_sheets(file_path)
    proj = BESSProject()

    def _val(sheet: str, row: int, col: int = 2, default: Any = None) -> Any:
        df = sheets.get(sheet)
        if df is None:
            return default
        try:
            value = df.iloc[row - 1, col - 1]
        except Exception:
            return default
        if pd.isna(value):
            return default
        return value

    proj.project_id = str(_val("PROJECT", 3, default=proj.project_id) or proj.project_id)
    proj.cliente = str(_val("PROJECT", 4, default="") or "")
    proj.submercado = str(_val("PROJECT", 5, default=proj.submercado) or proj.submercado)
    proj.data_inicio = _safe_datetime(_val("PROJECT", 6, default=proj.data_inicio), proj.data_inicio)
    proj.horizonte_anos = _as_int(_val("PROJECT", 7, default=proj.horizonte_anos), proj.horizonte_anos)
    proj.taxa_desconto = _as_float(_val("PROJECT", 8, default=proj.taxa_desconto), proj.taxa_desconto)
    proj.inflacao = _as_float(_val("PROJECT", 9, default=proj.inflacao), proj.inflacao)
    proj.tipo_projeto = str(_val("PROJECT", 10, default=proj.tipo_projeto) or proj.tipo_projeto)

    proj.potencia_mw = _as_float(_val("BESS_TECH", 3, default=proj.potencia_mw), proj.potencia_mw)
    proj.energia_mwh = _as_float(_val("BESS_TECH", 4, default=proj.energia_mwh), proj.energia_mwh)
    proj.duracao_h = _as_float(_val("BESS_TECH", 5, default=proj.duracao_h), proj.duracao_h)
    proj.eficiencia_rt = _as_float(_val("BESS_TECH", 6, default=proj.eficiencia_rt), proj.eficiencia_rt)
    proj.ciclos_dia_max = _as_int(_val("BESS_TECH", 7, default=proj.ciclos_dia_max), proj.ciclos_dia_max)
    proj.degradacao_anual = _as_float(_val("BESS_TECH", 8, default=proj.degradacao_anual), proj.degradacao_anual)
    proj.vida_util = _as_int(_val("BESS_TECH", 9, default=proj.vida_util), proj.vida_util)
    proj.soc_min = _as_float(_val("BESS_TECH", 10, default=proj.soc_min), proj.soc_min)
    proj.soc_max = _as_float(_val("BESS_TECH", 11, default=proj.soc_max), proj.soc_max)
    proj.c_rate_max = _as_float(_val("BESS_TECH", 12, default=proj.c_rate_max), proj.c_rate_max)

    proj.usar_pld_modelado = _as_bool(_val("MARKET", 3, default=proj.usar_pld_modelado), proj.usar_pld_modelado)
    proj.cenario_pld = str(_val("MARKET", 4, default=proj.cenario_pld) or proj.cenario_pld)
    proj.pld_fixo = _as_float(_val("MARKET", 5, default=proj.pld_fixo), proj.pld_fixo)
    proj.spread_tarifario = _as_float(_val("MARKET", 6, default=proj.spread_tarifario), proj.spread_tarifario)
    # Submercado de referência para PLD (linha 7 da aba MARKET – "Submercado PLD")
    _pld_sub = _val("MARKET", 7, default="")
    proj.pld_submercado = str(_pld_sub).strip() if _pld_sub and str(_pld_sub).strip() not in ("", "nan", "None") else ""

    proj.modo = str(_val("STRATEGY", 3, default=proj.modo) or proj.modo)
    proj.carga_inicio = _as_int(_val("STRATEGY", 5, default=proj.carga_inicio), proj.carga_inicio)
    proj.carga_fim = _as_int(_val("STRATEGY", 6, default=proj.carga_fim), proj.carga_fim)
    proj.descarga_inicio = _as_int(_val("STRATEGY", 7, default=proj.descarga_inicio), proj.descarga_inicio)
    proj.descarga_fim = _as_int(_val("STRATEGY", 8, default=proj.descarga_fim), proj.descarga_fim)
    proj.limite_demanda_kw = _as_float(_val("STRATEGY", 10, default=proj.limite_demanda_kw), proj.limite_demanda_kw)
    proj.prioridade = str(_val("STRATEGY", 11, default=proj.prioridade) or proj.prioridade)
    proj.usar_otimizacao_lp = _as_bool(_val("STRATEGY", 12, default=proj.usar_otimizacao_lp), proj.usar_otimizacao_lp)

    proj.capex_bess_kwh = _as_float(_val("CAPEX", 3, default=proj.capex_bess_kwh), proj.capex_bess_kwh)
    proj.capex_pcs_kw = _as_float(_val("CAPEX", 4, default=proj.capex_pcs_kw), proj.capex_pcs_kw)
    proj.capex_bop_kw = _as_float(_val("CAPEX", 5, default=proj.capex_bop_kw), proj.capex_bop_kw)
    proj.capex_engenharia = _as_float(_val("CAPEX", 6, default=proj.capex_engenharia), proj.capex_engenharia)
    proj.capex_instalacao = _as_float(_val("CAPEX", 7, default=proj.capex_instalacao), proj.capex_instalacao)

    proj.opex_fixo = _as_float(_val("OPEX", 3, default=proj.opex_fixo), proj.opex_fixo)
    proj.opex_variavel = _as_float(_val("OPEX", 4, default=proj.opex_variavel), proj.opex_variavel)
    proj.seguro_pct = _as_float(_val("OPEX", 5, default=proj.seguro_pct), proj.seguro_pct)
    proj.opex_om = _as_float(_val("OPEX", 6, default=proj.opex_om), proj.opex_om)

    proj.demanda_contratada = _as_float(_val("ALTERNATIVE", 4, default=proj.demanda_contratada), proj.demanda_contratada)
    proj.custo_energia_base = _as_float(_val("ALTERNATIVE", 5, default=proj.custo_energia_base), proj.custo_energia_base)
    proj.custo_demanda_base = _as_float(_val("ALTERNATIVE", 6, default=proj.custo_demanda_base), proj.custo_demanda_base)
    # Campos de exposição à liquidação de mercado (linhas 8-11 da aba ALTERNATIVE)
    proj.carga_media_mw      = _as_float(_val("ALTERNATIVE", 8, default=proj.carga_media_mw), proj.carga_media_mw)
    proj.contrato_mw         = _as_float(_val("ALTERNATIVE", 9, default=proj.contrato_mw), proj.contrato_mw)
    proj.flexibilidade_mw    = _as_float(_val("ALTERNATIVE", 10, default=proj.flexibilidade_mw), proj.flexibilidade_mw)
    proj.encargo_tusd_rs_mwh = _as_float(_val("ALTERNATIVE", 11, default=proj.encargo_tusd_rs_mwh), proj.encargo_tusd_rs_mwh)

    proj.n_simulacoes = _as_int(_val("SIMULATION", 3, default=proj.n_simulacoes), proj.n_simulacoes)
    proj.horizonte_horas = _as_int(_val("SIMULATION", 4, default=proj.horizonte_horas), proj.horizonte_horas)
    proj.usar_montecarlo = _as_bool(_val("SIMULATION", 5, default=proj.usar_montecarlo), proj.usar_montecarlo)
    proj.seed = _as_int(_val("SIMULATION", 6, default=proj.seed), proj.seed)
    proj.vol_pld = _as_float(_val("SIMULATION", 7, default=proj.vol_pld), proj.vol_pld)
    proj.vol_carga = _as_float(_val("SIMULATION", 8, default=proj.vol_carga), proj.vol_carga)

    if proj.potencia_mw > 0 and proj.energia_mwh > 0 and proj.duracao_h <= 0:
        proj.duracao_h = proj.energia_mwh / proj.potencia_mw

    if "LOAD_PROFILE" in sheets:
        try:
            lp_raw = sheets["LOAD_PROFILE"].copy()
            header_row = None
            for idx, row in lp_raw.iterrows():
                values = [str(v).strip().lower() for v in row if pd.notna(v)]
                if any("datetime" in v or "data" in v or "hora" in v for v in values):
                    header_row = idx
                    break
            if header_row is not None:
                lp = pd.read_excel(file_path, sheet_name="LOAD_PROFILE", header=header_row)
                lp.columns = [str(col).strip().lower().replace(" ", "_").replace("/", "_") for col in lp.columns]
                dt_col = next((c for c in lp.columns if "datetime" in c or "data" in c or c == "0"), None)
                if dt_col:
                    lp = lp.rename(columns={dt_col: "datetime"})
                    lp["datetime"] = pd.to_datetime(lp["datetime"], errors="coerce")
                    lp = lp[lp["datetime"].notna()].copy()
                proj.load_profile = lp.reset_index(drop=True)
        except Exception as exc:
            logger.warning("LOAD_PROFILE parse falhou: %s", exc)

    return proj


def validate_project(proj: BESSProject) -> List[str]:
    """Validate the project and return warnings/issues."""

    issues: List[str] = []
    if not (0.5 <= proj.eficiencia_rt <= 1.0):
        issues.append(f"Eficiencia RT={proj.eficiencia_rt} fora do intervalo [0.5, 1.0].")
    if proj.soc_min >= proj.soc_max:
        issues.append(f"SOC minimo ({proj.soc_min}%) maior ou igual ao maximo ({proj.soc_max}%).")
    if proj.potencia_mw <= 0 or proj.energia_mwh <= 0:
        issues.append("Potencia e energia devem ser positivas.")
    if proj.capex_total <= 0:
        issues.append("CAPEX total invalido.")
    if not (0 < proj.taxa_desconto <= 50):
        issues.append(f"Taxa de desconto={proj.taxa_desconto}% fora da faixa esperada.")
    if proj.ciclos_dia_max <= 0:
        issues.append("Ciclos maximos por dia devem ser positivos.")
    if proj.load_profile.empty:
        issues.append("Perfil de carga vazio: o motor usara o perfil padrao.")
    implied_duration = proj.energia_mwh / max(proj.potencia_mw, 0.001)
    if abs(implied_duration - proj.duracao_h) > 0.25:
        issues.append(
            f"Duracao informada ({proj.duracao_h:.2f}h) diverge da relacao energia/potencia ({implied_duration:.2f}h)."
        )
    if proj.horizonte_anos > proj.vida_util:
        issues.append(
            f"Horizonte financeiro ({proj.horizonte_anos}a) sera limitado pela vida util do ativo ({proj.vida_util}a)."
        )
    return issues


def _submercado_key(submercado: str) -> str:
    """Normaliza qualquer alias de submercado para chave interna do engine.

    Espelha exatamente ``_normalize_submercado_name`` de core_analysis.py e o
    mapeamento ``SUB_COL`` de app_premium.py.

    Entradas aceitas (case-insensitive, ignora espaços e barras):
        SE/CO, SE, SECO, SUDESTE, SUDESTECENTROOESTE → "seco"  (→ pld_se no app)
        SUL, S                                        → "s"     (→ pld_s  no app)
        NE, NORDESTE                                  → "ne"    (→ pld_ne no app)
        N, NORTE                                      → "n"     (→ pld_n  no app)
        "1"                                           → "seco"  (código CCEE)
        "2"                                           → "s"
        "3"                                           → "ne"
        "4"                                           → "n"
    """
    sub = str(submercado).strip().upper()
    # Remove separadores (core_analysis faz .replace("/","").replace("-","").replace(" ",""))
    sub_clean = sub.replace("/", "").replace("-", "").replace(" ", "")

    mapping: Dict[str, str] = {
        # Códigos numéricos CCEE
        "1": "seco", "2": "s", "3": "ne", "4": "n",
        # SE/CO e variantes
        "SE":   "seco", "SECO": "seco",
        "SECO": "seco",
        "SUDESTECENTROOESTE": "seco",
        "SUDESTE": "seco",
        "SECO": "seco",
        # Sul
        "SUL": "s", "S": "s",
        # Nordeste
        "NE": "ne", "NORDESTE": "ne",
        # Norte
        "N": "n", "NORTE": "n",
    }
    return mapping.get(sub_clean, mapping.get(sub, "seco"))


def _pld_col_from_submercado(submercado: str) -> str:
    """Retorna o nome da coluna pld_* usada pelo app_premium (DuckDB/Neon).

    app_premium.py usa SUB_COL = {
        "SE/CO": "pld_se", "NE": "pld_ne", "S": "pld_s", "N": "pld_n"
    }
    Esta função permite que o engine aponte para a coluna correta do DataFrame
    horário quando recebe um submercado de referência.
    """
    key = _submercado_key(submercado)
    return {"seco": "pld_se", "s": "pld_s", "ne": "pld_ne", "n": "pld_n"}.get(key, "pld_se")


def get_pld_official_day(
    target_date: str,
    submercado: str,
    pld_fixo_fallback: float = 200.0,
) -> np.ndarray:
    """Retorna o vetor de PLD oficial (24h) para ``target_date`` do DuckDB/Neon.

    Schema da tabela ``pld_historical`` (definido em integrated_collector_v2.py):
        data              TIMESTAMP   -- ex.: 2026-04-02 17:00:00
        submercado        VARCHAR     -- ex.: 'SE', 'SE/CO', 'SUDESTE', 'SUL', 'NE', 'N'
        pld               DOUBLE      -- valor em R$/MWh  ← coluna canônica
        ano               INTEGER
        mes               INTEGER
        hora              INTEGER     -- 0..23  ← granularidade horária preservada
        dia               INTEGER
        mes_referencia    INTEGER     -- YYYYMM
        periodo_comercializacao INTEGER

    Chaves de submercado (coluna ``submercado`` no banco):
        SE/CO  → aliases: SE, SUDESTE, SECO, SUDESTECENTROOESTE  → chave curta "se"
        SUL    → aliases: SUL, S                                  → chave curta "s"
        NE     → aliases: NE, NORDESTE                            → chave curta "ne"
        N      → aliases: N, NORTE                                → chave curta "n"

    Estratégia de query (espelha core_analysis.py _duckdb_fetchdf):
        SELECT hora, AVG(pld) FROM pld_historical
        WHERE CAST(data AS DATE) = '<date>'
          AND UPPER(TRIM(submercado)) IN ('<alias1>', '<alias2>', ...)
          AND pld > 0
        GROUP BY hora ORDER BY hora

    O campo ``hora`` (0-23) já existe na tabela — sem necessidade de date_trunc.
    O PLD é puro, sem spread, sem sazonalidade artificial.

    Retorna np.ndarray shape (24,), R$/MWh, clampado em [57.31, 1611.04].
    """
    from pathlib import Path as _Path

    target_ts = pd.to_datetime(target_date).normalize()
    date_str = target_ts.strftime("%Y-%m-%d")
    sub_key = _submercado_key(submercado)

    # Aliases exatos que o integrated_collector_v2 grava na coluna submercado
    # (dfx["submercado"] = dfx["submercado"].astype(str).str.upper().str.strip())
    _ALIASES: Dict[str, List[str]] = {
        "seco": ["SE/CO", "SE", "SUDESTE", "SECO", "SUDESTECENTROOESTE", "SUDESTE/CENTRO-OESTE"],
        "s":    ["SUL", "S"],
        "ne":   ["NE", "NORDESTE"],
        "n":    ["N", "NORTE"],
    }
    aliases = _ALIASES.get(sub_key, ["SE/CO", "SE", "SUDESTE", "SECO"])
    alias_sql = ", ".join(f"'{a}'" for a in aliases)

    # Query idêntica ao padrão de core_analysis.py:
    # filtra por CAST(data AS DATE), usa coluna `pld` (não pld_hora),
    # e agrupa por `hora` (coluna inteira 0-23) para evitar date_trunc.
    query_duck = f"""
        SELECT
            hora,
            AVG(pld) AS pld_media
        FROM pld_historical
        WHERE CAST(data AS DATE) = '{date_str}'
          AND UPPER(TRIM(submercado)) IN ({alias_sql})
          AND pld IS NOT NULL
          AND pld > 0
        GROUP BY hora
        ORDER BY hora
    """

    # Query Neon idêntica (usa mesma tabela/colunas)
    query_neon = f"""
        SELECT
            hora,
            AVG(pld) AS pld_media
        FROM pld_historical
        WHERE CAST(data AS DATE) = '{date_str}'::date
          AND UPPER(TRIM(submercado)) IN ({alias_sql})
          AND pld IS NOT NULL
          AND pld > 0
        GROUP BY hora
        ORDER BY hora
    """

    def _build_array(df_q: pd.DataFrame) -> Optional[np.ndarray]:
        """Converte DataFrame {hora INT, pld_media FLOAT} → array 24h."""
        if df_q.empty:
            return None
        df_q = df_q.copy()
        df_q["hora"] = pd.to_numeric(df_q["hora"], errors="coerce")
        df_q["pld_media"] = pd.to_numeric(df_q["pld_media"], errors="coerce")
        df_q = df_q.dropna(subset=["hora", "pld_media"])
        df_q = df_q[(df_q["hora"] >= 0) & (df_q["hora"] <= 23)]
        if df_q.empty or len(df_q) < 20:
            return None
        arr = np.full(24, pld_fixo_fallback, dtype=float)
        for _, row in df_q.iterrows():
            arr[int(row["hora"])] = float(row["pld_media"])
        return np.clip(arr, 57.31, 1611.04)

    # ── Tentativa 1: DuckDB local (data/kintuadi.duckdb) ─────────────────────
    duckdb_path = _Path("data/kintuadi.duckdb")
    if duckdb_path.exists():
        try:
            import duckdb as _ddb
            con = _ddb.connect(str(duckdb_path), read_only=True)
            try:
                df_q = con.execute(query_duck).df()
            finally:
                con.close()
            result = _build_array(df_q)
            if result is not None:
                logger.info("PLD D+1 (%s, %s): lido do DuckDB (%d registros)",
                            date_str, submercado, len(df_q))
                return result
            # Se tiver poucos registros, tenta também sem filtro de data
            # para pegar o dia mais recente disponível (caso o PLD de amanhã
            # já esteja no banco sob outra data)
            logger.debug("DuckDB: PLD para %s/%s: apenas %d h — tentando dia mais recente",
                         date_str, submercado, len(df_q))
        except Exception as _e:
            logger.debug("DuckDB PLD falhou (%s/%s): %s", date_str, submercado, _e)

    # ── Tentativa 2: Neon PostgreSQL ──────────────────────────────────────────
    try:
        import db_neon  # type: ignore
        df_q = db_neon.fetchdf(query_neon)
        result = _build_array(df_q)
        if result is not None:
            logger.info("PLD D+1 (%s, %s): lido do Neon (%d registros)",
                        date_str, submercado, len(df_q))
            return result
        logger.debug("Neon: PLD para %s/%s: apenas %d h", date_str, submercado, len(df_q))
    except Exception as _e:
        logger.debug("Neon PLD falhou (%s/%s): %s", date_str, submercado, _e)

    # ── Tentativa 3: Dia mais recente disponível no banco ─────────────────────
    # O PLD de amanhã pode ainda não ter sido publicado. Usa o último dia
    # disponível como proxy (melhor que spread artificial).
    query_latest_duck = f"""
        SELECT
            hora,
            AVG(pld) AS pld_media
        FROM pld_historical
        WHERE CAST(data AS DATE) = (
            SELECT MAX(CAST(data AS DATE))
            FROM pld_historical
            WHERE UPPER(TRIM(submercado)) IN ({alias_sql})
              AND pld > 0
        )
        AND UPPER(TRIM(submercado)) IN ({alias_sql})
        AND pld > 0
        GROUP BY hora
        ORDER BY hora
    """
    query_latest_neon = f"""
        SELECT
            hora,
            AVG(pld) AS pld_media
        FROM pld_historical
        WHERE CAST(data AS DATE) = (
            SELECT MAX(CAST(data AS DATE))
            FROM pld_historical
            WHERE UPPER(TRIM(submercado)) IN ({alias_sql})
              AND pld > 0
        )
        AND UPPER(TRIM(submercado)) IN ({alias_sql})
        AND pld > 0
        GROUP BY hora
        ORDER BY hora
    """
    if duckdb_path.exists():
        try:
            import duckdb as _ddb
            con = _ddb.connect(str(duckdb_path), read_only=True)
            try:
                df_q = con.execute(query_latest_duck).df()
            finally:
                con.close()
            result = _build_array(df_q)
            if result is not None:
                logger.warning("PLD D+1 (%s/%s): data não encontrada — usando último dia disponível",
                               date_str, submercado)
                return result
        except Exception:
            pass
    try:
        import db_neon  # type: ignore
        df_q = db_neon.fetchdf(query_latest_neon)
        result = _build_array(df_q)
        if result is not None:
            logger.warning("PLD D+1 (%s/%s): Neon — usando último dia disponível", date_str, submercado)
            return result
    except Exception:
        pass

    # ── Fallback final: PLD plano sem spread ─────────────────────────────────
    logger.warning(
        "PLD oficial não encontrado para %s/%s — fallback plano R$%.0f/MWh (sem spread)",
        date_str, submercado, pld_fixo_fallback,
    )
    return np.full(24, pld_fixo_fallback, dtype=float)


def _get_spectral_peak_hour() -> Optional[int]:
    try:
        from spectral_engine import get_current_fingerprint

        fp = get_current_fingerprint()
        if fp and getattr(fp, "regime", "unknown") != "unknown":
            peak_hour = int(round(float(getattr(fp, "peak_hour", 18))))
            return max(0, min(23, peak_hour))
    except Exception:
        return None
    return None


def get_pld_path(proj: BESSProject, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate an hourly PLD path for the BESS simulation.

    O submercado de referência é determinado por ``proj.pld_submercado`` quando
    informado no template (aba MARKET → "Submercado PLD"). Se não informado,
    usa ``proj.submercado`` (aba PROJECT).  Isso permite modelar, por exemplo, um
    projeto instalado no SE/CO que negocia no submercado Sul.
    """

    n = proj.horizonte_horas
    # Resolve qual submercado usar para referência de PLD
    ref_sub = proj.pld_submercado if proj.pld_submercado else proj.submercado
    base_prices: List[float] = []

    if proj.usar_pld_modelado:
        try:
            from pld_forecast_engine import forecast_short_term, get_latest_pmo_state

            pmo = get_latest_pmo_state()
            short = forecast_short_term(pmo)
            semanas = short.get("semanas", [])
            sub_key = _submercado_key(ref_sub)
            for semana in semanas:
                p50 = semana.get(f"pld_p50_{sub_key}")
                if p50 is not None:
                    base_prices.append(float(p50))
        except Exception:
            base_prices = []

    if not base_prices:
        base_prices = [proj.pld_fixo]

    hourly = np.zeros(n, dtype=float)
    weekly_len = max(1, int(np.ceil(n / max(len(base_prices), 1))))
    spread = max(proj.spread_tarifario, 1.01)
    for hour in range(n):
        week_idx = min(hour // weekly_len, len(base_prices) - 1)
        base = float(base_prices[week_idx])
        hod = hour % 24
        if 17 <= hod <= 20:
            seasonality = spread
        elif 0 <= hod <= 5:
            seasonality = 1.0 / spread
        else:
            seasonality = 1.0
        hourly[hour] = base * seasonality

    if rng is not None:
        sigma = max(float(proj.vol_pld), 0.0)
        if sigma > 0:
            noise = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=n)
            hourly = hourly * noise

    return np.clip(hourly, 57.31, 1611.04)


def get_tariff_path(proj: BESSProject) -> np.ndarray:
    """Return the hourly tariff series used for savings/shaving logic."""

    n = proj.horizonte_horas
    if not proj.load_profile.empty:
        col = next(
            (
                c
                for c in proj.load_profile.columns
                if "tarifa" in c.lower() or "energy" in c.lower() or "preco" in c.lower() or "r$" in c.lower()
            ),
            None,
        )
        if col:
            base = pd.to_numeric(proj.load_profile[col], errors="coerce").fillna(proj.pld_fixo).to_numpy()
            reps = (n // len(base)) + 1
            return np.tile(base, reps)[:n]

    tariff = np.full(n, proj.pld_fixo, dtype=float)
    spread = max(proj.spread_tarifario, 1.01)
    for hour in range(n):
        hod = hour % 24
        if 17 <= hod <= 20:
            tariff[hour] = proj.pld_fixo * spread
        elif 0 <= hod <= 5:
            tariff[hour] = proj.pld_fixo / spread
    return tariff


def get_load_path(proj: BESSProject, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate or reuse an hourly load profile."""

    n = proj.horizonte_horas
    if not proj.load_profile.empty:
        col = next((c for c in proj.load_profile.columns if "carga" in c.lower() or "load" in c.lower()), None)
        if col and len(proj.load_profile) > 0:
            base = pd.to_numeric(proj.load_profile[col], errors="coerce").fillna(5.0).to_numpy()
            reps = (n // len(base)) + 1
            load = np.tile(base, reps)[:n]
        else:
            load = np.full(n, 5.0, dtype=float)
    else:
        load = np.zeros(n, dtype=float)
        for hour in range(n):
            hod = hour % 24
            dow = (hour // 24) % 7
            if dow < 5:
                if 8 <= hod <= 18:
                    load[hour] = 8.0
                elif 6 <= hod <= 20:
                    load[hour] = 3.5
                else:
                    load[hour] = 2.0
            else:
                load[hour] = 2.5 if 8 <= hod <= 16 else 1.5

    if rng is not None:
        sigma = max(float(proj.vol_carga), 0.0)
        if sigma > 0:
            noise = rng.normal(1.0, sigma, n)
            load = load * np.clip(noise, 0.6, 1.4)

    return np.clip(load, 0.0, proj.potencia_mw * 5.0)


def optimize_dispatch_lp(
    proj: BESSProject,
    pld: np.ndarray,
    load: np.ndarray,
    year: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal hourly charge/discharge schedule via Linear Programming.

    Maximiza a receita de arbitragem líquida:
        max  Σ_t  PLD_t · discharge_t  −  PLD_t · charge_t

    sujeito a:
        SOC_t = SOC_{t-1} + charge_t · η  −  discharge_t       (dinâmica)
        soc_min ≤ SOC_t ≤ soc_max                               (limites de SOC)
        0 ≤ charge_t   ≤ potencia_max                           (potência de carga)
        0 ≤ discharge_t ≤ min(potencia_max, load_t)             (potência de descarga)
        discharge_t + charge_t ≤ potencia_max                   (não opera simultâneo)
        Σ_{t in dia d} discharge_t ≤ ciclos_dia_max · energia_util  (ciclos/dia)

    Retorna arrays numpy ``(charge, discharge)`` de tamanho ``len(pld)``.
    Faz fallback silencioso para arrays zeros se PuLP não estiver disponível.
    """
    try:
        import pulp  # noqa: F401
    except ImportError:
        logger.warning("PuLP não instalado – retornando despacho zero. Instale com: pip install pulp")
        n = min(len(pld), len(load), proj.horizonte_horas)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    import pulp

    n = min(len(pld), len(load), proj.horizonte_horas)
    pld_arr = np.asarray(pld, dtype=float)[:n]
    load_arr = np.asarray(load, dtype=float)[:n]

    degradation_factor = (1.0 - proj.degradacao_anual / 100.0) ** max(year - 1, 0)
    energia_util = proj.energia_util_mwh * degradation_factor
    soc_min_mwh = proj.soc_min / 100.0 * proj.energia_mwh * degradation_factor
    soc_max_mwh = proj.soc_max / 100.0 * proj.energia_mwh * degradation_factor
    potencia_max = float(min(proj.potencia_mw, proj.c_rate_max * max(proj.energia_mwh, 0.001)))
    eta = max(proj.eficiencia_rt, 0.01)

    prob = pulp.LpProblem("BESS_Dispatch", pulp.LpMaximize)

    charge = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=potencia_max) for t in range(n)]
    discharge = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=potencia_max) for t in range(n)]
    soc = [pulp.LpVariable(f"s_{t}", lowBound=soc_min_mwh, upBound=soc_max_mwh) for t in range(n)]

    # Objetivo
    prob += pulp.lpSum(pld_arr[t] * discharge[t] - pld_arr[t] * charge[t] for t in range(n))

    # SOC inicial (meio do intervalo útil)
    soc_init = soc_min_mwh + 0.5 * energia_util
    prob += soc[0] == soc_init + charge[0] * eta - discharge[0]

    for t in range(1, n):
        prob += soc[t] == soc[t - 1] + charge[t] * eta - discharge[t]

    # Não opera carga e descarga simultaneamente
    for t in range(n):
        prob += charge[t] + discharge[t] <= potencia_max
        # Descarga limitada pela carga local (se modo não for arbitragem pura, respeita a demanda)
        if proj.modo != "arbitragem":
            prob += discharge[t] <= float(load_arr[t]) + potencia_max * 0.0  # sem restrição de carga local no LP geral

    # Limite de ciclos por dia
    daily_budget = float(max(energia_util * proj.ciclos_dia_max, 0.0))
    n_days = (n + 23) // 24
    for d in range(n_days):
        h_start = d * 24
        h_end = min(h_start + 24, n)
        prob += pulp.lpSum(discharge[t] for t in range(h_start, h_end)) <= daily_budget

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    prob.solve(solver)

    charge_arr = np.array([max(pulp.value(charge[t]) or 0.0, 0.0) for t in range(n)], dtype=float)
    discharge_arr = np.array([max(pulp.value(discharge[t]) or 0.0, 0.0) for t in range(n)], dtype=float)

    return charge_arr, discharge_arr


def get_next_day_guidance(
    proj: BESSProject,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Guia de operação ótima para o DIA SEGUINTE com PLD oficial do banco.

    Fluxo:
    1. Determina data de operação (amanhã por padrão) e submercado de referência.
    2. Busca o PLD oficial do DuckDB/Neon para a data-alvo.
       - PLD PURO, sem spread, sem sazonalidade artificial.
       - A data no banco é a data de operação (registrado como D, publicado em D-1).
    3. Roda LP sobre as 24h do dia.
    4. Calcula receita de arbitragem + economia por liquidação de mercado (se configurado):
       - Comprar acima do contrato no PLD baixo (carregar BESS) → custo
       - Vender abaixo do contrato no PLD alto (descarregar BESS) → receita
       - Encargos TUSD/ESS sobre energia ciclada
    5. Retorna plano hora-a-hora e métricas financeiras do dia.
    """
    from datetime import date, timedelta

    today = date.today()
    op_date = pd.to_datetime(target_date).date() if target_date else today + timedelta(days=1)
    pub_date = op_date - timedelta(days=1)
    ref_sub = proj.pld_submercado if proj.pld_submercado else proj.submercado

    # ── PLD oficial do banco (puro, sem spread) ───────────────────────────────
    pld_24 = get_pld_official_day(
        target_date=str(op_date),
        submercado=ref_sub,
        pld_fixo_fallback=proj.pld_fixo,
    )
    fonte_pld = "DuckDB/Neon (oficial)" if pld_24.std() > 1.0 else f"Fallback plano R${proj.pld_fixo:.0f}/MWh"

    # ── Perfil de carga para o dia ────────────────────────────────────────────
    _proj24 = BESSProject(
        project_id=proj.project_id,
        submercado=proj.submercado,
        pld_submercado=ref_sub,
        potencia_mw=proj.potencia_mw,
        energia_mwh=proj.energia_mwh,
        eficiencia_rt=proj.eficiencia_rt,
        ciclos_dia_max=proj.ciclos_dia_max,
        degradacao_anual=proj.degradacao_anual,
        soc_min=proj.soc_min,
        soc_max=proj.soc_max,
        c_rate_max=proj.c_rate_max,
        modo=proj.modo,
        opex_variavel=proj.opex_variavel,
        carga_media_mw=proj.carga_media_mw,
        contrato_mw=proj.contrato_mw,
        flexibilidade_mw=proj.flexibilidade_mw,
        encargo_tusd_rs_mwh=proj.encargo_tusd_rs_mwh,
        horizonte_horas=24,
        data_inicio=str(op_date),
        # Mantém pld_fixo como fallback mas não usa spread — PLD já vem puro
        pld_fixo=proj.pld_fixo,
        spread_tarifario=1.0,
        usar_pld_modelado=False,
    )
    load_24 = get_load_path(_proj24)

    # ── Despacho LP sobre PLD oficial ─────────────────────────────────────────
    charge_arr, discharge_arr = optimize_dispatch_lp(_proj24, pld_24, load_24, year=1)

    # ── Reconstrução hora a hora com liquidação de mercado ────────────────────
    eta = _proj24.eficiencia_rt
    soc_min_mwh = _proj24.soc_min / 100.0 * _proj24.energia_mwh
    soc_max_mwh = _proj24.soc_max / 100.0 * _proj24.energia_mwh
    energia_util = _proj24.energia_util_mwh
    soc_cur = soc_min_mwh + 0.5 * energia_util

    usar_liquidacao = (proj.carga_media_mw > 0 and proj.contrato_mw > 0)
    carga_media = proj.carga_media_mw
    contrato = proj.contrato_mw
    flex = max(proj.flexibilidade_mw, 0.0)
    tusd = proj.encargo_tusd_rs_mwh

    plano: List[Dict[str, Any]] = []
    total_arbitragem = 0.0
    total_liquidacao = 0.0
    total_tusd = 0.0

    for h in range(24):
        c = float(charge_arr[h])
        d = float(discharge_arr[h])
        price = float(pld_24[h])

        soc_cur = float(np.clip(soc_cur + c * eta - d, soc_min_mwh, soc_max_mwh))

        # Receita de arbitragem pura (descarga vende ao PLD; carga compra ao PLD)
        receita_arb = d * price - c * price
        total_arbitragem += receita_arb

        # ── Liquidação de mercado ──────────────────────────────────────────────
        # Carga real na hora = carga_media ± ajuste BESS
        # Ao carregar: consome carga_media + c MW da rede → exposição = +c acima do contrato
        # Ao descarregar: consome carga_media - d MW da rede → exposição = -d abaixo do contrato
        # A diferença em relação ao contrato é liquidada pelo CCEE ao PLD da hora.
        liquidacao_h = 0.0
        carga_rede_h = 0.0
        if usar_liquidacao:
            # Consumo real da rede nesta hora
            carga_rede_h = max(0.0, carga_media + c - d)
            # Diferença em relação ao contrato (positivo = comprou mais; negativo = vendeu sobra)
            delta_contrato = carga_rede_h - contrato
            # Limita pela flexibilidade disponível
            delta_efetivo = float(np.clip(delta_contrato, -flex, flex))
            # Custo da liquidação: paga PLD quando comprou mais; recebe PLD quando vendeu menos
            liquidacao_h = -delta_efetivo * price  # negativo = custo; positivo = receita
            total_liquidacao += liquidacao_h

        # Encargo TUSD/ESS sobre energia ciclada (independe da liquidação)
        tusd_h = (c + d) * tusd
        total_tusd += tusd_h

        receita_total_h = receita_arb + liquidacao_h - tusd_h

        if c > 1e-4:
            acao = "CARREGAR"
        elif d > 1e-4:
            acao = "DESCARREGAR"
        else:
            acao = "OCIOSO"

        plano.append({
            "hora": h,
            "acao": acao,
            "carga_mw": round(c, 4),
            "descarga_mw": round(d, 4),
            "soc_mwh": round(soc_cur, 4),
            "pld_rs_mwh": round(price, 2),
            "receita_arbitragem_rs": round(receita_arb, 2),
            "liquidacao_mercado_rs": round(liquidacao_h, 2),
            "encargo_tusd_rs": round(tusd_h, 2),
            "receita_total_rs": round(receita_total_h, 2),
            "carga_rede_mw": round(carga_rede_h, 4) if usar_liquidacao else None,
        })

    receita_liquida_dia = total_arbitragem + total_liquidacao - total_tusd

    # ── Texto resumo ──────────────────────────────────────────────────────────
    horas_c = [p["hora"] for p in plano if p["acao"] == "CARREGAR"]
    horas_d = [p["hora"] for p in plano if p["acao"] == "DESCARREGAR"]
    _fh = lambda lst: ", ".join(f"{h:02d}h" for h in lst) if lst else "–"

    resumo = (
        f"Submercado: {ref_sub} | Fonte PLD: {fonte_pld}. "
        f"Data operação: {op_date.strftime('%d/%m/%Y')} (PLD publicado: {pub_date.strftime('%d/%m/%Y')}). "
        f"Horas de carga: {_fh(horas_c)} | Horas de descarga: {_fh(horas_d)}. "
        f"Receita arbitragem: R${total_arbitragem:,.2f} | "
        f"Liquidação mercado: R${total_liquidacao:,.2f} | "
        f"Encargos TUSD: R${total_tusd:,.2f} | "
        f"Resultado líquido do dia: R${receita_liquida_dia:,.2f}."
    )

    return {
        "submercado_referencia": ref_sub,
        "fonte_pld": fonte_pld,
        "data_operacao": str(op_date),
        "data_publicacao_pld": str(pub_date),
        "plano_horario": plano,
        "receita_arbitragem_rs": round(total_arbitragem, 2),
        "liquidacao_mercado_rs": round(total_liquidacao, 2),
        "encargos_tusd_rs": round(total_tusd, 2),
        "receita_liquida_dia_rs": round(receita_liquida_dia, 2),
        # Mantido por compatibilidade com código legado
        "receita_total_esperada_rs": round(receita_liquida_dia, 2),
        "energia_carregada_mwh": round(float(charge_arr.sum()), 4),
        "energia_descarregada_mwh": round(float(discharge_arr.sum()), 4),
        "resumo_texto": resumo,
    }


def optimize_for_irr(
    proj: BESSProject,
    pld: np.ndarray,
    load: np.ndarray,
    next_day_guidance: Optional[Dict[str, Any]] = None,
    finance_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Projeta a TIR do equity em 3 cenários extrapolando o Guia D+1 para toda a vida do projeto.

    Lógica:
    - O Guia D+1 fornece a receita líquida de UM dia típico (arbitragem + liquidação - TUSD).
    - Extrapolamos esse dia para 365 dias × horizonte do projeto, com degradação anual.
    - Três cenários:
        Pessimista : receita_dia × 0,6  (60% do guia D+1)
        Base       : receita_dia × 1,0  (guia D+1 idêntico todos os dias)
        Otimista   : receita_dia × 1,1  (110% do guia D+1)
    - Calcula TIR do equity, VPL e DSCR mínimo para cada cenário.
    - Fallback: se next_day_guidance não disponível, usa receita do Monte Carlo P50.

    Corrige o bug 'unsupported format string passed to NoneType' tratando IRR None.
    """
    finance = FinanceInputs(**(finance_overrides or {}))
    life = max(1, min(int(proj.horizonte_anos), int(proj.vida_util)))

    # ── Receita diária de referência ──────────────────────────────────────────
    receita_dia_ref: float = 0.0
    fonte_referencia = "Monte Carlo P50"

    if next_day_guidance and "receita_liquida_dia_rs" in next_day_guidance:
        receita_dia_ref = float(next_day_guidance.get("receita_liquida_dia_rs") or 0.0)
        fonte_referencia = f"Guia D+1 ({next_day_guidance.get('data_operacao', '?')})"
    elif next_day_guidance and "receita_total_esperada_rs" in next_day_guidance:
        receita_dia_ref = float(next_day_guidance.get("receita_total_esperada_rs") or 0.0)
        fonte_referencia = f"Guia D+1 ({next_day_guidance.get('data_operacao', '?')})"
    else:
        # Fallback: usa receita anual da simulação Monte Carlo (sem next_day_guidance)
        try:
            tariff = get_tariff_path(proj)
            mc_fallback = run_monte_carlo(proj, n_sims=max(50, min(proj.n_simulacoes, 200)))
            receita_anual = float(mc_fallback.get("receita_p50") or 0.0)
            receita_dia_ref = receita_anual / 365.0
            fonte_referencia = "Monte Carlo P50 (fallback)"
        except Exception as _e:
            logger.warning("optimize_for_irr fallback falhou: %s", _e)
            receita_dia_ref = 0.0

    # ── Três cenários ─────────────────────────────────────────────────────────
    cenarios = [
        ("pessimista", 0.6),
        ("base",       1.0),
        ("otimista",   1.1),
    ]

    resultados = {}
    for nome, fator in cenarios:
        receita_dia = receita_dia_ref * fator
        receita_ano1 = receita_dia * 365.0

        try:
            op_df = _build_operating_case(proj, receita_ano1, finance)
            annual_df, financing_s = _build_financing_schedule(op_df, proj.capex_total, finance)

            equity_outflow = financing_s["equity_amount_rs"] + financing_s["initial_mmra_rs"]
            project_cf = [-proj.capex_total] + op_df["project_fcf_rs"].tolist()
            equity_cf = [-equity_outflow] + annual_df["equity_cashflow_rs"].tolist()

            project_irr = _irr(project_cf)
            equity_irr  = _irr(equity_cf)
            discount    = proj.taxa_desconto / 100.0
            project_npv = round(_npv(discount, project_cf), 2)
            equity_npv  = round(_npv(discount, equity_cf), 2)

            resultados[nome] = {
                "fator": fator,
                "receita_dia_rs": round(receita_dia, 2),
                "receita_ano1_rs": round(receita_ano1, 2),
                "project_irr_pct": project_irr,
                "equity_irr_pct": equity_irr,
                "project_npv_rs": project_npv,
                "equity_npv_rs": equity_npv,
                "min_dscr": financing_s.get("min_dscr"),
                "llcr": financing_s.get("llcr"),
                "debt_amount_rs": financing_s.get("debt_amount_rs"),
            }
        except Exception as exc:
            logger.warning("optimize_for_irr cenário %s falhou: %s", nome, exc)
            resultados[nome] = {
                "fator": fator,
                "receita_dia_rs": round(receita_dia, 2),
                "receita_ano1_rs": round(receita_ano1, 2),
                "project_irr_pct": None,
                "equity_irr_pct": None,
                "project_npv_rs": None,
                "equity_npv_rs": None,
                "min_dscr": None,
                "llcr": None,
                "debt_amount_rs": None,
            }

    base = resultados.get("base", {})

    return {
        "fonte_referencia": fonte_referencia,
        "receita_dia_referencia_rs": round(receita_dia_ref, 2),
        "cenarios": resultados,
        # Atalhos para o dashboard
        "optimal_irr_pct": base.get("equity_irr_pct"),
        "optimal_utilizacao": 1.0,
        "optimal_potencia_mw": proj.potencia_mw,
        "optimal_energia_mwh": proj.energia_mwh,
        # Para compatibilidade com código de tabela anterior
        "all_utilization_results": [
            {
                "cenario": nome,
                "fator": r["fator"],
                "receita_dia_rs": r["receita_dia_rs"],
                "receita_ano1_rs": r["receita_ano1_rs"],
                "equity_irr_pct": r["equity_irr_pct"],
                "project_irr_pct": r["project_irr_pct"],
                "equity_npv_rs": r["equity_npv_rs"],
                "min_dscr": r["min_dscr"],
            }
            for nome, r in resultados.items()
        ],
    }


def simulate_year(
    proj: BESSProject,
    pld_hourly: np.ndarray,
    load_hourly: np.ndarray,
    year: int = 1,
    tariff_hourly: Optional[np.ndarray] = None,
    charge: Optional[np.ndarray] = None,
    discharge: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulate hourly BESS operation for one operating year.

    Se ``charge`` e ``discharge`` forem fornecidos (e.g. pelo LP), eles são usados
    diretamente e a lógica heurística interna é ignorada. O SOC ainda é rastreado
    para validação e log.
    """

    n_hours = min(len(pld_hourly), len(load_hourly), proj.horizonte_horas)
    pld_hourly = np.asarray(pld_hourly, dtype=float)[:n_hours]
    load_hourly = np.asarray(load_hourly, dtype=float)[:n_hours]
    tariff_hourly = get_tariff_path(proj) if tariff_hourly is None else np.asarray(tariff_hourly, dtype=float)
    tariff_hourly = np.tile(tariff_hourly, (n_hours // len(tariff_hourly)) + 1)[:n_hours]

    degradation_factor = (1.0 - proj.degradacao_anual / 100.0) ** max(year - 1, 0)
    energia_util = proj.energia_util_mwh * degradation_factor
    soc_min = proj.soc_min / 100.0 * proj.energia_mwh * degradation_factor
    soc_max = proj.soc_max / 100.0 * proj.energia_mwh * degradation_factor
    potencia_max = min(proj.potencia_mw, proj.c_rate_max * max(proj.energia_mwh, 0.1))

    soc = np.zeros(n_hours, dtype=float)
    charged = np.zeros(n_hours, dtype=float)
    discharged = np.zeros(n_hours, dtype=float)
    revenue = np.zeros(n_hours, dtype=float)
    cost_avoided = np.zeros(n_hours, dtype=float)
    charge_cost = np.zeros(n_hours, dtype=float)
    opex_var = np.zeros(n_hours, dtype=float)
    actions: List[str] = ["idle"] * n_hours

    # ── CAMINHO OTIMIZADO (LP) ─────────────────────────────────────────────────
    # Se vetores pré-computados forem fornecidos, usa-os diretamente e rastreia o SOC.
    if charge is not None and discharge is not None:
        charge_in = np.asarray(charge, dtype=float)[:n_hours]
        discharge_in = np.asarray(discharge, dtype=float)[:n_hours]
        current_soc = soc_min + 0.5 * energia_util
        eta = max(proj.eficiencia_rt, 0.01)
        for hour in range(n_hours):
            c = float(charge_in[hour])
            d = float(discharge_in[hour])
            price = float(pld_hourly[hour])
            current_soc = float(np.clip(current_soc + c * eta - d, soc_min, soc_max))
            charged[hour] = c
            discharged[hour] = d
            soc[hour] = current_soc
            charge_cost[hour] = c * price
            revenue[hour] = d * price
            opex_var[hour] = (c + d) * proj.opex_variavel
            if c > 1e-6:
                actions[hour] = "charge"
            elif d > 1e-6:
                actions[hour] = "discharge"
            else:
                actions[hour] = "idle"
    else:
        # ── CAMINHO HEURÍSTICO (fallback / backward compat) ───────────────────
        current_soc = soc_min + 0.5 * energia_util
        daily_discharge_budget = max(energia_util * proj.ciclos_dia_max, 0.0)
        discharged_today = 0.0
        last_day = -1
        spectral_peak = _get_spectral_peak_hour()

        for hour in range(n_hours):
            hod = hour % 24
            day = hour // 24
            price = float(pld_hourly[hour])
            load = float(load_hourly[hour])

            if day != last_day:
                discharged_today = 0.0
                last_day = day

            daily_headroom = max(daily_discharge_budget - discharged_today, 0.0)
            action = "idle"

            if proj.modo == "arbitragem":
                can_charge = proj.carga_inicio <= hod <= proj.carga_fim
                can_discharge = proj.descarga_inicio <= hod <= proj.descarga_fim
                if can_charge and current_soc < soc_max and daily_headroom > 0:
                    power = min(potencia_max, (soc_max - current_soc) / max(proj.eficiencia_rt, 0.01))
                    current_soc += power * proj.eficiencia_rt
                    charged[hour] = power
                    charge_cost[hour] = power * price
                    action = "charge"
                elif can_discharge and current_soc > soc_min and daily_headroom > 0:
                    power = min(potencia_max, current_soc - soc_min, daily_headroom)
                    current_soc -= power
                    discharged[hour] = power
                    discharged_today += power
                    revenue[hour] = power * price
                    action = "discharge"
            elif proj.modo == "peak_shaving":
                demand_kw = load * 1000.0
                if demand_kw > proj.limite_demanda_kw and current_soc > soc_min and daily_headroom > 0:
                    excess_mw = min(
                        (demand_kw - proj.limite_demanda_kw) / 1000.0,
                        potencia_max,
                        current_soc - soc_min,
                        daily_headroom,
                    )
                    current_soc -= excess_mw
                    discharged[hour] = excess_mw
                    discharged_today += excess_mw
                    cost_avoided[hour] = excess_mw * tariff_hourly[hour]
                    action = "shave"
                elif proj.carga_inicio <= hod <= proj.carga_fim and current_soc < soc_max:
                    power = min(potencia_max, (soc_max - current_soc) / max(proj.eficiencia_rt, 0.01))
                    current_soc += power * proj.eficiencia_rt
                    charged[hour] = power
                    charge_cost[hour] = power * price
                    action = "charge"
            else:
                demand_kw = load * 1000.0
                discharge_start = spectral_peak if spectral_peak is not None else proj.descarga_inicio
                within_discharge = discharge_start <= hod <= proj.descarga_fim
                hours_to_peak = (discharge_start - hod) % 24
                reserve_soc = soc_max * 0.80 if not within_discharge and hours_to_peak <= 8 else soc_min
                if within_discharge and current_soc > soc_min and daily_headroom > 0:
                    power = min(potencia_max, current_soc - soc_min, daily_headroom)
                    current_soc -= power
                    discharged[hour] = power
                    discharged_today += power
                    revenue[hour] = power * price
                    action = "discharge"
                elif proj.carga_inicio <= hod <= proj.carga_fim and current_soc < soc_max:
                    power = min(potencia_max, (soc_max - current_soc) / max(proj.eficiencia_rt, 0.01))
                    current_soc += power * proj.eficiencia_rt
                    charged[hour] = power
                    charge_cost[hour] = power * price
                    action = "charge"
                elif demand_kw > proj.limite_demanda_kw and current_soc > reserve_soc and daily_headroom > 0:
                    excess_mw = min(
                        (demand_kw - proj.limite_demanda_kw) / 1000.0,
                        potencia_max,
                        current_soc - reserve_soc,
                        daily_headroom,
                    )
                    current_soc -= excess_mw
                    discharged[hour] = excess_mw
                    discharged_today += excess_mw
                    cost_avoided[hour] = excess_mw * tariff_hourly[hour]
                    action = "shave"

            current_soc = float(np.clip(current_soc, soc_min, soc_max))
            soc[hour] = current_soc
            opex_var[hour] = (charged[hour] + discharged[hour]) * proj.opex_variavel
            actions[hour] = action

    start_ts = pd.to_datetime(proj.data_inicio, errors="coerce")
    if pd.isna(start_ts):
        start_ts = pd.Timestamp(datetime.now().date())
    timestamp = pd.date_range(start=start_ts, periods=n_hours, freq="h")
    net_margin = revenue + cost_avoided - charge_cost - opex_var

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "hour": np.arange(n_hours, dtype=int),
            "soc_mwh": soc,
            "charged_mwh": charged,
            "discharged_mwh": discharged,
            "revenue_rs": revenue,
            "cost_avoided_rs": cost_avoided,
            "charge_cost_rs": charge_cost,
            "opex_var_rs": opex_var,
            "net_margin_rs": net_margin,
            "pld_rs_mwh": pld_hourly,
            "tariff_rs_mwh": tariff_hourly,
            "load_mw": load_hourly,
            "action": actions,
        }
    )

    total_revenue = float(revenue.sum() + cost_avoided.sum())
    total_cost = float(charge_cost.sum() + opex_var.sum())
    hours_active = int((charged > 0).sum() + (discharged > 0).sum())
    cycles_equiv = float(discharged.sum() / max(energia_util, 0.1))

    summary = {
        "year": year,
        "energia_descarregada_mwh": round(float(discharged.sum()), 1),
        "energia_carregada_mwh": round(float(charged.sum()), 1),
        "receita_bruta_rs": round(total_revenue, 2),
        "custo_carga_rs": round(total_cost, 2),
        "receita_liquida_rs": round(total_revenue - total_cost, 2),
        "ciclos_equivalentes": round(cycles_equiv, 1),
        "degradacao_acumulada_pct": round((1.0 - degradation_factor) * 100.0, 1),
        "fator_capacidade_pct": round(float(discharged.sum() / max(proj.potencia_mw * n_hours, 0.1) * 100.0), 1),
        "horas_ativas": hours_active,
        "utilizacao_media_soc_pct": round(float(df["soc_mwh"].mean() / max(soc_max, 0.1) * 100.0), 1),
    }
    return df, summary


def run_monte_carlo(proj: BESSProject, n_sims: Optional[int] = None) -> Dict[str, Any]:
    """Run the full Monte Carlo merchant simulation."""

    n = int(n_sims or proj.n_simulacoes)
    rng = np.random.default_rng(proj.seed)
    tariff = get_tariff_path(proj)
    results: List[Dict[str, float]] = []

    for _ in range(n):
        pld = get_pld_path(proj, rng)
        load = get_load_path(proj, rng)
        _, summary = simulate_year(proj, pld, load, year=1, tariff_hourly=tariff)
        results.append(summary)

    df = pd.DataFrame(results)
    receita = df["receita_liquida_rs"] if "receita_liquida_rs" in df else pd.Series(dtype=float)
    return {
        "n_simulacoes": n,
        "receita_p10": round(float(receita.quantile(0.10)), 2) if not receita.empty else 0.0,
        "receita_p50": round(float(receita.quantile(0.50)), 2) if not receita.empty else 0.0,
        "receita_p90": round(float(receita.quantile(0.90)), 2) if not receita.empty else 0.0,
        "receita_media": round(float(receita.mean()), 2) if not receita.empty else 0.0,
        "receita_std": round(float(receita.std()), 2) if len(receita) > 1 else 0.0,
        "energia_media_mwh": round(float(df["energia_descarregada_mwh"].mean()), 1) if not df.empty else 0.0,
        "ciclos_medio": round(float(df["ciclos_equivalentes"].mean()), 1) if not df.empty else 0.0,
        "resultados": df,
    }


def _irr(cashflows: List[float]) -> Optional[float]:
    try:
        import numpy_financial as npf

        value = npf.irr(cashflows)
        return round(float(value) * 100.0, 2) if np.isfinite(value) else None
    except Exception:
        pass

    cfs = np.array(cashflows, dtype=float)
    periods = np.arange(len(cfs), dtype=float)
    rate = 0.10
    for _ in range(200):
        denom = (1.0 + rate) ** periods
        npv = float(np.sum(cfs / denom))
        dnpv = float(np.sum(-periods * cfs / ((1.0 + rate) ** (periods + 1.0))))
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-9:
            rate = new_rate
            break
        rate = float(np.clip(new_rate, -0.99, 100.0))
    return round(rate * 100.0, 2) if -0.99 < rate < 100.0 else None


def _npv(rate: float, cashflows: List[float]) -> float:
    return float(sum(cf / (1.0 + rate) ** idx for idx, cf in enumerate(cashflows)))


def _project_life_years(proj: BESSProject) -> int:
    return max(1, min(int(proj.horizonte_anos), int(proj.vida_util)))


def _build_operating_case(
    proj: BESSProject,
    receita_ano1: float,
    finance: FinanceInputs,
) -> pd.DataFrame:
    """Build an annual operating case used by the finance dashboard."""

    life_years = _project_life_years(proj)
    inflation = proj.inflacao / 100.0
    opex_base = proj.opex_anual
    initial_usable_energy = proj.energia_util_mwh
    pack_capex = proj.capex_bess_kwh * proj.energia_mwh * 1000.0
    augmentation_year = max(4, min(life_years, int(np.ceil(proj.vida_util * 0.5))))
    augmentation_cost = pack_capex * finance.augmentation_factor_pct / 100.0 if life_years >= 8 else 0.0

    rows: List[Dict[str, float]] = []
    for year in range(1, life_years + 1):
        degradation_factor = (1.0 - proj.degradacao_anual / 100.0) ** (year - 1)
        inflation_factor = (1.0 + inflation) ** (year - 1)
        revenue = receita_ano1 * degradation_factor * inflation_factor
        opex = opex_base * inflation_factor
        cfads = revenue - opex
        augmentation = augmentation_cost * inflation_factor if year == augmentation_year else 0.0
        rows.append(
            {
                "year": year,
                "revenue_rs": revenue,
                "opex_rs": opex,
                "cfads_rs": cfads,
                "augmentation_rs": augmentation,
                "project_fcf_rs": cfads - augmentation,
                "usable_energy_mwh": initial_usable_energy * degradation_factor,
                "degradation_pct": (1.0 - degradation_factor) * 100.0,
            }
        )

    return pd.DataFrame(rows)


def _build_financing_schedule(
    operating_df: pd.DataFrame,
    capex_total: float,
    finance: FinanceInputs,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """Build an indicative sculpted debt schedule and lender metrics."""

    df = operating_df.copy()
    debt_rate = finance.debt_cost_pct / 100.0
    tenor = max(1, min(finance.debt_tenor_years, len(df)))
    grace = max(0, min(finance.grace_years, tenor - 1))
    target_dscr = max(finance.target_dscr, 1.01)
    service_years = list(range(grace + 1, tenor + 1))
    cfads_service = df.loc[df["year"].isin(service_years), "cfads_rs"].astype(float).to_numpy()

    raw_capacity = 0.0
    for idx, cfads in enumerate(cfads_service, start=1):
        scheduled_service = max(cfads / target_dscr, 0.0)
        raw_capacity += scheduled_service / ((1.0 + debt_rate) ** idx)

    debt_cap_limit = capex_total * finance.gearing_max_pct / 100.0
    debt_amount = min(raw_capacity, debt_cap_limit)
    equity_amount = max(capex_total - debt_amount, 0.0)

    service_profile = np.array([max(cfads / target_dscr, 0.0) for cfads in cfads_service], dtype=float)
    if raw_capacity > 0:
        service_profile *= debt_amount / raw_capacity

    df["interest_rs"] = 0.0
    df["principal_rs"] = 0.0
    df["debt_service_rs"] = 0.0
    df["outstanding_open_rs"] = 0.0
    df["outstanding_close_rs"] = 0.0

    outstanding = debt_amount
    for year in range(1, tenor + 1):
        row_idx = df.index[df["year"] == year][0]
        df.at[row_idx, "outstanding_open_rs"] = outstanding
        interest = outstanding * debt_rate

        if year <= grace:
            principal = 0.0
            debt_service = interest
        else:
            profile_idx = year - grace - 1
            target_service = service_profile[profile_idx] if 0 <= profile_idx < len(service_profile) else 0.0
            principal = max(target_service - interest, 0.0)
            principal = min(principal, outstanding)
            debt_service = interest + principal
            if year == tenor and outstanding - principal > 1e-9:
                principal = outstanding
                debt_service = interest + principal

        outstanding = max(outstanding - principal, 0.0)
        df.at[row_idx, "interest_rs"] = interest
        df.at[row_idx, "principal_rs"] = principal
        df.at[row_idx, "debt_service_rs"] = debt_service
        df.at[row_idx, "outstanding_close_rs"] = outstanding

    df["mmra_required_rs"] = df["debt_service_rs"] * finance.mmra_months / 12.0
    df["dscr"] = np.where(df["debt_service_rs"] > 0, df["cfads_rs"] / df["debt_service_rs"], np.nan)

    initial_mmra = (
        float(df.loc[df["debt_service_rs"] > 0, "mmra_required_rs"].iloc[0])
        if (df["debt_service_rs"] > 0).any()
        else 0.0
    )
    mmra_release_year = (
        int(df.loc[df["debt_service_rs"] > 0, "year"].iloc[-1])
        if (df["debt_service_rs"] > 0).any()
        else None
    )

    df["equity_cashflow_rs"] = df["project_fcf_rs"] - df["debt_service_rs"]
    if mmra_release_year is not None:
        release_idx = df.index[df["year"] == mmra_release_year][0]
        df.at[release_idx, "equity_cashflow_rs"] += initial_mmra

    active = df.loc[df["debt_service_rs"] > 0].copy()
    min_dscr = float(active["dscr"].min()) if not active.empty else None
    avg_dscr = float(active["dscr"].mean()) if not active.empty else None

    llcr = None
    if debt_amount > 0 and not active.empty:
        pv_cfads = 0.0
        for idx, cfads in enumerate(active["cfads_rs"].tolist(), start=1):
            pv_cfads += cfads / ((1.0 + debt_rate) ** idx)
        llcr = pv_cfads / debt_amount if debt_amount > 0 else None

    financing_summary = {
        "debt_amount_rs": round(float(debt_amount), 2),
        "equity_amount_rs": round(float(equity_amount), 2),
        "initial_mmra_rs": round(float(initial_mmra), 2),
        "gearing_pct": round(float((debt_amount / capex_total) * 100.0), 2) if capex_total > 0 else 0.0,
        "target_dscr": finance.target_dscr,
        "min_dscr": round(float(min_dscr), 2) if min_dscr is not None else None,
        "avg_dscr": round(float(avg_dscr), 2) if avg_dscr is not None else None,
        "llcr": round(float(llcr), 2) if llcr is not None else None,
        "debt_tenor_years": tenor,
        "debt_cost_pct": finance.debt_cost_pct,
        "grace_years": grace,
        "mmra_months": finance.mmra_months,
    }
    return df, financing_summary


def _scenario_financials(
    proj: BESSProject,
    receita_ano1: float,
    finance: FinanceInputs,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    operating_df = _build_operating_case(proj, receita_ano1, finance)
    annual_df, financing_summary = _build_financing_schedule(operating_df, proj.capex_total, finance)

    project_cashflows = [-proj.capex_total] + annual_df["project_fcf_rs"].tolist()
    equity_outflow = financing_summary["equity_amount_rs"] + financing_summary["initial_mmra_rs"]
    equity_cashflows = [-equity_outflow] + annual_df["equity_cashflow_rs"].tolist()

    discount_rate = proj.taxa_desconto / 100.0
    project_irr = _irr(project_cashflows)
    equity_irr = _irr(equity_cashflows)
    project_npv = round(_npv(discount_rate, project_cashflows), 2)
    equity_npv = round(_npv(discount_rate, equity_cashflows), 2)

    cumulative = -proj.capex_total
    payback_year = None
    for year, cf in zip(annual_df["year"].tolist(), annual_df["project_fcf_rs"].tolist()):
        cumulative += cf / ((1.0 + discount_rate) ** year)
        if payback_year is None and cumulative >= 0:
            payback_year = int(year)

    scenario = {
        "receita_ano1": round(float(receita_ano1), 2),
        "capex_total": round(float(proj.capex_total), 2),
        "opex_anual": round(float(proj.opex_anual), 2),
        "project_irr_pct": project_irr,
        "equity_irr_pct": equity_irr,
        "irr_pct": equity_irr,
        "project_npv_rs": project_npv,
        "equity_npv_rs": equity_npv,
        "npv_rs": equity_npv,
        "payback_anos": payback_year,
        "vida_util": _project_life_years(proj),
        "cashflows": [round(float(cf), 2) for cf in project_cashflows],
        "equity_cashflows": [round(float(cf), 2) for cf in equity_cashflows],
        "min_dscr": financing_summary["min_dscr"],
        "llcr": financing_summary["llcr"],
        "debt_amount_rs": financing_summary["debt_amount_rs"],
        "equity_amount_rs": financing_summary["equity_amount_rs"],
        "initial_mmra_rs": financing_summary["initial_mmra_rs"],
        "financing_summary": financing_summary,
    }
    return scenario, annual_df


def compute_financials(
    proj: BESSProject,
    mc_result: Dict[str, Any],
    finance_inputs: Optional[FinanceInputs] = None,
) -> Dict[str, Any]:
    """Compute project-finance style metrics for BESS scenarios."""

    finance = finance_inputs or FinanceInputs()
    labels = [("pessimista", "receita_p10"), ("base", "receita_p50"), ("otimista", "receita_p90")]

    scenarios: Dict[str, Dict[str, Any]] = {}
    annual_tables: Dict[str, pd.DataFrame] = {}
    for label, revenue_key in labels:
        scenario, annual_df = _scenario_financials(proj, float(mc_result.get(revenue_key, 0.0)), finance)
        scenarios[label] = scenario
        annual_tables[label] = annual_df

    project_irrs: List[float] = []
    equity_irrs: List[float] = []
    min_dscrs: List[float] = []
    if "resultados" in mc_result and isinstance(mc_result["resultados"], pd.DataFrame):
        for revenue in mc_result["resultados"]["receita_liquida_rs"].tolist():
            scenario, _ = _scenario_financials(proj, float(revenue), finance)
            if scenario["project_irr_pct"] is not None:
                project_irrs.append(float(scenario["project_irr_pct"]))
            if scenario["equity_irr_pct"] is not None:
                equity_irrs.append(float(scenario["equity_irr_pct"]))
            if scenario["min_dscr"] is not None:
                min_dscrs.append(float(scenario["min_dscr"]))

    project_arr = np.array(project_irrs, dtype=float) if project_irrs else np.array([], dtype=float)
    equity_arr = np.array(equity_irrs, dtype=float) if equity_irrs else np.array([], dtype=float)
    dscr_arr = np.array(min_dscrs, dtype=float) if min_dscrs else np.array([], dtype=float)

    def _quantile(arr: np.ndarray, q: float) -> Optional[float]:
        if arr.size == 0:
            return None
        return round(float(np.percentile(arr, q)), 2)

    distribution = {
        "project_irr_p10": _quantile(project_arr, 10),
        "project_irr_p50": _quantile(project_arr, 50),
        "project_irr_p90": _quantile(project_arr, 90),
        "equity_irr_p10": _quantile(equity_arr, 10),
        "equity_irr_p50": _quantile(equity_arr, 50),
        "equity_irr_p90": _quantile(equity_arr, 90),
        "IRR_P10": _quantile(equity_arr, 10),
        "IRR_P50": _quantile(equity_arr, 50),
        "IRR_P90": _quantile(equity_arr, 90),
        "IRR_mean": round(float(equity_arr.mean()), 2) if equity_arr.size else None,
        "IRR_std": round(float(equity_arr.std()), 2) if equity_arr.size else None,
        "prob_positive_irr": round(float((equity_arr > 0).mean() * 100.0), 1) if equity_arr.size else None,
        "prob_positive_project_irr": round(float((project_arr > 0).mean() * 100.0), 1) if project_arr.size else None,
        "prob_min_dscr_above_target": round(float((dscr_arr >= finance.target_dscr).mean() * 100.0), 1)
        if dscr_arr.size
        else None,
    }

    base_financing = scenarios["base"]["financing_summary"]
    scenario_rows = []
    for label in ["pessimista", "base", "otimista"]:
        s = scenarios[label]
        scenario_rows.append(
            {
                "cenario": label.title(),
                "receita_ano1_rs": s["receita_ano1"],
                "project_irr_pct": s["project_irr_pct"],
                "equity_irr_pct": s["equity_irr_pct"],
                "project_npv_rs": s["project_npv_rs"],
                "equity_npv_rs": s["equity_npv_rs"],
                "payback_anos": s["payback_anos"],
                "min_dscr": s["min_dscr"],
                "llcr": s["llcr"],
                "debt_amount_rs": s["debt_amount_rs"],
            }
        )

    return {
        "scenarios": scenarios,
        "distribution": distribution,
        "financing": base_financing,
        "assumptions": asdict(finance),
        "scenario_table": pd.DataFrame(scenario_rows),
        "annual_base_case": annual_tables["base"],
        "annual_pessimistic": annual_tables["pessimista"],
        "annual_optimistic": annual_tables["otimista"],
        "samples": {
            "project_irr_pct": project_irrs,
            "equity_irr_pct": equity_irrs,
            "min_dscr": min_dscrs,
        },
        "capex_total": round(float(proj.capex_total), 2),
        "opex_anual": round(float(proj.opex_anual), 2),
        "economia_anual_base": round(float(mc_result.get("receita_p50", 0.0)), 2),
    }


def _serialize_df(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype(str)
    return out.to_dict(orient="records")


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project": result["project"],
        "validation": result["validation"],
        "dispatch_mode": result.get("dispatch_mode", "heuristico"),
        "pld_submercado_referencia": result.get("pld_submercado_referencia", ""),
        "deterministic": {
            "summary": result["deterministic"]["summary"],
            "hourly_preview": _serialize_df(result["deterministic"]["hourly"], max_rows=336),
        },
        "monte_carlo": {k: v for k, v in result["monte_carlo"].items() if k != "resultados"},
        "financials": {
            "scenarios": result["financials"]["scenarios"],
            "distribution": result["financials"]["distribution"],
            "financing": result["financials"]["financing"],
            "assumptions": result["financials"]["assumptions"],
            "scenario_table": _serialize_df(result["financials"]["scenario_table"]),
            "annual_base_case": _serialize_df(result["financials"]["annual_base_case"]),
        },
        "next_day_guidance": {
            k: v for k, v in result.get("next_day_guidance", {}).items()
            if k != "plano_horario"  # omit large list from preview; available via CSV download
        },
        "next_day_plan": result.get("next_day_guidance", {}).get("plano_horario", []),
        "irr_optimization": result.get("irr_optimization", {}),
        "timestamp": result["timestamp"],
    }


def run_bess_simulation(
    file_path: str,
    n_sims: Optional[int] = None,
    finance_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end BESS pipeline."""

    print(f"\n{'=' * 68}")
    print("  Kintuadi Energy · BESS Simulation Engine")
    print(f"{'=' * 68}")

    proj = parse_excel(file_path)
    issues = validate_project(proj)

    print("\n[1/5] Projeto")
    print(f"  {proj.project_id} | {proj.potencia_mw:.1f} MW / {proj.energia_mwh:.1f} MWh")
    print(f"  Modo: {proj.modo} | Submercado: {proj.submercado}")
    print(f"  CAPEX: R${proj.capex_total:,.0f} | OPEX: R${proj.opex_anual:,.0f}/ano")

    print("\n[2/5] Validacao")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  Sem avisos criticos.")

    print("\n[3/5] Simulacao deterministica")
    pld = get_pld_path(proj)
    load = get_load_path(proj)
    tariff = get_tariff_path(proj)

    # Despacho ótimo via LP (se habilitado) ou heurístico (fallback)
    ref_sub = proj.pld_submercado if proj.pld_submercado else proj.submercado
    if proj.usar_otimizacao_lp:
        print(f"  Modo LP ativo | Submercado PLD de referência: {ref_sub}")
        try:
            charge_opt, discharge_opt = optimize_dispatch_lp(proj, pld, load, year=1)
            det_hourly, det_summary = simulate_year(
                proj, pld, load, year=1, tariff_hourly=tariff,
                charge=charge_opt, discharge=discharge_opt,
            )
            det_summary["dispatch_mode"] = "lp_otimizado"
        except Exception as exc:
            logger.warning("LP dispatch falhou (%s) – usando heurística.", exc)
            det_hourly, det_summary = simulate_year(proj, pld, load, year=1, tariff_hourly=tariff)
            det_summary["dispatch_mode"] = "heuristico_fallback"
    else:
        det_hourly, det_summary = simulate_year(proj, pld, load, year=1, tariff_hourly=tariff)
        det_summary["dispatch_mode"] = "heuristico"

    print(f"  Receita liquida ano 1: R${det_summary['receita_liquida_rs']:,.0f}")
    print(f"  Energia descarregada: {det_summary['energia_descarregada_mwh']:,.1f} MWh")
    print(f"  Modo despacho: {det_summary['dispatch_mode']}")

    n = int(n_sims or proj.n_simulacoes)
    print(f"\n[4/5] Monte Carlo ({n} simulacoes)")
    mc = run_monte_carlo(proj, n)
    print(f"  Receita P10: R${mc['receita_p10']:,.0f}")
    print(f"  Receita P50: R${mc['receita_p50']:,.0f}")
    print(f"  Receita P90: R${mc['receita_p90']:,.0f}")

    print("\n[5/5] Finance")
    finance_inputs = FinanceInputs(**(finance_overrides or {}))
    financials = compute_financials(proj, mc, finance_inputs)
    base = financials["scenarios"]["base"]
    print(f"  Equity IRR base: {base['equity_irr_pct']}%")
    print(f"  Equity NPV base: R${base['equity_npv_rs']:,.0f}")
    print(f"  Min DSCR base: {base['min_dscr']}")
    print(f"  LLCR base: {base['llcr']}")

    # Guia de operação para o dia seguinte (baseado no PLD publicado hoje)
    print("\n[+] Guia de operação — dia seguinte")
    try:
        next_day = get_next_day_guidance(proj)
        print(f"  {next_day['resumo_texto']}")
    except Exception as exc:
        logger.warning("next_day_guidance falhou: %s", exc)
        next_day = {"erro": str(exc)}

    # Otimização de TIR (se LP habilitado)
    irr_opt: Dict[str, Any] = {}
    if proj.usar_otimizacao_lp:
        print("\n[+] Projecao de TIR — 3 cenarios D+1")
        try:
            irr_opt = optimize_for_irr(
                proj, pld, load,
                next_day_guidance=next_day,
                finance_overrides=finance_overrides,
            )
            base_irr = irr_opt.get("optimal_irr_pct")
            irr_str = f"{base_irr:.2f}%" if base_irr is not None else "N/D"
            print(f"  Fonte: {irr_opt.get('fonte_referencia')} | TIR equity base: {irr_str}")
        except Exception as exc:
            logger.warning("optimize_for_irr falhou: %s", exc)
            irr_opt = {"erro": str(exc)}

    print(f"\n{'=' * 68}\n")

    return {
        "project": {k: v for k, v in proj.__dict__.items() if k != "load_profile"},
        "validation": issues,
        "deterministic": {"summary": det_summary, "hourly": det_hourly},
        "monte_carlo": mc,
        "financials": financials,
        "next_day_guidance": next_day,
        "irr_optimization": {
            k: v for k, v in irr_opt.items()
            if k not in ("best",)  # exclui arrays numpy do JSON principal
        } if irr_opt else {},
        "dispatch_mode": det_summary.get("dispatch_mode", "heuristico"),
        "pld_submercado_referencia": proj.pld_submercado if proj.pld_submercado else proj.submercado,
        "timestamp": datetime.now().isoformat(),
    }


def _chart_layout(title: str, height: int = 360) -> Dict[str, Any]:
    return {
        "title": title,
        "template": "plotly_dark",
        "paper_bgcolor": PLOT_BG,
        "plot_bgcolor": CARD_BG,
        "font": {"color": TEXT},
        "height": height,
        "margin": {"l": 30, "r": 20, "t": 60, "b": 30},
    }


def _plot_scenario_bars(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    df = financials["scenario_table"].copy()
    fig = go.Figure()
    fig.add_bar(
        x=df["cenario"],
        y=df["equity_irr_pct"],
        name="TIR do equity",
        marker_color=[RED, ACCENT, GREEN],
        text=[_fmt_pct(v) for v in df["equity_irr_pct"]],
        textposition="outside",
    )
    fig.update_layout(**_chart_layout("Cenários de retorno", 330))
    fig.update_yaxes(title="TIR do equity (%)", gridcolor=GRID)
    return fig


def _plot_capital_stack(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    financing = financials["financing"]
    labels = ["Dívida", "Equity do patrocinador", "MMRA inicial"]
    values = [
        max(float(financing.get("debt_amount_rs", 0.0)), 0.0),
        max(float(financing.get("equity_amount_rs", 0.0)), 0.0),
        max(float(financing.get("initial_mmra_rs", 0.0)), 0.0),
    ]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker={"colors": [BLUE, ACCENT, GREEN]},
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(**_chart_layout("Estrutura de capital", 330))
    return fig


def _plot_dscr(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    annual = financials["annual_base_case"].copy()
    target = financials["financing"]["target_dscr"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=annual["year"],
            y=annual["dscr"],
            name="DSCR",
            mode="lines+markers",
            line={"color": BLUE, "width": 3},
        )
    )
    fig.add_hline(y=target, line_dash="dash", line_color=ACCENT, annotation_text=f"Alvo {target:.2f}x")
    fig.update_layout(**_chart_layout("Perfil anual de DSCR", 330))
    fig.update_yaxes(title="x", gridcolor=GRID)
    fig.update_xaxes(title="Ano")
    return fig


def _plot_cashflow_bridge(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    annual = financials["annual_base_case"].copy()
    fig = go.Figure()
    fig.add_bar(x=annual["year"], y=annual["cfads_rs"], name="CFADS", marker_color=GREEN)
    fig.add_bar(x=annual["year"], y=annual["debt_service_rs"], name="Serviço da dívida", marker_color=RED)
    fig.add_trace(
        go.Scatter(
            x=annual["year"],
            y=annual["equity_cashflow_rs"],
            name="Caixa do equity",
            mode="lines+markers",
            line={"color": ACCENT, "width": 3},
            yaxis="y2",
        )
    )
    fig.update_layout(
        **_chart_layout("CFADS versus serviço da dívida", 360),
        barmode="group",
        yaxis2={"overlaying": "y", "side": "right", "title": "Caixa do equity"},
    )
    fig.update_yaxes(title="R$")
    fig.update_xaxes(title="Ano")
    return fig


def _plot_degradation(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    annual = financials["annual_base_case"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=annual["year"],
            y=annual["usable_energy_mwh"],
            name="Energia útil",
            mode="lines+markers",
            line={"color": BLUE, "width": 3},
        )
    )
    fig.add_trace(
        go.Bar(
            x=annual["year"],
            y=annual["augmentation_rs"] / 1e6,
            name="CAPEX de augmentation",
            marker_color=ACCENT,
            yaxis="y2",
        )
    )
    fig.update_layout(
        **_chart_layout("Degradação e augmentation", 360),
        yaxis={"title": "Energia útil (MWh)", "gridcolor": GRID},
        yaxis2={"overlaying": "y", "side": "right", "title": "Augmentation (R$ MM)"},
        bargap=0.30,
    )
    return fig


def _plot_hourly_operations(det_hourly: pd.DataFrame, hours_to_show: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = det_hourly.head(hours_to_show).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["pld_rs_mwh"],
            name="PLD",
            line={"color": ACCENT, "width": 2.5},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["soc_mwh"],
            name="SOC",
            line={"color": BLUE, "width": 2},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["discharged_mwh"],
            name="Descarga",
            marker_color=GREEN,
            opacity=0.45,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=-df["charged_mwh"],
            name="Carga",
            marker_color=RED,
            opacity=0.35,
        ),
        secondary_y=True,
    )
    fig.update_layout(**_chart_layout("Visão horária de despacho", 420), barmode="relative")
    fig.update_yaxes(title_text="PLD (R$/MWh)", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="SOC / MWh", secondary_y=True)
    return fig


def _plot_distribution(financials: Dict[str, Any]):
    import plotly.graph_objects as go

    values = financials["samples"]["equity_irr_pct"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=24,
            marker_color=BLUE,
            opacity=0.85,
            name="TIR do equity",
        )
    )
    fig.update_layout(**_chart_layout("Distribuição Monte Carlo da TIR do equity", 330))
    fig.update_xaxes(title="TIR do equity (%)")
    fig.update_yaxes(title="Frequência", gridcolor=GRID)
    return fig


def _render_intro(st, template_path: Path):
    st.markdown("## Dashboard Financeiro para Projetos BESS")
    st.caption("Análises merchant e behind-the-meter com visão de bancabilidade e retorno.")

    hero_col, kpi_col = st.columns([1.6, 1.0])
    with hero_col:
        st.markdown(
            """
            O armazenamento em baterias é uma das teses mais relevantes em energia e infraestrutura no momento.

            Este módulo amplia a plataforma de um simulador operacional para uma leitura mais próxima de
            project finance: lógica de operação, bandas de receita via Monte Carlo, dimensionamento de
            dívida, DSCR, LLCR, estrutura de capital e tabelas exportáveis.
            """
        )
        st.markdown(
            """
            **O que esta versão já entrega**

            - Simula o despacho horário do BESS para arbitragem, peak shaving e modo híbrido.
            - Constrói faixas de receita merchant com Monte Carlo.
            - Converte o caso operacional em uma visão anual de fluxo de caixa com lógica de credor.
            - Apresenta sizing de dívida, MMRA, DSCR, LLCR, TIR do equity e VPL do projeto.
            - Exporta tabelas horárias e anuais para análise detalhada.
            """
        )
    with kpi_col:
        st.markdown(
            """
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px 18px 8px 18px;">
              <div style="font-size:12px;color:#9ca3af;text-transform:uppercase;letter-spacing:0.08em;">Dentro do modelo</div>
              <div style="margin-top:10px;font-size:14px;line-height:1.7;">
                Duração e degradação<br/>
                Bandas de receita via Monte Carlo<br/>
                Sizing da dívida e métricas do credor<br/>
                MMRA e waterfall do equity<br/>
                Despacho horário e tabelas exportáveis
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if template_path.exists():
        st.download_button(
            "Baixar template BESS",
            template_path.read_bytes(),
            file_name=template_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="bess_template_download",
        )


def render_bess_tab():
    """Render the BESS tab inside the premium Streamlit app."""

    import plotly.graph_objects as go
    import streamlit as st

    # Aceita v2 (com LP e submercado PLD) e v1 como fallback
    _base = Path(__file__).parent
    template_path = next(
        (p for p in [
            _base / "bess_project_input_template_v2.xlsx",
            _base / "bess_project_input_template_v1.xlsx",
        ] if p.exists()),
        _base / "bess_project_input_template_v2.xlsx",
    )
    _render_intro(st, template_path)

    uploaded = st.file_uploader(
        "Carregar template de entrada BESS (v1 ou v2)",
        type=["xlsx"],
        key="bess_upload",
        help="Use o template v2 para habilitar despacho LP e seleção de submercado PLD.",
    )

    upload_sig = None if uploaded is None else f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("bess_upload_sig") != upload_sig:
        st.session_state["bess_upload_sig"] = upload_sig
        st.session_state.pop("bess_result", None)
        st.session_state.pop("bess_project_preview", None)

    if uploaded is None:
        st.info("Carregue o template BESS (v2 recomendado) para montar o dashboard da simulação.")
        return

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    try:
        proj = parse_excel(tmp_path)
        issues = validate_project(proj)
        st.session_state["bess_project_preview"] = proj

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Potência", f"{proj.potencia_mw:.1f} MW")
        k2.metric("Energia", f"{proj.energia_mwh:.1f} MWh")
        k3.metric("Duração", f"{proj.duracao_h:.1f} h")
        k4.metric("CAPEX", _fmt_currency(proj.capex_total, scale=1e6, suffix=" MM"))
        k5.metric("Estratégia", proj.modo.title())

        # Linha de configuração: submercado PLD e modo de despacho
        _ref_sub_disp = proj.pld_submercado if proj.pld_submercado else proj.submercado
        _lp_badge = "✅ LP ativo" if proj.usar_otimizacao_lp else "⚙️ Heurístico"
        st.caption(
            f"**Submercado PLD de referência:** {_ref_sub_disp} &nbsp;·&nbsp; "
            f"**Despacho:** {_lp_badge} &nbsp;·&nbsp; "
            f"**Submercado do projeto:** {proj.submercado}"
        )

        if issues:
            with st.expander("Notas de validação do projeto", expanded=False):
                for issue in issues:
                    st.warning(issue)

        ctrl1, ctrl2 = st.columns([1.0, 1.3])
        with ctrl1:
            n_sims = st.slider(
                "Número de simulações Monte Carlo",
                min_value=100,
                max_value=5000,
                value=min(max(int(proj.n_simulacoes), 250), 1500),
                step=100,
                key="bess_nsims",
            )
            hours_window = st.select_slider(
                "Janela do gráfico horário",
                options=[48, 72, 96, 168, 336],
                value=168,
                key="bess_hour_window",
            )

        with ctrl2:
            with st.expander("Premissas de financiamento", expanded=True):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    gearing = st.slider("Alavancagem máxima (%)", 30, 85, 70, 1, key="bess_gearing")
                    debt_cost = st.slider("Custo da dívida (%)", 6.0, 20.0, 12.0, 0.25, key="bess_debt_cost")
                with fc2:
                    tenor = st.slider("Prazo da dívida (anos)", 4, 18, 10, 1, key="bess_tenor")
                    grace = st.slider("Carência (anos)", 0, 3, 1, 1, key="bess_grace")
                with fc3:
                    target_dscr = st.slider("DSCR alvo", 1.10, 2.00, 1.35, 0.05, key="bess_target_dscr")
                    mmra = st.slider("MMRA (meses)", 0, 12, 6, 1, key="bess_mmra")
                augmentation_factor = st.slider(
                    "Fator indicativo de augmentation (% do CAPEX do battery pack)",
                    0,
                    100,
                    50,
                    5,
                    key="bess_augmentation_factor",
                )

        finance_overrides = {
            "gearing_max_pct": float(gearing),
            "debt_cost_pct": float(debt_cost),
            "debt_tenor_years": int(tenor),
            "grace_years": int(grace),
            "target_dscr": float(target_dscr),
            "mmra_months": int(mmra),
            "augmentation_factor_pct": float(augmentation_factor),
        }

        if st.button("Rodar simulação BESS", key="bess_run"):
            with st.spinner(f"Executando {n_sims} simulações e montando o dashboard financeiro..."):
                st.session_state["bess_result"] = run_bess_simulation(
                    tmp_path,
                    n_sims=n_sims,
                    finance_overrides=finance_overrides,
                )

        result = st.session_state.get("bess_result")
        if not result:
            st.info("Ajuste as premissas acima e clique em `Rodar simulação BESS` para gerar o dashboard.")
            return

        financials = result["financials"]
        base = financials["scenarios"]["base"]
        financing = financials["financing"]
        dist = financials["distribution"]
        det_summary = result["deterministic"]["summary"]
        det_hourly = result["deterministic"]["hourly"]
        annual_base = financials["annual_base_case"].copy()

        # Novos campos do pipeline v2
        next_day_guidance = result.get("next_day_guidance", {})
        irr_opt = result.get("irr_optimization", {})
        dispatch_mode = result.get("dispatch_mode", "heuristico")
        pld_ref_sub = result.get("pld_submercado_referencia", proj.submercado)

        st.markdown("---")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("TIR do equity (base)", _fmt_pct(base["equity_irr_pct"]))
        m2.metric("VPL do projeto", _fmt_currency(base["project_npv_rs"], scale=1e6, suffix=" MM"))
        m3.metric("Dimensionamento da dívida", _fmt_currency(financing["debt_amount_rs"], scale=1e6, suffix=" MM"))
        m4.metric("DSCR mínimo", f"{financing['min_dscr']}x" if financing["min_dscr"] is not None else "-")
        m5.metric("LLCR", f"{financing['llcr']}x" if financing["llcr"] is not None else "-")
        m6.metric(
            "Prob. de TIR positiva do equity",
            f"{dist['prob_positive_irr']}%" if dist["prob_positive_irr"] is not None else "-",
        )

        # Badge do modo de despacho
        _mode_color = {"lp_otimizado": GREEN, "heuristico_fallback": ACCENT, "heuristico": BLUE}.get(dispatch_mode, BLUE)
        st.markdown(
            f"<span style='background:{_mode_color}22;border:1px solid {_mode_color};"
            f"border-radius:6px;padding:3px 10px;font-size:12px;color:{_mode_color};font-weight:600'>"
            f"Despacho: {dispatch_mode.replace('_',' ').title()}"
            f"</span>&nbsp;&nbsp;"
            f"<span style='color:#9ca3af;font-size:12px'>Submercado PLD de referência: <b>{pld_ref_sub}</b></span>",
            unsafe_allow_html=True,
        )

        tab_overview, tab_bank, tab_ops, tab_next_day, tab_irr_opt, tab_export = st.tabs(
            ["Tese de investimento", "Bancabilidade", "Operação", "🗓 Guia D+1", "📈 TIR Otimizada", "Exportações"]
        )

        with tab_overview:
            o1, o2 = st.columns(2)
            with o1:
                st.plotly_chart(_plot_scenario_bars(financials), use_container_width=True, key="bess_scenario_bars")
            with o2:
                st.plotly_chart(_plot_capital_stack(financials), use_container_width=True, key="bess_capital_stack")

            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(_plot_distribution(financials), use_container_width=True, key="bess_irr_hist")
            with d2:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=["Determinístico A1", "MC P10", "MC P50", "MC P90"],
                        y=[
                            det_summary["receita_liquida_rs"],
                            result["monte_carlo"]["receita_p10"],
                            result["monte_carlo"]["receita_p50"],
                            result["monte_carlo"]["receita_p90"],
                        ],
                        marker_color=[BLUE, RED, ACCENT, GREEN],
                        text=[
                            _fmt_currency(det_summary["receita_liquida_rs"], scale=1e6, suffix=" MM"),
                            _fmt_currency(result["monte_carlo"]["receita_p10"], scale=1e6, suffix=" MM"),
                            _fmt_currency(result["monte_carlo"]["receita_p50"], scale=1e6, suffix=" MM"),
                            _fmt_currency(result["monte_carlo"]["receita_p90"], scale=1e6, suffix=" MM"),
                        ],
                        textposition="outside",
                    )
                )
                fig.update_layout(**_chart_layout("Faixa de receita", 330))
                fig.update_yaxes(title="R$", gridcolor=GRID)
                st.plotly_chart(fig, use_container_width=True, key="bess_revenue_range")

            st.markdown("### Resumo de cenários")
            scenario_table = financials["scenario_table"].copy()
            display = scenario_table.rename(
                columns={
                    "cenario": "Cenário",
                    "receita_ano1_rs": "Receita ano 1 (R$)",
                    "project_irr_pct": "TIR do projeto (%)",
                    "equity_irr_pct": "TIR do equity (%)",
                    "project_npv_rs": "VPL do projeto (R$)",
                    "equity_npv_rs": "VPL do equity (R$)",
                    "payback_anos": "Payback (anos)",
                    "min_dscr": "DSCR mínimo",
                    "llcr": "LLCR",
                    "debt_amount_rs": "Dimensionamento da dívida (R$)",
                }
            )
            st.dataframe(display, use_container_width=True)

        with tab_bank:
            b1, b2 = st.columns(2)
            with b1:
                st.plotly_chart(_plot_dscr(financials), use_container_width=True, key="bess_dscr")
            with b2:
                st.plotly_chart(_plot_cashflow_bridge(financials), use_container_width=True, key="bess_cash_bridge")

            bank_metrics = pd.DataFrame(
                [
                    {"Metric": "Valor da dívida", "Value": _fmt_currency(financing["debt_amount_rs"], scale=1e6, suffix=" MM")},
                    {"Metric": "Equity do patrocinador", "Value": _fmt_currency(financing["equity_amount_rs"], scale=1e6, suffix=" MM")},
                    {"Metric": "MMRA inicial", "Value": _fmt_currency(financing["initial_mmra_rs"], scale=1e6, suffix=" MM")},
                    {"Metric": "Alavancagem", "Value": _fmt_pct(financing["gearing_pct"])},
                    {"Metric": "Custo da dívida", "Value": _fmt_pct(financing["debt_cost_pct"])},
                    {"Metric": "DSCR alvo", "Value": f"{financing['target_dscr']:.2f}x"},
                    {"Metric": "DSCR mínimo", "Value": f"{financing['min_dscr']:.2f}x" if financing["min_dscr"] is not None else "-"},
                    {"Metric": "DSCR médio", "Value": f"{financing['avg_dscr']:.2f}x" if financing["avg_dscr"] is not None else "-"},
                    {"Metric": "LLCR", "Value": f"{financing['llcr']:.2f}x" if financing["llcr"] is not None else "-"},
                    {
                        "Metric": "Prob. de DSCR mínimo acima do alvo",
                        "Value": f"{dist['prob_min_dscr_above_target']}%" if dist["prob_min_dscr_above_target"] is not None else "-",
                    },
                ]
            )
            st.dataframe(bank_metrics, use_container_width=True)
            st.markdown("### Visão anual do credor")
            st.dataframe(
                annual_base[
                    [
                        "year",
                        "revenue_rs",
                        "opex_rs",
                        "cfads_rs",
                        "augmentation_rs",
                        "debt_service_rs",
                        "dscr",
                        "mmra_required_rs",
                        "equity_cashflow_rs",
                    ]
                ],
                use_container_width=True,
            )

        with tab_ops:
            st.plotly_chart(
                _plot_hourly_operations(det_hourly, hours_window),
                use_container_width=True,
                key="bess_hourly_ops",
            )
            st.plotly_chart(_plot_degradation(financials), use_container_width=True, key="bess_degradation")

            op1, op2, op3, op4 = st.columns(4)
            op1.metric("Receita líquida determinística", _fmt_currency(det_summary["receita_liquida_rs"], scale=1e6, suffix=" MM"))
            op2.metric("Energia descarregada", f"{det_summary['energia_descarregada_mwh']:,.1f} MWh")
            op3.metric("Ciclos equivalentes", f"{det_summary['ciclos_equivalentes']:,.1f}")
            op4.metric("Fator de capacidade", f"{det_summary['fator_capacidade_pct']:,.1f}%")

            st.markdown(
                """
                **Escopo atual do modelo**

                Este dashboard já converte os resultados operacionais em uma visão financeira com degradação,
                reserva para augmentation, MMRA, sizing de dívida e retorno do equity. Como próximos saltos,
                vale considerar uma camada explícita de receita contratada via CTA e uma lógica de carregamento
                associada a curtailment renovável, para aproximar o módulo de uma modelagem completa de project finance.
                """
            )

        # ── Tab: Guia de Operação D+1 ──────────────────────────────────────────
        with tab_next_day:
            if next_day_guidance and "erro" not in next_day_guidance:
                nd = next_day_guidance
                st.markdown(f"### Plano de operação — {nd.get('data_operacao', 'N/D')}")
                st.caption(
                    f"PLD **oficial** do banco ({nd.get('fonte_pld', 'DuckDB/Neon')}) — puro, sem spread. "
                    f"Publicado em **{nd.get('data_publicacao_pld', 'N/D')}** para operação do dia seguinte. "
                    f"Submercado de referência: **{nd.get('submercado_referencia', pld_ref_sub)}**."
                )

                nd_c1, nd_c2, nd_c3, nd_c4 = st.columns(4)
                nd_c1.metric("Resultado líquido do dia", _fmt_currency(nd.get("receita_liquida_dia_rs") or nd.get("receita_total_esperada_rs", 0)))
                nd_c2.metric("Receita de arbitragem", _fmt_currency(nd.get("receita_arbitragem_rs", 0)))
                nd_c3.metric("Liquidação de mercado", _fmt_currency(nd.get("liquidacao_mercado_rs", 0)))
                nd_c4.metric("Encargos TUSD/ESS", _fmt_currency(-(nd.get("encargos_tusd_rs", 0))))

                nd_e1, nd_e2 = st.columns(2)
                nd_e1.metric("Energia carregada", f"{nd.get('energia_carregada_mwh', 0):.3f} MWh")
                nd_e2.metric("Energia descarregada", f"{nd.get('energia_descarregada_mwh', 0):.3f} MWh")

                plano = nd.get("plano_horario", [])
                if plano:
                    df_plan = pd.DataFrame(plano)

                    # Gráfico despacho hora a hora
                    _ACTION_COLOR = {"CARREGAR": RED, "DESCARREGAR": GREEN, "OCIOSO": "#374151"}
                    fig_nd = go.Figure()
                    for acao, color in _ACTION_COLOR.items():
                        mask = df_plan["acao"] == acao
                        if mask.any():
                            y_val = df_plan.loc[mask, "descarga_mw"] - df_plan.loc[mask, "carga_mw"]
                            fig_nd.add_bar(x=df_plan.loc[mask, "hora"], y=y_val, name=acao, marker_color=color)
                    fig_nd.add_scatter(
                        x=df_plan["hora"], y=df_plan["pld_rs_mwh"],
                        name="PLD oficial (R$/MWh)", mode="lines+markers",
                        line={"color": ACCENT, "width": 2.5}, yaxis="y2",
                    )
                    fig_nd.update_layout(
                        **_chart_layout(f"Despacho ótimo D+1 — {nd.get('data_operacao', '')} | PLD oficial", 360),
                        barmode="relative",
                        yaxis={"title": "Potência (MW, + descarga / – carga)", "gridcolor": GRID},
                        yaxis2={"overlaying": "y", "side": "right", "title": "PLD (R$/MWh)"},
                    )
                    st.plotly_chart(fig_nd, use_container_width=True, key="bess_next_day_chart")

                    # SOC
                    fig_soc = go.Figure()
                    fig_soc.add_scatter(
                        x=df_plan["hora"], y=df_plan["soc_mwh"],
                        mode="lines+markers", name="SOC (MWh)",
                        line={"color": BLUE, "width": 2.5},
                    )
                    fig_soc.update_layout(**_chart_layout("Estado de Carga (SOC) — D+1", 260))
                    fig_soc.update_yaxes(title="MWh", gridcolor=GRID)
                    st.plotly_chart(fig_soc, use_container_width=True, key="bess_soc_nextday")

                    # Gráfico de receita hora a hora
                    _receita_col = "receita_total_rs" if "receita_total_rs" in df_plan.columns else "receita_esperada_rs"
                    if _receita_col in df_plan.columns:
                        fig_rec = go.Figure()
                        fig_rec.add_bar(
                            x=df_plan["hora"],
                            y=df_plan[_receita_col],
                            marker_color=[GREEN if v >= 0 else RED for v in df_plan[_receita_col]],
                            name="Receita líquida (R$)",
                        )
                        fig_rec.update_layout(**_chart_layout("Receita líquida por hora (R$)", 240))
                        fig_rec.update_yaxes(title="R$", gridcolor=GRID)
                        st.plotly_chart(fig_rec, use_container_width=True, key="bess_receita_hora_chart")

                    with st.expander("Plano hora a hora (detalhado)"):
                        rename_map = {
                            "hora": "Hora", "acao": "Ação", "carga_mw": "Carga (MW)",
                            "descarga_mw": "Descarga (MW)", "soc_mwh": "SOC (MWh)",
                            "pld_rs_mwh": "PLD (R$/MWh)",
                            "receita_arbitragem_rs": "Arbitragem (R$)",
                            "liquidacao_mercado_rs": "Liquidação Mercado (R$)",
                            "encargo_tusd_rs": "Encargo TUSD (R$)",
                            "receita_total_rs": "Receita Total (R$)",
                            "receita_esperada_rs": "Receita (R$)",
                            "carga_rede_mw": "Carga Rede (MW)",
                        }
                        st.dataframe(
                            df_plan.rename(columns={k: v for k, v in rename_map.items() if k in df_plan.columns}),
                            use_container_width=True,
                        )

                    # Nota sobre liquidação de mercado
                    if proj.carga_media_mw > 0 and proj.contrato_mw > 0:
                        st.info(
                            f"**Exposição à liquidação de mercado ativa.** "
                            f"Carga média: {proj.carga_media_mw:.1f} MW | Contrato: {proj.contrato_mw:.1f} MW | "
                            f"Flexibilidade: ±{proj.flexibilidade_mw:.1f} MW | "
                            f"Encargo TUSD: R${proj.encargo_tusd_rs_mwh:.0f}/MWh.  \n"
                            "O projeto compra energia acima do contrato no PLD baixo (carregar BESS) e "
                            "injeta/reduz consumo no PLD alto (descarregar BESS), liquidando a diferença no CCEE."
                        )
                    else:
                        st.caption(
                            "💡 Configure 'Carga Média (MW)' e 'Contrato (MW)' na aba ALTERNATIVE do template "
                            "para modelar a exposição à liquidação de mercado pelo PLD."
                        )

                st.info(nd.get("resumo_texto", ""))

            elif next_day_guidance and "erro" in next_day_guidance:
                st.warning(f"Guia D+1 não disponível: {next_day_guidance['erro']}")
            else:
                st.info("Rode a simulação para gerar o guia de operação do dia seguinte.")

        # ── Tab: Projeção de TIR — 3 cenários D+1 ────────────────────────────
        with tab_irr_opt:
            if irr_opt and "erro" not in irr_opt and irr_opt.get("cenarios"):
                cenarios = irr_opt["cenarios"]
                fonte = irr_opt.get("fonte_referencia", "–")
                rec_dia = irr_opt.get("receita_dia_referencia_rs", 0.0)

                st.markdown("### Projeção de TIR — extrapolação do Guia D+1")
                st.caption(
                    f"**Fonte de referência:** {fonte}  ·  "
                    f"**Receita líquida diária de referência:** {_fmt_currency(rec_dia)}  \n"
                    "Cada cenário replica o resultado do Guia D+1 para todos os dias da vida do projeto, "
                    "com degradação anual aplicada. **Pessimista:** 60% · **Base:** 100% · **Otimista:** 110%."
                )

                # KPIs dos 3 cenários lado a lado
                sc_cols = st.columns(3)
                _SC_COLORS = {"pessimista": RED, "base": ACCENT, "otimista": GREEN}
                _SC_LABELS = {"pessimista": "⬇ Pessimista (×0,6)", "base": "◆ Base (×1,0)", "otimista": "⬆ Otimista (×1,1)"}
                for col_ui, (nome, r) in zip(sc_cols, cenarios.items()):
                    irr_v = r.get("equity_irr_pct")
                    npv_v = r.get("equity_npv_rs")
                    dscr_v = r.get("min_dscr")
                    with col_ui:
                        st.markdown(
                            f"<div style='background:#111827;border:1px solid #1f2937;"
                            f"border-top:3px solid {_SC_COLORS[nome]};border-radius:10px;"
                            f"padding:14px 12px;'>"
                            f"<div style='font-size:11px;color:#9ca3af;text-transform:uppercase;"
                            f"letter-spacing:.06em'>{_SC_LABELS[nome]}</div>"
                            f"<div style='font-size:22px;font-weight:700;color:{_SC_COLORS[nome]};margin-top:6px'>"
                            f"{_fmt_pct(irr_v) if irr_v is not None else '—'}</div>"
                            f"<div style='font-size:11px;color:#9ca3af;margin-top:4px'>TIR equity</div>"
                            f"<div style='font-size:14px;color:#e5e7eb;margin-top:8px'>"
                            f"VPL: {_fmt_currency(npv_v, scale=1e6, suffix=' MM') if npv_v else '—'}</div>"
                            f"<div style='font-size:12px;color:#9ca3af'>"
                            f"DSCR mín: {f'{dscr_v:.2f}x' if dscr_v else '—'}</div>"
                            f"<div style='font-size:12px;color:#9ca3af'>"
                            f"Receita/dia: {_fmt_currency(r.get('receita_dia_rs', 0))}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # Gráfico de barras TIR por cenário
                nomes_label = [_SC_LABELS[n] for n in cenarios]
                irrs = [cenarios[n].get("equity_irr_pct") for n in cenarios]
                cores = [_SC_COLORS[n] for n in cenarios]

                if any(v is not None for v in irrs):
                    fig_irr = go.Figure()
                    fig_irr.add_bar(
                        x=nomes_label,
                        y=[v if v is not None else 0 for v in irrs],
                        marker_color=cores,
                        text=[_fmt_pct(v) if v is not None else "N/D" for v in irrs],
                        textposition="outside",
                    )
                    fig_irr.update_layout(**_chart_layout("TIR do equity — 3 cenários D+1", 340))
                    fig_irr.update_yaxes(title="TIR do equity (%)", gridcolor=GRID)
                    st.plotly_chart(fig_irr, use_container_width=True, key="bess_irr_cenarios_chart")

                # Gráfico de VPL
                npvs = [cenarios[n].get("equity_npv_rs") for n in cenarios]
                if any(v is not None for v in npvs):
                    fig_npv = go.Figure()
                    fig_npv.add_bar(
                        x=nomes_label,
                        y=[v / 1e6 if v is not None else 0 for v in npvs],
                        marker_color=cores,
                        text=[_fmt_currency(v, scale=1e6, suffix=" MM") if v is not None else "N/D" for v in npvs],
                        textposition="outside",
                    )
                    fig_npv.update_layout(**_chart_layout("VPL do equity (R$ MM) — 3 cenários D+1", 300))
                    fig_npv.update_yaxes(title="VPL (R$ MM)", gridcolor=GRID)
                    st.plotly_chart(fig_npv, use_container_width=True, key="bess_npv_cenarios_chart")

                # Tabela consolidada
                with st.expander("Tabela consolidada dos cenários"):
                    rows_tbl = []
                    for nome, r in cenarios.items():
                        rows_tbl.append({
                            "Cenário": _SC_LABELS[nome],
                            "Fator": f"×{r['fator']:.1f}",
                            "Receita/dia (R$)": _fmt_currency(r.get("receita_dia_rs")),
                            "Receita Ano 1 (R$)": _fmt_currency(r.get("receita_ano1_rs"), scale=1e6, suffix=" MM"),
                            "TIR Projeto (%)": _fmt_pct(r.get("project_irr_pct")),
                            "TIR Equity (%)": _fmt_pct(r.get("equity_irr_pct")),
                            "VPL Equity (R$ MM)": _fmt_currency(r.get("equity_npv_rs"), scale=1e6, suffix=" MM"),
                            "DSCR Mín.": f"{r['min_dscr']:.2f}x" if r.get("min_dscr") else "—",
                        })
                    st.dataframe(pd.DataFrame(rows_tbl).set_index("Cenário"), use_container_width=True)

            elif irr_opt and "erro" in irr_opt:
                st.warning(f"Projeção de TIR não disponível: {irr_opt['erro']}")
            elif not proj.usar_otimizacao_lp:
                st.info("Habilite 'Usar Otimização LP' no template (aba STRATEGY) para ativar esta análise.")
            else:
                st.info("Rode a simulação para ver a projeção de TIR em 3 cenários.")

        with tab_export:
            export_payload = _serialize_result(result)
            st.download_button(
                "Baixar resultados (JSON)",
                json.dumps(export_payload, indent=2, ensure_ascii=False, default=str),
                file_name=f"bess_result_{proj.project_id}.json",
                mime="application/json",
                key="bess_download_json",
            )
            st.download_button(
                "Baixar caso base anual (CSV)",
                annual_base.to_csv(index=False).encode("utf-8"),
                file_name=f"bess_annual_base_{proj.project_id}.csv",
                mime="text/csv",
                key="bess_download_annual_csv",
            )
            st.download_button(
                "Baixar despacho horário (CSV)",
                det_hourly.to_csv(index=False).encode("utf-8"),
                file_name=f"bess_dispatch_{proj.project_id}.csv",
                mime="text/csv",
                key="bess_download_hourly_csv",
            )
            st.download_button(
                "Baixar tabela de cenários (CSV)",
                financials["scenario_table"].to_csv(index=False).encode("utf-8"),
                file_name=f"bess_scenarios_{proj.project_id}.csv",
                mime="text/csv",
                key="bess_download_scenarios_csv",
            )
            # Plano D+1 (CSV)
            _nd_plan = next_day_guidance.get("plano_horario", []) if next_day_guidance else []
            if _nd_plan:
                st.download_button(
                    "Baixar plano D+1 (CSV)",
                    pd.DataFrame(_nd_plan).to_csv(index=False).encode("utf-8"),
                    file_name=f"bess_plano_d1_{proj.project_id}.csv",
                    mime="text/csv",
                    key="bess_download_nextday_csv",
                )
            # Otimização de TIR (CSV)
            _irr_rows = irr_opt.get("all_utilization_results", []) if irr_opt else []
            if _irr_rows:
                st.download_button(
                    "Baixar otimização TIR (CSV)",
                    pd.DataFrame(_irr_rows).to_csv(index=False).encode("utf-8"),
                    file_name=f"bess_irr_opt_{proj.project_id}.csv",
                    mime="text/csv",
                    key="bess_download_irr_opt_csv",
                )
            st.dataframe(det_hourly.head(min(hours_window, 96)), use_container_width=True)

    except Exception as exc:
        st.error(f"Erro ao processar o modelo BESS: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kintuadi Energy · BESS Simulation Engine")
    parser.add_argument("--file", required=True, help="Excel input template")
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    parser.add_argument("--sims", type=int, default=None, help="Monte Carlo path count")
    args = parser.parse_args()

    result = run_bess_simulation(args.file, args.sims)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(_serialize_result(result), fh, indent=2, ensure_ascii=False, default=str)
        print(f"Resultado salvo em {args.output}")
