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
    sub = str(submercado).strip().lower()
    mapping = {
        "se/co": "seco",
        "se": "seco",
        "seco": "seco",
        "sul": "s",
        "s": "s",
        "ne": "ne",
        "nordeste": "ne",
        "n": "n",
        "norte": "n",
    }
    return mapping.get(sub, "seco")
 
 
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
    """Gera o guia de operação ótima para o DIA SEGUINTE baseado no PLD publicado.
 
    O PLD do dia D é publicado no dia D-1. Esta função:
    1. Resolve qual submercado usar (``proj.pld_submercado`` ou ``proj.submercado``).
    2. Gera o vetor de PLD para as 24h do dia-alvo (D+1 a partir de hoje).
    3. Roda o despacho LP para as 24 horas.
    4. Retorna um plano hora-a-hora com ação recomendada e valor esperado.
 
    Parâmetros
    ----------
    proj : BESSProject
        Objeto de projeto já carregado.
    target_date : str, opcional
        Data-alvo no formato "AAAA-MM-DD". Padrão: amanhã.
 
    Retorna
    -------
    dict com chaves:
        - submercado_referencia: str
        - data_operacao: str
        - data_publicacao_pld: str
        - plano_horario: list[dict]  (hora, acao, carga_mw, descarga_mw, soc_mwh, pld_r$_mwh, receita_esperada_r$)
        - receita_total_esperada_rs: float
        - energia_carregada_mwh: float
        - energia_descarregada_mwh: float
        - resumo_texto: str
    """
    from datetime import date, timedelta
 
    today = date.today()
    if target_date:
        op_date = pd.to_datetime(target_date).date()
    else:
        op_date = today + timedelta(days=1)
 
    pub_date = op_date - timedelta(days=1)
    ref_sub = proj.pld_submercado if proj.pld_submercado else proj.submercado
 
    # Gera PLD para 24h usando o motor existente (proj configurado para 24h)
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
        usar_pld_modelado=proj.usar_pld_modelado,
        cenario_pld=proj.cenario_pld,
        pld_fixo=proj.pld_fixo,
        spread_tarifario=proj.spread_tarifario,
        modo=proj.modo,
        opex_variavel=proj.opex_variavel,
        horizonte_horas=24,
        data_inicio=str(op_date),
    )
 
    pld_24 = get_pld_path(_proj24)
    load_24 = get_load_path(_proj24)
 
    charge_arr, discharge_arr = optimize_dispatch_lp(_proj24, pld_24, load_24, year=1)
 
    # Reconstrói SOC para o plano
    degradation_factor = 1.0
    energia_util = _proj24.energia_util_mwh
    soc_min_mwh = _proj24.soc_min / 100.0 * _proj24.energia_mwh
    soc_max_mwh = _proj24.soc_max / 100.0 * _proj24.energia_mwh
    soc_cur = soc_min_mwh + 0.5 * energia_util
    eta = _proj24.eficiencia_rt
 
    plano = []
    total_receita = 0.0
    for h in range(24):
        c = float(charge_arr[h])
        d = float(discharge_arr[h])
        soc_cur = float(np.clip(soc_cur + c * eta - d, soc_min_mwh, soc_max_mwh))
        price = float(pld_24[h])
        receita_h = d * price - c * price
        total_receita += receita_h
        if c > 1e-4:
            acao = "CARREGAR"
        elif d > 1e-4:
            acao = "DESCARREGAR"
        else:
            acao = "OCIOSO"
        plano.append(
            {
                "hora": h,
                "acao": acao,
                "carga_mw": round(c, 4),
                "descarga_mw": round(d, 4),
                "soc_mwh": round(soc_cur, 4),
                "pld_rs_mwh": round(price, 2),
                "receita_esperada_rs": round(receita_h, 2),
            }
        )
 
    # Texto resumo
    horas_carga = [p["hora"] for p in plano if p["acao"] == "CARREGAR"]
    horas_desc = [p["hora"] for p in plano if p["acao"] == "DESCARREGAR"]
 
    def _fmt_horas(lst):
        if not lst:
            return "–"
        return ", ".join(f"{h:02d}h" for h in lst)
 
    resumo = (
        f"Submercado de referência: {ref_sub}. "
        f"PLD publicado em {pub_date.strftime('%d/%m/%Y')} para operação de {op_date.strftime('%d/%m/%Y')}. "
        f"Horas de carga: {_fmt_horas(horas_carga)}. "
        f"Horas de descarga: {_fmt_horas(horas_desc)}. "
        f"Receita esperada: R${total_receita:,.2f}."
    )
 
    return {
        "submercado_referencia": ref_sub,
        "data_operacao": str(op_date),
        "data_publicacao_pld": str(pub_date),
        "plano_horario": plano,
        "receita_total_esperada_rs": round(total_receita, 2),
        "energia_carregada_mwh": round(float(charge_arr.sum()), 4),
        "energia_descarregada_mwh": round(float(discharge_arr.sum()), 4),
        "resumo_texto": resumo,
    }
 
 
def optimize_for_irr(
    proj: BESSProject,
    pld: np.ndarray,
    load: np.ndarray,
    utilization_steps: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Outer-loop: itera sobre fatores de utilização para maximizar a TIR do equity.
 
    Para cada fator ``u`` em ``utilization_steps``:
      - Escala potencia_mw e energia_mwh por u
      - Roda despacho LP
      - Simula o ano e calcula a receita
      - Computa TIR do equity
 
    Retorna o melhor fator e o resultado completo da configuração ótima.
    """
    if utilization_steps is None:
        utilization_steps = [round(u, 2) for u in np.arange(0.4, 1.01, 0.1)]
 
    tariff = get_tariff_path(proj)
    finance = FinanceInputs()
    best: Dict[str, Any] = {"equity_irr_pct": None, "utilizacao": None}
    all_results: List[Dict[str, Any]] = []
 
    for u in utilization_steps:
        # Clone com capacidade escalada
        _p = BESSProject(**{
            k: v for k, v in proj.__dict__.items() if k != "load_profile"
        })
        _p.load_profile = proj.load_profile
        _p.potencia_mw = proj.potencia_mw * u
        _p.energia_mwh = proj.energia_mwh * u
 
        try:
            charge_u, discharge_u = optimize_dispatch_lp(_p, pld, load, year=1)
            _, summary_u = simulate_year(_p, pld, load, year=1, tariff_hourly=tariff, charge=charge_u, discharge=discharge_u)
            receita_u = float(summary_u["receita_liquida_rs"])
 
            # TIR simplificada (usa apenas receita do ano 1 replicada)
            operating_df = _build_operating_case(_p, receita_u, finance)
            annual_df, _ = _build_financing_schedule(operating_df, _p.capex_total, finance)
            equity_outflow = _p.capex_total * (1.0 - finance.gearing_max_pct / 100.0)
            equity_cf = [-equity_outflow] + annual_df["equity_cashflow_rs"].tolist()
            irr = _irr(equity_cf)
        except Exception as exc:
            logger.warning("optimize_for_irr: erro em u=%.2f – %s", u, exc)
            irr = None
            receita_u = 0.0
            summary_u = {}
            charge_u = np.zeros(proj.horizonte_horas)
            discharge_u = np.zeros(proj.horizonte_horas)
 
        record = {
            "utilizacao": u,
            "potencia_mw": round(_p.potencia_mw, 3),
            "energia_mwh": round(_p.energia_mwh, 3),
            "receita_rs": round(receita_u, 2),
            "equity_irr_pct": irr,
        }
        all_results.append(record)
 
        if irr is not None and (best["equity_irr_pct"] is None or irr > best["equity_irr_pct"]):
            best = {
                **record,
                "charge": charge_u,
                "discharge": discharge_u,
                "summary": summary_u,
            }
 
    return {
        "best": best,
        "all_utilization_results": all_results,
        "optimal_utilizacao": best.get("utilizacao"),
        "optimal_irr_pct": best.get("equity_irr_pct"),
        "optimal_potencia_mw": best.get("potencia_mw"),
        "optimal_energia_mwh": best.get("energia_mwh"),
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
        print("\n[+] Otimizacao de TIR por fator de utilizacao")
        try:
            irr_opt = optimize_for_irr(proj, pld, load)
            print(f"  Melhor utilizacao: {irr_opt['optimal_utilizacao']:.0%} | TIR equity: {irr_opt['optimal_irr_pct']}%")
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
    from datetime import datetime, timedelta
    
    template_path = Path(__file__).with_name("bess_project_input_template_v1.xlsx")
    _render_intro(st, template_path)
    
    uploaded = st.file_uploader("Carregar template de entrada BESS", type=["xlsx"], key="bess_upload")
    
    upload_sig = None if uploaded is None else f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("bess_upload_sig") != upload_sig:
        st.session_state["bess_upload_sig"] = upload_sig
        st.session_state.pop("bess_result", None)
        st.session_state.pop("bess_project_preview", None)
    
    if uploaded is None:
        st.info("Carregue o arquivo `bess_project_input_template_v1.xlsx` para montar o dashboard da simulação.")
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
        
        # Exibir submercado de referência do PLD
        ref_sub = proj.pld_submercado if proj.pld_submercado else proj.submercado
        st.caption(f"Submercado de referência para PLD: **{ref_sub}**")
        
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
        
        # Botão principal da simulação
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
        
        # =========================================================================
        # NOVA SEÇÃO: Indicadores do modo de despacho e otimização
        # =========================================================================
        st.markdown("---")
        st.markdown("### ⚙️ Configuração da Simulação")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            dispatch_mode = result.get("dispatch_mode", "heuristico")
            mode_labels = {
                "lp_otimizado": "✅ Otimização Linear (LP)",
                "heuristico": "⚙️ Heurístico",
                "heuristico_fallback": "⚠️ Heurístico (fallback)"
            }
            st.metric(
                "Modo de Despacho",
                mode_labels.get(dispatch_mode, dispatch_mode),
                help="LP otimizado maximiza receita via programação linear. Heurístico segue regras pré-definidas."
            )
        
        with col_d2:
            pld_ref = result.get("pld_submercado_referencia", proj.submercado)
            st.metric("Submercado PLD", pld_ref, help="Submercado usado como referência para formação do PLD")
        
        with col_d3:
            # Exibir se LP foi usado
            if proj.usar_otimizacao_lp:
                st.success("Otimização LP habilitada no template")
            else:
                st.info("Otimização LP desabilitada (usar_otimizacao_lp = FALSE)")
        
        # =========================================================================
        # NOVA SEÇÃO: Guia de Operação para o Dia Seguinte
        # =========================================================================
        next_day = result.get("next_day_guidance", {})
        if next_day and "erro" not in next_day:
            st.markdown("---")
            st.markdown("### 📅 Guia de Operação para o Dia Seguinte")
            st.info(next_day.get("resumo_texto", ""))
            
            # Expandir para ver o plano horário
            with st.expander("Ver plano horário detalhado", expanded=False):
                plano = next_day.get("plano_horario", [])
                if plano:
                    plano_df = pd.DataFrame(plano)
                    plano_df["hora_str"] = plano_df["hora"].apply(lambda x: f"{x:02d}:00")
                    plano_df = plano_df[["hora_str", "acao", "carga_mw", "descarga_mw", "soc_mwh", "pld_rs_mwh", "receita_esperada_rs"]]
                    plano_df.columns = ["Hora", "Ação", "Carga (MW)", "Descarga (MW)", "SOC (MWh)", "PLD (R$/MWh)", "Receita Esperada (R$)"]
                    st.dataframe(plano_df, use_container_width=True)
                    
                    # Gráfico do guia
                    fig_guide = go.Figure()
                    fig_guide.add_trace(go.Bar(
                        x=plano_df["Hora"],
                        y=plano_df["Descarga (MW)"],
                        name="Descarga",
                        marker_color=GREEN,
                    ))
                    fig_guide.add_trace(go.Bar(
                        x=plano_df["Hora"],
                        y=-plano_df["Carga (MW)"],
                        name="Carga",
                        marker_color=RED,
                    ))
                    fig_guide.add_trace(go.Scatter(
                        x=plano_df["Hora"],
                        y=plano_df["PLD (R$/MWh)"],
                        name="PLD",
                        yaxis="y2",
                        line=dict(color=ACCENT, width=2),
                    ))
                    fig_guide.update_layout(
                        title="Recomendação de Operação para Amanhã",
                        template="plotly_dark",
                        paper_bgcolor=PLOT_BG,
                        plot_bgcolor=CARD_BG,
                        height=400,
                        yaxis=dict(title="Potência (MW)", gridcolor=GRID),
                        yaxis2=dict(title="PLD (R$/MWh)", overlaying="y", side="right"),
                        barmode="relative",
                    )
                    st.plotly_chart(fig_guide, use_container_width=True, key="bess_next_day_chart")
                    
                    st.caption(
                        f"📊 PLD publicado em {next_day.get('data_publicacao_pld', '—')} | "
                        f"Operação prevista para {next_day.get('data_operacao', '—')} | "
                        f"Receita esperada total: R${next_day.get('receita_total_esperada_rs', 0):,.2f}"
                    )
        
        # =========================================================================
        # NOVA SEÇÃO: Otimização de TIR por Fator de Utilização
        # =========================================================================
        irr_opt = result.get("irr_optimization", {})
        if irr_opt and "erro" not in irr_opt:
            st.markdown("---")
            st.markdown("### 📈 Otimização de TIR por Fator de Utilização")
            st.info(
                f"**Melhor fator de utilização:** {irr_opt.get('optimal_utilizacao', 0)*100:.0f}%  \n"
                f"**TIR do equity otimizada:** {irr_opt.get('optimal_irr_pct', 0)}%  \n"
                f"**Potência ótima:** {irr_opt.get('optimal_potencia_mw', 0):.1f} MW  \n"
                f"**Energia ótima:** {irr_opt.get('optimal_energia_mwh', 0):.1f} MWh"
            )
            
            # Gráfico da curva de otimização
            all_results = irr_opt.get("all_utilization_results", [])
            if all_results:
                opt_df = pd.DataFrame(all_results)
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(
                    x=opt_df["utilizacao"] * 100,
                    y=opt_df["equity_irr_pct"],
                    mode="lines+markers",
                    line=dict(color=ACCENT, width=2),
                    marker=dict(size=8),
                    name="TIR do equity",
                ))
                # Marcar ponto ótimo
                best_u = irr_opt.get("optimal_utilizacao", 0) * 100
                best_irr = irr_opt.get("optimal_irr_pct", 0)
                fig_opt.add_trace(go.Scatter(
                    x=[best_u],
                    y=[best_irr],
                    mode="markers",
                    marker=dict(size=12, color=GREEN, symbol="star"),
                    name=f"Ótimo: {best_u:.0f}%",
                ))
                fig_opt.update_layout(
                    title="TIR do equity × Fator de Utilização",
                    template="plotly_dark",
                    paper_bgcolor=PLOT_BG,
                    plot_bgcolor=CARD_BG,
                    height=350,
                    xaxis=dict(title="Fator de Utilização (%)", gridcolor=GRID),
                    yaxis=dict(title="TIR do equity (%)", gridcolor=GRID),
                )
                st.plotly_chart(fig_opt, use_container_width=True, key="bess_irr_optimization")
                
                st.caption(
                    "O gráfico mostra como a TIR do equity varia com a redução da capacidade instalada. "
                    "Em projetos com receita marginal limitada, reduzir o CAPEX pode aumentar o retorno."
                )
        
        # =========================================================================
        # Dashboard principal (existente)
        # =========================================================================
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
        
        tab_overview, tab_bank, tab_ops, tab_export = st.tabs(
            ["Tese de investimento", "Bancabilidade", "Operação", "Exportações"]
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
                reserva para augmentation, MMRA, sizing de dívida e retorno do equity. 
                
                **Novidades nesta versão:**
                - **Otimização Linear (LP):** despacho ótimo que maximiza a receita de arbitragem
                - **Guia de Operação para o Dia Seguinte:** plano horário baseado no PLD publicado
                - **Otimização de TIR:** identifica o fator de utilização ideal para maximizar retorno do equity
                """
            )
        
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
            
            # Download do guia do dia seguinte
            if next_day and "erro" not in next_day:
                next_day_df = pd.DataFrame(next_day.get("plano_horario", []))
                if not next_day_df.empty:
                    st.download_button(
                        "Baixar guia do dia seguinte (CSV)",
                        next_day_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"bess_next_day_guide_{proj.project_id}.csv",
                        mime="text/csv",
                        key="bess_download_next_day_csv",
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