# -*- coding: utf-8 -*-
"""
premium_engine.py — Energy Price Justification Engine v2
══════════════════════════════════════════════════════════
Módulo premium MAÁTria Energia.

Modelo:  P(PLD ≥ preço | X, T)
onde:
    X = estado físico do SIN  {EAR, ENA, térmica, carga, curtailment}
    T = dimensão temporal     {mês, hora do dia}

Dois modos de análise:
    • ATUAL   — P(PLD | estado físico atual, sazonalidade do período)
    • HISTÓRICO — P(PLD | sazonalidade, sem condicionar ao estado físico)

API CCEE (dadosabertos.ccee.org.br): dados em memória, sem persistência.
"""
from __future__ import annotations

import calendar
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ─── sklearn / scipy (opcionais) ─────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from scipy.optimize import minimize
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ─── constantes ──────────────────────────────────────────────────────────────
_N_SIMULATIONS  = 8_000
_MIN_TRAIN_ROWS = 200          # reduzido para filtros sazonais
_CCEE_API_BASE  = "https://dadosabertos.ccee.org.br/api/3/action"
_SUBMERCADOS    = ["SE", "NE", "S", "N"]

_MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",    4: "Abril",
    5: "Maio",    6: "Junho",     7: "Julho",     8: "Agosto",
    9: "Setembro",10: "Outubro",  11: "Novembro", 12: "Dezembro",
}
_TRIMESTRES = {"Q1 (Jan–Mar)": [1,2,3], "Q2 (Abr–Jun)": [4,5,6],
               "Q3 (Jul–Set)": [7,8,9], "Q4 (Out–Dez)": [10,11,12]}

_COLORS = {
    "gold": "#c8a44d", "blue": "#60a5fa", "green": "#34d399",
    "red":  "#f87171", "purple": "#a78bfa",
    "bg":   "#0b1222", "panel": "#111827",
}

def _hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_REGIME_BREAKS = [pd.Timestamp("2023-08-01"), pd.Timestamp("2025-03-01")]
_REGIME_LABELS = ["pré-solar (2021–ago/2023)", "transição (ago/2023–fev/2025)", "curtailment (mar/2025–hoje)"]
_DECAY_HALF_LIFE_YEARS = 1.5

# ─── CCEE resource IDs ───────────────────────────────────────────────────────
_CCEE_RESOURCES = {
    2023: "67390467-e175-402f-8bf1-491a80d01a01",
    2024: "05c25b5e-aeed-4494-a203-93d68d070b2a",
    2025: "f5e2e2ce-9388-458f-86d2-c038dc18d997",
    2026: "7143897d-d1b7-445e-ba53-5864e5a99688",
}
_CCEE_MONTANTE_COLS = [
    "MONTANTE_MODULADO_CONTRATO_CCEARQ",
    "MONTANTE_MODULADO_CONTRATO_CCEARD",
    "MONTANTE_MODULADO_CONTRATO_PROINFA",
    "MONTANTE_MODULADO_CONTRATO_ITAIPU",
    "MONTANTE_MODULADO_CONTRATO_CCEAL",
    "MONTANTE_MODULADO_CONTRATO_CCGF",
    "MONTANTE_MODULADO_CONTRATO_CCEN",
]
_CCEE_SUB_MAP = {
    "SE": "SE", "SUDESTE": "SE", "S": "S", "SUL": "S",
    "NE": "NE", "NORDESTE": "NE", "N": "N", "NORTE": "N",
}
_CCEE_SUB_API = {
    "SE": "SUDESTE", "NE": "NORDESTE", "S": "SUL", "N": "NORTE",
}


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalFilter:
    """Define o recorte temporal para análise sazonal."""
    tipo:     str              # "mensal" | "trimestral" | "anual"
    meses:    List[int]        # meses a incluir (1–12)
    horas:    Optional[List[int]] = None   # horas do dia (0–23); None = todas
    label:    str = ""         # rótulo legível

    @classmethod
    def mensal(cls, mes: int) -> "TemporalFilter":
        return cls(tipo="mensal", meses=[mes],
                   label=_MESES_PT.get(mes, str(mes)))

    @classmethod
    def trimestral(cls, label: str, meses: List[int]) -> "TemporalFilter":
        return cls(tipo="trimestral", meses=meses, label=label)

    @classmethod
    def anual(cls) -> "TemporalFilter":
        return cls(tipo="anual", meses=list(range(1, 13)), label="Ano completo")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra o DataFrame pelo recorte temporal definido."""
        out = df[df.index.month.isin(self.meses)].copy()
        if self.horas is not None:
            out = out[out.index.hour.isin(self.horas)]
        return out

    def describe(self) -> str:
        parts = [self.label]
        if self.horas:
            parts.append(f"horas {self.horas[0]}h–{self.horas[-1]}h")
        return " | ".join(parts)


@dataclass
class PhysicalScenario:
    """Cenário físico do SIN (real ou implícito)."""
    ear_pct:           float = np.nan
    ena_pct:           float = np.nan
    thermal_ratio:     float = np.nan
    load_ratio:        float = np.nan
    curtailment_ratio: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.ear_pct, self.ena_pct,
                         self.thermal_ratio, self.load_ratio,
                         self.curtailment_ratio], dtype=float)

    @classmethod
    def from_array(cls, a: np.ndarray) -> "PhysicalScenario":
        return cls(*[round(float(v), 4) for v in a])

    def is_valid(self) -> bool:
        return not np.isnan(self.ear_pct) and not np.isnan(self.load_ratio)


@dataclass
class JustificationResult:
    """Resultado completo da análise."""
    preco_analisado:             float
    submercado:                  str
    modo:                        str          # "atual" | "historico"
    filtro_temporal:             TemporalFilter
    pld_fundamental:             float
    fundamental_gap:             float
    probabilidade_justificacao:  float
    classificacao:               str
    cenario_fisico_implicito:    PhysicalScenario
    cenario_fisico_atual:        PhysicalScenario
    pld_simulado_dist:           np.ndarray
    n_obs_filtro:                int          # observações após filtro sazonal
    percentis:                   Dict[str, float]
    model_r2:                    float = np.nan
    model_cv_score:              float = np.nan
    res_std:                     float = np.nan
    feature_importance:          Dict[str, float] = None
    exposicao_spot:              Optional[Dict[str, Any]] = None
    aviso:                       str = ""


# ══════════════════════════════════════════════════════════════════════════════
# PREPARAÇÃO DO DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _assign_regime(ts: "pd.DatetimeIndex") -> np.ndarray:
    regime = np.zeros(len(ts), dtype=float)
    for i, brk in enumerate(_REGIME_BREAKS):
        regime[ts >= brk] = i + 1
    return regime


def _temporal_weights(ts: "pd.Index", half_life_years: float = _DECAY_HALF_LIFE_YEARS) -> np.ndarray:
    try:
        ts_dt = pd.to_datetime(ts)
        age   = (pd.Timestamp.now() - ts_dt).total_seconds() / (365.25*24*3600)
        lam   = np.log(2) / half_life_years
        return np.exp(-lam * np.clip(age.values, 0, None)).astype(float)
    except Exception:
        return np.ones(len(ts), dtype=float)


def _annual_price_index(df: pd.DataFrame, submercado: str) -> pd.Series:
    """
    Normalização temporal: índice de preço anual.

        PLD_normalizado = PLD / média_anual_do_ano

    Remove o drift estrutural de nível de preços entre anos,
    permitindo que o modelo aprenda relações físicas estáveis
    independente do patamar absoluto de cada ano.

    Retorna série com mesmo índice do df — valor 1.0 = nível médio do ano.
    """
    sm_col = {"SE": "pld_se", "NE": "pld_ne", "S": "pld_s", "N": "pld_n"}
    pld_col = sm_col.get(submercado.upper(), "pld")
    if pld_col not in df.columns:
        pld_col = "pld" if "pld" in df.columns else None
    if pld_col is None:
        return pd.Series(1.0, index=df.index)

    pld = pd.to_numeric(df[pld_col], errors="coerce")
    # Média anual por ano calendário
    annual_mean = pld.groupby(pld.index.year).transform("mean")
    # Índice = PLD / média_do_ano (1.0 = nível médio do ano)
    idx = (pld / annual_mean.replace(0, np.nan)).clip(0.05, 10.0)
    return idx.fillna(1.0)


def _structural_drift(df: pd.DataFrame) -> pd.Series:
    if "infra_marginal_rent" not in df.columns:
        return pd.Series(1.0, index=df.index)
    imr    = pd.to_numeric(df["infra_marginal_rent"], errors="coerce")
    smooth = imr.rolling(720, min_periods=24).mean().ffill().bfill()
    base   = float(smooth.iloc[:2160].median()) if len(smooth) >= 2160 else float(smooth.median())
    if not base or np.isnan(base):
        return pd.Series(1.0, index=df.index)
    return (smooth / base).clip(0.1, 5.0)


def _build_features(df: pd.DataFrame, submercado: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extrai X e y do DataFrame horário do app.py.

    Features: EAR_percent, ENA_percent, thermal_generation_ratio,
              load_ratio, curtailment_ratio, mes (1–12), hora (0–23)

    O mês e a hora capturam a sazonalidade diretamente como features,
    permitindo que o modelo aprenda P(PLD | X, T).
    """
    sm_col = {"SE": "pld_se", "NE": "pld_ne", "S": "pld_s", "N": "pld_n"}
    pld_col = sm_col.get(submercado.upper(), "pld")
    if pld_col not in df.columns:
        pld_col = "pld"

    feat = pd.DataFrame(index=df.index)

    # ── Features físicas ──────────────────────────────────────────────────────
    feat["EAR_percent"] = pd.to_numeric(
        df.get("ear_pct", pd.Series(np.nan, index=df.index)), errors="coerce"
    )

    if "ENA_norm" in df.columns:
        feat["ENA_percent"] = pd.to_numeric(df["ENA_norm"], errors="coerce") * 100
    elif "ena_arm" in df.columns:
        ena = pd.to_numeric(df["ena_arm"], errors="coerce")
        p90 = float(ena.quantile(0.90)) if not ena.dropna().empty else np.nan
        feat["ENA_percent"] = (ena / p90 * 100).clip(0, 150) if p90 and p90 > 0 else np.nan
    else:
        feat["ENA_percent"] = np.nan

    if "thermal" in df.columns and "geracao_total" in df.columns:
        feat["thermal_generation_ratio"] = (
            pd.to_numeric(df["thermal"], errors="coerce") /
            pd.to_numeric(df["geracao_total"], errors="coerce").replace(0, np.nan)
        ).clip(0, 1)
    elif "Thermal_inflex_ratio" in df.columns:
        feat["thermal_generation_ratio"] = pd.to_numeric(
            df["Thermal_inflex_ratio"], errors="coerce").clip(0, 1)
    else:
        feat["thermal_generation_ratio"] = np.nan

    if "Load_norm" in df.columns:
        feat["load_ratio"] = pd.to_numeric(df["Load_norm"], errors="coerce").clip(0, 1.5)
    else:
        feat["load_ratio"] = np.nan

    if "Curtailment_norm" in df.columns:
        feat["curtailment_ratio"] = pd.to_numeric(
            df["Curtailment_norm"], errors="coerce").clip(0, 1).fillna(0.0)
    elif "curtail_total" in df.columns and "avail_ren" in df.columns:
        feat["curtailment_ratio"] = (
            pd.to_numeric(df["curtail_total"], errors="coerce").fillna(0) /
            pd.to_numeric(df["avail_ren"], errors="coerce").replace(0, np.nan)
        ).clip(0, 1).fillna(0.0)
    else:
        feat["curtailment_ratio"] = 0.0

    # ── Features temporais — capturam sazonalidade diretamente ────────────────
    feat["mes"]              = df.index.month.astype(float)
    feat["hora"]             = df.index.hour.astype(float)
    feat["regime"]           = _assign_regime(df.index)
    feat["structural_drift"] = _structural_drift(df).values
    feat["price_index"]      = _annual_price_index(df, submercado).values

    # ── Target ────────────────────────────────────────────────────────────────
    target = (
        pd.to_numeric(df[pld_col], errors="coerce")
        if pld_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    combined = feat.copy()
    combined["_y"]  = target
    combined["_ts"] = combined.index
    combined = combined.dropna(subset=["_y", "EAR_percent"])
    combined = combined.fillna(combined.median(numeric_only=True))
    combined = combined[combined["_y"] > 0]
    combined.index = pd.to_datetime(combined["_ts"])
    combined = combined.drop(columns=["_ts"])
    return combined.drop(columns=["_y"]), combined["_y"]


# ══════════════════════════════════════════════════════════════════════════════
# MODELO FUNDAMENTAL  —  PLD = f(X, T) + ε
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def _train_model(
    df_hash: str,
    submercado: str,
    _df: pd.DataFrame,
) -> Optional[dict]:
    """
    Treina GradientBoostingRegressor no histórico completo.

    Features incluem mês e hora → modelo aprende P(PLD | X, T).
    Resíduos ε_i = PLD_real_i − PLD_hat_i guardam a incerteza residual
    após condicionar ao estado físico E ao tempo.
    """
    if not _SKLEARN_OK:
        return None

    X, y = _build_features(_df, submercado)
    if len(X) < _MIN_TRAIN_ROWS:
        return None

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=15, random_state=42,
        ))
    ])

    weights = _temporal_weights(X.index)
    weights = weights / weights.mean()
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    pipe.fit(X, y, gbr__sample_weight=weights)
    y_pred = pipe.predict(X)
    residuals = (y.values - y_pred).astype(float)
    regime_stats: dict = {}
    if "regime" in X.columns:
        for ri, rlabel in enumerate(_REGIME_LABELS):
            mask = X["regime"] == ri
            if mask.sum() >= 10:
                res_r = residuals[mask.values]
                regime_stats[rlabel] = {"n": int(mask.sum()), "pld_med": round(float(y[mask].median()),1), "pld_p90": round(float(np.percentile(y[mask],90)),1), "res_std": round(float(np.std(res_r)),1)}
    current_drift = float(_structural_drift(_df).iloc[-1]) if "infra_marginal_rent" in _df.columns else 1.0

    # Média anual atual — usada para reconverter PLD normalizado → absoluto
    sm_col = {"SE": "pld_se", "NE": "pld_ne", "S": "pld_s", "N": "pld_n"}
    pld_col_m = sm_col.get(submercado, "pld")
    if pld_col_m not in _df.columns:
        pld_col_m = "pld"
    if pld_col_m in _df.columns:
        _pld_s = pd.to_numeric(_df[pld_col_m], errors="coerce").dropna()
        current_year_mean = float(
            _pld_s[_pld_s.index.year == pd.Timestamp.now().year].mean()
        ) if not _pld_s.empty else float(_pld_s.iloc[-720:].mean()) if len(_pld_s) >= 720 else float(_pld_s.mean())
    else:
        current_year_mean = 200.0   # fallback neutro
    if np.isnan(current_year_mean) or current_year_mean <= 0:
        current_year_mean = float(_pld_s.iloc[-720:].mean()) if len(_pld_s) >= 720 else 200.0

    return {
        "model":              pipe,
        "feature_names":      list(X.columns),
        "residuals":          residuals,
        "r2":                 float(pipe.score(X, y)),
        "cv_score":           float(cv_scores.mean()),
        "cv_std":             float(cv_scores.std()),
        "n_obs":              len(X),
        "res_std":            float(np.std(residuals)),
        "res_mean":           float(np.mean(residuals)),
        "current_drift":      round(current_drift, 3),
        "current_year_mean":  round(current_year_mean, 2),
        "regime_stats":       regime_stats,
        "feature_importance": dict(zip(
            X.columns,
            pipe.named_steps["gbr"].feature_importances_.round(4).tolist()
        )),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO SAZONAL  —  sub-conjunto histórico comparável
# ══════════════════════════════════════════════════════════════════════════════

def _seasonal_residuals(
    model_dict: dict,
    df: pd.DataFrame,
    submercado: str,
    tf: TemporalFilter,
) -> np.ndarray:
    """
    Extrai resíduos apenas dos períodos históricos comparáveis ao filtro.

    Ex: para agosto → usa apenas resíduos de todos os agostos 2021–2025.
    Isso cria a distribuição:  ε | mês=agosto

    Combinado com PLD_hat(X_atual, mês=agosto), obtemos:
        P(PLD | X_atual, mês=agosto)
    """
    X_full, y_full = _build_features(df, submercado)
    if X_full.empty:
        return model_dict["residuals"]

    # Filtrar pelo mês/hora do filtro temporal
    mask = X_full["mes"].isin(tf.meses)
    if tf.horas is not None:
        mask &= X_full["hora"].isin(tf.horas)

    X_filt = X_full[mask]
    y_filt = y_full[mask]

    if len(X_filt) < 30:
        # Poucos dados no filtro — usar resíduos globais com aviso
        return model_dict["residuals"]

    y_pred_filt = model_dict["model"].predict(X_filt)
    return (y_filt.values - y_pred_filt).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO FÍSICO ATUAL
# ══════════════════════════════════════════════════════════════════════════════

def _current_scenario(df: pd.DataFrame, submercado: str,
                      tf: TemporalFilter) -> PhysicalScenario:
    """
    Extrai estado físico da última hora disponível no filtro sazonal.
    Usa a mesma pipeline de features do modelo para consistência.
    """
    X, _ = _build_features(df, submercado)
    if X.empty:
        return PhysicalScenario()

    # Última linha do filtro sazonal
    mask = X["mes"].isin(tf.meses)
    if tf.horas is not None:
        mask &= X["hora"].isin(tf.horas)
    X_filt = X[mask]

    last = X_filt.iloc[-1] if not X_filt.empty else X.iloc[-1]
    return PhysicalScenario(
        ear_pct=float(last.get("EAR_percent", np.nan)),
        ena_pct=float(last.get("ENA_percent", np.nan)),
        thermal_ratio=float(last.get("thermal_generation_ratio", np.nan)),
        load_ratio=float(last.get("load_ratio", np.nan)),
        curtailment_ratio=float(last.get("curtailment_ratio", 0.0)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIMULAÇÃO MONTE CARLO CONDICIONAL  —  P(PLD | X, T)
# ══════════════════════════════════════════════════════════════════════════════

def _monte_carlo(
    model_dict: dict,
    scenario: PhysicalScenario,
    tf: TemporalFilter,
    residuals: np.ndarray,
    n_sim: int = _N_SIMULATIONS,
) -> Tuple[float, np.ndarray]:
    """
    Simulação P(PLD | X_atual, T):

        PLD_sim = PLD_hat(X_atual, T) + ε_sazonal

    onde:
        PLD_hat  = modelo avaliado no estado físico atual + mês/hora do filtro
        ε_sazonal = resíduos históricos filtrados pelo mesmo período

    Isso garante que a distribuição simulada seja condicional tanto ao
    estado físico QUANTO à sazonalidade — a distribuição mais precisa possível.
    """
    # Ponto representativo do filtro temporal (mediana dos meses/horas)
    mes_rep  = float(np.median(tf.meses))
    hora_rep = float(np.median(tf.horas)) if tf.horas else 12.0

    cur_drift  = model_dict.get("current_drift", 1.0)
    cur_regime = float(len(_REGIME_BREAKS))
    # price_index = 1.0 significa nível médio do ano atual (sem viés)
    feat_row = {
        "EAR_percent": _safe_val(scenario.ear_pct, 50.0),
        "ENA_percent": _safe_val(scenario.ena_pct, 50.0),
        "thermal_generation_ratio": _safe_val(scenario.thermal_ratio, 0.20),
        "load_ratio": _safe_val(scenario.load_ratio, 0.70),
        "curtailment_ratio": _safe_val(scenario.curtailment_ratio, 0.00),
        "mes": mes_rep, "hora": hora_rep,
        "regime": cur_regime, "structural_drift": cur_drift,
        "price_index": 1.0,
    }
    X_cur = pd.DataFrame([feat_row])
    X_cur = X_cur[[f for f in model_dict["feature_names"] if f in X_cur.columns]]
    pld_hat = float(model_dict["model"].predict(X_cur)[0])

    # Bootstrap empírico dos resíduos sazonais
    rng = np.random.default_rng()
    eps = rng.choice(residuals, size=n_sim, replace=True)

    simulated = np.clip(pld_hat + eps, 0.0, None)
    return pld_hat, simulated


def _safe_val(v: float, default: float) -> float:
    return default if (v is None or np.isnan(v)) else float(v)


# ══════════════════════════════════════════════════════════════════════════════
# MODO HISTÓRICO PURO  —  P(PLD | T)
# ══════════════════════════════════════════════════════════════════════════════

def _historical_distribution(
    df: pd.DataFrame,
    submercado: str,
    tf: TemporalFilter,
) -> Tuple[np.ndarray, int]:
    """
    Modo histórico: distribui PLD real observado no mesmo período sazonal.

    Retorna (array de PLD históricos filtrados, n_obs).
    Não condiciona ao estado físico — útil como referência comparativa.
    """
    sm_col = {"SE": "pld_se", "NE": "pld_ne", "S": "pld_s", "N": "pld_n"}
    pld_col = sm_col.get(submercado.upper(), "pld")
    if pld_col not in df.columns:
        pld_col = "pld"
    if pld_col not in df.columns:
        return np.array([]), 0

    s = pd.to_numeric(df[pld_col], errors="coerce").dropna()
    mask = s.index.month.isin(tf.meses)
    if tf.horas is not None:
        mask &= s.index.hour.isin(tf.horas)
    filtered = s[mask].values
    return filtered, len(filtered)


# ══════════════════════════════════════════════════════════════════════════════
# INFERÊNCIA DO CENÁRIO FÍSICO IMPLÍCITO
# ══════════════════════════════════════════════════════════════════════════════

def _implied_scenario(
    model_dict: dict,
    target_price: float,
    current: PhysicalScenario,
    tf: TemporalFilter,
) -> PhysicalScenario:
    """
    Resolve:  min (f(X, T) − target_price)²   sujeito a bounds físicos.

    Responde: "que tipo de crise seria necessária para justificar esse preço?"
    """
    if not _SKLEARN_OK:
        return PhysicalScenario()

    mes_rep  = float(np.median(tf.meses))
    hora_rep = float(np.median(tf.horas)) if tf.horas else 12.0

    feat_names  = model_dict["feature_names"]
    cur_drift   = model_dict.get("current_drift", 1.0)
    cur_regime  = float(len(_REGIME_BREAKS))   # regime atual fixo

    # Mapa completo de features com seus valores iniciais e bounds
    # regime e structural_drift são fixos (não otimizamos sobre eles —
    # queremos o cenário físico implícito mantendo o regime atual)
    feat_config = {
        "EAR_percent":              (_safe_val(current.ear_pct,           50.0),  (0.0,  100.0)),
        "ENA_percent":              (_safe_val(current.ena_pct,           50.0),  (0.0,  150.0)),
        "thermal_generation_ratio": (_safe_val(current.thermal_ratio,     0.20),  (0.0,    1.0)),
        "load_ratio":               (_safe_val(current.load_ratio,        0.70),  (0.3,    1.2)),
        "curtailment_ratio":        (_safe_val(current.curtailment_ratio, 0.00),  (0.0,    1.0)),
        "mes":                      (mes_rep,   (mes_rep,  mes_rep)),   # fixo
        "hora":                     (hora_rep,  (hora_rep, hora_rep)),  # fixo
        "regime":                   (cur_regime,(cur_regime, cur_regime)),  # fixo
        "structural_drift":         (cur_drift, (cur_drift,  cur_drift)),   # fixo
        "price_index":              (1.0,       (1.0, 1.0)),                # fixo em 1.0
    }

    # Selecionar apenas features que o modelo conhece, na ordem correta
    active = [(f, feat_config[f]) for f in feat_names if f in feat_config]
    x0     = np.array([v for _, (v, _) in active])
    bounds = [b for _, (_, b) in active]

    def obj(x: np.ndarray) -> float:
        row = {f: float(v) for f, v in zip([f for f, _ in active], x)}
        return (float(model_dict["model"].predict(pd.DataFrame([row]))[0]) - target_price) ** 2

    opt = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10})
    x = np.clip(opt.x, [b[0] for b in bounds], [b[1] for b in bounds])
    x_map = {f: float(v) for f, v in zip([f for f, _ in active], x)}

    return PhysicalScenario(
        ear_pct=round(x_map.get("EAR_percent",             50.0), 1),
        ena_pct=round(x_map.get("ENA_percent",             50.0), 1),
        thermal_ratio=round(x_map.get("thermal_generation_ratio", 0.20), 4),
        load_ratio=round(x_map.get("load_ratio",           0.70), 4),
        curtailment_ratio=round(x_map.get("curtailment_ratio", 0.00), 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# API CCEE
# ══════════════════════════════════════════════════════════════════════════════

def _ccee_get(resource_id: str, filters: dict = None,
              limit: int = 1000, offset: int = 0, timeout: int = 15) -> "Optional[pd.DataFrame]":
    """
    Acessa a API CCEE com headers de browser + cookie de sessão.

    O cookie CKAN é necessário em IPs residenciais (Cloudflare).
    Em servidores cloud (Render) geralmente funciona sem cookie.

    Configurar no Render → Environment Variables:
        CCEE_COOKIE — valor completo do cookie copiado do browser
                      (renovar quando expirar, ~24h)
    """
    import json as _json
    params: dict = {"resource_id": resource_id, "limit": limit, "offset": offset}
    if filters:
        params["filters"] = _json.dumps(filters)

    ccee_cookie = os.getenv("CCEE_COOKIE", "")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
        ),
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":           "pt-BR,pt;q=0.9,en;q=0.8",
        "sec-ch-ua":                 '"Not:A-Brand";v="99", "Microsoft Edge";v="145", "Chromium";v="145"',
        "sec-ch-ua-mobile":          "?0",
        "sec-ch-ua-platform":        '"Windows"',
        "sec-fetch-dest":            "document",
        "sec-fetch-mode":            "navigate",
        "sec-fetch-site":            "none",
        "upgrade-insecure-requests": "1",
    }
    if ccee_cookie:
        headers["Cookie"] = ccee_cookie

    try:
        resp = requests.get(
            f"{_CCEE_API_BASE}/datastore_search",
            params=params, headers=headers, timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("success"):
            return None
        records = data.get("result", {}).get("records", [])
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return None


def _ccee_fetch_all(resource_id: str, filters: dict = None,
                    max_records: int = 10_000, timeout: int = 20) -> pd.DataFrame:
    PAGE, frames = 1000, []
    for offset in range(0, max_records, PAGE):
        df = _ccee_get(resource_id, filters=filters, limit=PAGE, offset=offset, timeout=timeout)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ccee_parse(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "SUBMERCADO" in df.columns:
        df["SUBMERCADO"] = df["SUBMERCADO"].str.upper().str.strip().map(
            lambda x: _CCEE_SUB_MAP.get(x, x))
    if "MES_REFERENCIA" in df.columns:
        df["MES_REFERENCIA"] = pd.to_datetime(
            df["MES_REFERENCIA"].astype(str).str[:6], format="%Y%m", errors="coerce")
    for col in _CCEE_MONTANTE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "PERIODO_COMERCIALIZACAO" in df.columns:
        df["PERIODO_COMERCIALIZACAO"] = pd.to_numeric(df["PERIODO_COMERCIALIZACAO"], errors="coerce")
    return df


def fetch_ccee_market_data(submercado: str = "SE", timeout: int = 20) -> "Dict[str, Any]":
    """Dados de comercialização em memória — sem persistência."""
    from datetime import datetime as _dt
    result = {
        "montante_total_mwh": None, "montante_medio_mwh": None,
        "exposicao_spot_proxy": None, "montante_por_tipo": {},
        "historico_meses": [], "submercado": submercado,
        "mes_referencia": None, "fonte": "ccee_dadosabertos",
        "erro": None, "disponivel": False,
    }
    sub_api = _CCEE_SUB_API.get(submercado.upper(), submercado.upper())
    df_all  = pd.DataFrame()
    for ano in [_dt.now().year, _dt.now().year - 1]:
        rid = _CCEE_RESOURCES.get(ano)
        if not rid:
            continue
        df_all = _ccee_fetch_all(rid, filters={"SUBMERCADO": sub_api},
                                 max_records=10_000, timeout=timeout)
        if not df_all.empty:
            break
    if df_all.empty:
        result["erro"] = "Sem dados na API da CCEE para o submercado selecionado"
        return result
    df_all   = _ccee_parse(df_all)
    num_cols = [c for c in _CCEE_MONTANTE_COLS if c in df_all.columns]
    if not num_cols or "MES_REFERENCIA" not in df_all.columns:
        result["erro"] = "Colunas esperadas não encontradas"
        return result
    mensal = (df_all.groupby("MES_REFERENCIA")[num_cols].sum()
                .reset_index().sort_values("MES_REFERENCIA", ascending=False))
    mensal["MONTANTE_TOTAL"] = mensal[num_cols].sum(axis=1)
    if "MONTANTE_MODULADO_CONTRATO_CCEARD" in mensal.columns:
        mensal["EXPOSICAO_SPOT_PROXY"] = (
            mensal["MONTANTE_MODULADO_CONTRATO_CCEARD"] /
            mensal["MONTANTE_TOTAL"].replace(0, np.nan)).clip(0, 1)
    u     = mensal.iloc[0]
    total = float(u.get("MONTANTE_TOTAL", 0) or 0)
    n_per = int(df_all[df_all["MES_REFERENCIA"] == u["MES_REFERENCIA"]]
                       ["PERIODO_COMERCIALIZACAO"].nunique()) or 1
    result.update({
        "mes_referencia":       str(u["MES_REFERENCIA"])[:7],
        "montante_total_mwh":   round(total, 0),
        "montante_medio_mwh":   round(total / n_per, 2),
        "exposicao_spot_proxy": round(float(u.get("EXPOSICAO_SPOT_PROXY", 0) or 0), 4),
        "montante_por_tipo": {
            col.replace("MONTANTE_MODULADO_CONTRATO_", ""): round(float(u.get(col, 0) or 0), 0)
            for col in num_cols
        },
        "historico_meses": mensal[["MES_REFERENCIA", "MONTANTE_TOTAL"]
                                  + (["EXPOSICAO_SPOT_PROXY"]
                                     if "EXPOSICAO_SPOT_PROXY" in mensal.columns else [])
                                  ].head(6).to_dict("records"),
        "disponivel": True,
    })
    return result

# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_justification_engine(
    df: pd.DataFrame,
    preco_energia: float,
    submercado: str = "SE",
    tf: Optional[TemporalFilter] = None,
    modo: str = "atual",          # "atual" | "historico"
    fetch_ccee: bool = True,
    n_sim: int = _N_SIMULATIONS,
) -> Optional[JustificationResult]:
    """
    P(PLD ≥ preço | X, T)  ou  P(PLD ≥ preço | T)

    Parâmetros:
        df            : DataFrame horário do _build_hourly_df_cached
        preco_energia : preço a analisar (R$/MWh)
        submercado    : SE | NE | S | N
        tf            : filtro temporal sazonal (None = ano completo)
        modo          : "atual" condicionado ao estado físico presente;
                        "historico" usa apenas distribuição observada
        fetch_ccee    : consultar API CCEE
        n_sim         : simulações Monte Carlo
    """
    if not _SKLEARN_OK:
        st.error("scikit-learn não instalado. Adicione ao requirements.txt:\n"
                 "scikit-learn\nscipy")
        return None

    if df.empty or len(df) < _MIN_TRAIN_ROWS:
        return None

    if tf is None:
        tf = TemporalFilter.anual()

    # ── Modo histórico puro ──────────────────────────────────────────────────
    if modo == "historico":
        hist, n_obs = _historical_distribution(df, submercado, tf)
        if len(hist) < 30:
            st.warning("Dados históricos insuficientes para o filtro sazonal selecionado.")
            return None
        pct = {
            "p25": float(np.percentile(hist, 25)),
            "p50": float(np.percentile(hist, 50)),
            "p75": float(np.percentile(hist, 75)),
            "p90": float(np.percentile(hist, 90)),
            "p95": float(np.percentile(hist, 95)),
        }
        prob    = float(np.mean(hist >= preco_energia))
        classif = _classify(prob)
        pld_med = pct["p50"]
        gap     = preco_energia - pld_med
        rng     = np.random.default_rng()
        simulated = rng.choice(hist, size=min(n_sim, len(hist) * 10), replace=True)
        return JustificationResult(
            preco_analisado=preco_energia,
            submercado=submercado,
            modo="historico",
            filtro_temporal=tf,
            pld_fundamental=round(pld_med, 2),
            fundamental_gap=round(gap, 2),
            probabilidade_justificacao=round(prob * 100, 1),
            classificacao=classif,
            cenario_fisico_implicito=PhysicalScenario(),
            cenario_fisico_atual=PhysicalScenario(),
            pld_simulado_dist=simulated,
            n_obs_filtro=n_obs,
            percentis=pct,
            exposicao_spot=fetch_ccee_market_data(submercado) if fetch_ccee else None,
        )

    # ── Modo atual — P(PLD | X_atual, T) ────────────────────────────────────
    df_hash = str(len(df)) + str(df.index[-1])
    model_dict = _train_model(df_hash, submercado, df)
    if model_dict is None:
        st.warning("Dados insuficientes para treinar o modelo fundamental.")
        return None

    current    = _current_scenario(df, submercado, tf)
    residuals  = _seasonal_residuals(model_dict, df, submercado, tf)
    pld_hat, simulated = _monte_carlo(model_dict, current, tf, residuals, n_sim)

    n_obs_filt = len(residuals)
    prob       = float(np.mean(simulated >= preco_energia))
    classif    = _classify(prob)
    implied    = _implied_scenario(model_dict, preco_energia, current, tf)

    pct = {
        "p25": float(np.percentile(simulated, 25)),
        "p50": float(np.percentile(simulated, 50)),
        "p75": float(np.percentile(simulated, 75)),
        "p90": float(np.percentile(simulated, 90)),
        "p95": float(np.percentile(simulated, 95)),
    }

    aviso = ""
    if n_obs_filt < 100:
        aviso = f"⚠️ Poucos dados no filtro sazonal ({n_obs_filt} obs) — " \
                "resíduos globais usados como fallback."

    try:
        st.session_state["_engine_drift"]        = model_dict.get("current_drift", 1.0)
        st.session_state["_engine_regime_stats"] = model_dict.get("regime_stats", {})
        st.session_state["_engine_year_mean"]    = model_dict.get("current_year_mean", None)
    except Exception:
        pass

    return JustificationResult(
        preco_analisado=preco_energia,
        submercado=submercado,
        modo="atual",
        filtro_temporal=tf,
        pld_fundamental=round(pld_hat, 2),
        fundamental_gap=round(preco_energia - pld_hat, 2),
        probabilidade_justificacao=round(prob * 100, 1),
        classificacao=classif,
        cenario_fisico_implicito=implied,
        cenario_fisico_atual=current,
        pld_simulado_dist=simulated,
        n_obs_filtro=n_obs_filt,
        percentis=pct,
        model_r2=round(model_dict["r2"], 3),
        model_cv_score=round(model_dict["cv_score"], 3),
        res_std=round(model_dict["res_std"], 2),
        feature_importance=model_dict["feature_importance"],
        exposicao_spot=fetch_ccee_market_data(submercado) if fetch_ccee else None,
        aviso=aviso,
    )


def _classify(prob: float) -> str:
    if prob >= 0.70: return "razoável"
    if prob >= 0.40: return "neutro"
    if prob >= 0.20: return "estressado"
    return "extremo"


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

def _plot_distribution(result: JustificationResult) -> go.Figure:
    dist  = result.pld_simulado_dist
    preco = result.preco_analisado
    fund  = result.pld_fundamental

    bar_color = {
        "razoável": _COLORS["green"], "neutro": _COLORS["blue"],
        "estressado": _COLORS["gold"], "extremo": _COLORS["red"],
    }.get(result.classificacao, _COLORS["blue"])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dist, nbinsx=80, name="P(PLD | X, T)",
        marker_color=bar_color, opacity=0.60,
        hovertemplate="R$ %{x:.0f}/MWh<br>Freq: %{y}<extra></extra>",
    ))
    above = dist[dist >= preco]
    if len(above) > 0:
        fig.add_trace(go.Histogram(
            x=above, nbinsx=40, opacity=0.55,
            marker_color=_COLORS["gold"],
            name=f"P(PLD ≥ preço) = {result.probabilidade_justificacao:.1f}%",
        ))

    # Linha do preço analisado
    fig.add_vline(x=preco, line_color=_COLORS["gold"], line_width=2.5,
                  annotation_text=f"Preço: R${preco:.0f}",
                  annotation_font_color=_COLORS["gold"],
                  annotation_position="top right")

    # Linha do fundamental (apenas modo atual)
    if result.modo == "atual":
        fig.add_vline(x=fund, line_color=_COLORS["blue"], line_width=2,
                      line_dash="dash",
                      annotation_text=f"Fundamental: R${fund:.0f}",
                      annotation_font_color=_COLORS["blue"],
                      annotation_position="top left")

    # Percentis como linhas discretas
    for pname, pval, pcol in [
        ("p50", result.percentis["p50"], "#9ca3af"),
        ("p90", result.percentis["p90"], "#6b7280"),
    ]:
        fig.add_vline(x=pval, line_color=pcol, line_width=1,
                      line_dash="dot",
                      annotation_text=f"{pname.upper()}: R${pval:.0f}",
                      annotation_font_color=pcol, annotation_font_size=10,
                      annotation_position="bottom right")

    p5, p95 = np.percentile(dist, 2), np.percentile(dist, 98)
    modo_label = "Estado atual + sazonalidade" if result.modo == "atual" else "Histórico sazonal"
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], barmode="overlay", height=400,
        margin=dict(l=40, r=20, t=55, b=40),
        title=dict(
            text=(f"P(PLD | {result.filtro_temporal.describe()}, {result.submercado})  "
                  f"— <span style='font-size:11px;color:{bar_color}'>"
                  f"{result.classificacao.upper()} | {modo_label}</span>"),
            font=dict(size=13, color="#e5e7eb"),
        ),
        xaxis=dict(title="PLD (R$/MWh)", range=[max(0, p5-30), p95+80],
                   gridcolor="#1f2937"),
        yaxis=dict(title="Frequência", gridcolor="#1f2937"),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
    )
    return fig


def _plot_radar(current: PhysicalScenario, implied: PhysicalScenario) -> go.Figure:
    cats = ["EAR %", "ENA %", "Térmica", "Carga", "Curtailment"]

    def _n(sc):
        return [
            _safe_val(sc.ear_pct, 50) / 100,
            _safe_val(sc.ena_pct, 50) / 100,
            _safe_val(sc.thermal_ratio, 0.2),
            _safe_val(sc.load_ratio, 0.7),
            sc.curtailment_ratio,
        ]

    fig = go.Figure()
    for vals, name, color in [
        (_n(current), "Cenário atual",    _COLORS["blue"]),
        (_n(implied), "Cenário implícito", _COLORS["gold"]),
    ]:
        r = vals + [vals[0]]
        t = cats + [cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=t, fill="toself", name=name,
            line_color=color, fillcolor=_hex_rgba(color, 0.20),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#374151"),
            angularaxis=dict(gridcolor="#374151"),
            bgcolor=_COLORS["panel"],
        ),
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        height=320, margin=dict(l=30, r=30, t=45, b=20),
        title=dict(text="Cenário físico: atual vs implícito", font=dict(size=13)),
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def _plot_feature_importance(fi: Dict[str, float]) -> go.Figure:
    labels_pt = {
        "EAR_percent": "EAR (%)",
        "ENA_percent": "ENA (% MLT)",
        "thermal_generation_ratio": "Despacho Térmico",
        "load_ratio": "Pressão de Carga",
        "curtailment_ratio": "Curtailment",
        "mes": "Mês (sazonalidade)",
        "hora": "Hora (perfil diário)",
    }
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names = [labels_pt.get(k, k) for k, _ in items]
    vals  = [v for _, v in items]
    colors = [_COLORS["gold"] if "Mês" in n or "Hora" in n else _COLORS["blue"]
              for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors, opacity=0.85,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], height=280,
        margin=dict(l=10, r=10, t=40, b=20),
        title=dict(text="Importância das features no modelo", font=dict(size=12)),
        xaxis=dict(title="Importância", gridcolor="#1f2937"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 12 — VISUALIZAÇÕES DE CUSTO FÍSICO E ENCARGOS
# ══════════════════════════════════════════════════════════════════════════════

_IMR_OUTLIER_FLOOR = -250_000_000  # R$/h — filtro de outliers do heatmap

def _plot_imr_heatmap(df: pd.DataFrame,
                      imr_col: str = "infra_marginal_physical") -> go.Figure:
    """Heatmap hora × dia do IMR selecionado.
    Remove outliers abaixo de -250M R$/h antes de plotar.
    """
    if imr_col not in df.columns or df.empty:
        return go.Figure()
    s = pd.to_numeric(df[imr_col], errors="coerce").dropna()
    if s.empty:
        return go.Figure()
    # Filtro de outliers
    s = s[s >= _IMR_OUTLIER_FLOOR]
    if s.empty:
        return go.Figure()
    pivot = s.to_frame("val")
    pivot["date"] = pivot.index.date
    pivot["hour"] = pivot.index.hour
    mat = pivot.pivot_table(index="hour", columns="date", values="val", aggfunc="mean")
    label_map = {
        "infra_marginal_physical": "IMR Físico (R$/h)",
        "infra_marginal_market":   "IMR Mercado (R$/h)",
        "infra_marginal_system":   "IMR Sistêmico (R$/h)",
    }
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(d) for d in mat.columns],
        y=[f"{h:02d}h" for h in mat.index],
        colorscale="RdYlGn",
        colorbar=dict(title="R$/h", thickness=12),
        hovertemplate="Data: %{x}<br>Hora: %{y}<br>Valor: R$ %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], height=380,
        margin=dict(l=40, r=20, t=45, b=30),
        title=dict(text=f"Heatmap — {label_map.get(imr_col, imr_col)}",
                   font=dict(size=13, color="#e5e7eb")),
        xaxis=dict(nticks=20, gridcolor="#1f2937"),
        yaxis=dict(autorange="reversed", gridcolor="#1f2937"),
    )
    return fig


def _plot_spdi_series(df: pd.DataFrame) -> go.Figure:
    """Série temporal do SPDI com zonas de alerta."""
    if "spdi" not in df.columns or df.empty:
        return go.Figure()
    s = pd.to_numeric(df["spdi"], errors="coerce").dropna()
    daily = s.resample("D").median()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values, mode="lines", name="SPDI",
        line=dict(color=_COLORS["gold"], width=1.8),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>SPDI: %{y:.2f}×<extra></extra>",
    ))
    for level, color, label in [
        (1.0, "#34d399", "Alinhado (1.0)"),
        (1.3, "#c8a44d", "Prêmio estrutural (1.3)"),
        (1.6, "#f87171", "Distorção forte (1.6)"),
    ]:
        fig.add_hline(y=level, line_color=color, line_dash="dot", line_width=1,
                      annotation_text=label, annotation_font_color=color,
                      annotation_font_size=10)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], height=320,
        margin=dict(l=40, r=20, t=45, b=30),
        title=dict(text="Structural Price Distortion Index (SPDI)",
                   font=dict(size=13, color="#e5e7eb")),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(title="SPDI (×)", gridcolor="#1f2937"),
    )
    return fig


def _plot_ess_vs_market(df: pd.DataFrame) -> go.Figure:
    """ESS cost vs Market cost — série diária."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for col, name, color in [
        ("market_cost_Rh",              "Custo de Mercado (R$/h)",  _COLORS["blue"]),
        ("physical_generation_cost_Rh", "Custo Físico (R$/h)",      _COLORS["green"]),
        ("ess_cost_Rh",                 "Encargo ESS (R$/h)",        _COLORS["gold"]),
        ("hidden_system_cost",          "Custo Oculto (R$/h)",       _COLORS["purple"]),
    ]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna().resample("D").mean()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"{name}: R$ %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], height=360,
        margin=dict(l=40, r=20, t=45, b=30),
        title=dict(text="Decomposição de Custos do Sistema (média diária)",
                   font=dict(size=13, color="#e5e7eb")),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(title="R$/h", gridcolor="#1f2937"),
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
    )
    return fig


def _plot_eii_series(df: pd.DataFrame) -> go.Figure:
    """Encargo Intensity Index — série diária."""
    if "encargo_intensity_index" not in df.columns or df.empty:
        return go.Figure()
    s = pd.to_numeric(df["encargo_intensity_index"], errors="coerce").dropna()
    daily = s.resample("D").median()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values * 100, mode="lines", name="EII (%)",
        line=dict(color=_COLORS["purple"], width=1.8),
        fill="tozeroy", fillcolor=_hex_rgba(_COLORS["purple"], 0.13),
        hovertemplate="Data: %{x|%d/%m/%Y}<br>EII: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=5,  line_color=_COLORS["gold"], line_dash="dot",
                  annotation_text="Estresse (5%)",           annotation_font_color=_COLORS["gold"])
    fig.add_hline(y=10, line_color=_COLORS["red"],  line_dash="dot",
                  annotation_text="Intervenção elevada (10%)", annotation_font_color=_COLORS["red"])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_COLORS["bg"],
        plot_bgcolor=_COLORS["panel"], height=300,
        margin=dict(l=40, r=20, t=45, b=30),
        title=dict(text="Encargo Intensity Index — ESS / Custo de Mercado",
                   font=dict(size=13, color="#e5e7eb")),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(title="EII (%)", gridcolor="#1f2937"),
    )
    return fig


def render_charges_tab(df: pd.DataFrame) -> None:
    """
    Tab premium de Encargos & Custo Físico.
    Chamar com: with tabs[N]: render_charges_tab(df)
    """
    st.markdown("""
    <div style='background:#111827;border-left:3px solid #a78bfa;border-radius:8px;
    padding:14px 18px;margin-bottom:16px'>
    <strong style='color:#a78bfa'>⚡ Encargos ESS & Custo Físico do Sistema</strong><br>
    <span style='font-size:.82rem;color:#9ca3af'>
    Compara o preço de mercado (PLD × carga) com o custo físico real de operação
    (geração térmica × CVU + hidro × CMO + encargos ESS).
    Dados ESS obtidos via API CCEE — processados em memória.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Verificar colunas novas
    new_cols = ["ess_cost_Rh", "physical_generation_cost_Rh",
                "market_cost_Rh", "spdi", "encargo_intensity_index"]
    missing = [col for col in new_cols if col not in df.columns]
    if missing:
        st.warning(
            f"Colunas ainda não calculadas: `{'`, `'.join(missing)}`. "
            "Verifique se `ccee_charges.py` está instalado e a API CCEE está acessível."
        )

    # Seletor de IMR
    imr_options = {k: v for k, v in {
        "infra_marginal_physical": "IMR Físico ← recomendado",
        "infra_marginal_market":   "IMR Mercado",
        "infra_marginal_system":   "IMR Sistêmico",
    }.items() if k in df.columns}

    imr_col = "infra_marginal_physical"
    if imr_options:
        imr_col = st.selectbox(
            "Métrica de Renda Infra-Marginal",
            options=list(imr_options.keys()),
            format_func=lambda x: imr_options.get(x, x),
        )

    # KPIs
    def _last30(col):
        if col not in df.columns:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        cut = s.index.max() - pd.Timedelta(days=30) if not s.empty else s.index.min()
        return s[s.index >= cut]

    k1, k2, k3, k4, k5 = st.columns(5)
    imr_s  = _last30(imr_col)
    spdi_s = _last30("spdi")
    eii_s  = _last30("encargo_intensity_index")
    hsc_s  = _last30("Hidden_system_cost_R$/MWh")
    ess_s  = _last30("ess_cost_Rh")

    def _fmt(s, pct=False, suffix=""):
        if s.empty or s.isna().all():
            return "N/D"
        v = float(s.median())
        return f"{v:.1%}" if pct else f"R$ {v:,.0f}{suffix}"

    k1.metric("IMR Corrigido mediana (30d)",
              _fmt(_last30("infra_marginal_rent_corrigido")),
              help="IMR = Receita − (T_total + Encargos)")
    k2.metric("Structural Gap mediana (30d)",
              _fmt(_last30("Structural_gap_R$/MWh"), suffix="/MWh"),
              help="PLD − Custo real unitário")
    k3.metric("SPDI mediana (30d)",
              "N/D" if spdi_s.empty else f"{float(spdi_s.median()):.2f}×")
    k4.metric("Hidden Cost mediana (30d)",
              _fmt(hsc_s, suffix="/MWh"))
    k5.metric("Encargo Intensity (30d)", _fmt(eii_s, pct=True))

    ka, kb, kc = st.columns(3)
    ka.metric("Encargos Total mediana", _fmt(_last30("Encargos_total_R$/h"), suffix="/h"))
    kb.metric("Custo Real mediana",     _fmt(_last30("Custo_real_R$/h"),     suffix="/h"))
    kc.metric("Custo Real Unitário",    _fmt(_last30("Custo_real_R$/MWh"),   suffix="/MWh"))

    st.markdown("---")

    gtabs = st.tabs([
        "🌡 Heatmap IMR", "📈 SPDI", "💰 Decomposição Custos",
        "📊 Encargo Intensity", "🔍 Fechamento Econômico", "📋 Tabela resumo",
    ])

    with gtabs[0]:
        st.plotly_chart(_plot_imr_heatmap(df, imr_col), use_container_width=True, key="charges_heatmap")
        if imr_col in df.columns:
            _n_out = int((pd.to_numeric(df[imr_col], errors="coerce").dropna() < _IMR_OUTLIER_FLOOR).sum())
            if _n_out > 0:
                st.caption(f"⚠️ {_n_out} registros abaixo de R$ -250M/h removidos (outliers).")
            st.markdown("- IMR ≈ 0 → preço economicamente consistente com o sistema físico  \n- IMR > 0 (verde) → presença de renda infra-marginal / prêmio estrutural  \n- IMR < 0 (vermelho) → custo sistêmico elevado ou distorções operativas não refletidas no preço")

    with gtabs[1]:
        st.plotly_chart(_plot_spdi_series(df), use_container_width=True, key="charges_spdi")
        st.markdown("**≈ 1,0** alinhado · **> 1,3** prêmio estrutural · **> 1,6** distorção forte")

    with gtabs[2]:
        st.plotly_chart(_plot_ess_vs_market(df), use_container_width=True, key="charges_ess_market")
        st.caption("Custo Oculto = ESS + GFOM (despacho fora do mérito).")

    with gtabs[3]:
        st.plotly_chart(_plot_eii_series(df), use_container_width=True, key="charges_eii")
        st.markdown("**< 5%** normal · **5–10%** estresse · **> 10%** intervenção elevada")

    with gtabs[4]:
        st.markdown("#### Fechamento Econômico Completo")
        st.caption("Separação entre (a) custo físico modelado, (b) encargos ocultos e (c) receita de mercado.")

        # Structural Gap série diária
        _sg_col = "Structural_gap_R$/MWh"
        _cr_col = "Custo_real_R$/MWh"
        if _sg_col in df.columns or _cr_col in df.columns:
            _fig_gap = go.Figure()
            if "pld" in df.columns:
                _s = pd.to_numeric(df["pld"], errors="coerce").dropna().resample("D").mean()
                _fig_gap.add_trace(go.Scatter(x=_s.index, y=_s.values,
                    name="PLD (R$/MWh)", line=dict(color="#f59e0b", width=2)))
            if _cr_col in df.columns:
                _s = pd.to_numeric(df[_cr_col], errors="coerce").dropna().resample("D").mean()
                _fig_gap.add_trace(go.Scatter(x=_s.index, y=_s.values,
                    name="Custo Real (R$/MWh)", line=dict(color="#f87171", width=2)))
            if _sg_col in df.columns:
                _daily_sg = pd.to_numeric(df[_sg_col], errors="coerce").dropna().resample("D").mean()
                _fig_gap.add_trace(go.Bar(x=_daily_sg.index, y=_daily_sg.values,
                    name="Structural Gap", opacity=0.6,
                    marker_color=["#34d399" if v >= 0 else "#f87171"
                                  for v in _daily_sg.values]))
            _fig_gap.add_hline(y=0, line_color="#6b7280", line_dash="dot")
            _fig_gap.update_layout(
                template="plotly_dark", paper_bgcolor=_COLORS["bg"],
                plot_bgcolor=_COLORS["panel"], height=360,
                margin=dict(l=40, r=20, t=45, b=30),
                title=dict(text="PLD vs Custo Real + Structural Gap (R$/MWh)",
                           font=dict(size=13, color="#e5e7eb")),
                xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(_fig_gap, use_container_width=True, key="charges_structural_gap")

        # Cascata de custo
        st.markdown("##### Cascata de custo (médias do período)")
        _wf = []
        for lbl, col, sign in [
            ("Receita (sin_cost)",      "market_cost_Rh",               1),
            ("− Custo Físico (T_total)","t_total",                      -1),
            ("− Encargos",              "Encargos_total_R$/h",          -1),
            ("= IMR Corrigido",         "infra_marginal_rent_corrigido", 1),
        ]:
            if col in df.columns:
                v = float(pd.to_numeric(df[col], errors="coerce").mean())
                _wf.append({"Componente": lbl, "R$/h (média)": round(v * sign, 0)})
        if _wf:
            st.dataframe(pd.DataFrame(_wf).set_index("Componente"), use_container_width=True)

        st.info(
            "**IMR Corrigido** = Receita − (T_total + Encargos)  \n"
            "Quando IMR Corrigido < IMR Original → encargos absorvendo prêmio aparente."
        )

    with gtabs[5]:
        cols_show = [
            ("market_cost_Rh",                 "Custo Mercado (R$/h)"),
            ("physical_generation_cost_Rh",    "Custo Físico (R$/h)"),
            ("Encargos_total_R$/h",            "Encargos Total (R$/h)"),
            ("ESS_total_R$",                   "ESS Total (R$/h)"),
            ("constrained_off_R$",             "Constrained-Off (R$/h)"),
            ("constrained_on_R$",              "Constrained-On (R$/h)"),
            ("seguranca_energetica_R$",        "Segurança Energética (R$/h)"),
            ("reserva_operativa_R$",           "Reserva Operativa (R$/h)"),
            ("Custo_real_R$/h",                "Custo Real do Sistema (R$/h)"),
            ("Custo_real_R$/MWh",              "Custo Real Unitário (R$/MWh)"),
            ("Hidden_system_cost_R$/MWh",      "Hidden System Cost (R$/MWh)"),
            ("infra_marginal_rent_corrigido",  "IMR Corrigido (R$/h)"),
            ("infra_marginal_physical",        "IMR Físico (R$/h)"),
            ("infra_marginal_market",          "IMR Mercado (R$/h)"),
            ("Structural_gap_R$/MWh",          "Structural Gap (R$/MWh)"),
            ("spdi",                           "SPDI (×)"),
            ("encargo_intensity_index",        "EII (%)"),
            ("structural_drift",               "Structural Drift"),
        ]
        rows = []
        for col, lbl in cols_show:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append({
                "Indicador":    lbl,
                "Mínimo":       f"{s.min():,.2f}",
                "Mediana":      f"{s.median():,.2f}",
                "Média":        f"{s.mean():,.2f}",
                "Máximo":       f"{s.max():,.2f}",
                "Últimas 720h": f"{s.iloc[-720:].mean():,.2f}" if len(s) >= 720 else "N/D",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Indicador"), use_container_width=True)
        else:
            st.info("Dados de encargos não disponíveis — verifique a conexão com a API CCEE.")


def render_premium_tab(df: pd.DataFrame) -> None:
    """Renderiza a tab premium. Chamar dentro de `with tabs[N]:` no app.py."""

    st.markdown("""
    <style>
    .epje-header{background:linear-gradient(135deg,#0b1222,#111827);
        border:1px solid #c8a44d44;border-radius:12px;padding:20px 24px;margin-bottom:20px}
    .epje-title{font-size:1.35rem;font-weight:700;color:#c8a44d;
        letter-spacing:.06em;text-transform:uppercase;margin:0}
    .epje-sub{font-size:.8rem;color:#9ca3af;margin:4px 0 0 0}
    .epje-badge{display:inline-block;background:#c8a44d22;border:1px solid #c8a44d66;
        color:#c8a44d;font-size:.68rem;font-weight:700;letter-spacing:.12em;
        padding:2px 8px;border-radius:4px;text-transform:uppercase;margin-bottom:8px}
    .epje-card{background:#111827;border-left:3px solid #c8a44d;
        border-radius:8px;padding:14px 18px;margin:8px 0}
    .clsf-razoavel{color:#34d399;font-weight:700;font-size:1.05rem}
    .clsf-neutro{color:#60a5fa;font-weight:700;font-size:1.05rem}
    .clsf-estressado{color:#c8a44d;font-weight:700;font-size:1.05rem}
    .clsf-extremo{color:#f87171;font-weight:700;font-size:1.05rem}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='epje-header'>
      <div class='epje-badge'>✦ Premium</div>
      <p class='epje-title'>Energy Price Justification Engine</p>
      <p class='epje-sub'>
        Avalia se um preço de energia é coerente com o cenário físico do SIN.<br>
        Modelo: <code>P(PLD ≥ preço | estado físico, sazonalidade)</code>
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not _SKLEARN_OK:
        st.error("⚠️ Dependências ausentes.\n```\nscikit-learn\nscipy\n```")
        return
    if df.empty:
        st.warning("Sem dados. Selecione um período com dados disponíveis.")
        return

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.markdown("#### ① Parâmetros do preço")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        preco = st.number_input("Preço a analisar (R$/MWh)",
                                min_value=0.0, max_value=5000.0,
                                value=300.0, step=10.0, format="%.2f")
    with c2:
        submercado = st.selectbox("Submercado", _SUBMERCADOS)
    with c3:
        modo = st.radio("Modo de análise",
                        ["Cenário atual", "Histórico sazonal"],
                        help=("**Cenário atual**: P(PLD | estado físico presente + sazonalidade)\n\n"
                              "**Histórico sazonal**: P(PLD | sazonalidade) — distribuição observada"))
        modo_key = "atual" if "atual" in modo else "historico"

    st.markdown("#### ② Período de entrega")
    tipo_periodo = st.radio("Granularidade", ["Mês", "Trimestre", "Ano completo"],
                            horizontal=True)

    tf: Optional[TemporalFilter] = None
    if tipo_periodo == "Mês":
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mes_sel = st.selectbox("Mês", list(_MESES_PT.values()),
                                   index=pd.Timestamp.now().month - 1)
            mes_num = [k for k, v in _MESES_PT.items() if v == mes_sel][0]
        with col_m2:
            filtrar_hora = st.checkbox("Filtrar por perfil horário")
            if filtrar_hora:
                hora_ini, hora_fim = st.slider("Horas do dia", 0, 23, (0, 23))
                horas = list(range(hora_ini, hora_fim + 1))
            else:
                horas = None
        tf = TemporalFilter.mensal(mes_num)
        tf.horas = horas

    elif tipo_periodo == "Trimestre":
        trim_sel = st.selectbox("Trimestre", list(_TRIMESTRES.keys()))
        tf = TemporalFilter.trimestral(trim_sel, _TRIMESTRES[trim_sel])

    else:
        tf = TemporalFilter.anual()

    st.caption(f"📅 Filtro sazonal: **{tf.describe()}** — "
               f"usa todos os anos históricos disponíveis para este período")

    st.markdown("#### ③ Dados de mercado CCEE")
    usar_ccee = st.toggle("Consultar API da CCEE (montantes modulados, em memória)",
                          value=False)

    with st.expander("⚙️ Configurações avançadas"):
        n_sim = st.select_slider("Simulações Monte Carlo",
                                 [1000, 2000, 5000, 8000, 10000],
                                 value=_N_SIMULATIONS)

    st.markdown("---")

    if st.button("▶ Executar análise", type="primary"):
        with st.spinner(f"Calculando P(PLD ≥ {preco:.0f} | {tf.describe()}, {submercado})…"):
            result = run_justification_engine(
                df=df, preco_energia=preco, submercado=submercado,
                tf=tf, modo=modo_key, fetch_ccee=usar_ccee, n_sim=n_sim,
            )
        if result is None:
            st.error("Dados insuficientes para o filtro selecionado.")
            return
        _render_results(result)



def _render_drift_regime(result: "JustificationResult") -> None:
    """
    Tab que mostra explicitamente como o drift estrutural e os regimes
    de mercado estão sendo considerados no modelo.
    """
    if result.modo != "atual":
        st.info("Análise de regime disponível apenas no modo **Cenário atual**.")
        return

    drift_val     = st.session_state.get("_engine_drift", 1.0)
    regime_stats  = st.session_state.get("_engine_regime_stats", {})
    year_mean     = st.session_state.get("_engine_year_mean", None)
    fi            = result.feature_importance or {}

    # ── Cabeçalho explicativo ─────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#111827;border-left:3px solid #c8a44d;border-radius:8px;
    padding:14px 18px;margin-bottom:16px'>
    <strong style='color:#c8a44d'>Como o modelo trata o drift estrutural de preços</strong><br>
    <span style='font-size:.85rem;color:#9ca3af'>
    O PLD brasileiro não é estacionário — vem subindo estruturalmente desde 2021,
    acelerando após março/2025. O modelo incorpora isso em <strong>quatro camadas</strong>:
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 Cards das abordagens ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Decay temporal
        weight_today  = 1.0
        weight_1y     = round(2 ** (-1 / 1.5), 2)
        weight_3y     = round(2 ** (-3 / 1.5), 2)
        st.markdown(f"""
        <div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:14px;margin-bottom:10px'>
        <div style='color:#60a5fa;font-weight:700;margin-bottom:6px'>⚖️ 1. Decay Temporal</div>
        <div style='font-size:.8rem;color:#9ca3af'>
        Observações recentes têm peso maior no treino.<br>
        Meia-vida = <strong style='color:#e5e7eb'>1,5 anos</strong>
        </div>
        <div style='margin-top:8px;font-size:.78rem'>
        <span style='color:#34d399'>● Hoje: peso 1,00</span><br>
        <span style='color:#60a5fa'>● 1,5 anos atrás: peso {weight_1y:.2f}</span><br>
        <span style='color:#6b7280'>● 3 anos atrás: peso {weight_3y:.2f}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Normalização temporal
        norm_status = "✅ Ativa" if "price_index" in fi else "⚠️ Sem dados de PLD"
        norm_imp    = fi.get("price_index", 0)
        st.markdown(f"""
        <div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:14px;margin-bottom:10px'>
        <div style='color:#60a5fa;font-weight:700;margin-bottom:6px'>📐 2. Normalização Temporal</div>
        <div style='font-size:.8rem;color:#9ca3af'>
        PLD normalizado = PLD / média_anual<br>
        Remove drift de nível entre anos.<br>
        Status: <strong style='color:#34d399'>{norm_status}</strong>
        </div>
        {"" if not year_mean else f"<div style='margin-top:6px;font-size:.78rem;color:#e5e7eb'>Média ano atual: <strong>R$ {year_mean:.0f}/MWh</strong></div>"}
        <div style='font-size:.75rem;color:#6b7280;margin-top:4px'>Importância no modelo: {norm_imp:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Structural drift
        drift_delta = (drift_val - 1) * 100
        drift_color = "#f87171" if drift_val > 1.3 else "#c8a44d" if drift_val > 1.1 else "#34d399"
        imr_imp     = fi.get("structural_drift", 0)
        st.markdown(f"""
        <div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:14px;margin-bottom:10px'>
        <div style='color:#c8a44d;font-weight:700;margin-bottom:6px'>📊 3. Infra-Marginal Rent como Feature</div>
        <div style='font-size:.8rem;color:#9ca3af'>
        IMR suavizado (30d) / base 2021<br>
        Captura reprecificação estrutural do mercado.
        </div>
        <div style='margin-top:8px'>
        <span style='font-size:1.3rem;font-weight:700;color:{drift_color}'>{drift_val:.2f}×</span>
        <span style='font-size:.8rem;color:{drift_color}'> ({drift_delta:+.0f}% vs base 2021)</span>
        </div>
        <div style='font-size:.75rem;color:#6b7280;margin-top:4px'>Importância no modelo: {imr_imp:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Regimes
        regime_imp = fi.get("regime", 0)
        n_regimes  = len(regime_stats)
        st.markdown(f"""
        <div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:14px;margin-bottom:10px'>
        <div style='color:#a78bfa;font-weight:700;margin-bottom:6px'>🔀 4. Regimes de Mercado</div>
        <div style='font-size:.8rem;color:#9ca3af'>
        3 regimes detectados automaticamente:<br>
        pré-solar → transição → curtailment
        </div>
        <div style='margin-top:6px;font-size:.78rem'>
        <span style='color:#6b7280'>● pré-solar  2021–ago/2023</span><br>
        <span style='color:#60a5fa'>● transição  ago/2023–fev/2025</span><br>
        <span style='color:#f87171'>● curtailment  mar/2025–hoje ← regime atual</span>
        </div>
        <div style='font-size:.75rem;color:#6b7280;margin-top:4px'>Importância no modelo: {regime_imp:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabela de PLD por regime ──────────────────────────────────────────────
    if regime_stats:
        st.markdown("#### PLD por regime — comparativo histórico")
        rows = []
        for label, stats in regime_stats.items():
            is_current = "curtailment" in label.lower()
            rows.append({
                "Regime":          ("🔴 " if is_current else "") + label,
                "Observações":     f"{stats['n']:,}",
                "PLD mediana":     f"R$ {stats['pld_med']:,.0f}",
                "PLD P90":         f"R$ {stats['pld_p90']:,.0f}",
                "Incerteza (σ)":   f"R$ {stats['res_std']:,.1f}",
                "Regime atual":    "← você está aqui" if is_current else "",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)
        st.caption(
            "O modelo é treinado com **peso maior para o regime atual** (curtailment). "
            "A incerteza σ representa o desvio padrão dos resíduos em cada regime — "
            "maior σ = menos previsível naquele período."
        )

    # ── Insight sobre o drift atual ───────────────────────────────────────────
    st.markdown("#### Interpretação do Structural Price Drift")
    if drift_val >= 1.3:
        msg = (f"O mercado está operando **{drift_delta:.0f}% acima** do nível base de 2021. "
               "Isso indica reprecificação estrutural significativa — provavelmente combinação "
               "de expansão solar com curtailment, prêmio de risco crescente e restrições "
               "de transmissão em horas de pico. **O modelo já incorpora esse nível elevado** "
               "via feature `structural_drift` e peso maior para dados recentes.")
        color = "#f87171"
    elif drift_val >= 1.1:
        msg = (f"O mercado está operando **{drift_delta:.0f}% acima** do nível base de 2021. "
               "Drift moderado — o mercado está se ajustando à nova realidade operativa "
               "com mais renováveis e curtailment. O modelo captura essa tendência.")
        color = "#c8a44d"
    else:
        msg = ("O mercado está próximo do nível base histórico. "
               "O drift estrutural é baixo — a distribuição histórica é mais representativa.")
        color = "#34d399"

    st.markdown(
        f"<div style='background:#111827;border-left:3px solid {color};"
        f"border-radius:8px;padding:12px 16px;font-size:.85rem;color:#e5e7eb'>"
        f"{msg}</div>",
        unsafe_allow_html=True,
    )

    # ── Impacto no cálculo ────────────────────────────────────────────────────
    st.markdown("#### Impacto no cálculo probabilístico")
    st.markdown(f"""
Sem os ajustes de drift, o modelo usaria a distribuição histórica bruta
(2021–hoje com pesos iguais), o que **subestimaria** os preços atuais.

Com os 4 ajustes combinados:

| Ajuste | Efeito no PLD estimado |
|---|---|
| Decay temporal (meia-vida 1,5a) | Pondera mais dados de 2024–2026 |
| Normalização anual (price_index) | Remove diferença de patamar entre anos |
| IMR como feature (drift {drift_val:.2f}×) | Captura reprecificação de mercado |
| Regime curtailment (mar/2025+) | Aprende comportamento do regime atual |

O resultado é um **PLD fundamental mais preciso** e uma **distribuição simulada
condizente com o nível de preços vigente** — não com a média histórica 2021–2024.
    """)


def _render_results(result: JustificationResult) -> None:
    if result.aviso:
        st.warning(result.aviso)

    c_map = {"razoável": "#34d399", "neutro": "#60a5fa",
              "estressado": "#c8a44d", "extremo": "#f87171"}
    c_col = c_map.get(result.classificacao, "#9ca3af")
    gap_s = "+" if result.fundamental_gap >= 0 else ""

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Preço analisado",    f"R$ {result.preco_analisado:,.0f}")
    k2.metric("PLD " + ("fundamental" if result.modo=="atual" else "mediana histórica"),
              f"R$ {result.pld_fundamental:,.0f}")
    k3.metric("Gap",
              f"{gap_s}R$ {abs(result.fundamental_gap):,.0f}",
              delta=f"{gap_s}{result.fundamental_gap:.0f}",
              delta_color="inverse")
    k4.metric("P(justificação)", f"{result.probabilidade_justificacao:.1f}%")
    k5.metric("Obs. sazonais", f"{result.n_obs_filtro:,}")

    st.markdown(
        f"<div class='epje-card'>Classificação: "
        f"<span class='clsf-{result.classificacao}'>{result.classificacao.upper()}</span>"
        f" &nbsp;|&nbsp; Período: <strong>{result.filtro_temporal.describe()}</strong>"
        f" &nbsp;|&nbsp; Modo: <strong>{'Estado atual' if result.modo=='atual' else 'Histórico puro'}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Tabs de resultado ────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Distribuição", "📋 Percentis", "🕸 Cenário físico",
        "📈 Drift & Regime", "⚙️ Modelo", "🏛 CCEE", "📖 Interpretação",
    ])

    with tabs[0]:
        st.plotly_chart(_plot_distribution(result), use_container_width=True)

    with tabs[1]:
        pct = result.percentis
        st.markdown(f"""
| Percentil | PLD (R$/MWh) | vs. Preço analisado |
|---|---|---|
| P25 | R$ {pct['p25']:,.0f} | {"abaixo" if pct['p25'] < result.preco_analisado else "acima"} |
| P50 | R$ {pct['p50']:,.0f} | {"abaixo" if pct['p50'] < result.preco_analisado else "acima"} |
| P75 | R$ {pct['p75']:,.0f} | {"abaixo" if pct['p75'] < result.preco_analisado else "acima"} |
| P90 | R$ {pct['p90']:,.0f} | {"abaixo" if pct['p90'] < result.preco_analisado else "acima"} |
| P95 | R$ {pct['p95']:,.0f} | {"abaixo" if pct['p95'] < result.preco_analisado else "acima"} |
""")
        st.caption(f"Distribuição simulada — {len(result.pld_simulado_dist):,} cenários.")

    with tabs[2]:
        if result.modo == "atual":
            st.plotly_chart(
                _plot_radar(result.cenario_fisico_atual,
                            result.cenario_fisico_implicito),
                use_container_width=True,
            )
            cur, imp = result.cenario_fisico_atual, result.cenario_fisico_implicito
            rows = [
                ("EAR (%)",         cur.ear_pct,        imp.ear_pct,        "{:.1f}%"),
                ("ENA (% MLT)",     cur.ena_pct,        imp.ena_pct,        "{:.1f}%"),
                ("Despacho térmico",cur.thermal_ratio,  imp.thermal_ratio,  "{:.1%}"),
                ("Pressão de carga",cur.load_ratio,     imp.load_ratio,     "{:.1%}"),
                ("Curtailment",     cur.curtailment_ratio,imp.curtailment_ratio,"{:.1%}"),
            ]
            tdf = pd.DataFrame(rows, columns=["Variável","Atual","Implícito","_f"])
            for i, row in tdf.iterrows():
                fmt = row["_f"]
                for col in ["Atual","Implícito"]:
                    try:
                        tdf.at[i, col] = fmt.format(row[col]) if not np.isnan(float(row[col])) else "N/D"
                    except Exception:
                        tdf.at[i, col] = "N/D"
            st.dataframe(tdf.drop(columns=["_f"]).set_index("Variável"),
                         use_container_width=True)
            imp = result.cenario_fisico_implicito
            st.markdown(
                f"<div class='epje-card'>"
                f"🔍 <strong>Para justificar R$ {result.preco_analisado:,.0f}/MWh</strong> "
                f"seria necessário: EAR ≈ {imp.ear_pct:.0f}%, "
                f"ENA ≈ {imp.ena_pct:.0f}% MLT, "
                f"despacho térmico ≈ {imp.thermal_ratio:.0%} da geração total."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Cenário físico implícito disponível apenas no modo **Cenário atual**.")

    with tabs[3]:
        _render_drift_regime(result)

    with tabs[4]:
        if result.modo == "atual" and result.feature_importance:
            c1, c2 = st.columns(2)
            c1.metric("R² treino",       f"{result.model_r2:.3f}")
            c2.metric("R² validação cruzada", f"{result.model_cv_score:.3f}")
            st.plotly_chart(_plot_feature_importance(result.feature_importance),
                            use_container_width=True)
            st.caption(
                f"Desvio padrão dos resíduos: R$ {result.res_std:.1f}/MWh — "
                f"representa a incerteza do modelo após condicionar ao estado físico e período."
            )
        else:
            st.info("Diagnóstico do modelo disponível apenas no modo **Cenário atual**.")

    with tabs[5]:
        ccee = result.exposicao_spot
        if ccee and ccee.get("disponivel"):
            st.success(f"API CCEE — dados do submercado **{ccee['submercado']}**, "
                       f"mês de referência **{ccee.get('mes_referencia','')}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Montante total (MWh)", f"{ccee.get('montante_total_mwh',0):,.0f}")
            m2.metric("Montante médio/hora",  f"{ccee.get('montante_medio_mwh',0):,.1f} MWh")
            m3.metric("Exposição spot proxy (CCEAR-D/Total)",
                      f"{ccee.get('exposicao_spot_proxy',0):.1%}")
            if ccee.get("montante_por_tipo"):
                st.markdown("**Montante por tipo de contrato (MWh):**")
                st.dataframe(
                    pd.DataFrame([ccee["montante_por_tipo"]]).T
                      .rename(columns={0: "MWh"})
                      .sort_values("MWh", ascending=False),
                    use_container_width=True,
                )
            st.caption("⚠️ Dados CCEE mantidos apenas em memória — não persistidos.")
        elif ccee and ccee.get("erro"):
            st.caption(f"API CCEE indisponível: {ccee['erro']}")
        else:
            st.info("Ative 'Consultar API da CCEE' nos parâmetros para ver dados de comercialização.")

    with tabs[6]:
        modo_desc = ("estado físico atual + sazonalidade do período"
                     if result.modo == "atual"
                     else "distribuição histórica sazonal observada")
        st.markdown(f"""
**Preço analisado:** R$ {result.preco_analisado:,.2f}/MWh
**Submercado:** {result.submercado}
**Período:** {result.filtro_temporal.describe()}
**Modo:** {modo_desc}
**Observações históricas no filtro:** {result.n_obs_filtro:,}

---

**P(PLD ≥ {result.preco_analisado:,.0f}) = {result.probabilidade_justificacao:.1f}%**

Dentre os {len(result.pld_simulado_dist):,} cenários simulados para
**{result.filtro_temporal.describe()}** no submercado **{result.submercado}**,
{result.probabilidade_justificacao:.1f}% resultaram em PLD ≥ R$ {result.preco_analisado:,.0f}/MWh.

**Classificação `{result.classificacao.upper()}`:**
{"✅ Preço consistente com o histórico sazonal nas condições físicas atuais." if result.classificacao == "razoável" else
 "⚠️ Preço moderadamente acima do esperado — possível prêmio de risco ou expectativa de tensionamento." if result.classificacao == "neutro" else
 "⚡ Preço significativamente acima do esperado — requer cenário de estresse físico para se materializar." if result.classificacao == "estressado" else
 "🔴 Preço extremo — só se justificaria em crise hidrológica severa ou colapso de oferta."}

**Por que o período importa:**
O PLD brasileiro tem forte sazonalidade — o mesmo preço de R$ {result.preco_analisado:,.0f}/MWh
pode ser {'razoável' if result.classificacao in ['razoável','neutro'] else 'incomum'}
em **{result.filtro_temporal.label}** e ter probabilidade muito diferente em outros meses.
Esta análise usa apenas o histórico de **{result.filtro_temporal.describe()}** (2021–presente)
para garantir comparabilidade estatística.
""")
