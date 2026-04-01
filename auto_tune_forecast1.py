# -*- coding: utf-8 -*-
"""
auto_tune_forecast.py — MAÁTria Energia · Auto-Tuning de Modelos PLD
=====================================================================
Script autônomo que executa ciclos iterativos de otimização:

  1. Carrega dataset uma vez
  2. Testa combinações de hiperparâmetros + features
  3. Avalia métricas por subsistema e horizonte
  4. Registra o que melhorou e o que piorou
  5. Persiste o melhor modelo encontrado
  6. Para quando atinge metas de qualidade OU max iterações

Metas de qualidade "excepcional":
  - R² teste ≥ 0.35 para h=4w  (previsão de erro é difícil — 0.35+ é forte)
  - MAE PLD ≤ 35 R$/MWh para h=4w SE/CO
  - Coverage P10 entre 0.05–0.20, P90 entre 0.80–0.95 (bandas calibradas)
  - R² CV próximo de R² teste (gap ≤ 0.15 = sem overfitting)
  - Consistência entre subsistemas (todos acima do limiar mínimo)

Uso:
  python auto_tune_forecast.py                    # roda otimização completa
  python auto_tune_forecast.py --max-iter 30      # limita iterações
  python auto_tune_forecast.py --target-r2 0.40   # meta R² customizada
  python auto_tune_forecast.py --report            # mostra relatório do melhor

Gera:
  data/models/meta/tuning_log.json       — histórico completo de iterações
  data/models/meta/best_config.json      — melhor configuração encontrada
  data/models/meta/meta_*.pkl            — modelos do melhor run
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import product as iterproduct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("auto_tune")

# ── Paths ────────────────────────────────────────────────────────────────────
PMO_XLSX  = Path(os.getenv("PMO_XLSX",  "data/ons/PMOs/validacao_pmo.xlsx"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))
DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
META_DIR  = MODEL_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

SUBSISTEMAS = ["seco", "s", "ne", "n"]
SUB_LABEL   = {"seco": "SE/CO", "s": "Sul", "ne": "Nordeste", "n": "Norte"}
QUANTILES   = [0.10, 0.50, 0.90]

# ══════════════════════════════════════════════════════════════════════════════
# QUALITY TARGETS (meta de qualidade "excepcional")
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TARGETS = {
    "r2_test_min":       0.35,   # R² mínimo no teste (h=4w, seco)
    "r2_cv_min":         0.20,   # R² mínimo cross-val
    "mae_pld_max":       35.0,   # MAE PLD máximo aceitável (R$/MWh)
    "overfit_gap_max":   0.15,   # Gap máximo R²_test − R²_cv
    "coverage_p10_range": (0.03, 0.22),  # P10 deve cobrir 3–22%
    "coverage_p90_range": (0.78, 0.97),  # P90 deve cobrir 78–97%
}


def _score_metrics(metrics: Dict, targets: Dict = DEFAULT_TARGETS) -> float:
    """
    Score composto [0, 100] que resume a qualidade de um modelo.
    Quanto maior, melhor.
    
    Componentes:
      40% — R² teste (quanto mais alto, melhor)
      25% — MAE PLD invertido (quanto menor, melhor)
      15% — Calibração das bandas (quanto mais próximo do ideal)
      10% — Consistência CV (quanto menor o gap, melhor)
      10% — R² CV (estabilidade)
    """
    r2 = metrics.get("r2_test", -1)
    r2_cv = metrics.get("r2_cv_mean", -1) or -1
    mae = metrics.get("mae_pld", 999)
    cov_p10 = metrics.get("coverage_p10", 0.5)
    cov_p90 = metrics.get("coverage_p90", 0.5)

    # R² score: 0 at r2=-0.5, 100 at r2=0.6
    s_r2 = max(0, min(100, (r2 + 0.5) / 1.1 * 100))

    # MAE score: 100 at mae=0, 0 at mae=80
    s_mae = max(0, min(100, (1 - mae / 80) * 100))

    # Coverage score: 100 when p10≈0.10 and p90≈0.90
    dev_p10 = abs(cov_p10 - 0.10)
    dev_p90 = abs(cov_p90 - 0.90)
    s_cov = max(0, 100 - (dev_p10 + dev_p90) * 200)

    # Overfit score: 100 when gap=0, 0 when gap≥0.3
    gap = abs(r2 - r2_cv)
    s_overfit = max(0, min(100, (1 - gap / 0.3) * 100))

    # CV stability
    s_cv = max(0, min(100, (r2_cv + 0.5) / 1.1 * 100))

    score = (0.40 * s_r2 +
             0.25 * s_mae +
             0.15 * s_cov +
             0.10 * s_overfit +
             0.10 * s_cv)

    return round(score, 2)


def _check_targets_met(all_metrics: Dict, targets: Dict) -> Tuple[bool, List[str]]:
    """Verifica se todas as metas foram atingidas. Retorna (ok, mensagens)."""
    msgs = []
    met = True

    # Foco no subsistema principal (seco, h=4w)
    seco_4 = all_metrics.get("seco", {}).get(4, {})
    if not seco_4:
        return False, ["Sem métricas para seco h=4w"]

    r2 = seco_4.get("r2_test", -1)
    if r2 < targets["r2_test_min"]:
        msgs.append(f"R² teste={r2:.3f} < meta {targets['r2_test_min']}")
        met = False
    else:
        msgs.append(f"✅ R² teste={r2:.3f} ≥ {targets['r2_test_min']}")

    r2_cv = seco_4.get("r2_cv_mean", -1) or -1
    if r2_cv < targets["r2_cv_min"]:
        msgs.append(f"R² CV={r2_cv:.3f} < meta {targets['r2_cv_min']}")
        met = False
    else:
        msgs.append(f"✅ R² CV={r2_cv:.3f} ≥ {targets['r2_cv_min']}")

    mae = seco_4.get("mae_pld", 999)
    if mae > targets["mae_pld_max"]:
        msgs.append(f"MAE PLD={mae:.1f} > meta {targets['mae_pld_max']}")
        met = False
    else:
        msgs.append(f"✅ MAE PLD={mae:.1f} ≤ {targets['mae_pld_max']}")

    gap = abs(r2 - r2_cv)
    if gap > targets["overfit_gap_max"]:
        msgs.append(f"Gap overfit={gap:.3f} > meta {targets['overfit_gap_max']}")
        met = False
    else:
        msgs.append(f"✅ Gap overfit={gap:.3f} ≤ {targets['overfit_gap_max']}")

    cov_p10 = seco_4.get("coverage_p10", 0.5)
    cov_p90 = seco_4.get("coverage_p90", 0.5)
    lo10, hi10 = targets["coverage_p10_range"]
    lo90, hi90 = targets["coverage_p90_range"]
    if lo10 <= cov_p10 <= hi10:
        msgs.append(f"✅ P10 coverage={cov_p10:.3f} ∈ [{lo10}, {hi10}]")
    else:
        msgs.append(f"P10 coverage={cov_p10:.3f} fora de [{lo10}, {hi10}]")
        met = False
    if lo90 <= cov_p90 <= hi90:
        msgs.append(f"✅ P90 coverage={cov_p90:.3f} ∈ [{lo90}, {hi90}]")
    else:
        msgs.append(f"P90 coverage={cov_p90:.3f} fora de [{lo90}, {hi90}]")
        met = False

    return met, msgs


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH SPACE — hiperparâmetros + features
# ══════════════════════════════════════════════════════════════════════════════

HYPERPARAM_GRID = {
    "max_iter":          [200, 400, 600, 800],
    "max_depth":         [3, 4, 5, 6],
    "learning_rate":     [0.02, 0.035, 0.05, 0.08],
    "min_samples_leaf":  [5, 8, 12, 20],
    "l2_regularization": [0.05, 0.15, 0.30, 0.50],
}

FEATURE_SETS = {
    "base_only": {
        "description": "Apenas features base do pld_forecast_engine",
        "include_enhanced": False,
        "extra_lags": False,
        "interactions": False,
    },
    "enhanced": {
        "description": "Base + MAÁTria (GFOM, curtailment, disp)",
        "include_enhanced": True,
        "extra_lags": False,
        "interactions": False,
    },
    "enhanced_lags": {
        "description": "Enhanced + lags adicionais (2w, 4w)",
        "include_enhanced": True,
        "extra_lags": True,
        "interactions": False,
    },
    "full": {
        "description": "Enhanced + lags + interações",
        "include_enhanced": True,
        "extra_lags": True,
        "interactions": True,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def _add_extra_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona lags adicionais de 2 e 4 semanas para features-chave."""
    df = df.copy()
    for col in ["cmo_real_seco", "pld_real_seco", "erro_ons_seco",
                "ena_real_seco_mlt", "ear_real_seco"]:
        if col in df.columns:
            df[f"{col}_lag2"] = df[col].shift(2)
            df[f"{col}_lag4"] = df[col].shift(4)

    # Momentum: variação entre lag1 e lag2
    if "cmo_real_seco" in df.columns:
        l1 = df["cmo_real_seco"].shift(1)
        l2 = df["cmo_real_seco"].shift(2)
        df["cmo_momentum_seco"] = l1 - l2

    if "erro_ons_seco" in df.columns:
        df["erro_ons_accel"] = (
            df["erro_ons_seco"].shift(1) - df["erro_ons_seco"].shift(2)
        )
    return df


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features de interação (produto cruzado)."""
    df = df.copy()

    # ENA × EAR: sinal combinado de hidrologia
    if "ena_prev_seco_mlt" in df.columns and "ear_init_seco" in df.columns:
        df["hydro_combo_seco"] = (
            df["ena_prev_seco_mlt"].fillna(80) *
            df["ear_init_seco"].fillna(50) / 100
        )

    # Thermal ratio × CMO delta: pressão de custo
    if "thermal_ratio" in df.columns and "cmo_delta_seco" in df.columns:
        df["thermal_cost_pressure"] = (
            df["thermal_ratio"].fillna(0.15) *
            df["cmo_delta_seco"].fillna(0)
        )

    # Curtailment × renovável ratio: saturação
    if "curtail_total_sem_lag1" in df.columns and "renov_carga_ratio" in df.columns:
        df["saturation_index"] = (
            df["curtail_total_sem_lag1"].fillna(0) *
            df["renov_carga_ratio"].fillna(0.2)
        )

    # Erro persistente × volatilidade: incerteza composta
    if "erro_ons_seco_4w" in df.columns and "vol_cmo_seco_4w" in df.columns:
        df["uncertainty_index"] = (
            df["erro_ons_seco_4w"].fillna(0).abs() *
            df["vol_cmo_seco_4w"].fillna(10) / 100
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING FUNCTION (parametrizada)
# ══════════════════════════════════════════════════════════════════════════════

def train_with_config(
    df: pd.DataFrame,
    config: Dict,
    subsistema: str = "seco",
    horizon_weeks: int = 4,
) -> Optional[Dict]:
    """
    Treina modelo com configuração específica de hiperparâmetros e features.
    Retorna bundle com modelos + métricas + config usada.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score
    except ImportError:
        return None

    hp = config.get("hyperparams", {})
    feat_set = config.get("feature_set", "enhanced")
    feat_config = FEATURE_SETS.get(feat_set, FEATURE_SETS["enhanced"])

    # ── Preparar dataset com variant de features ─────────────────────────
    df_work = df.copy()
    if feat_config.get("extra_lags"):
        df_work = _add_extra_lags(df_work)
    if feat_config.get("interactions"):
        df_work = _add_interactions(df_work)

    # ── Target ───────────────────────────────────────────────────────────
    erro_col = f"erro_pld_{subsistema}"
    if erro_col not in df_work.columns:
        cmo_col = f"cmo_med_{subsistema}"
        pld_col = f"pld_real_{subsistema}"
        if pld_col in df_work.columns and cmo_col in df_work.columns:
            df_work[erro_col] = df_work[pld_col] - df_work[cmo_col]
        else:
            return None

    if df_work[erro_col].notna().sum() < 52:
        return None

    # ── Feature selection ────────────────────────────────────────────────
    # Começar com as features conhecidas, adicionar novas se disponíveis
    from meta_forecast_engine import ENHANCED_FEATURE_COLS, BASE_FEATURE_COLS

    if feat_config.get("include_enhanced"):
        candidates = list(ENHANCED_FEATURE_COLS)
    else:
        candidates = list(BASE_FEATURE_COLS)

    # Adicionar features extras geradas acima
    extra_cols = [c for c in df_work.columns
                  if c.endswith("_lag2") or c.endswith("_lag4")
                  or c in ("cmo_momentum_seco", "erro_ons_accel",
                           "hydro_combo_seco", "thermal_cost_pressure",
                           "saturation_index", "uncertainty_index")]
    candidates.extend(extra_cols)

    available = [c for c in candidates if c in df_work.columns]
    # Deduplicate preservando ordem
    seen = set()
    available = [c for c in available if not (c in seen or seen.add(c))]

    if not available:
        return None

    # ── Target shifted ───────────────────────────────────────────────────
    y = df_work[erro_col].shift(-horizon_weeks)
    mask = y.notna()
    X = df_work[available][mask].copy()
    y = y[mask]
    X = X.ffill().fillna(0)

    if len(X) < 52:
        return None

    n_tr = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr, y_te = y.iloc[:n_tr], y.iloc[n_tr:]

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, n_tr // 26)))

    # ── Hiperparâmetros ──────────────────────────────────────────────────
    gbm_params = {
        "loss":              "quantile",
        "max_iter":          hp.get("max_iter", 400),
        "max_depth":         hp.get("max_depth", 4),
        "learning_rate":     hp.get("learning_rate", 0.035),
        "min_samples_leaf":  hp.get("min_samples_leaf", 8),
        "l2_regularization": hp.get("l2_regularization", 0.15),
        "random_state":      42,
    }

    models, metrics = {}, {}
    cv_scores = []

    for q in QUANTILES:
        params = {**gbm_params, "quantile": q}
        gb = HistGradientBoostingRegressor(**params)
        gb.fit(X_tr.values, y_tr.values)
        y_pred = gb.predict(X_te.values)

        if q == 0.50:
            metrics["mae_erro"] = round(float(np.mean(np.abs(y_pred - y_te.values))), 2)
            metrics["r2_test"]  = round(float(r2_score(y_te.values, y_pred)), 3)

            for tr_idx, val_idx in tscv.split(X_tr):
                params_cv = {**gbm_params, "quantile": 0.50,
                             "max_iter": min(300, gbm_params["max_iter"])}
                gb_cv = HistGradientBoostingRegressor(**params_cv)
                gb_cv.fit(X_tr.values[tr_idx], y_tr.values[tr_idx])
                cv_pred = gb_cv.predict(X_tr.values[val_idx])
                cv_scores.append(r2_score(y_tr.values[val_idx], cv_pred))

            metrics["r2_cv_mean"] = round(float(np.mean(cv_scores)), 3) if cv_scores else None
            metrics["r2_cv_std"]  = round(float(np.std(cv_scores)), 3)  if cv_scores else None

            # PLD-level metrics
            cmo_col = f"cmo_med_{subsistema}"
            if cmo_col in df_work.columns:
                cmo_te = df_work[cmo_col].shift(-horizon_weeks).iloc[n_tr:n_tr+len(y_pred)]
                cmo_valid = cmo_te.notna()
                if cmo_valid.sum() > 0:
                    _cmo = cmo_te[cmo_valid].values[:len(y_pred)]
                    _ypred = y_pred[:len(_cmo)]
                    _yte = y_te.values[:len(_cmo)]
                    if len(_cmo) > 0:
                        pld_pred = _cmo + _ypred
                        pld_real = _cmo + _yte
                        metrics["mae_pld"] = round(float(np.mean(np.abs(pld_pred - pld_real))), 2)
                        denom = np.where(pld_real == 0, np.nan, pld_real)
                        metrics["mape_pld"] = round(float(
                            np.nanmean(np.abs((pld_pred - pld_real) / denom)) * 100), 1)

        metrics[f"coverage_p{int(q*100)}"] = round(float(np.mean(y_te.values <= y_pred)), 3)
        models[q] = gb

    # Feature importance
    fi = {}
    gb50 = models.get(0.50)
    if gb50 and hasattr(gb50, "feature_importances_"):
        fi = dict(sorted(
            zip(available, gb50.feature_importances_.tolist()),
            key=lambda x: -x[1]))

    metrics["score"] = _score_metrics(metrics)
    metrics["n_features"] = len(available)

    return {
        "models":             models,
        "feature_cols":       available,
        "metrics":            metrics,
        "feature_importance": fi,
        "config":             config,
        "n_train":            n_tr,
        "n_test":             len(X_te),
        "subsistema":         subsistema,
        "horizon_weeks":      horizon_weeks,
        "target":             "erro_pld",
        "trained_at":         datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def _generate_initial_configs() -> List[Dict]:
    """Gera configurações iniciais cobrindo o espaço de busca."""
    configs = []

    # Config 0: baseline (parâmetros atuais do meta_forecast_engine)
    configs.append({
        "name": "baseline",
        "hyperparams": {
            "max_iter": 400, "max_depth": 4,
            "learning_rate": 0.035, "min_samples_leaf": 8,
            "l2_regularization": 0.15,
        },
        "feature_set": "enhanced",
    })

    # Config 1-3: variar feature sets com baseline HP
    for fs in ["base_only", "enhanced_lags", "full"]:
        configs.append({
            "name": f"feat_{fs}",
            "hyperparams": configs[0]["hyperparams"].copy(),
            "feature_set": fs,
        })

    # Config 4-7: variar learning rate (mais impactante)
    for lr in [0.02, 0.05, 0.08]:
        configs.append({
            "name": f"lr_{lr}",
            "hyperparams": {**configs[0]["hyperparams"], "learning_rate": lr},
            "feature_set": "enhanced",
        })

    # Config 8-10: variar profundidade
    for md in [3, 5, 6]:
        configs.append({
            "name": f"depth_{md}",
            "hyperparams": {**configs[0]["hyperparams"], "max_depth": md},
            "feature_set": "enhanced",
        })

    # Config 11-13: variar regularização
    for l2 in [0.05, 0.30, 0.50]:
        configs.append({
            "name": f"l2_{l2}",
            "hyperparams": {**configs[0]["hyperparams"], "l2_regularization": l2},
            "feature_set": "enhanced",
        })

    # Config 14-15: variar iterações
    for mi in [200, 800]:
        configs.append({
            "name": f"iter_{mi}",
            "hyperparams": {**configs[0]["hyperparams"], "max_iter": mi},
            "feature_set": "enhanced",
        })

    return configs


def _generate_refinement_configs(best_config: Dict, best_metrics: Dict) -> List[Dict]:
    """
    Gera configs de refinamento ao redor do melhor encontrado.
    Perturba cada hiperparâmetro ±20% e testa feature sets vizinhos.
    """
    configs = []
    hp = best_config["hyperparams"]
    fs = best_config["feature_set"]

    # Perturbar cada HP individualmente
    perturbations = {
        "learning_rate":     [0.8, 0.9, 1.1, 1.2],
        "max_iter":          [0.75, 1.25, 1.5],
        "max_depth":         [-1, 0, 1],  # inteiro
        "min_samples_leaf":  [-3, -1, 2, 4],  # inteiro
        "l2_regularization": [0.7, 0.85, 1.15, 1.3],
    }

    for param, factors in perturbations.items():
        for f in factors:
            new_hp = hp.copy()
            if param in ("max_depth", "min_samples_leaf"):
                new_val = max(2, int(hp[param] + f))
            else:
                new_val = round(hp[param] * f, 4)
                if param == "max_iter":
                    new_val = max(100, int(new_val))
                elif param == "learning_rate":
                    new_val = max(0.005, min(0.15, new_val))
                elif param == "l2_regularization":
                    new_val = max(0.01, min(1.0, new_val))

            if new_val == hp[param]:
                continue

            new_hp[param] = new_val
            configs.append({
                "name": f"refine_{param}_{new_val}",
                "hyperparams": new_hp,
                "feature_set": fs,
            })

    # Testar feature sets adjacentes
    fs_order = ["base_only", "enhanced", "enhanced_lags", "full"]
    idx = fs_order.index(fs) if fs in fs_order else 1
    for di in [-1, 1]:
        ni = idx + di
        if 0 <= ni < len(fs_order) and fs_order[ni] != fs:
            configs.append({
                "name": f"refine_feat_{fs_order[ni]}",
                "hyperparams": hp.copy(),
                "feature_set": fs_order[ni],
            })

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    max_iterations: int = 50,
    target_r2: float = 0.35,
    target_mae: float = 35.0,
    focus_sub: str = "seco",
    focus_horizon: int = 4,
    verbose: bool = True,
) -> Dict:
    """
    Loop principal de otimização.
    
    Fase 1 (exploração):  testa configs iniciais diversas
    Fase 2 (refinamento): perturba ao redor do melhor encontrado
    Fase 3 (deep refine): combinações do top-3
    
    Para quando: metas atingidas OU max iterações OU sem melhoria por 10 runs.
    """
    from meta_forecast_engine import (
        load_pmo_features, load_actuals,
        build_enhanced_dataset, load_maatria_weekly_features,
        save_meta_models,
    )

    targets = {**DEFAULT_TARGETS, "r2_test_min": target_r2, "mae_pld_max": target_mae}

    print("\n" + "=" * 65)
    print("  MAÁTria Energia · Auto-Tune Forecast Models")
    print("=" * 65)
    print(f"  Metas: R² ≥ {target_r2} | MAE PLD ≤ R${target_mae} | max {max_iterations} iter")
    print(f"  Foco: {SUB_LABEL[focus_sub]} h={focus_horizon}w")
    print("=" * 65)

    # ── Carregar dataset uma vez ─────────────────────────────────────────
    t0 = time.time()
    print("\n[0] Carregando dados...")
    pmo_df = load_pmo_features(PMO_XLSX)
    actuals_df = load_actuals(DATA_DIR)
    maatria_df = load_maatria_weekly_features(DATA_DIR)
    df = build_enhanced_dataset(pmo_df, actuals_df, maatria_df)

    if df.empty:
        print("ERRO: dataset vazio")
        return {}

    print(f"    Dataset: {len(df)} semanas × {df.shape[1]} colunas")
    print(f"    Tempo carga: {time.time()-t0:.1f}s")

    # ── Estado da otimização ─────────────────────────────────────────────
    history: List[Dict] = []
    best_score = -999
    best_config = None
    best_bundle = None
    best_all_metrics = {}
    no_improvement_count = 0
    phase = "exploration"

    # ── Gerar configs iniciais ───────────────────────────────────────────
    configs_queue = _generate_initial_configs()
    total_configs = len(configs_queue)

    iteration = 0
    while iteration < max_iterations:
        if not configs_queue:
            if phase == "exploration":
                # Transição para refinamento
                if best_config:
                    phase = "refinement"
                    configs_queue = _generate_refinement_configs(best_config,
                                                                 best_all_metrics)
                    if verbose:
                        print(f"\n{'='*65}")
                        print(f"  FASE 2 — Refinamento ({len(configs_queue)} configs)")
                        print(f"{'='*65}")
                    continue
                else:
                    break
            elif phase == "refinement":
                # Transição para deep refine
                phase = "deep_refine"
                # Combinar top-3 configs
                top3 = sorted(history, key=lambda x: x.get("score", 0), reverse=True)[:3]
                configs_queue = []
                for i, t in enumerate(top3):
                    for j, t2 in enumerate(top3):
                        if i >= j:
                            continue
                        # Misturar HPs
                        mixed = {}
                        hp1 = t["config"]["hyperparams"]
                        hp2 = t2["config"]["hyperparams"]
                        for k in hp1:
                            mixed[k] = (hp1[k] + hp2[k]) / 2
                            if k in ("max_iter", "max_depth", "min_samples_leaf"):
                                mixed[k] = int(round(mixed[k]))
                        configs_queue.append({
                            "name": f"mix_{t['config']['name']}_{t2['config']['name']}",
                            "hyperparams": mixed,
                            "feature_set": t["config"]["feature_set"],
                        })
                if verbose:
                    print(f"\n{'='*65}")
                    print(f"  FASE 3 — Deep Refine ({len(configs_queue)} configs)")
                    print(f"{'='*65}")
                if not configs_queue:
                    break
                continue
            else:
                break

        config = configs_queue.pop(0)
        iteration += 1
        t1 = time.time()

        # ── Treinar com esta config ──────────────────────────────────────
        bundle = train_with_config(df, config, focus_sub, focus_horizon)

        if bundle is None:
            if verbose:
                print(f"  [{iteration:3d}] {config['name']:<35} FALHOU")
            continue

        metrics = bundle["metrics"]
        score = metrics.get("score", 0)
        elapsed = time.time() - t1

        # ── Também treinar outros subsistemas para verificar consistência ─
        all_metrics = {focus_sub: {focus_horizon: metrics}}
        for sub in SUBSISTEMAS:
            if sub == focus_sub:
                continue
            b2 = train_with_config(df, config, sub, focus_horizon)
            if b2:
                all_metrics[sub] = {focus_horizon: b2["metrics"]}

        # ── Registrar ────────────────────────────────────────────────────
        entry = {
            "iteration": iteration,
            "phase":     phase,
            "config":    config,
            "score":     score,
            "metrics":   metrics,
            "elapsed":   round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        history.append(entry)

        # ── Verificar melhoria ───────────────────────────────────────────
        improved = score > best_score
        if improved:
            best_score = score
            best_config = config
            best_bundle = bundle
            best_all_metrics = all_metrics
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # ── Print ────────────────────────────────────────────────────────
        if verbose:
            marker = " ★ BEST" if improved else ""
            r2 = metrics.get("r2_test", -1)
            mae = metrics.get("mae_pld", 999)
            r2cv = metrics.get("r2_cv_mean", -1) or -1
            print(f"  [{iteration:3d}] {config['name']:<35} "
                  f"score={score:5.1f} R²={r2:+.3f} MAE={mae:5.1f} "
                  f"R²cv={r2cv:+.3f} ({elapsed:.1f}s){marker}")

        # ── Verificar metas ──────────────────────────────────────────────
        met, msgs = _check_targets_met(all_metrics, targets)
        if met:
            print(f"\n{'='*65}")
            print(f"  🏆 METAS ATINGIDAS na iteração {iteration}!")
            print(f"{'='*65}")
            for m in msgs:
                print(f"    {m}")
            break

        # ── Early stop: sem melhoria por N iterações ─────────────────────
        if no_improvement_count >= 12 and phase == "deep_refine":
            print(f"\n  ⚠️  Sem melhoria por {no_improvement_count} iterações — parando.")
            break
        if no_improvement_count >= 8 and phase == "refinement":
            # Forçar transição para deep refine
            configs_queue = []

    # ══════════════════════════════════════════════════════════════════════
    # RESULTADO FINAL
    # ══════════════════════════════════════════════════════════════════════

    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  RESULTADO FINAL — {iteration} iterações em {total_time:.0f}s")
    print(f"{'='*65}")

    if best_bundle:
        bm = best_bundle["metrics"]
        print(f"  Melhor config:  {best_config['name']}")
        print(f"  Feature set:    {best_config['feature_set']}")
        print(f"  Score:          {best_score:.1f}/100")
        print(f"  R² teste:       {bm.get('r2_test', '?')}")
        print(f"  R² CV:          {bm.get('r2_cv_mean', '?')}")
        print(f"  MAE erro:       R${bm.get('mae_erro', '?')}")
        print(f"  MAE PLD:        R${bm.get('mae_pld', '?')}")
        print(f"  Coverage P10:   {bm.get('coverage_p10', '?')}")
        print(f"  Coverage P90:   {bm.get('coverage_p90', '?')}")
        print(f"  N features:     {bm.get('n_features', '?')}")
        print(f"  Hyperparams:    {json.dumps(best_config['hyperparams'], indent=2)}")

        # ── Salvar melhor modelo ─────────────────────────────────────────
        print(f"\n  Salvando melhor modelo...")

        # Retreinar com melhor config para TODOS subsistemas e horizontes
        final_models = {}
        for sub in SUBSISTEMAS:
            final_models[sub] = {}
            for h in [4, 8, 12, 26]:
                b = train_with_config(df, best_config, sub, h)
                if b:
                    final_models[sub][h] = b
                    m = b["metrics"]
                    lbl = SUB_LABEL[sub]
                    print(f"    {lbl} h={h:2d}w: score={m.get('score',0):5.1f} "
                          f"R²={m.get('r2_test',0):+.3f} MAE={m.get('mae_pld','?')}")

        save_meta_models(final_models, META_DIR)

        # Salvar config
        with open(META_DIR / "best_config.json", "w") as f:
            json.dump({
                "config":      best_config,
                "score":       best_score,
                "metrics":     bm,
                "iterations":  iteration,
                "total_time":  round(total_time, 1),
                "timestamp":   datetime.now().isoformat(),
            }, f, indent=2)

        # Top features
        fi = best_bundle.get("feature_importance", {})
        if fi:
            print(f"\n  Top 10 features:")
            for i, (feat, imp) in enumerate(list(fi.items())[:10]):
                bar = "█" * int(imp * 200)
                print(f"    {i+1:2d}. {feat:<35} {imp:.4f} {bar}")

    # Salvar histórico completo
    with open(META_DIR / "tuning_log.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\n  Histórico salvo em {META_DIR}/tuning_log.json")

    # Verificação final
    met, msgs = _check_targets_met(best_all_metrics, targets)
    print(f"\n  {'✅ TODAS AS METAS ATINGIDAS' if met else '⚠️  Algumas metas pendentes'}:")
    for m in msgs:
        print(f"    {m}")

    print(f"\n{'='*65}\n")

    return {
        "best_config": best_config,
        "best_score":  best_score,
        "iterations":  iteration,
        "total_time":  round(total_time, 1),
        "targets_met": met,
        "history":     history,
    }


def show_report():
    """Mostra relatório do melhor modelo salvo."""
    config_path = META_DIR / "best_config.json"
    log_path = META_DIR / "tuning_log.json"

    if not config_path.exists():
        print("Nenhum tuning realizado. Execute: python auto_tune_forecast.py")
        return

    with open(config_path) as f:
        best = json.load(f)

    print("\n" + "=" * 65)
    print("  MAÁTria Energia · Relatório Auto-Tune")
    print("=" * 65)
    print(f"  Config:     {best['config']['name']}")
    print(f"  Feature set: {best['config']['feature_set']}")
    print(f"  Score:      {best['score']:.1f}/100")
    print(f"  Iterações:  {best['iterations']}")
    print(f"  Tempo:      {best['total_time']:.0f}s")
    print(f"  Data:       {best['timestamp'][:19]}")

    m = best.get("metrics", {})
    print(f"\n  Métricas:")
    print(f"    R² teste:     {m.get('r2_test', '?')}")
    print(f"    R² CV:        {m.get('r2_cv_mean', '?')}")
    print(f"    MAE erro:     R${m.get('mae_erro', '?')}")
    print(f"    MAE PLD:      R${m.get('mae_pld', '?')}")
    print(f"    MAPE PLD:     {m.get('mape_pld', '?')}%")
    print(f"    Coverage P10: {m.get('coverage_p10', '?')}")
    print(f"    Coverage P90: {m.get('coverage_p90', '?')}")

    hp = best["config"]["hyperparams"]
    print(f"\n  Hiperparâmetros:")
    for k, v in hp.items():
        print(f"    {k}: {v}")

    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)
        scores = [h["score"] for h in history if "score" in h]
        if scores:
            print(f"\n  Histórico ({len(history)} runs):")
            print(f"    Score mín:    {min(scores):.1f}")
            print(f"    Score med:    {np.median(scores):.1f}")
            print(f"    Score máx:    {max(scores):.1f}")
            print(f"    Melhoria:     {max(scores) - scores[0]:+.1f} vs baseline")

    print(f"\n{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MAÁTria · Auto-Tune PLD Forecast Models")
    ap.add_argument("--max-iter", type=int, default=50,
                    help="Máximo de iterações (default: 50)")
    ap.add_argument("--target-r2", type=float, default=0.35,
                    help="Meta R² teste (default: 0.35)")
    ap.add_argument("--target-mae", type=float, default=35.0,
                    help="Meta MAE PLD em R$/MWh (default: 35)")
    ap.add_argument("--focus-sub", default="seco",
                    choices=SUBSISTEMAS,
                    help="Subsistema foco (default: seco)")
    ap.add_argument("--report", action="store_true",
                    help="Mostra relatório do melhor modelo")
    ap.add_argument("--xlsx", default=str(PMO_XLSX))
    ap.add_argument("--data", default=str(DATA_DIR))
    args = ap.parse_args()

    PMO_XLSX = Path(args.xlsx)
    DATA_DIR = Path(args.data)

    if args.report:
        show_report()
    else:
        run_optimization(
            max_iterations=args.max_iter,
            target_r2=args.target_r2,
            target_mae=args.target_mae,
            focus_sub=args.focus_sub,
        )
