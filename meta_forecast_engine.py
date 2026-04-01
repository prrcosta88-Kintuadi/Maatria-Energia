# -*- coding: utf-8 -*-
"""
meta_forecast_engine.py — MAÁTria Energia · Sistema Híbrido de Precificação PLD
================================================================================
Pipeline de 3 camadas:
  Camada 1: ML Forecast Engine  (aprende o erro sistemático do ONS)
  Camada 2: Price Justification Engine  (modelo físico-econômico)
  Camada 3: Meta-Model  (combina ambos com ponderação dinâmica)

Outputs:
  - PLD_predito_ML       (previsão pura do modelo ML)
  - PLD_fundamental       (previsão estrutural do PJE)
  - PLD_final             (combinação ponderada)
  - confidence_score      (0–1)
  - market_regime         (inflated / compressed / balanced)

Integração:
  - Reutiliza pld_forecast_engine.py para dados PMO e actuals
  - Reutiliza premium_engine.py para P_justification e SPDI
  - Adiciona features ausentes: erro_ear, erro_carga, volatilidades,
    GFOM, curtailment, IMR semanal do DuckDB
  - Target preferido: erro_pld = pld_real − cmo_previsto

Uso:
  python meta_forecast_engine.py --build          # treina modelos
  python meta_forecast_engine.py --forecast       # gera previsão híbrida
  python meta_forecast_engine.py --diagnose       # verifica fontes de dados
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PMO_XLSX  = Path(os.getenv("PMO_XLSX",  "data/ons/PMOs/validacao_pmo.xlsx"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))
DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
META_DIR  = MODEL_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)

SUBSISTEMAS = ["seco", "s", "ne", "n"]
SUB_LABEL   = {"seco": "SE/CO", "s": "Sul", "ne": "Nordeste", "n": "Norte"}
QUANTILES   = [0.10, 0.50, 0.90]

# ── Importação dos motores existentes ────────────────────────────────────────
try:
    from pld_forecast_engine import (
        load_pmo_features, load_actuals, build_training_dataset,
        _build_derived_features_only, forecast_short_term,
        get_latest_pmo_state, _compute_erro_lag1,
        load_models as load_base_models,
        forecast_medium_long_term, _apply_regime_adjustment,
        FEATURE_COLS as BASE_FEATURE_COLS,
        BAND_CALIBRATION, SEASONAL_BIAS_R,
    )
    _BASE_ENGINE_OK = True
except ImportError:
    _BASE_ENGINE_OK = False
    BASE_FEATURE_COLS = []

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ENHANCED DATA INGESTION (features MAÁTria do DuckDB)
# ══════════════════════════════════════════════════════════════════════════════

def load_maatria_weekly_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Carrega features exclusivas MAÁTria do kintuadi.duckdb em base semanal:
      - GFOM % semanal (despacho fora de mérito)
      - Curtailment total semanal (MW cortado)
      - Disponibilidade hidro semanal
      - Geração renovável semanal (eólica + solar)
    
    Todas alinhadas por segunda-feira (início da semana operativa).
    """
    duckdb_path = data_dir / "kintuadi.duckdb"
    if not duckdb_path.exists():
        return pd.DataFrame()

    try:
        import duckdb as _ddb
        con = _ddb.connect(str(duckdb_path), read_only=True)
        tables = set(con.execute("SHOW TABLES").df()["name"].tolist())
        result = pd.DataFrame()

        # ── GFOM semanal (SIN agregado) ──────────────────────────────────
        if "despacho_gfom" in tables:
            try:
                gfom_df = con.execute("""
                    SELECT
                        date_trunc('week', din_instante) + INTERVAL '1 day' AS semana,
                        AVG(val_verifgfom)      AS gfom_sem,
                        AVG(val_verifgeracao)    AS gfom_ger_sem,
                        AVG(val_verifordemmerito) AS thermal_merit_sem
                    FROM despacho_gfom
                    WHERE din_instante IS NOT NULL AND ano >= 2021
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not gfom_df.empty:
                    gfom_df["semana"] = pd.to_datetime(gfom_df["semana"])
                    gfom_df = gfom_df.set_index("semana")
                    gfom_df["gfom_pct_sem"] = (
                        gfom_df["gfom_sem"] /
                        gfom_df["gfom_ger_sem"].replace(0, np.nan) * 100
                    ).clip(0, 100)
                    for c in gfom_df.columns:
                        result[c] = pd.to_numeric(gfom_df[c], errors="coerce")
            except Exception as e:
                logger.warning(f"GFOM semanal: {e}")

        # ── Curtailment semanal (eólica + solar) ─────────────────────────
        for fonte, tabela in [("wind", "restricao_eolica"),
                              ("solar", "restricao_fotovoltaica")]:
            if tabela not in tables:
                continue
            try:
                curt_df = con.execute(f"""
                    SELECT
                        date_trunc('week', din_instante) + INTERVAL '1 day' AS semana,
                        AVG(val_geracaolimitada - val_geracao) AS curtail_avg
                    FROM {tabela}
                    WHERE din_instante IS NOT NULL AND ano >= 2021
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not curt_df.empty:
                    curt_df["semana"] = pd.to_datetime(curt_df["semana"])
                    result[f"curtail_{fonte}_sem"] = (
                        curt_df.set_index("semana")["curtail_avg"]
                        .clip(lower=0)
                    )
            except Exception as e:
                logger.warning(f"Curtailment {fonte}: {e}")

        if "curtail_wind_sem" in result.columns or "curtail_solar_sem" in result.columns:
            result["curtail_total_sem"] = (
                result.get("curtail_wind_sem", 0).fillna(0) +
                result.get("curtail_solar_sem", 0).fillna(0)
            )

        # ── Geração renovável semanal (para net_load proxy) ─────────────
        if "geracao_usina_horaria" in tables:
            try:
                ren_df = con.execute("""
                    SELECT
                        date_trunc('week', din_instante) + INTERVAL '1 day' AS semana,
                        AVG(CASE WHEN UPPER(nom_tipousina) IN
                            ('FOTOVOLTAICA','SOLAR','EOLIELÉTRICA','EÓLICA',
                             'EOLICA','EOLIELETRICA') THEN val_geracao END
                        ) AS renov_avg_sem,
                        AVG(val_geracao) AS geracao_total_sem
                    FROM geracao_usina_horaria
                    WHERE din_instante IS NOT NULL AND ano >= 2021
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not ren_df.empty:
                    ren_df["semana"] = pd.to_datetime(ren_df["semana"])
                    ren_df = ren_df.set_index("semana")
                    for c in ren_df.columns:
                        result[c] = pd.to_numeric(ren_df[c], errors="coerce")
            except Exception:
                pass

        # ── Disponibilidade hidro semanal ────────────────────────────────
        if "disponibilidade_usina" in tables:
            try:
                disp_df = con.execute("""
                    SELECT
                        date_trunc('week', din_instante) + INTERVAL '1 day' AS semana,
                        AVG(CASE WHEN UPPER(id_tipousina) IN ('UHE','PCH','CGH')
                            THEN val_dispsincronizada END) AS disp_hydro_sem
                    FROM disponibilidade_usina
                    WHERE din_instante IS NOT NULL AND ano >= 2021
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not disp_df.empty:
                    disp_df["semana"] = pd.to_datetime(disp_df["semana"])
                    result["disp_hydro_sem"] = (
                        disp_df.set_index("semana")["disp_hydro_sem"]
                    )
            except Exception:
                pass

        con.close()

        if result.empty:
            return pd.DataFrame()

        result.index = pd.DatetimeIndex(result.index)
        return result[result.index >= "2021-01-01"].sort_index()

    except Exception as e:
        logger.error(f"MAÁTria features: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ENHANCED FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# Features adicionais ao FEATURE_COLS base
ENHANCED_FEATURE_COLS = list(BASE_FEATURE_COLS) + [
    # Erros do ONS adicionais (lag1 — sem data leakage)
    "erro_ear_seco_lag1",
    "erro_carga_sin_lag1",
    # Volatilidades (rolling 4 semanas)
    "vol_cmo_seco_4w",
    "vol_ena_seco_4w",
    # Features MAÁTria (lag1 — valores da semana anterior)
    "gfom_pct_sem_lag1",
    "curtail_total_sem_lag1",
    "disp_hydro_sem_lag1",
    "renov_avg_sem_lag1",
    # Ratio renovável / carga
    "renov_carga_ratio",
    # ── Autoregressivas (lag do próprio erro — autocorr=0.874!) ──────
    # NÃO é leakage: no momento t, sabemos o erro em t-1, t-2, t-4.
    # O modelo prevê erro em t+H, usando o padrão de erros passados.
    "ar_erro_pld_seco_lag1",
    "ar_erro_pld_seco_lag2",
    "ar_erro_pld_seco_lag4",
    "ar_erro_pld_seco_ma4",    # média móvel 4 semanas
    "ar_erro_cmo_seco_lag1",
    "ar_erro_cmo_seco_lag2",
]


def build_enhanced_dataset(
    pmo_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    maatria_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói dataset de treino COMPLETO:
    1. Base: features PMO + actuals (via pld_forecast_engine)
    2. Enhanced: features MAÁTria + erros adicionais + volatilidades
    3. Target: erro_pld = pld_real − cmo_previsto (preferido)
    
    NENHUMA feature usa dados do período corrente (tudo lag ≥ 1).
    """
    # ── Passo 1: dataset base do pld_forecast_engine ─────────────────────
    if _BASE_ENGINE_OK:
        df = build_training_dataset(pmo_df, actuals_df)
    else:
        df = pmo_df.copy()
        if not actuals_df.empty:
            df = df.join(actuals_df, how="left", rsuffix="_act")

    if df.empty:
        return df

    # ── Passo 2: juntar features MAÁTria ─────────────────────────────────
    if not maatria_df.empty:
        # Alinhar por segunda-feira
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        if not isinstance(maatria_df.index, pd.DatetimeIndex):
            maatria_df.index = pd.to_datetime(maatria_df.index, errors="coerce")

        for col in maatria_df.columns:
            if col not in df.columns:
                df[col] = maatria_df[col].reindex(df.index, method="nearest",
                                                   tolerance=pd.Timedelta("3D"))

    # ── Passo 3: erros adicionais do ONS (SEMPRE com lag1) ──────────────
    # erro_ear: quanto a previsão de EAR errou na semana passada
    for sub in SUBSISTEMAS:
        ep = f"ear_init_{sub}"
        er = f"ear_real_{sub}"
        if ep in df.columns and er in df.columns:
            df[f"erro_ear_{sub}_lag1"] = (
                df[ep].shift(1) - df[er].shift(1)
            )

    # erro_carga: quanto a previsão de carga errou na semana passada
    if "carga_sem1_sin" in df.columns and "carga_real_sin" in df.columns:
        df["erro_carga_sin_lag1"] = (
            df["carga_sem1_sin"].shift(1) - df["carga_real_sin"].shift(1)
        )
    else:
        df["erro_carga_sin_lag1"] = np.nan

    # ── Passo 4: volatilidades rolling (4 semanas) ──────────────────────
    for sub in SUBSISTEMAS:
        cmo_col = f"cmo_real_{sub}"
        if cmo_col in df.columns:
            df[f"vol_cmo_{sub}_4w"] = (
                df[cmo_col].rolling(4, min_periods=2).std()
            )

        ena_col = f"ena_real_{sub}_mlt"
        if ena_col in df.columns:
            df[f"vol_ena_{sub}_4w"] = (
                df[ena_col].rolling(4, min_periods=2).std()
            )

    # ── Passo 5: features MAÁTria com lag1 (sem data leakage) ──────────
    for col in ["gfom_pct_sem", "curtail_total_sem",
                "disp_hydro_sem", "renov_avg_sem"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # ── Passo 6: ratio renovável / carga ────────────────────────────────
    if "renov_avg_sem" in df.columns and "carga_real_sin" in df.columns:
        df["renov_carga_ratio"] = (
            df["renov_avg_sem_lag1"].fillna(0) /
            df["carga_real_sin"].shift(1).replace(0, np.nan)
        ).clip(0, 1)
    else:
        df["renov_carga_ratio"] = np.nan

    # ── Passo 7: targets — múltiplas formulações ────────────────────────
    # PROBLEMA IDENTIFICADO: AVG(PLD horário) vs CMO semanal é injusto.
    # PLD horário tem picos extremos que inflam a média.
    # O ONS prevê uma tendência central semanal, não picos horários.
    for sub in SUBSISTEMAS:
        cmo_prev = f"cmo_med_{sub}"     # CMO previsto pelo ONS (semanal)
        pld_mean = f"pld_real_{sub}"     # AVG(PLD horário) — enviesado por picos
        pld_p50  = f"pld_real_{sub}_p50" # MEDIANA(PLD horário) — robusto
        cmo_real = f"cmo_real_{sub}"     # AVG(CMO real semi-horário)
        cmo_real_p50 = f"cmo_real_{sub}_p50" # MEDIANA(CMO real)

        if cmo_prev in df.columns:
            # Target A: erro com PLD médio (ORIGINAL — enviesado)
            if pld_mean in df.columns:
                df[f"erro_pld_mean_{sub}"] = df[pld_mean] - df[cmo_prev]

            # Target B: erro com PLD mediano (RECOMENDADO — robusto)
            if pld_p50 in df.columns:
                df[f"erro_pld_{sub}"] = df[pld_p50] - df[cmo_prev]
            elif pld_mean in df.columns:
                # Fallback: se mediana não disponível, usar média
                df[f"erro_pld_{sub}"] = df[pld_mean] - df[cmo_prev]

            # Target C: erro CMO puro (MAIS PREVISÍVEL — apples vs apples)
            if cmo_real in df.columns:
                df[f"erro_cmo_{sub}"] = df[cmo_real] - df[cmo_prev]
            if cmo_real_p50 in df.columns:
                df[f"erro_cmo_p50_{sub}"] = df[cmo_real_p50] - df[cmo_prev]

    # ── Passo 8: features AUTOREGRESSIVAS (o preditor mais forte) ──────
    # Autocorr lag1 = 0.874 → o erro da semana passada é o melhor preditor
    # do erro futuro. NÃO é leakage: shift(1) usa apenas dados passados.
    for sub in SUBSISTEMAS:
        ecol = f"erro_pld_{sub}"
        if ecol in df.columns:
            df[f"ar_erro_pld_{sub}_lag1"] = df[ecol].shift(1)
            df[f"ar_erro_pld_{sub}_lag2"] = df[ecol].shift(2)
            df[f"ar_erro_pld_{sub}_lag4"] = df[ecol].shift(4)
            df[f"ar_erro_pld_{sub}_ma4"]  = df[ecol].rolling(4, min_periods=2).mean().shift(1)

        ecmo = f"erro_cmo_{sub}"
        if ecmo in df.columns:
            df[f"ar_erro_cmo_{sub}_lag1"] = df[ecmo].shift(1)
            df[f"ar_erro_cmo_{sub}_lag2"] = df[ecmo].shift(2)

    # Diagnóstico de targets
    for sub in ["seco"]:
        for tgt in [f"erro_pld_mean_{sub}", f"erro_pld_{sub}", f"erro_cmo_{sub}"]:
            if tgt in df.columns:
                s = df[tgt].dropna()
                if not s.empty:
                    print(f"    Target {tgt}: média={s.mean():.1f} std={s.std():.1f} "
                          f"mediana={s.median():.1f} |outliers 3σ|={((s-s.mean()).abs()>3*s.std()).sum()}")

    n_feats = sum(1 for c in ENHANCED_FEATURE_COLS if c in df.columns)
    n_base = sum(1 for c in BASE_FEATURE_COLS if c in df.columns)
    print(f"  Enhanced dataset: {len(df)} semanas | "
          f"{n_base} base + {n_feats - n_base} novas features | "
          f"{df.shape[1]} colunas total")

    return df.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ERROR-BASED MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_error_model(
    df: pd.DataFrame,
    subsistema: str = "seco",
    horizon_weeks: int = 4,
) -> Optional[Dict]:
    """
    Treina modelo GBM quantílico com TARGET = erro_pld (preferido).
    
    Target: erro_pld_{sub} = pld_real − cmo_previsto
    Predição: PLD_predito_ML = cmo_previsto + predicted_error
    
    Vantagem sobre target direto (pld_real):
    - O modelo precisa aprender apenas o DESVIO, não o nível absoluto
    - CMO do ONS já é um bom estimador (R²=0.88) — o modelo refina
    - Menor variância, menor risco de overfitting
    
    Split temporal 80/20, sem shuffle. TimeSeriesSplit para métricas.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("  pip install scikit-learn")
        return None

    # Target: erro_pld shifted H semanas à frente
    erro_col = f"erro_pld_{subsistema}"
    pld_col = f"pld_real_{subsistema}"

    # Fallback: se erro_pld não existe, calcular
    if erro_col not in df.columns:
        cmo_col = f"cmo_med_{subsistema}"
        if pld_col in df.columns and cmo_col in df.columns:
            df[erro_col] = df[pld_col] - df[cmo_col]
        else:
            return None

    if df[erro_col].notna().sum() < 52:
        return None

    # Features: usar enhanced se disponíveis, senão base
    available = [c for c in ENHANCED_FEATURE_COLS if c in df.columns]
    if not available:
        available = [c for c in BASE_FEATURE_COLS if c in df.columns]
    if not available:
        return None

    # Target shifted H semanas
    y = df[erro_col].shift(-horizon_weeks)
    mask = y.notna()
    X = df[available][mask].copy()
    y = y[mask]

    # Limpar NaN nas features (fill forward + 0)
    X = X.ffill().fillna(0)

    if len(X) < 52:
        return None

    # ── Split temporal 80/20 ─────────────────────────────────────────────
    n_tr = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr, y_te = y.iloc[:n_tr], y.iloc[n_tr:]

    # ── TimeSeriesSplit cross-validation para métricas robustas ──────────
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, n_tr // 26)))
    cv_scores = []

    # ── Treinar modelos quantílicos ──────────────────────────────────────
    models = {}
    metrics = {}

    for q in QUANTILES:
        gb = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=400, max_depth=4,
            learning_rate=0.035, min_samples_leaf=8,
            l2_regularization=0.15, random_state=42,
        )
        gb.fit(X_tr.values, y_tr.values)
        y_pred = gb.predict(X_te.values)

        if q == 0.50:
            metrics["mae_erro"] = round(float(np.mean(np.abs(y_pred - y_te.values))), 2)
            metrics["r2_test"]  = round(float(r2_score(y_te.values, y_pred)), 3)

            # Cross-val R² no treino
            for tr_idx, val_idx in tscv.split(X_tr):
                gb_cv = HistGradientBoostingRegressor(
                    loss="quantile", quantile=0.50,
                    max_iter=300, max_depth=4,
                    learning_rate=0.035, min_samples_leaf=8,
                    l2_regularization=0.15, random_state=42,
                )
                gb_cv.fit(X_tr.values[tr_idx], y_tr.values[tr_idx])
                cv_pred = gb_cv.predict(X_tr.values[val_idx])
                cv_scores.append(r2_score(y_tr.values[val_idx], cv_pred))

            metrics["r2_cv_mean"] = round(float(np.mean(cv_scores)), 3) if cv_scores else None
            metrics["r2_cv_std"]  = round(float(np.std(cv_scores)), 3)  if cv_scores else None

            # PLD-level metrics (reconstruct: PLD = CMO + erro)
            cmo_col = f"cmo_med_{subsistema}"
            if cmo_col in df.columns:
                cmo_te = df[cmo_col].shift(-horizon_weeks).iloc[n_tr:len(X)]
                cmo_te = cmo_te[cmo_te.notna()].values[:len(y_pred)]
                if len(cmo_te) == len(y_pred):
                    pld_pred = cmo_te + y_pred
                    pld_real = cmo_te + y_te.values[:len(cmo_te)]
                    metrics["mae_pld"]  = round(float(np.mean(np.abs(pld_pred - pld_real))), 2)
                    metrics["mape_pld"] = round(float(
                        np.nanmean(np.abs((pld_pred - pld_real) /
                                          np.where(pld_real == 0, np.nan, pld_real))) * 100
                    ), 1)

        metrics[f"coverage_p{int(q*100)}"] = round(float(np.mean(y_te.values <= y_pred)), 3)
        models[q] = gb

    # ── Feature importance ───────────────────────────────────────────────
    gb50 = models[0.50]
    if hasattr(gb50, "feature_importances_"):
        fi = dict(sorted(
            zip(available, gb50.feature_importances_.tolist()),
            key=lambda x: -x[1]
        ))
    else:
        fi = {}

    lbl = SUB_LABEL.get(subsistema, subsistema)
    print(f"  {lbl} h={horizon_weeks}w: "
          f"MAE_erro={metrics.get('mae_erro')} "
          f"R²_test={metrics.get('r2_test')} "
          f"R²_cv={metrics.get('r2_cv_mean')} "
          f"MAE_PLD={metrics.get('mae_pld')} "
          f"n={n_tr}+{len(X_te)}")

    return {
        "models":           models,
        "feature_cols":     available,
        "metrics":          metrics,
        "feature_importance": fi,
        "n_train":          n_tr,
        "n_test":           len(X_te),
        "subsistema":       subsistema,
        "horizon_weeks":    horizon_weeks,
        "target":           "erro_pld",
        "trained_at":       datetime.now().isoformat(),
    }


def train_all_error_models(df: pd.DataFrame) -> Dict:
    """Treina modelos erro-based para todos os subsistemas e horizontes."""
    all_models = {}
    for sub in SUBSISTEMAS:
        all_models[sub] = {}
        for h in [4, 8, 12, 26]:
            r = train_error_model(df, subsistema=sub, horizon_weeks=h)
            if r:
                all_models[sub][h] = r
    return all_models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ML PREDICTION (error-based → PLD)
# ══════════════════════════════════════════════════════════════════════════════

def predict_pld_ml(
    features_now: pd.Series,
    all_models: Dict,
    cmo_by_sub: Dict[str, float],
) -> Dict[str, Dict]:
    """
    PLD_predito_ML = cmo_previsto + predicted_error
    
    Retorna por subsistema e horizonte:
      {sub: {h: {"p10": x, "p50": x, "p90": x}}}
    """
    results = {}
    for sub in SUBSISTEMAS:
        results[sub] = {}
        if sub not in all_models:
            continue

        cmo = cmo_by_sub.get(sub)
        if cmo is None:
            continue

        for h, bundle in all_models[sub].items():
            X = np.array([
                float(features_now.get(c, 0) if pd.notna(features_now.get(c)) else 0)
                for c in bundle["feature_cols"]
            ]).reshape(1, -1)

            preds = {}
            for q, gb in bundle["models"].items():
                erro_pred = float(gb.predict(X)[0])
                pld_pred = max(0.0, cmo + erro_pred)
                preds[q] = round(pld_pred, 2)

            results[sub][h] = {
                "pld_p10": preds.get(0.10),
                "pld_p50": preds.get(0.50),
                "pld_p90": preds.get(0.90),
                "erro_p50": round(preds.get(0.50, 0) - cmo, 2),
                "cmo_base": cmo,
            }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PJE INTEGRATION (bridge to premium_engine)
# ══════════════════════════════════════════════════════════════════════════════

def get_pje_context(hourly_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Extrai outputs do Price Justification Engine do DataFrame horário.
    Usa as últimas 720h (30 dias) para estabilidade.
    
    Returns:
        pld_fundamental: estimativa estrutural de preço (mediana CMO ponderado)
        p_justification: probabilidade de justificação (0–1)
        spdi: Structural Price Distortion Index
        imr: Infra-Marginal Rent médio
        structural_gap: gap estrutural R$/MWh
    """
    if hourly_df is None or hourly_df.empty:
        return {}

    last = hourly_df.last("720h") if len(hourly_df) > 720 else hourly_df

    def _med(col):
        s = pd.to_numeric(last.get(col, pd.Series(dtype=float)), errors="coerce")
        return float(s.median()) if s.notna().any() else None

    def _last(col):
        s = pd.to_numeric(last.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        return float(s.iloc[-1]) if not s.empty else None

    spdi = _last("spdi")
    drift = _last("structural_drift")
    gap = _med("Structural_gap_R$/MWh")
    imr = _med("infra_marginal_rent")

    # PLD fundamental = CMO dominante mediano (estimativa estrutural do sistema)
    pld_fund = _med("cmo_dominante")
    pld_obs = _med("pld")

    # P_justification simplificado:
    # Se SPDI ~ 1.0 → preço justificado (alta confiança no estrutural)
    # Se SPDI muito diferente de 1.0 → preço distorcido (menor confiança)
    p_just = None
    if spdi is not None:
        # Transformação: SPDI=1.0 → p=0.8, SPDI=0.5 ou 1.5 → p=0.3
        p_just = max(0.0, min(1.0, 1.0 - abs(spdi - 1.0) * 1.5))

    return {
        "pld_fundamental":    pld_fund,
        "p_justification":    p_just,
        "spdi":               spdi,
        "structural_drift":   drift,
        "structural_gap":     gap,
        "imr":                imr,
        "pld_observado":      pld_obs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — META-MODEL (combinação ponderada dinâmica)
# ══════════════════════════════════════════════════════════════════════════════

def meta_combine(
    pld_ml: float,
    pld_fundamental: float,
    p_justification: float,
    horizon_weeks: int = 4,
) -> Tuple[float, float, float]:
    """
    Combina previsão ML com previsão estrutural (PJE).
    
    Lógica de ponderação:
    - P_justification alta (>0.7): confia mais no estrutural
    - P_justification baixa (<0.4): confia mais no ML
    - Entre: pesos iguais
    
    Horizonte longo: mais peso no ML (estrutural perde precisão com distância).
    
    Returns:
        (pld_final, w_ml, w_pje)
    """
    if p_justification > 0.7:
        w_pje = 0.7
        w_ml = 0.3
    elif p_justification < 0.4:
        w_pje = 0.3
        w_ml = 0.7
    else:
        w_pje = 0.5
        w_ml = 0.5

    # Horizonte longo: deslocar peso para ML
    # (PJE reflete condição atual; ML projeta tendência)
    if horizon_weeks >= 12:
        shift = min(0.15, (horizon_weeks - 12) * 0.01)
        w_ml += shift
        w_pje -= shift

    # Normalizar
    total = w_ml + w_pje
    w_ml /= total
    w_pje /= total

    pld_final = round(w_ml * pld_ml + w_pje * pld_fundamental, 2)
    return pld_final, round(w_ml, 3), round(w_pje, 3)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MARKET REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(spdi: Optional[float]) -> str:
    """
    Classifica o regime de mercado com base no SPDI.
    
    SPDI > 1.4  → "inflated"    (preço acima do custo físico)
    SPDI < 0.9  → "compressed"  (preço abaixo do custo físico)
    else        → "balanced"    (equilíbrio)
    """
    if spdi is None:
        return "unknown"
    if spdi > 1.4:
        return "inflated"
    elif spdi < 0.9:
        return "compressed"
    return "balanced"


REGIME_LABELS = {
    "inflated":   "Preço Inflado",
    "compressed": "Preço Comprimido",
    "balanced":   "Equilíbrio",
    "unknown":    "Sem Dados",
}

REGIME_EMOJI = {
    "inflated": "🔴", "compressed": "🟢",
    "balanced": "🟡", "unknown":    "⚪",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — CONFIDENCE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(
    p_justification: Optional[float],
    pld_ml: float,
    pld_fundamental: float,
    model_r2: Optional[float] = None,
) -> float:
    """
    Score de confiança normalizado [0, 1]:
    
    50% — P_justification (quanto o preço é estruturalmente justificado)
    30% — concordância ML vs PJE (quanto mais próximos, mais confiável)
    20% — R² do modelo ML no validation set
    """
    # Componente 1: P_justification
    c1 = p_justification if p_justification is not None else 0.5

    # Componente 2: concordância ML vs PJE
    if pld_fundamental and pld_fundamental > 0:
        divergence = abs(pld_ml - pld_fundamental) / pld_fundamental
        c2 = max(0.0, 1.0 - divergence)
    else:
        c2 = 0.5

    # Componente 3: qualidade do modelo
    c3 = max(0.0, min(1.0, model_r2)) if model_r2 is not None else 0.5

    confidence = 0.50 * c1 + 0.30 * c2 + 0.20 * c3
    return round(max(0.0, min(1.0, confidence)), 3)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def save_meta_models(all_models: Dict, meta_dir: Path = META_DIR) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for sub, horizons in all_models.items():
        for h, bundle in horizons.items():
            with open(meta_dir / f"meta_{sub}_h{h}.pkl", "wb") as f:
                pickle.dump(bundle, f)

    meta = {
        sub: {
            str(h): {k: v for k, v in b.items() if k != "models"}
            for h, b in hh.items()
        }
        for sub, hh in all_models.items()
    }
    with open(meta_dir / "meta_models_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Meta-modelos salvos em {meta_dir}")


def load_meta_models(meta_dir: Path = META_DIR) -> Dict:
    all_models = {}
    for sub in SUBSISTEMAS:
        all_models[sub] = {}
        for h in [4, 8, 12, 26]:
            p = meta_dir / f"meta_{sub}_h{h}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    all_models[sub][h] = pickle.load(f)
    return all_models


def build_and_train_hybrid(
    xlsx_path: Path = PMO_XLSX,
    model_dir: Path = MODEL_DIR,
    data_dir:  Path = DATA_DIR,
) -> Dict:
    """
    Pipeline completo de build:
    1. Carrega features PMO
    2. Carrega actuals (DuckDB → Neon → cache)
    3. Carrega features MAÁTria (DuckDB)
    4. Constrói dataset enhanced
    5. Treina modelos erro-based
    6. Salva modelos e metadata
    """
    print("\n" + "=" * 60)
    print("MAÁTria Energia · Hybrid PLD Forecast Engine · Build")
    print("=" * 60)

    if not _BASE_ENGINE_OK:
        print("ERRO: pld_forecast_engine.py não encontrado")
        return {}

    print("\n[1/5] Carregando features do PMO...")
    pmo_df = load_pmo_features(xlsx_path)

    print("\n[2/5] Carregando actuals (local → Neon → cache)...")
    actuals_df = load_actuals(data_dir)

    print("\n[3/5] Carregando features MAÁTria (DuckDB)...")
    maatria_df = load_maatria_weekly_features(data_dir)
    if not maatria_df.empty:
        print(f"  MAÁTria: {len(maatria_df)} semanas | {maatria_df.shape[1]} features")
    else:
        print("  MAÁTria: sem dados (GFOM/curtailment/disp indisponíveis)")

    print("\n[4/5] Construindo dataset enhanced...")
    df = build_enhanced_dataset(pmo_df, actuals_df, maatria_df)

    # Salvar dataset para diagnóstico
    meta_dir = model_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_dir / "enhanced_dataset.csv")
    df.to_pickle(meta_dir / "enhanced_dataset.pkl")
    print(f"  Dataset salvo em {meta_dir}/")

    print("\n[5/5] Treinando modelos GBM (target: erro_pld)...")
    all_models = train_all_error_models(df)
    save_meta_models(all_models, meta_dir)

    n = sum(len(v) for v in all_models.values())
    print(f"\n✅ Build: {n} modelos erro-based "
          f"({len(SUBSISTEMAS)} subs × {n // max(len(SUBSISTEMAS), 1)} horizontes)")

    return all_models


def run_hybrid_forecast(
    hourly_df: Optional[pd.DataFrame] = None,
    meta_models: Optional[Dict] = None,
    base_models: Optional[Dict] = None,
    xlsx_path: Path = PMO_XLSX,
    data_dir: Path = DATA_DIR,
) -> Dict:
    """
    Pipeline completo de previsão híbrida:
    1. Curto prazo: CMO ONS + bandas calibradas (pld_forecast_engine)
    2. Médio/longo prazo ML: modelos erro-based
    3. PJE context: SPDI, IMR, P_justification do hourly_df
    4. Meta-model: combinação ponderada
    5. Confidence score + regime
    """
    if not _BASE_ENGINE_OK:
        return {"erro": "pld_forecast_engine.py não encontrado"}

    # ── Carregar modelos ─────────────────────────────────────────────────
    if meta_models is None:
        meta_models = load_meta_models()
    if base_models is None:
        base_models = load_base_models(MODEL_DIR)

    has_meta = any(meta_models.get(s) for s in SUBSISTEMAS)
    has_base = any(base_models.get(s) for s in SUBSISTEMAS)

    # ── PMO state ────────────────────────────────────────────────────────
    pmo_state = get_latest_pmo_state(xlsx_path)
    ena_mlt = pmo_state.get("_ena_mlt_by_sub")
    erro_lag1 = _compute_erro_lag1(data_dir)

    # ── Curto prazo (inalterado — bandas calibradas são excelentes) ──────
    short = forecast_short_term(pmo_state,
                                 ena_mlt_by_sub=ena_mlt,
                                 erro_lag1_by_sub=erro_lag1)

    # ── PJE context ──────────────────────────────────────────────────────
    pje = get_pje_context(hourly_df)
    regime = classify_regime(pje.get("spdi"))

    # ── Features atuais para médio/longo prazo ───────────────────────────
    pmo_df = load_pmo_features(xlsx_path)
    if not pmo_df.empty:
        valid = pmo_df[pmo_df.get("cmo_med_seco", pd.Series(dtype=float)).notna()]
        base = valid.iloc[-1].copy() if not valid.empty else pmo_df.iloc[-1].copy()
    else:
        base = pd.Series(dtype=float)

    tmp = base.to_frame().T.copy()
    _build_derived_features_only(tmp)
    features_now = tmp.iloc[0].copy()

    # Sobrescrever com valores frescos do PMO
    sub_map_xls = {"seco": "SE/CO", "s": "Sul", "ne": "NE", "n": "Norte"}
    cmo_by_sub = {}
    for sub, lbl in sub_map_xls.items():
        v = pmo_state.get(f"{lbl} Med.Sem. (R$/MWh)")
        if v is not None:
            features_now[f"cmo_med_{sub}"] = float(v)
            cmo_by_sub[sub] = float(v)
        if ena_mlt and ena_mlt.get(sub) is not None:
            features_now[f"ena_prev_{sub}_mlt"] = float(ena_mlt[sub])

    # Features MAÁTria recentes (lag1 da última semana)
    maatria = load_maatria_weekly_features(data_dir)
    if not maatria.empty:
        last_week = maatria.iloc[-1]
        for col in ["gfom_pct_sem", "curtail_total_sem",
                     "disp_hydro_sem", "renov_avg_sem"]:
            if col in last_week.index and pd.notna(last_week[col]):
                features_now[f"{col}_lag1"] = float(last_week[col])

    # Erro lag1 como feature
    if erro_lag1 and erro_lag1.get("seco") is not None:
        features_now["erro_ons_seco"] = float(erro_lag1["seco"])
        features_now["cmo_erro_seco_lag1"] = float(erro_lag1["seco"])

    # Sazonalidade
    mes_atual = datetime.now().month
    features_now["mes_sin"] = np.sin(2 * np.pi * mes_atual / 12)
    features_now["mes_cos"] = np.cos(2 * np.pi * mes_atual / 12)

    # Regime econômico
    if pje.get("spdi") is not None:
        features_now["spdi"] = pje["spdi"]
    if pje.get("structural_drift") is not None:
        features_now["structural_drift"] = pje["structural_drift"]

    # ── ML predictions (error-based) ─────────────────────────────────────
    ml_preds = {}
    if has_meta:
        ml_preds = predict_pld_ml(features_now, meta_models, cmo_by_sub)

    # ── Base model predictions (fallback) ────────────────────────────────
    base_preds = {}
    if has_base:
        base_ml = forecast_medium_long_term(features_now, base_models,
                                             pje if pje else None)
        base_preds = base_ml.get("subsistemas", {})

    # ── Meta-model: combinar ML + PJE ────────────────────────────────────
    horizon_labels = {4: "4 semanas", 8: "8 semanas",
                      12: "3 meses", 26: "6 meses"}

    combined_results = {}
    for sub in SUBSISTEMAS:
        combined_results[sub] = {}

        pld_fundamental = pje.get("pld_fundamental")
        p_just = pje.get("p_justification", 0.5)

        for h in [4, 8, 12, 26]:
            # PLD ML (preferir meta, fallback base)
            ml_h = ml_preds.get(sub, {}).get(h, {})
            base_h = base_preds.get(sub, {}).get(h, {})

            pld_ml_p50 = ml_h.get("pld_p50") or base_h.get("pld_p50")
            pld_ml_p10 = ml_h.get("pld_p10") or base_h.get("pld_p10")
            pld_ml_p90 = ml_h.get("pld_p90") or base_h.get("pld_p90")

            if pld_ml_p50 is None:
                continue

            # Meta-combine
            if pld_fundamental and p_just is not None:
                pld_final, w_ml, w_pje = meta_combine(
                    pld_ml_p50, pld_fundamental, p_just, h)
            else:
                pld_final = pld_ml_p50
                w_ml, w_pje = 1.0, 0.0

            # Model R² para confidence
            model_r2 = None
            bundle = meta_models.get(sub, {}).get(h)
            if bundle:
                model_r2 = bundle.get("metrics", {}).get("r2_test")
            if model_r2 is None:
                bundle_b = base_models.get(sub, {}).get(h)
                if bundle_b and isinstance(bundle_b, dict):
                    model_r2 = bundle_b.get("metrics", {}).get("r2_test")

            # Confidence
            confidence = compute_confidence(
                p_just, pld_ml_p50,
                pld_fundamental if pld_fundamental else pld_ml_p50,
                model_r2,
            )

            combined_results[sub][h] = {
                "horizonte_semanas": h,
                "horizonte_label":   horizon_labels.get(h, f"{h}w"),
                "pld_ml_p10":        pld_ml_p10,
                "pld_ml_p50":        pld_ml_p50,
                "pld_ml_p90":        pld_ml_p90,
                "pld_fundamental":   pld_fundamental,
                "pld_final":         pld_final,
                "w_ml":              w_ml,
                "w_pje":             w_pje,
                "confidence":        confidence,
                "regime":            regime,
                "regime_label":      REGIME_LABELS.get(regime, regime),
                "erro_p50":          ml_h.get("erro_p50"),
                "cmo_base":          ml_h.get("cmo_base") or cmo_by_sub.get(sub),
            }

    return {
        "timestamp":        datetime.now().isoformat(),
        "semana_pmo":       str(pmo_state.get("Semana início", "")),
        "regime":           regime,
        "regime_label":     REGIME_LABELS.get(regime, regime),
        "pje_context":      pje,
        "short_term":       short,
        "hybrid_forecast":  combined_results,
        "subsistemas":      SUB_LABEL,
        "model_type":       "error-based" if has_meta else "base",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — STREAMLIT VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def render_hybrid_forecast_tab(
    hourly_df: pd.DataFrame,
    meta_models: Optional[Dict] = None,
    base_models: Optional[Dict] = None,
) -> None:
    """
    Tab Streamlit completa: Previsão Híbrida de PLD.
    
    Plots:
    1. ML vs PJE vs Final (barras por horizonte)
    2. Bandas de confiança (curto prazo)
    3. Feature importance
    4. Regime e confidence dashboard
    """
    import streamlit as st
    import plotly.graph_objects as go

    st.markdown("## 🔮 Previsão Híbrida de PLD")

    with st.spinner("Calculando previsão híbrida..."):
        forecast = run_hybrid_forecast(
            hourly_df=hourly_df,
            meta_models=meta_models,
            base_models=base_models,
        )

    if "erro" in forecast:
        st.error(f"Erro: {forecast['erro']}")
        return

    pje = forecast.get("pje_context", {})
    regime = forecast.get("regime", "unknown")

    # ── KPIs de regime ───────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Regime de Mercado",
              f"{REGIME_EMOJI.get(regime, '')} {REGIME_LABELS.get(regime, regime)}",
              f"SPDI {pje.get('spdi', 0):.2f}×" if pje.get("spdi") else "—")

    k2.metric("PLD Fundamental",
              f"R${pje.get('pld_fundamental', 0):.0f}/MWh"
              if pje.get("pld_fundamental") else "N/D")

    k3.metric("P(Justificação)",
              f"{pje.get('p_justification', 0):.0%}"
              if pje.get("p_justification") is not None else "N/D")

    k4.metric("IMR Médio",
              f"R${pje.get('imr', 0):,.0f}/h"
              if pje.get("imr") is not None else "N/D")

    st.markdown("---")

    # ── Tabs de visualização ─────────────────────────────────────────────
    tab_short, tab_ml, tab_fi, tab_diag = st.tabs([
        "⚡ Curto prazo (1–5 sem.)",
        "📈 ML vs PJE vs Final",
        "🎯 Feature importance",
        "⚙️ Diagnóstico",
    ])

    # ── Tab 1: Curto prazo (inalterado — bandas calibradas) ──────────────
    with tab_short:
        short = forecast.get("short_term", {})
        semanas = short.get("semanas", [])
        st.caption(f"CMO ONS + bandas calibradas — PMO ref.: {forecast.get('semana_pmo', '?')}")

        if not semanas:
            st.info("Dados de curto prazo não disponíveis.")
        else:
            for sub in SUBSISTEMAS:
                lbl = SUB_LABEL[sub]
                rows = [w for w in semanas if w.get(f"pld_p50_{sub}") is not None]
                if not rows:
                    continue
                sems = [w["semana"] for w in rows]
                p10s = [w[f"pld_p10_{sub}"] for w in rows]
                p50s = [w[f"pld_p50_{sub}"] for w in rows]
                p90s = [w[f"pld_p90_{sub}"] for w in rows]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=sems, y=[p90 - p10 for p10, p90 in zip(p10s, p90s)],
                    base=p10s, name="Banda P10–P90",
                    marker_color="#1e3a5f", opacity=0.55,
                ))
                fig.add_trace(go.Scatter(
                    x=sems, y=p50s, name="P50",
                    mode="lines+markers",
                    line=dict(color="#c8a44d", width=2),
                    marker=dict(size=8),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0f14", plot_bgcolor="#111827",
                    height=260, margin=dict(l=40, r=20, t=40, b=30),
                    title=dict(text=f"{lbl} — PLD curto prazo (R$/MWh)",
                               font=dict(size=12, color="#e5e7eb")),
                    yaxis=dict(title="R$/MWh", gridcolor="#1f2937"),
                    xaxis=dict(gridcolor="#1f2937"),
                    legend=dict(orientation="h", y=1.12),
                    barmode="overlay",
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"hybrid_short_{sub}")

    # ── Tab 2: ML vs PJE vs Final ────────────────────────────────────────
    with tab_ml:
        hybrid = forecast.get("hybrid_forecast", {})
        pld_fund = pje.get("pld_fundamental")

        if not any(hybrid.values()):
            st.warning("Modelos não treinados. Execute:")
            st.code("python meta_forecast_engine.py --build", language="bash")
        else:
            for sub in SUBSISTEMAS:
                lbl = SUB_LABEL[sub]
                sub_data = hybrid.get(sub, {})
                if not sub_data:
                    continue

                horizons = sorted(sub_data.keys())
                hlabels = [sub_data[h]["horizonte_label"] for h in horizons]
                ml_vals = [sub_data[h]["pld_ml_p50"] for h in horizons]
                final_vals = [sub_data[h]["pld_final"] for h in horizons]
                confs = [sub_data[h]["confidence"] for h in horizons]

                fig = go.Figure()

                # Banda P10-P90 do ML
                p10s = [sub_data[h].get("pld_ml_p10", 0) for h in horizons]
                p90s = [sub_data[h].get("pld_ml_p90", 0) for h in horizons]
                fig.add_trace(go.Scatter(
                    x=hlabels + hlabels[::-1],
                    y=[v or 0 for v in p90s] + [v or 0 for v in p10s[::-1]],
                    fill="toself", fillcolor="rgba(30,58,95,0.3)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Banda ML P10–P90",
                ))

                # ML P50
                fig.add_trace(go.Scatter(
                    x=hlabels, y=ml_vals, name="ML (erro-based)",
                    mode="lines+markers",
                    line=dict(color="#60a5fa", width=2),
                    marker=dict(size=8),
                ))

                # PJE (linha horizontal)
                if pld_fund:
                    fig.add_trace(go.Scatter(
                        x=hlabels, y=[pld_fund] * len(hlabels),
                        name="PJE (fundamental)",
                        mode="lines",
                        line=dict(color="#34d399", width=2, dash="dash"),
                    ))

                # Final (combinado)
                fig.add_trace(go.Scatter(
                    x=hlabels, y=final_vals, name="Final (híbrido)",
                    mode="lines+markers",
                    line=dict(color="#c8a44d", width=3),
                    marker=dict(size=10, symbol="diamond"),
                ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0f14", plot_bgcolor="#111827",
                    height=300, margin=dict(l=40, r=20, t=40, b=30),
                    title=dict(text=f"{lbl} — ML vs PJE vs Final (R$/MWh)",
                               font=dict(size=12, color="#e5e7eb")),
                    yaxis=dict(title="R$/MWh", gridcolor="#1f2937"),
                    xaxis=dict(gridcolor="#1f2937"),
                    legend=dict(orientation="h", y=1.15),
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"hybrid_ml_{sub}")

                # Tabela de detalhes
                tbl = []
                for h in horizons:
                    d = sub_data[h]
                    tbl.append({
                        "Horizonte":   d["horizonte_label"],
                        "ML P50":      f"R${d['pld_ml_p50']:.0f}" if d['pld_ml_p50'] else "—",
                        "PJE":         f"R${pld_fund:.0f}" if pld_fund else "—",
                        "Final":       f"R${d['pld_final']:.0f}",
                        "Peso ML":     f"{d['w_ml']:.0%}",
                        "Peso PJE":    f"{d['w_pje']:.0%}",
                        "Confiança":   f"{d['confidence']:.0%}",
                        "Regime":      d.get("regime_label", "—"),
                    })
                if tbl:
                    st.dataframe(pd.DataFrame(tbl).set_index("Horizonte"),
                                 use_container_width=True)

    # ── Tab 3: Feature importance ────────────────────────────────────────
    with tab_fi:
        loaded = meta_models or load_meta_models()
        any_fi = False

        for sub in SUBSISTEMAS:
            for h in [4, 8]:  # tentar h=4 e h=8
                bundle = loaded.get(sub, {}).get(h)
                if not bundle:
                    continue
                fi = bundle.get("feature_importance", {})
                if not fi:
                    continue

                any_fi = True
                lbl = SUB_LABEL[sub]

                top = dict(list(fi.items())[:15])
                fig = go.Figure(go.Bar(
                    y=list(top.keys())[::-1],
                    x=list(top.values())[::-1],
                    orientation="h",
                    marker_color="#c8a44d",
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0b0f14", plot_bgcolor="#111827",
                    height=400, margin=dict(l=200, r=20, t=40, b=30),
                    title=dict(text=f"{lbl} — Top features (h={h}w)",
                               font=dict(size=12, color="#e5e7eb")),
                    xaxis=dict(title="Importância", gridcolor="#1f2937"),
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"hybrid_fi_{sub}_{h}")
                break  # só mostrar um horizonte por subsistema

        if not any_fi:
            # Diagnóstico: mostrar o que foi encontrado
            n_models = sum(len(v) for v in loaded.values() if isinstance(v, dict))
            if n_models > 0:
                st.warning(f"{n_models} modelos carregados mas sem feature importance.")
                st.caption("Isso pode acontecer se os modelos foram treinados com uma versão antiga. "
                           "Retreine com o botão na aba Diagnóstico.")
            else:
                st.info("Nenhum modelo encontrado. Execute no terminal:")
                st.code("python auto_tune_forecast.py", language="bash")

    # ── Tab 4: Diagnóstico ───────────────────────────────────────────────
    with tab_diag:
        st.markdown("### Fontes de dados")
        meta_meta_path = META_DIR / "meta_models_meta.json"
        if meta_meta_path.exists():
            with open(meta_meta_path) as f:
                meta_info = json.load(f)
            for sub, horizons in meta_info.items():
                lbl = SUB_LABEL.get(sub, sub)
                for h, info in horizons.items():
                    m = info.get("metrics", {})
                    st.text(
                        f"{lbl} h={h}w: "
                        f"MAE_erro={m.get('mae_erro')} "
                        f"R²={m.get('r2_test')} "
                        f"R²_cv={m.get('r2_cv_mean')} "
                        f"MAE_PLD={m.get('mae_pld')} "
                        f"n={info.get('n_train')}+{info.get('n_test')} "
                        f"target={info.get('target')}"
                    )
        else:
            st.info("Nenhum modelo meta treinado.")

        st.markdown("### Contexto PJE")
        for k, v in pje.items():
            if v is not None:
                st.text(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        st.markdown("### Configuração")
        st.text(f"  PMO: {PMO_XLSX}")
        st.text(f"  Models: {MODEL_DIR}")
        st.text(f"  Meta: {META_DIR}")
        st.text(f"  Data: {DATA_DIR}")
        st.text(f"  Engine base: {'✅' if _BASE_ENGINE_OK else '❌'}")
        st.text(f"  Meta models: {'✅' if any(load_meta_models().get(s) for s in SUBSISTEMAS) else '❌'}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(
        description="MAÁTria · Hybrid PLD Forecast Engine (Meta-Model)")
    ap.add_argument("--build", action="store_true",
                    help="Constrói dataset enhanced e treina modelos erro-based")
    ap.add_argument("--forecast", action="store_true",
                    help="Gera previsão híbrida e imprime JSON")
    ap.add_argument("--diagnose", action="store_true",
                    help="Verifica fontes de dados")
    ap.add_argument("--xlsx", default=str(PMO_XLSX))
    ap.add_argument("--models", default=str(MODEL_DIR))
    ap.add_argument("--data", default=str(DATA_DIR))
    args = ap.parse_args()

    PMO_XLSX = Path(args.xlsx)
    MODEL_DIR = Path(args.models)
    DATA_DIR = Path(args.data)
    META_DIR = MODEL_DIR / "meta"
    META_DIR.mkdir(parents=True, exist_ok=True)

    if args.diagnose:
        print("\n=== Diagnóstico — Hybrid Forecast Engine ===")
        sources = {
            "PMO Excel":              PMO_XLSX.exists(),
            "DuckDB local":           (DATA_DIR / "kintuadi.duckdb").exists(),
            "Meta models":            (META_DIR / "meta_seco_h4.pkl").exists(),
            "Base models":            (MODEL_DIR / "gbm_seco_h4.pkl").exists(),
            "Enhanced dataset":       (META_DIR / "enhanced_dataset.csv").exists(),
            "Base engine importável": _BASE_ENGINE_OK,
            "Neon DATABASE_URL":      bool(os.getenv("DATABASE_URL", "")),
        }
        for label, ok in sources.items():
            print(f"  {'✅' if ok else '❌'} {label}")

        if (DATA_DIR / "kintuadi.duckdb").exists():
            print("\n=== Features MAÁTria disponíveis ===")
            maatria = load_maatria_weekly_features(DATA_DIR)
            if not maatria.empty:
                for col in maatria.columns:
                    n = maatria[col].notna().sum()
                    print(f"  {col}: {n} semanas")
            else:
                print("  Nenhuma feature MAÁTria encontrada")

    elif args.build:
        build_and_train_hybrid(PMO_XLSX, MODEL_DIR, DATA_DIR)

    elif args.forecast:
        result = run_hybrid_forecast(
            hourly_df=None,
            xlsx_path=PMO_XLSX,
            data_dir=DATA_DIR,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    else:
        ap.print_help()
