from __future__ import annotations

"""
Adaptive PLD Forward Engine.

Hybrid forward engine for Brazilian PLD that combines:
  1. Physical system simulation
  2. Monte Carlo paths
  3. Structural regime detection
  4. Regime-aware SPDI modeling
  5. SPDI trend vs shock decomposition
  6. Representative scenario extraction
  7. Forecast-error self-recalibration

The engine works on hourly data and enforces the structural relationship:

    PLD = custo_fisico * SPDI

`custo_fisico` is modeled as the structural system cost in R$/MWh, while SPDI
captures the market distortion regime around that structural cost.
"""

import argparse
import logging
import os
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

try:
    import duckdb
except ImportError:  # pragma: no cover - optional at import time
    duckdb = None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - optional at import time
    psycopg2 = None

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover - required at runtime
    rpt = None


warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)


LOGGER = logging.getLogger(__name__)
EPS = 1e-9
MODEL_VERSION = "adaptive_pld_forward_v4"

LOCAL_TABLE_NAME = "adaptive_pld_forward_hourly"
AUTH_TABLE_NAME = "maat_adaptive_pld_forward_hourly"
QUALITY_LOCAL_TABLE_NAME = "adaptive_pld_forward_quality_hourly"
QUALITY_AUTH_TABLE_NAME = "maat_adaptive_pld_forward_quality_hourly"
QUALITY_SUMMARY_LOCAL_TABLE_NAME = "adaptive_pld_forward_quality_summary"
QUALITY_SUMMARY_AUTH_TABLE_NAME = "maat_adaptive_pld_forward_quality_summary"

REGIME_LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}
REGIME_ID_TO_LABEL = {value: key for key, value in REGIME_LABEL_TO_ID.items()}
QUALITY_SUMMARY_WINDOWS = (30, 90, 180)


@dataclass
class ScenarioSummary:
    trajectory: list[float]
    final_value: float


@dataclass
class AdaptivePLDForwardResult:
    run_id: str
    generated_at: pd.Timestamp
    hourly_table: pd.DataFrame
    scenarios: Dict[str, ScenarioSummary]
    distribution: Dict[str, float]
    regime_probabilities: Dict[str, float]
    confidence_score: float
    current_era: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios": {
                name: {
                    "trajectory": summary.trajectory,
                    "final_value": summary.final_value,
                }
                for name, summary in self.scenarios.items()
            },
            "distribution": self.distribution,
            "regime_probabilities": self.regime_probabilities,
            "confidence_score": self.confidence_score,
            "current_era": self.current_era,
        }


class AdaptivePLDForwardEngine:
    """
    Production-grade PLD forward engine driven by physical state and SPDI.

    The engine expects hourly data with at least the physical state variables
    required to rebuild:
      - net_load
      - custo_fisico
      - SPDI
      - ISR
      - curtailment

    When some derived columns are missing, the engine rebuilds them from the
    project's existing economic decomposition logic.
    """

    regime_feature_cols = [
        "ear",
        "ena",
        "net_load",
        "thermal_share",
        "curtailment",
        "isr",
    ]

    dynamic_feature_cols = [
        "ear",
        "ena",
        "load",
        "solar",
        "wind",
        "net_load",
        "thermal_share",
        "curtailment_ratio",
        "isr",
        "month",
        "hour",
    ]

    def __init__(
        self,
        duckdb_path: str | Path = "data/kintuadi.duckdb",
        pmo_xlsx_path: str | Path = "data/ons/PMOs/validacao_pmo.xlsx",
        n_paths: int = 1000,
        horizon_hours: int = 24 * 30 * 6,
        random_state: int = 42,
        zscore_reversion_threshold: float = 2.0,
        shock_decay_factor: float = 0.65,
        spike_zscore_threshold: float = 1.1,
        spike_persistence: float = 0.58,
        pmo_guidance_hours: int = 24 * 7,
        recent_profile_hours: int = 24 * 14,
        pmo_guidance_weight: float = 0.82,
        spdi_recent_weight: float = 0.78,
        feedback_lookback_days: int = 180,
        feedback_half_life_days: int = 21,
        feedback_strength_max: float = 0.55,
        feedback_min_weight: float = 12.0,
        feedback_ratio_clip: float = 0.30,
        quality_lookback_days: int = 365,
        season_transition_days: int = 8,
        cost_anchor_weight: float = 0.22,
        cost_tail_clip: float = 1.18,
        # Ornstein-Uhlenbeck mean-reversion para custo_fisico
        # Evita divergência exponencial em horizontes longos
        ou_kappa: float = 0.004,        # velocidade de reversão por hora (~half-life 7 dias)
        ou_mu_quantile: float = 0.50,   # quantil histórico usado como nível de equilíbrio
        ou_long_run_clip: float = 3.0,  # clip máximo em relação à mu de longo prazo
        min_era_size_hours: int = 24 * 30,
        max_eras: int = 6,
    ) -> None:
        self.duckdb_path = Path(duckdb_path)
        self.pmo_xlsx_path = Path(pmo_xlsx_path)
        self.n_paths = int(n_paths)
        self.horizon_hours = int(horizon_hours)
        self.random_state = int(random_state)
        self.zscore_reversion_threshold = float(zscore_reversion_threshold)
        self.shock_decay_factor = float(shock_decay_factor)
        self.spike_zscore_threshold = float(spike_zscore_threshold)
        self.spike_persistence = float(spike_persistence)
        self.pmo_guidance_hours = int(pmo_guidance_hours)
        self.recent_profile_hours = int(recent_profile_hours)
        self.pmo_guidance_weight = float(pmo_guidance_weight)
        self.spdi_recent_weight = float(spdi_recent_weight)
        self.feedback_lookback_days = int(feedback_lookback_days)
        self.feedback_half_life_days = int(feedback_half_life_days)
        self.feedback_strength_max = float(feedback_strength_max)
        self.feedback_min_weight = float(feedback_min_weight)
        self.feedback_ratio_clip = float(feedback_ratio_clip)
        self.quality_lookback_days = int(quality_lookback_days)
        self.season_transition_days = int(season_transition_days)
        self.cost_anchor_weight = float(cost_anchor_weight)
        self.cost_tail_clip = float(cost_tail_clip)
        self.ou_kappa = float(ou_kappa)
        self.ou_mu_quantile = float(ou_mu_quantile)
        self.ou_long_run_clip = float(ou_long_run_clip)
        self.min_era_size_hours = int(min_era_size_hours)
        self.max_eras = int(max_eras)

        self.history_source_: str = "unknown"
        self.history_: pd.DataFrame | None = None
        self.prepared_history_: pd.DataFrame | None = None

        self.latest_era_id_: int = 0
        self.regime_classifiers_: Dict[int, Dict[str, Any]] = {}
        self.era_classifier_: Dict[str, Any] | None = None

        self.spdi_base_surface_: Dict[str, Any] = {}
        self.cost_surface_: Dict[str, Any] = {}
        self.shock_surface_: Dict[str, Any] = {}
        self.spike_surface_: Dict[str, Any] = {}
        self.climatology_: Dict[str, Any] = {}
        self.recent_profile_surface_: Dict[str, Any] = {}
        self.pmo_guidance_: Dict[str, Any] = {}
        self.feedback_surface_: Dict[str, Any] = {}
        self.model_bundles_: Dict[str, Dict[str, Any]] = {}
        self.feature_medians_: Dict[str, float] = {}
        self.bounds_: Dict[str, tuple[float, float]] = {}

    def load_hourly_history(self) -> pd.DataFrame:
        """
        Load the canonical hourly history.

        Priority:
          1. Local DuckDB `data/kintuadi.duckdb`
          2. Operational Neon PostgreSQL via `db_neon.py`
        """
        local_df = self._load_hourly_history_from_duckdb()
        if not local_df.empty:
            self.history_source_ = "duckdb_local"
            return local_df

        neon_df = self._load_hourly_history_from_neon()
        if not neon_df.empty:
            self.history_source_ = "neon_operational"
            return neon_df

        raise RuntimeError(
            "Unable to load hourly history from DuckDB or operational Neon."
        )

    def fit(self, hourly_df: pd.DataFrame | None = None) -> "AdaptivePLDForwardEngine":
        """
        Fit all structural layers of the engine.
        """
        if rpt is None:
            raise ImportError(
                "ruptures is required for AdaptivePLDForwardEngine. "
                "Install it with `pip install ruptures`."
            )

        history = hourly_df.copy() if hourly_df is not None else self.load_hourly_history()
        prepared = self._prepare_history(history)
        if prepared.empty or len(prepared) < 24 * 14:
            raise ValueError(
                "Not enough hourly history to fit the forward engine. "
                "At least two weeks of hourly data are required."
            )

        prepared["era_id"] = self._detect_eras(prepared["spdi_ma24"])
        prepared = self._fit_regimes(prepared)
        prepared = self._build_spdi_layers(prepared)

        self._build_climatology(prepared)
        self._build_recent_profile_surface(prepared)
        self._load_pmo_guidance(prepared)
        self._fit_feedback_surface(prepared)
        self._fit_physical_models(prepared)
        self._fit_era_classifier(prepared)

        self.history_ = history
        self.prepared_history_ = prepared
        self.latest_era_id_ = int(prepared["era_id"].iloc[-1])
        self.feature_medians_ = (
            prepared[self.dynamic_feature_cols + self.regime_feature_cols]
            .median(numeric_only=True)
            .to_dict()
        )
        self.bounds_ = {
            "ear": self._series_bounds(prepared["ear"]),
            "thermal_share": self._series_bounds(prepared["thermal_share"], lower=0.0, upper=1.0),
            "custo_fisico": self._series_bounds(prepared["custo_fisico"], lower=1.0),
            "spdi": self._series_bounds(prepared["spdi"], lower=0.05, upper=10.0),
        }
        # Nível de equilíbrio de longo prazo do custo_fisico (Ornstein-Uhlenbeck)
        _cf_clean = pd.to_numeric(prepared.get("custo_fisico", pd.Series(dtype=float)), errors="coerce").dropna()
        self.ou_mu_   = float(_cf_clean.quantile(self.ou_mu_quantile))  if not _cf_clean.empty else 150.0
        self.ou_sigma_ = float(_cf_clean.std())                          if not _cf_clean.empty else 30.0
        return self

    def forecast(
        self,
        horizon_hours: int | None = None,
        n_paths: int | None = None,
        persist: bool = True,
    ) -> AdaptivePLDForwardResult:
        """
        Run the full Monte Carlo forward process and optionally persist results.
        """
        if self.prepared_history_ is None:
            self.fit()

        prepared = self.prepared_history_.copy()
        horizon = int(horizon_hours or self.horizon_hours)
        paths = int(n_paths or self.n_paths)
        current_state = self._select_current_state(prepared)
        current_era = self._predict_current_era(current_state)
        current_probs = self._predict_regime_probabilities(current_era, current_state.to_frame().T)[0]

        sim = self._simulate_paths(
            prepared=prepared,
            current_state=current_state,
            current_era=current_era,
            horizon_hours=horizon,
            n_paths=paths,
        )

        scenarios = self._extract_scenarios(sim["pld_paths"])
        distribution = {
            "P10": round(float(np.percentile(sim["pld_paths"][:, -1], 10)), 4),
            "P50": round(float(np.percentile(sim["pld_paths"][:, -1], 50)), 4),
            "P90": round(float(np.percentile(sim["pld_paths"][:, -1], 90)), 4),
        }
        dispersion_confidence = float(
            self._confidence_from_band(distribution["P10"], distribution["P50"], distribution["P90"])
        )
        feedback_confidence = float(np.nanmean(sim.get("feedback_reliability_by_step", np.array([1.0], dtype=float))))
        confidence_score = round(dispersion_confidence * (0.75 + 0.25 * feedback_confidence), 4)

        generated_at = pd.Timestamp.utcnow().tz_localize(None)
        run_id = f"{MODEL_VERSION}_{generated_at.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        hourly_table = self._build_hourly_table(
            generated_at=generated_at,
            run_id=run_id,
            current_era=current_era,
            sim=sim,
            scenarios=scenarios,
            confidence_score=confidence_score,
            n_paths=paths,
        )

        result = AdaptivePLDForwardResult(
            run_id=run_id,
            generated_at=generated_at,
            hourly_table=hourly_table,
            scenarios=scenarios,
            distribution=distribution,
            regime_probabilities={
                "low": round(float(current_probs[0]), 4),
                "mid": round(float(current_probs[1]), 4),
                "high": round(float(current_probs[2]), 4),
            },
            confidence_score=confidence_score,
            current_era=current_era,
        )

        if persist:
            self.persist_hourly_table(hourly_table)

        return result

    def run(
        self,
        hourly_df: pd.DataFrame | None = None,
        horizon_hours: int | None = None,
        n_paths: int | None = None,
        persist: bool = True,
    ) -> AdaptivePLDForwardResult:
        """
        Fit and forecast in one call.
        """
        self.fit(hourly_df)
        return self.forecast(horizon_hours=horizon_hours, n_paths=n_paths, persist=persist)

    def persist_hourly_table(self, hourly_table: pd.DataFrame) -> None:
        """
        Persist the hourly result table to:
          1. Local DuckDB in `data/kintuadi.duckdb`
          2. Neon AUTH database following the same URL resolution as monetization.py
        """
        self._persist_to_duckdb(hourly_table)
        try:
            self._persist_to_auth_neon(hourly_table)
        except Exception as exc:
            LOGGER.warning("Adaptive PLD forward AUTH persistence failed: %s", exc)
        try:
            self.refresh_quality_history(persist=True)
        except Exception as exc:
            LOGGER.warning("Adaptive PLD quality refresh failed: %s", exc)

    def refresh_quality_history(self, persist: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a realized-vs-forecast quality ledger for the adaptive forward.

        The detailed observation table stores every realized forecast hour for each
        archived run. The summary table exposes the current MAE / MAPE / bias /
        coverage profile by lead bucket so the UI and operators can track
        forecast quality over time.
        """
        if self.prepared_history_ is None:
            self.fit()

        quality_obs = self._build_quality_observation_table(self.prepared_history_)
        quality_summary = self._build_quality_summary_table(quality_obs)
        if persist:
            self._persist_quality_to_duckdb(quality_obs, quality_summary)
            self._persist_quality_to_auth_neon(quality_obs, quality_summary)
        return quality_obs, quality_summary

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_hourly_history_from_duckdb(self) -> pd.DataFrame:
        if duckdb is None or not self.duckdb_path.exists():
            return pd.DataFrame()

        con = duckdb.connect(str(self.duckdb_path), read_only=True)
        try:
            try:
                tables = set(con.execute("SHOW TABLES").df()["name"].str.lower().tolist())
            except Exception:
                tables = set()

            df = pd.DataFrame()

            def join_series(series: pd.Series) -> None:
                nonlocal df
                if series.empty:
                    return
                df = df.join(series, how="outer") if not df.empty else series.to_frame()

            def ts_series(series: pd.Series) -> pd.Series:
                series.index = pd.to_datetime(series.index, errors="coerce")
                if getattr(series.index, "tz", None):
                    series.index = series.index.tz_localize(None)
                return series.dropna()

            if "pld_historical" in tables:
                pld_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', data) AS instante,
                        UPPER(TRIM(submercado)) AS submercado,
                        AVG(pld) AS pld_hora
                    FROM pld_historical
                    WHERE data IS NOT NULL AND pld > 0
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
                if not pld_df.empty:
                    pld_df["instante"] = pd.to_datetime(pld_df["instante"], errors="coerce")
                    pld_df = pld_df.dropna(subset=["instante", "submercado"])
                    pld_df["pld_hora"] = pd.to_numeric(pld_df["pld_hora"], errors="coerce")
                    sub_cols = {
                        "SUDESTE": "pld_se",
                        "SUDESTE/CENTRO-OESTE": "pld_se",
                        "SE": "pld_se",
                        "SE/CO": "pld_se",
                        "NORDESTE": "pld_ne",
                        "NE": "pld_ne",
                        "SUL": "pld_s",
                        "S": "pld_s",
                        "NORTE": "pld_n",
                        "N": "pld_n",
                    }
                    for submarket, col_name in sub_cols.items():
                        sub_s = (
                            pld_df[pld_df["submercado"] == submarket]
                            .set_index("instante")["pld_hora"]
                            .rename(col_name)
                        )
                        if not sub_s.empty:
                            join_series(ts_series(sub_s))
                    join_series(ts_series(pld_df.groupby("instante")["pld_hora"].mean().rename("pld")))

            if "cmo" in tables:
                cmo_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        UPPER(TRIM(id_subsistema)) AS sub,
                        AVG(val_cmo) AS cmo
                    FROM cmo
                    WHERE din_instante IS NOT NULL AND val_cmo > 0
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
                if not cmo_df.empty:
                    cmo_df["instante"] = pd.to_datetime(cmo_df["instante"], errors="coerce")
                    cmo_df = cmo_df.dropna(subset=["instante", "sub"])
                    cmo_df["cmo"] = pd.to_numeric(cmo_df["cmo"], errors="coerce")
                    sub_map = {
                        "SE": "se",
                        "SUDESTE": "se",
                        "SECO": "se",
                        "NE": "ne",
                        "NORDESTE": "ne",
                        "S": "s",
                        "SUL": "s",
                        "N": "n",
                        "NORTE": "n",
                    }
                    cmo_df["sk"] = cmo_df["sub"].map(sub_map)
                    cmo_df = cmo_df.dropna(subset=["sk"])
                    for sk in ("se", "ne", "s", "n"):
                        sub_s = (
                            cmo_df[cmo_df["sk"] == sk]
                            .set_index("instante")["cmo"]
                            .rename(f"cmo_{sk}")
                        )
                        if not sub_s.empty:
                            join_series(ts_series(sub_s))
                    dominant = (
                        cmo_df[cmo_df["sk"] == "se"]
                        .set_index("instante")["cmo"]
                        .rename("cmo_dominante")
                    )
                    if not dominant.empty:
                        join_series(ts_series(dominant))

            if "geracao_usina_horaria" in tables:
                gen_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        CASE
                            WHEN UPPER(nom_tipousina) IN ('FOTOVOLTAICA', 'SOLAR', 'FOTOVOLT') THEN 'solar'
                            WHEN UPPER(nom_tipousina) IN ('EOLIELÉTRICA', 'EÓLICA', 'EOLICA', 'EOLIELETRICA', 'EOLIELÉTRICO', 'EOL') THEN 'wind'
                            WHEN UPPER(nom_tipousina) IN ('HIDROELÉTRICA', 'HIDROELETRICA', 'HIDRÁULICA', 'HIDRAULICA', 'UHE', 'PCH', 'CGH', 'HIDRO') THEN 'hydro'
                            WHEN UPPER(nom_tipousina) IN ('TÉRMICA', 'TERMICA', 'UTE', 'BIOMASSA', 'GAS', 'GÁS', 'OLEO', 'ÓLEO', 'CARVAO', 'CARVÃO', 'DERIVADOS') THEN 'thermal'
                            WHEN UPPER(nom_tipousina) IN ('NUCLEAR', 'UTN') THEN 'nuclear'
                            ELSE 'other'
                        END AS tipo,
                        SUM(val_geracao) AS valor
                    FROM geracao_usina_horaria
                    WHERE din_instante IS NOT NULL AND val_geracao IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
            elif "geracao_tipo_hora" in tables:
                gen_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        LOWER(TRIM(tipo_geracao)) AS tipo,
                        SUM(val_geracao_mw) AS valor
                    FROM geracao_tipo_hora
                    WHERE din_instante IS NOT NULL AND val_geracao_mw IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
            else:
                gen_df = pd.DataFrame()
            if not gen_df.empty:
                gen_df["instante"] = pd.to_datetime(gen_df["instante"], errors="coerce")
                gen_df = gen_df.dropna(subset=["instante"])
                gen_df["valor"] = pd.to_numeric(gen_df["valor"], errors="coerce")
                gen_df["tipo"] = gen_df["tipo"].astype(str).str.lower().str.strip()
                for generation_type in ("solar", "wind", "hydro", "thermal", "nuclear"):
                    sub = gen_df[gen_df["tipo"] == generation_type]
                    if not sub.empty:
                        join_series(
                            ts_series(
                                sub.groupby("instante")["valor"].sum().rename(generation_type)
                            )
                        )

            if "curva_carga" in tables:
                load_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        UPPER(TRIM(id_subsistema)) AS id_sub,
                        SUM(val_cargaenergiahomwmed) AS valor
                    FROM curva_carga
                    WHERE din_instante IS NOT NULL AND val_cargaenergiahomwmed IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
                if not load_df.empty:
                    load_df["instante"] = pd.to_datetime(load_df["instante"], errors="coerce")
                    load_df = load_df.dropna(subset=["instante", "id_sub"])
                    load_df["valor"] = pd.to_numeric(load_df["valor"], errors="coerce")
                    sub_map = {
                        "SE": "se",
                        "SUDESTE": "se",
                        "NE": "ne",
                        "NORDESTE": "ne",
                        "S": "s",
                        "SUL": "s",
                        "N": "n",
                        "NORTE": "n",
                    }
                    load_df["sk"] = load_df["id_sub"].map(sub_map)
                    load_df = load_df.dropna(subset=["sk"])
                    for sk in ("se", "ne", "s", "n"):
                        sub = load_df[load_df["sk"] == sk]
                        if not sub.empty:
                            join_series(
                                ts_series(
                                    sub.groupby("instante")["valor"].sum().rename(f"load_{sk}")
                                )
                            )
                    join_series(ts_series(load_df.groupby("instante")["valor"].sum().rename("load")))

            if "ear_diario_subsistema" in tables:
                ear_df = con.execute(
                    """
                    SELECT
                        date_trunc('day', ear_data) AS data,
                        SUM(ear_verif_subsistema_mwmes) AS ear,
                        SUM(ear_max_subsistema) AS earmaxp
                    FROM ear_diario_subsistema
                    WHERE ear_data IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).df()
                if not ear_df.empty:
                    ear_df["data"] = pd.to_datetime(ear_df["data"], errors="coerce").dt.normalize()
                    ear_df = ear_df.dropna(subset=["data"])
                    ear_df["ear"] = pd.to_numeric(ear_df["ear"], errors="coerce")
                    ear_df["earmaxp"] = pd.to_numeric(ear_df["earmaxp"], errors="coerce")
                    ear_df["ear_pct"] = ear_df["ear"] / ear_df["earmaxp"].replace(0, np.nan) * 100
                    join_series(ts_series(ear_df.set_index("data")["ear_pct"].rename("ear_pct").resample("h").ffill()))

            if "ena_diario_subsistema" in tables:
                ena_df = con.execute(
                    """
                    SELECT
                        date_trunc('day', ena_data) AS data,
                        SUM(ena_bruta_regiao_mwmed) AS ena_bruta,
                        SUM(ena_armazenavel_regiao_mwmed) AS ena_arm
                    FROM ena_diario_subsistema
                    WHERE ena_data IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).df()
                if not ena_df.empty:
                    ena_df["data"] = pd.to_datetime(ena_df["data"], errors="coerce").dt.normalize()
                    ena_df = ena_df.dropna(subset=["data"])
                    ena_df = ena_df.set_index("data")
                    join_series(ts_series(pd.to_numeric(ena_df["ena_bruta"], errors="coerce").rename("ena_bruta").resample("h").ffill()))
                    join_series(ts_series(pd.to_numeric(ena_df["ena_arm"], errors="coerce").rename("ena_arm").resample("h").ffill()))

            if "disponibilidade_usina" in tables:
                disp_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        CASE
                            WHEN UPPER(id_tipousina) IN ('UHE', 'PCH', 'CGH') THEN 'hydro'
                            WHEN UPPER(id_tipousina) = 'UTE' THEN 'thermal'
                            WHEN UPPER(id_tipousina) = 'UTN' THEN 'nuclear'
                            WHEN UPPER(id_tipousina) = 'EOL' THEN 'wind'
                            WHEN UPPER(id_tipousina) = 'UFV' THEN 'solar'
                            ELSE 'other'
                        END AS tipo,
                        SUM(val_dispsincronizada) AS disp_sinc
                    FROM disponibilidade_usina
                    WHERE din_instante IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
            elif "disponibilidade_tipo_hora" in tables:
                disp_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        LOWER(TRIM(tipo_geracao)) AS tipo,
                        SUM(val_disp_sincronizada) AS disp_sinc
                    FROM disponibilidade_tipo_hora
                    WHERE din_instante IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
            else:
                disp_df = pd.DataFrame()
            if not disp_df.empty:
                disp_df["instante"] = pd.to_datetime(disp_df["instante"], errors="coerce")
                disp_df = disp_df.dropna(subset=["instante"])
                disp_df["disp_sinc"] = pd.to_numeric(disp_df["disp_sinc"], errors="coerce")
                disp_df["tipo"] = disp_df["tipo"].astype(str).str.lower().str.strip()
                for generation_type, col_name in (
                    ("hydro", "disp_hydro"),
                    ("thermal", "disp_thermal"),
                    ("nuclear", "disp_nuclear"),
                    ("solar", "disp_solar"),
                    ("wind", "disp_wind"),
                ):
                    sub = disp_df[disp_df["tipo"] == generation_type]
                    if not sub.empty:
                        join_series(
                            ts_series(
                                sub.groupby("instante")["disp_sinc"].sum().rename(col_name)
                            )
                        )
                join_series(ts_series(disp_df.groupby("instante")["disp_sinc"].sum().rename("disp_total")))

            if {"restricao_eolica", "restricao_fotovoltaica"}.intersection(tables):
                for source, table_name in (("wind", "restricao_eolica"), ("solar", "restricao_fotovoltaica")):
                    if table_name not in tables:
                        continue
                    restriction_df = con.execute(
                        f"""
                        SELECT
                            date_trunc('hour', din_instante) AS instante,
                            SUM(val_geracao) AS gerado,
                            SUM(val_geracaolimitada) AS limitado,
                            SUM(val_disponibilidade) AS disponivel
                        FROM {table_name}
                        WHERE din_instante IS NOT NULL
                        GROUP BY 1
                        ORDER BY 1
                        """
                    ).df()
                    if restriction_df.empty:
                        continue
                    restriction_df["instante"] = pd.to_datetime(restriction_df["instante"], errors="coerce")
                    restriction_df = restriction_df.dropna(subset=["instante"]).set_index("instante")
                    generated = pd.to_numeric(restriction_df["gerado"], errors="coerce")
                    limited = pd.to_numeric(restriction_df["limitado"], errors="coerce")
                    available = pd.to_numeric(restriction_df["disponivel"], errors="coerce")
                    join_series(ts_series((limited - generated).clip(lower=0).rename(f"curtail_{source}")))
                    join_series(ts_series(available.rename(f"avail_{source}")))
            elif "restricao_renovavel" in tables:
                restriction_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        LOWER(TRIM(fonte)) AS fonte,
                        SUM(val_geracao) AS gerado,
                        SUM(val_geracaolimitada) AS limitado,
                        SUM(val_disponibilidade) AS disponivel
                    FROM restricao_renovavel
                    WHERE din_instante IS NOT NULL
                    GROUP BY 1, 2
                    ORDER BY 1, 2
                    """
                ).df()
                if not restriction_df.empty:
                    restriction_df["instante"] = pd.to_datetime(restriction_df["instante"], errors="coerce")
                    restriction_df = restriction_df.dropna(subset=["instante", "fonte"])
                    for source in ("solar", "wind"):
                        sub = restriction_df[restriction_df["fonte"] == source].set_index("instante")
                        if sub.empty:
                            continue
                        generated = pd.to_numeric(sub["gerado"], errors="coerce")
                        limited = pd.to_numeric(sub["limitado"], errors="coerce")
                        available = pd.to_numeric(sub["disponivel"], errors="coerce")
                        join_series(ts_series((limited - generated).clip(lower=0).rename(f"curtail_{source}")))
                        join_series(ts_series(available.rename(f"avail_{source}")))

            if "despacho_gfom" in tables:
                gfom_df = con.execute(
                    """
                    SELECT
                        date_trunc('hour', din_instante) AS instante,
                        SUM(val_verifgeracao) AS gfom_ger,
                        SUM(val_verifconstrainedoff) AS constrained_off,
                        SUM(val_verifinflexibilidade) AS thermal_inflex_gfom,
                        SUM(val_verifordemmerito) AS thermal_merit,
                        SUM(val_verifgfom) AS gfom
                    FROM despacho_gfom
                    WHERE din_instante IS NOT NULL
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).df()
                if not gfom_df.empty:
                    gfom_df["instante"] = pd.to_datetime(gfom_df["instante"], errors="coerce")
                    gfom_df = gfom_df.dropna(subset=["instante"]).set_index("instante")
                    for column_name in ("gfom_ger", "constrained_off", "thermal_inflex_gfom", "thermal_merit", "gfom"):
                        if column_name in gfom_df.columns:
                            join_series(
                                ts_series(
                                    pd.to_numeric(gfom_df[column_name], errors="coerce").rename(column_name)
                                )
                            )

            if "cvu_usina_termica" in tables:
                cvu_df = con.execute(
                    """
                    SELECT
                        dat_fimsemana AS instante,
                        AVG(val_cvu) AS cvu_semana
                    FROM cvu_usina_termica
                    WHERE val_cvu > 0
                    GROUP BY 1
                    ORDER BY 1
                    """
                ).df()
                if not cvu_df.empty:
                    cvu_df["instante"] = pd.to_datetime(cvu_df["instante"], errors="coerce")
                    cvu_df = cvu_df.dropna(subset=["instante"]).set_index("instante")
                    join_series(ts_series(pd.to_numeric(cvu_df["cvu_semana"], errors="coerce").rename("cvu_semana")))

            if df.empty:
                return pd.DataFrame()
            return self._finalize_loaded_history(df.sort_index())
        finally:
            con.close()

    def _load_hourly_history_from_neon(self) -> pd.DataFrame:
        try:
            import db_neon
        except Exception:
            return pd.DataFrame()

        if not getattr(db_neon, "is_configured", lambda: False)():
            return pd.DataFrame()

        df = pd.DataFrame()

        def join_series(series: pd.Series) -> None:
            nonlocal df
            if series.empty:
                return
            df = df.join(series, how="outer") if not df.empty else series.to_frame()

        def ts_series(series: pd.Series) -> pd.Series:
            series.index = pd.to_datetime(series.index, errors="coerce")
            if getattr(series.index, "tz", None):
                series.index = series.index.tz_localize(None)
            return series.dropna()

        pld_df = db_neon.fetchdf(
            """
            SELECT
                MAKE_DATE(
                    CAST(SUBSTR(CAST(mes_referencia AS TEXT), 1, 4) AS INTEGER),
                    CAST(SUBSTR(CAST(mes_referencia AS TEXT), 5, 2) AS INTEGER),
                    dia
                ) + (hora * INTERVAL '1 hour') AS ts,
                submercado,
                AVG(pld_hora) AS pld_hora
            FROM pld_historical
            WHERE mes_referencia IS NOT NULL AND pld_hora IS NOT NULL
            GROUP BY mes_referencia, dia, hora, submercado
            ORDER BY ts, submercado
            """
        )
        if not pld_df.empty:
            pld_df["ts"] = pd.to_datetime(pld_df["ts"], errors="coerce")
            pld_df = pld_df.dropna(subset=["ts", "submercado"])
            pld_df["pld_hora"] = pd.to_numeric(pld_df["pld_hora"], errors="coerce")
            sub_cols = {
                "SUDESTE": "pld_se",
                "SE": "pld_se",
                "NORDESTE": "pld_ne",
                "NE": "pld_ne",
                "SUL": "pld_s",
                "S": "pld_s",
                "NORTE": "pld_n",
                "N": "pld_n",
            }
            for submarket, col_name in sub_cols.items():
                sub_s = pld_df[pld_df["submercado"] == submarket].set_index("ts")["pld_hora"].rename(col_name)
                if not sub_s.empty:
                    join_series(ts_series(sub_s))
            join_series(ts_series(pld_df.groupby("ts")["pld_hora"].mean().rename("pld")))

        cmo_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                UPPER(TRIM(id_subsistema)) AS sub,
                val_cmo AS cmo
            FROM cmo
            WHERE din_instante IS NOT NULL AND val_cmo IS NOT NULL
            ORDER BY din_instante
            """
        )
        if not cmo_df.empty:
            cmo_df["instante"] = pd.to_datetime(cmo_df["instante"], errors="coerce")
            cmo_df = cmo_df.dropna(subset=["instante", "sub"])
            cmo_df["cmo"] = pd.to_numeric(cmo_df["cmo"], errors="coerce")
            sub_map = {
                "SE": "se",
                "SUDESTE": "se",
                "NE": "ne",
                "NORDESTE": "ne",
                "S": "s",
                "SUL": "s",
                "N": "n",
                "NORTE": "n",
            }
            cmo_df["sk"] = cmo_df["sub"].map(sub_map)
            cmo_df = cmo_df.dropna(subset=["sk"])
            for sk in ("se", "ne", "s", "n"):
                sub_s = cmo_df[cmo_df["sk"] == sk].set_index("instante")["cmo"].rename(f"cmo_{sk}")
                if not sub_s.empty:
                    join_series(ts_series(sub_s))
            dominant = cmo_df[cmo_df["sk"] == "se"].set_index("instante")["cmo"].rename("cmo_dominante")
            if not dominant.empty:
                join_series(ts_series(dominant))

        generation_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                LOWER(TRIM(tipo_geracao)) AS tipo,
                SUM(val_geracao_mw) AS valor
            FROM geracao_tipo_hora
            WHERE din_instante IS NOT NULL AND val_geracao_mw IS NOT NULL
            GROUP BY din_instante, tipo_geracao
            ORDER BY din_instante
            """
        )
        if not generation_df.empty:
            generation_df["instante"] = pd.to_datetime(generation_df["instante"], errors="coerce")
            generation_df = generation_df.dropna(subset=["instante"])
            generation_df["valor"] = pd.to_numeric(generation_df["valor"], errors="coerce")
            generation_df["tipo"] = generation_df["tipo"].astype(str).str.lower().str.strip()
            for generation_type in ("solar", "wind", "hydro", "thermal", "nuclear"):
                sub = generation_df[generation_df["tipo"] == generation_type]
                if not sub.empty:
                    join_series(ts_series(sub.groupby("instante")["valor"].sum().rename(generation_type)))

        load_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                UPPER(TRIM(id_subsistema)) AS id_subsistema,
                SUM(val_cargaenergiahomwmed) AS valor
            FROM curva_carga
            WHERE din_instante IS NOT NULL AND val_cargaenergiahomwmed IS NOT NULL
            GROUP BY din_instante, id_subsistema
            ORDER BY din_instante
            """
        )
        if not load_df.empty:
            load_df["instante"] = pd.to_datetime(load_df["instante"], errors="coerce")
            load_df = load_df.dropna(subset=["instante", "id_subsistema"])
            load_df["valor"] = pd.to_numeric(load_df["valor"], errors="coerce")
            sub_map = {
                "SE": "se",
                "SUDESTE": "se",
                "NE": "ne",
                "NORDESTE": "ne",
                "S": "s",
                "SUL": "s",
                "N": "n",
                "NORTE": "n",
            }
            load_df["sk"] = load_df["id_subsistema"].map(sub_map)
            load_df = load_df.dropna(subset=["sk"])
            for sk in ("se", "ne", "s", "n"):
                sub = load_df[load_df["sk"] == sk]
                if not sub.empty:
                    join_series(ts_series(sub.groupby("instante")["valor"].sum().rename(f"load_{sk}")))
            join_series(ts_series(load_df.groupby("instante")["valor"].sum().rename("load")))

        ear_df = db_neon.fetchdf(
            """
            SELECT
                ear_data AS data,
                SUM(ear_verif_subsistema_mwmes) AS ear,
                SUM(ear_max_subsistema) AS earmaxp
            FROM ear_diario_subsistema
            WHERE ear_data IS NOT NULL
            GROUP BY ear_data
            ORDER BY ear_data
            """
        )
        if not ear_df.empty:
            ear_df["data"] = pd.to_datetime(ear_df["data"], errors="coerce").dt.normalize()
            ear_df = ear_df.dropna(subset=["data"])
            ear_df["ear"] = pd.to_numeric(ear_df["ear"], errors="coerce")
            ear_df["earmaxp"] = pd.to_numeric(ear_df["earmaxp"], errors="coerce")
            ear_df["ear_pct"] = ear_df["ear"] / ear_df["earmaxp"].replace(0, np.nan) * 100
            join_series(ts_series(ear_df.set_index("data")["ear_pct"].rename("ear_pct").resample("h").ffill()))

        ena_df = db_neon.fetchdf(
            """
            SELECT
                ena_data AS data,
                SUM(ena_bruta_regiao_mwmed) AS ena_bruta,
                SUM(ena_armazenavel_regiao_mwmed) AS ena_arm
            FROM ena_diario_subsistema
            WHERE ena_data IS NOT NULL
            GROUP BY ena_data
            ORDER BY ena_data
            """
        )
        if not ena_df.empty:
            ena_df["data"] = pd.to_datetime(ena_df["data"], errors="coerce").dt.normalize()
            ena_df = ena_df.dropna(subset=["data"]).set_index("data")
            join_series(ts_series(pd.to_numeric(ena_df["ena_bruta"], errors="coerce").rename("ena_bruta").resample("h").ffill()))
            join_series(ts_series(pd.to_numeric(ena_df["ena_arm"], errors="coerce").rename("ena_arm").resample("h").ffill()))

        availability_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                LOWER(TRIM(tipo_geracao)) AS tipo,
                SUM(val_disp_sincronizada) AS disp_sinc
            FROM disponibilidade_tipo_hora
            WHERE din_instante IS NOT NULL
            GROUP BY din_instante, tipo_geracao
            ORDER BY din_instante
            """
        )
        if not availability_df.empty:
            availability_df["instante"] = pd.to_datetime(availability_df["instante"], errors="coerce")
            availability_df = availability_df.dropna(subset=["instante"])
            availability_df["disp_sinc"] = pd.to_numeric(availability_df["disp_sinc"], errors="coerce")
            availability_df["tipo"] = availability_df["tipo"].astype(str).str.lower().str.strip()
            for generation_type, col_name in (
                ("hydro", "disp_hydro"),
                ("thermal", "disp_thermal"),
                ("nuclear", "disp_nuclear"),
                ("solar", "disp_solar"),
                ("wind", "disp_wind"),
            ):
                sub = availability_df[availability_df["tipo"] == generation_type]
                if not sub.empty:
                    join_series(ts_series(sub.groupby("instante")["disp_sinc"].sum().rename(col_name)))
            join_series(ts_series(availability_df.groupby("instante")["disp_sinc"].sum().rename("disp_total")))

        restriction_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                LOWER(TRIM(fonte)) AS fonte,
                SUM(val_geracao) AS gerado,
                SUM(val_geracaolimitada) AS limitado,
                SUM(val_disponibilidade) AS disponivel
            FROM restricao_renovavel
            WHERE din_instante IS NOT NULL
            GROUP BY din_instante, fonte
            ORDER BY din_instante
            """
        )
        if not restriction_df.empty:
            restriction_df["instante"] = pd.to_datetime(restriction_df["instante"], errors="coerce")
            restriction_df = restriction_df.dropna(subset=["instante", "fonte"])
            for source in ("solar", "wind"):
                sub = restriction_df[restriction_df["fonte"] == source].set_index("instante")
                if sub.empty:
                    continue
                generated = pd.to_numeric(sub["gerado"], errors="coerce")
                limited = pd.to_numeric(sub["limitado"], errors="coerce")
                available = pd.to_numeric(sub["disponivel"], errors="coerce")
                join_series(ts_series((limited - generated).clip(lower=0).rename(f"curtail_{source}")))
                join_series(ts_series(available.rename(f"avail_{source}")))

        gfom_df = db_neon.fetchdf(
            """
            SELECT
                din_instante AS instante,
                val_verifgeracao AS gfom_ger,
                val_verifconstrainedoff AS constrained_off,
                val_verifinflexibilidade AS thermal_inflex_gfom,
                val_verifordemmerito AS thermal_merit,
                val_verifgfom AS gfom
            FROM despacho_gfom
            WHERE din_instante IS NOT NULL
            ORDER BY din_instante
            """
        )
        if not gfom_df.empty:
            gfom_df["instante"] = pd.to_datetime(gfom_df["instante"], errors="coerce")
            gfom_df = gfom_df.dropna(subset=["instante"]).set_index("instante")
            for column_name in ("gfom_ger", "constrained_off", "thermal_inflex_gfom", "thermal_merit", "gfom"):
                if column_name in gfom_df.columns:
                    join_series(ts_series(pd.to_numeric(gfom_df[column_name], errors="coerce").rename(column_name)))

        cvu_df = db_neon.fetchdf(
            """
            SELECT
                dat_fimsemana AS instante,
                AVG(val_cvu) AS cvu_semana
            FROM cvu_usina_termica
            WHERE val_cvu > 0
            GROUP BY dat_fimsemana
            ORDER BY dat_fimsemana
            """
        )
        if not cvu_df.empty:
            cvu_df["instante"] = pd.to_datetime(cvu_df["instante"], errors="coerce")
            cvu_df = cvu_df.dropna(subset=["instante"]).set_index("instante")
            join_series(ts_series(pd.to_numeric(cvu_df["cvu_semana"], errors="coerce").rename("cvu_semana")))

        if df.empty:
            return pd.DataFrame()
        return self._finalize_loaded_history(df.sort_index())

    def _finalize_loaded_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rebuild derived hourly columns from the project's canonical operational logic.
        """
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        def col(name: str) -> pd.Series:
            return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(0.0, index=df.index)

        load_s = col("load")
        solar_s = col("solar")
        wind_s = col("wind")
        hydro_s = col("hydro")
        thermal_s = col("thermal")
        nuclear_s = col("nuclear")

        df["load"] = load_s
        df["net_load"] = load_s - solar_s.fillna(0) - wind_s.fillna(0)
        df["renov_total"] = solar_s.fillna(0) + wind_s.fillna(0)
        df["geracao_total"] = (
            solar_s.fillna(0)
            + wind_s.fillna(0)
            + hydro_s.fillna(0)
            + thermal_s.fillna(0)
            + nuclear_s.fillna(0)
        )

        sin_cost = pd.Series(0.0, index=df.index)
        any_submarket = False
        for sub_key in ("se", "ne", "s", "n"):
            load_col = f"load_{sub_key}"
            pld_col = f"pld_{sub_key}"
            if load_col in df.columns and pld_col in df.columns:
                sin_cost += col(load_col).fillna(0) * col(pld_col).fillna(0)
                any_submarket = True
        if not any_submarket:
            sin_cost = load_s.fillna(0) * col("pld").fillna(0)
        df["sin_cost"] = sin_cost.where(sin_cost > 0, np.nan)

        if "cvu_semana" in df.columns:
            df["cvu_semana"] = pd.to_numeric(df["cvu_semana"], errors="coerce").ffill().bfill()

        thermal_merit = col("thermal_merit")
        cvu = col("cvu_semana")
        df["thermal_share"] = (
            thermal_s.fillna(0) / df["geracao_total"].replace(0, np.nan)
        ).clip(lower=0, upper=1)
        df["thermal_real_cost"] = thermal_s.fillna(0) * cvu.fillna(0)
        df["thermal_merit_cost"] = thermal_merit.fillna(0) * cvu.fillna(0)

        disp_hydro = col("disp_hydro") if "disp_hydro" in df.columns else hydro_s
        hydro_preserved = (disp_hydro - hydro_s.fillna(0)).clip(lower=0)

        cmo = col("cmo_dominante")
        if "cmo_dominante" not in df.columns:
            cmo_candidates = [c for c in ("cmo_se", "cmo_ne", "cmo_s", "cmo_n") if c in df.columns]
            if cmo_candidates:
                cmo = pd.concat([col(c) for c in cmo_candidates], axis=1).mean(axis=1)
            else:
                cmo = col("cmo")
        df["cmo"] = cmo

        curtail_solar = col("curtail_solar")
        curtail_wind = col("curtail_wind")
        df["curtail_total"] = (curtail_solar.fillna(0) + curtail_wind.fillna(0)).clip(lower=0)
        avail_ren = col("avail_solar").fillna(0) + col("avail_wind").fillna(0)
        if (avail_ren > 0).any():
            df["avail_ren"] = avail_ren
        else:
            df["avail_ren"] = df["renov_total"].fillna(0) + df["curtail_total"].fillna(0)

        df["t_prudencia"] = np.where(
            cmo > col("pld"),
            hydro_preserved * (cmo - col("pld")),
            0.0,
        )
        df["t_hidro"] = hydro_preserved * cmo.fillna(0)
        df["t_eletric"] = df["thermal_merit_cost"]
        df["t_sistemica"] = (col("pld") - cmo).abs() * df["net_load"].clip(lower=0).fillna(0)
        df["t_total"] = df["t_eletric"] + df["t_hidro"] + df["t_prudencia"] + df["t_sistemica"]

        df["custo_fisico_rh"] = df["t_total"]
        df["custo_fisico"] = (df["custo_fisico_rh"] / load_s.replace(0, np.nan)).clip(lower=0.01)
        df["imr"] = col("pld") - df["custo_fisico"]
        df["spdi"] = (col("pld") / df["custo_fisico"].replace(0, np.nan)).clip(lower=0.05, upper=10.0)

        df["isr"] = (
            df["avail_ren"].fillna(df["renov_total"])
            / df["net_load"].abs().replace(0, np.nan)
        ).clip(lower=0, upper=25)
        df["curtailment_ratio"] = (
            df["curtail_total"].fillna(0)
            / df["avail_ren"].replace(0, np.nan)
        ).clip(lower=0, upper=1).fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Feature engineering and structural layers
    # ------------------------------------------------------------------
    def _prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        df = history.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"], errors="coerce")
            elif "ts" in df.columns:
                df.index = pd.to_datetime(df["ts"], errors="coerce")
            elif "data" in df.columns:
                df.index = pd.to_datetime(df["data"], errors="coerce")
            else:
                raise ValueError("Hourly history must have a DatetimeIndex or a timestamp column.")

        df = df[~df.index.isna()].copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        standardized = pd.DataFrame(index=df.index)
        standardized["ear"] = self._coalesce_numeric(df, ["EAR", "ear", "ear_pct"])
        standardized["ena"] = self._coalesce_numeric(df, ["ENA", "ena", "ena_arm", "ena_bruta"])
        standardized["load"] = self._coalesce_numeric(df, ["load", "Load", "carga_total"])
        standardized["solar"] = self._coalesce_numeric(df, ["solar", "Solar"])
        standardized["wind"] = self._coalesce_numeric(df, ["wind", "Wind"])
        standardized["thermal"] = self._coalesce_numeric(df, ["thermal", "Thermal"])
        standardized["hydro"] = self._coalesce_numeric(df, ["hydro", "Hydro"])
        standardized["nuclear"] = self._coalesce_numeric(df, ["nuclear", "Nuclear"])
        for column_name in ("solar", "wind", "thermal", "hydro", "nuclear"):
            standardized[column_name] = standardized[column_name].fillna(0.0)
        standardized["pld"] = self._coalesce_numeric(df, ["PLD", "pld", "pld_ponderado"])
        standardized["cmo"] = self._coalesce_numeric(df, ["CMO", "cmo", "cmo_dominante", "cmo_se"])
        standardized["net_load"] = standardized["load"] - (
            standardized["solar"].fillna(0) + standardized["wind"].fillna(0)
        )

        provided_custo = self._coalesce_numeric(df, ["custo_fisico"])
        t_total = self._coalesce_numeric(df, ["t_total", "T_total"])
        thermal_merit_cost = self._coalesce_numeric(df, ["thermal_merit_cost"])
        thermal_real_cost = self._coalesce_numeric(df, ["thermal_real_cost"])
        hydro_preserved = self._coalesce_numeric(df, ["Hydro_preserved"])
        if hydro_preserved.isna().all():
            hydro_preserved = (
                self._coalesce_numeric(df, ["disp_hydro"]).fillna(standardized["hydro"])
                - standardized["hydro"].fillna(0)
            ).clip(lower=0)

        if thermal_merit_cost.isna().all():
            thermal_merit_cost = thermal_real_cost
        if thermal_merit_cost.isna().all():
            thermal_merit_cost = standardized["thermal"] * self._coalesce_numeric(df, ["cvu_semana"])

        rebuilt_t_total = (
            thermal_merit_cost.fillna(0)
            + hydro_preserved.fillna(0) * standardized["cmo"].fillna(0)
            + np.where(
                standardized["cmo"] > standardized["pld"],
                hydro_preserved.fillna(0) * (standardized["cmo"] - standardized["pld"]),
                0.0,
            )
            + (standardized["pld"] - standardized["cmo"]).abs()
            * standardized["net_load"].clip(lower=0).fillna(0)
        )
        t_total = t_total.fillna(rebuilt_t_total)
        standardized["custo_fisico"] = provided_custo
        standardized.loc[standardized["custo_fisico"].isna(), "custo_fisico"] = (
            t_total / standardized["load"].replace(0, np.nan)
        )
        standardized["custo_fisico"] = standardized["custo_fisico"].replace([np.inf, -np.inf], np.nan)
        standardized["custo_fisico"] = standardized["custo_fisico"].clip(lower=0.01)

        standardized["spdi"] = self._coalesce_numeric(df, ["SPDI", "spdi"])
        standardized.loc[standardized["spdi"].isna(), "spdi"] = (
            standardized["pld"] / standardized["custo_fisico"].replace(0, np.nan)
        )
        standardized["spdi"] = standardized["spdi"].clip(lower=0.05, upper=10.0)

        standardized["imr"] = self._coalesce_numeric(df, ["IMR", "imr"])
        standardized.loc[standardized["imr"].isna(), "imr"] = (
            standardized["pld"] - standardized["custo_fisico"]
        )

        standardized["avail_ren"] = self._coalesce_numeric(df, ["avail_ren"])
        standardized.loc[standardized["avail_ren"].isna(), "avail_ren"] = (
            standardized["solar"].fillna(0)
            + standardized["wind"].fillna(0)
            + self._coalesce_numeric(df, ["curtail_total", "curtailment"]).fillna(0)
        )

        standardized["curtailment"] = self._coalesce_numeric(df, ["curtail_total", "curtailment", "Curtailment"])
        standardized["curtailment"] = standardized["curtailment"].fillna(0).clip(lower=0)
        standardized["curtailment_ratio"] = (
            standardized["curtailment"] / standardized["avail_ren"].replace(0, np.nan)
        ).clip(lower=0, upper=1).fillna(0.0)

        standardized["thermal_share"] = (
            standardized["thermal"] / standardized["load"].replace(0, np.nan)
        ).clip(lower=0, upper=1)
        if "geracao_total" in df.columns:
            geracao_total = pd.to_numeric(df["geracao_total"], errors="coerce")
            thermal_share_from_generation = (
                standardized["thermal"] / geracao_total.replace(0, np.nan)
            ).clip(lower=0, upper=1)
            standardized["thermal_share"] = standardized["thermal_share"].fillna(thermal_share_from_generation)

        standardized["isr"] = self._coalesce_numeric(df, ["ISR", "isr"])
        standardized.loc[standardized["isr"].isna(), "isr"] = (
            standardized["avail_ren"] / standardized["net_load"].abs().replace(0, np.nan)
        )
        standardized["isr"] = standardized["isr"].clip(lower=0, upper=25)

        standardized["month"] = standardized.index.month.astype(int)
        standardized["hour"] = standardized.index.hour.astype(int)
        standardized["spdi_ma24"] = standardized["spdi"].rolling(24, min_periods=6).mean()
        standardized["spdi_ma168"] = standardized["spdi"].rolling(168, min_periods=24).mean()
        standardized["shock_std24"] = standardized["spdi"].rolling(24, min_periods=6).std(ddof=0)
        standardized["shock"] = standardized["spdi"] - standardized["spdi_ma24"]
        standardized["zscore"] = standardized["shock"] / standardized["shock_std24"].replace(0, np.nan)
        standardized["divergence"] = standardized["spdi"] / standardized["spdi_ma24"].replace(0, np.nan)

        standardized = standardized.replace([np.inf, -np.inf], np.nan)
        standardized = standardized.ffill().bfill()
        standardized = standardized.dropna(
            subset=["ear", "ena", "load", "pld", "custo_fisico", "spdi", "spdi_ma24"]
        )
        standardized = standardized[standardized["load"] > 0]
        standardized = standardized[standardized["spdi"] < 9.95]

        return standardized

    def _detect_eras(self, spdi_trend: pd.Series) -> pd.Series:
        signal = pd.to_numeric(spdi_trend, errors="coerce").ffill().bfill().to_numpy(dtype=float)
        if len(signal) < max(self.min_era_size_hours * 2, 24 * 45):
            return pd.Series(0, index=spdi_trend.index, dtype=int)

        scaled = ((signal - np.nanmean(signal)) / max(np.nanstd(signal), EPS)).reshape(-1, 1)
        min_size = min(self.min_era_size_hours, max(24 * 14, len(signal) // 5))

        try:
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(scaled)
            penalty = max(1.0, float(np.log(len(signal)) * np.nanstd(scaled) * 3.0))
            breakpoints = [b for b in algo.predict(pen=penalty) if b < len(signal)]
        except Exception:
            breakpoints = []

        if len(breakpoints) > max(self.max_eras - 1, 0):
            breakpoints = breakpoints[: self.max_eras - 1]

        era_ids = np.zeros(len(signal), dtype=int)
        start = 0
        for era_id, breakpoint in enumerate(breakpoints, start=1):
            era_ids[start:breakpoint] = era_id - 1
            start = breakpoint
        if breakpoints:
            era_ids[start:] = len(breakpoints)

        return pd.Series(era_ids, index=spdi_trend.index, dtype=int)

    def _fit_regimes(self, prepared: pd.DataFrame) -> pd.DataFrame:
        df = prepared.copy()
        df["regime_id"] = 1
        df["regime_label"] = "mid"
        self.regime_classifiers_ = {}

        for era_id, era_df in df.groupby("era_id"):
            features = era_df[self.regime_feature_cols].replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill()
            features = features.dropna()
            if len(features) < 24:
                self.regime_classifiers_[int(era_id)] = self._constant_classifier_bundle(1, classes=[0, 1, 2])
                continue

            model = KMeans(n_clusters=3, random_state=self.random_state, n_init=20)
            clusters = model.fit_predict(features.to_numpy(dtype=float))
            centers = pd.DataFrame(model.cluster_centers_, columns=self.regime_feature_cols)
            stress_score = (
                centers["net_load"]
                + centers["thermal_share"]
                + centers["curtailment"]
                - centers["ear"]
                - centers["ena"]
                - centers["isr"]
            )
            ordered_clusters = stress_score.sort_values().index.tolist()
            cluster_to_regime = {
                ordered_clusters[0]: REGIME_LABEL_TO_ID["low"],
                ordered_clusters[1]: REGIME_LABEL_TO_ID["mid"],
                ordered_clusters[2]: REGIME_LABEL_TO_ID["high"],
            }
            regime_ids = pd.Series(clusters, index=features.index).map(cluster_to_regime).astype(int)
            df.loc[regime_ids.index, "regime_id"] = regime_ids
            df.loc[regime_ids.index, "regime_label"] = regime_ids.map(REGIME_ID_TO_LABEL)

            clf_bundle = self._fit_classifier_bundle(
                frame=df.loc[features.index].copy(),
                feature_cols=self.regime_feature_cols,
                target_col="regime_id",
                classes=[0, 1, 2],
            )
            self.regime_classifiers_[int(era_id)] = clf_bundle

        return df

    @staticmethod
    def _group_sample_arrays(
        frame: pd.DataFrame,
        group_cols: list[str],
        value_col: str,
    ) -> Dict[Any, np.ndarray]:
        if frame.empty:
            return {}
        return (
            frame.groupby(group_cols)[value_col]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )

    def _build_spike_surfaces(self, prepared: pd.DataFrame) -> pd.DataFrame:
        df = prepared.copy()
        shock = pd.to_numeric(df["shock"], errors="coerce").fillna(0.0)
        zscore = pd.to_numeric(df["zscore"], errors="coerce").fillna(0.0)

        group_cols = ["era_id", "regime_id", "month", "hour"]
        global_q85 = float(shock.quantile(0.85))
        group_q85 = df.groupby(group_cols)["shock"].transform(lambda s: s.quantile(0.85))
        spike_threshold = np.maximum(group_q85.fillna(global_q85), global_q85)

        df["positive_spike_flag"] = (
            (shock > 0.0)
            & (shock >= spike_threshold)
            & (zscore >= self.spike_zscore_threshold)
        ).astype(int)

        prev_ts = pd.Series(df.index, index=df.index).shift(1)
        contiguous = ((pd.Series(df.index, index=df.index) - prev_ts) == pd.Timedelta(hours=1)).fillna(False)
        df["prev_positive_spike_flag"] = df["positive_spike_flag"].shift(1).fillna(0).astype(int)
        df.loc[~contiguous, "prev_positive_spike_flag"] = 0

        normal_df = df[df["positive_spike_flag"] == 0]
        spike_df = df[df["positive_spike_flag"] == 1]

        spike_global = np.asarray(pd.to_numeric(spike_df["shock"], errors="coerce").dropna(), dtype=float)
        transition_cols = group_cols + ["prev_positive_spike_flag"]
        transition_full = (
            df.groupby(transition_cols)["positive_spike_flag"]
            .mean()
            .to_dict()
        )
        transition_era_regime_month = (
            df.groupby(["era_id", "regime_id", "month", "prev_positive_spike_flag"])["positive_spike_flag"]
            .mean()
            .to_dict()
        )
        transition_era_regime = (
            df.groupby(["era_id", "regime_id", "prev_positive_spike_flag"])["positive_spike_flag"]
            .mean()
            .to_dict()
        )
        transition_regime_month_hour = (
            df.groupby(["regime_id", "month", "hour", "prev_positive_spike_flag"])["positive_spike_flag"]
            .mean()
            .to_dict()
        )
        transition_month_hour = (
            df.groupby(["month", "hour", "prev_positive_spike_flag"])["positive_spike_flag"]
            .mean()
            .to_dict()
        )
        transition_global = (
            df.groupby(["prev_positive_spike_flag"])["positive_spike_flag"]
            .mean()
            .to_dict()
        )

        self.shock_surface_.update(
            {
                "samples_normal_full": self._group_sample_arrays(normal_df, group_cols, "shock"),
                "samples_normal_era_regime_month": self._group_sample_arrays(
                    normal_df, ["era_id", "regime_id", "month"], "shock"
                ),
                "samples_normal_era_regime": self._group_sample_arrays(
                    normal_df, ["era_id", "regime_id"], "shock"
                ),
                "samples_normal_regime_month_hour": self._group_sample_arrays(
                    normal_df, ["regime_id", "month", "hour"], "shock"
                ),
                "samples_normal_month_hour": self._group_sample_arrays(
                    normal_df, ["month", "hour"], "shock"
                ),
                "samples_normal_global": np.asarray(
                    pd.to_numeric(normal_df["shock"], errors="coerce").dropna(),
                    dtype=float,
                ),
            }
        )

        self.spike_surface_ = {
            "samples_full": self._group_sample_arrays(spike_df, group_cols, "shock"),
            "samples_era_regime_month": self._group_sample_arrays(
                spike_df, ["era_id", "regime_id", "month"], "shock"
            ),
            "samples_era_regime": self._group_sample_arrays(
                spike_df, ["era_id", "regime_id"], "shock"
            ),
            "samples_regime_month_hour": self._group_sample_arrays(
                spike_df, ["regime_id", "month", "hour"], "shock"
            ),
            "samples_month_hour": self._group_sample_arrays(
                spike_df, ["month", "hour"], "shock"
            ),
            "samples_global": spike_global,
            "transition_full": transition_full,
            "transition_era_regime_month": transition_era_regime_month,
            "transition_era_regime": transition_era_regime,
            "transition_regime_month_hour": transition_regime_month_hour,
            "transition_month_hour": transition_month_hour,
            "transition_global": transition_global,
        }
        return df

    def _build_spdi_layers(self, prepared: pd.DataFrame) -> pd.DataFrame:
        df = prepared.copy()
        df["month"] = df.index.month.astype(int)
        df["hour"] = df.index.hour.astype(int)

        base_group = (
            df.groupby(["era_id", "regime_id", "month", "hour"])["spdi_ma24"]
            .median()
            .to_dict()
        )
        base_by_era_regime_month = (
            df.groupby(["era_id", "regime_id", "month"])["spdi_ma24"]
            .median()
            .to_dict()
        )
        base_by_era_regime = (
            df.groupby(["era_id", "regime_id"])["spdi_ma24"]
            .median()
            .to_dict()
        )
        base_by_regime_month_hour = (
            df.groupby(["regime_id", "month", "hour"])["spdi_ma24"]
            .median()
            .to_dict()
        )
        base_by_month_hour = (
            df.groupby(["month", "hour"])["spdi_ma24"]
            .median()
            .to_dict()
        )
        base_global = float(df["spdi_ma24"].median())

        cost_stats_full = (
            df.groupby(["era_id", "regime_id", "month", "hour"])["custo_fisico"]
            .agg(
                p10=lambda s: s.quantile(0.10),
                p50="median",
                p90=lambda s: s.quantile(0.90),
                p95=lambda s: s.quantile(0.95),
            )
            .ffill()
            .bfill()
        )
        cost_stats_era_regime_month = (
            df.groupby(["era_id", "regime_id", "month"])["custo_fisico"]
            .agg(
                p10=lambda s: s.quantile(0.10),
                p50="median",
                p90=lambda s: s.quantile(0.90),
                p95=lambda s: s.quantile(0.95),
            )
            .ffill()
            .bfill()
        )
        cost_stats_era_regime = (
            df.groupby(["era_id", "regime_id"])["custo_fisico"]
            .agg(
                p10=lambda s: s.quantile(0.10),
                p50="median",
                p90=lambda s: s.quantile(0.90),
                p95=lambda s: s.quantile(0.95),
            )
            .ffill()
            .bfill()
        )
        cost_stats_regime_month_hour = (
            df.groupby(["regime_id", "month", "hour"])["custo_fisico"]
            .agg(
                p10=lambda s: s.quantile(0.10),
                p50="median",
                p90=lambda s: s.quantile(0.90),
                p95=lambda s: s.quantile(0.95),
            )
            .ffill()
            .bfill()
        )
        cost_stats_month_hour = (
            df.groupby(["month", "hour"])["custo_fisico"]
            .agg(
                p10=lambda s: s.quantile(0.10),
                p50="median",
                p90=lambda s: s.quantile(0.90),
                p95=lambda s: s.quantile(0.95),
            )
            .ffill()
            .bfill()
        )
        cost_global = {
            "p10": float(df["custo_fisico"].quantile(0.10)),
            "p50": float(df["custo_fisico"].median()),
            "p90": float(df["custo_fisico"].quantile(0.90)),
            "p95": float(df["custo_fisico"].quantile(0.95)),
        }

        shock_stats = (
            df.groupby(["era_id", "regime_id", "month", "hour"])["shock"]
            .agg(
                mean="mean",
                std="std",
                p05=lambda s: s.quantile(0.05),
                p25=lambda s: s.quantile(0.25),
                p50=lambda s: s.quantile(0.50),
                p75=lambda s: s.quantile(0.75),
                p95=lambda s: s.quantile(0.95),
                count="count",
            )
            .fillna(0.0)
        )
        shock_samples = (
            df.groupby(["era_id", "regime_id", "month", "hour"])["shock"]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )
        shock_by_era_regime_month = (
            df.groupby(["era_id", "regime_id", "month"])["shock"]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )
        shock_by_era_regime = (
            df.groupby(["era_id", "regime_id"])["shock"]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )
        shock_by_regime_month_hour = (
            df.groupby(["regime_id", "month", "hour"])["shock"]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )
        shock_by_month_hour = (
            df.groupby(["month", "hour"])["shock"]
            .apply(lambda s: np.asarray(pd.to_numeric(s, errors="coerce").dropna(), dtype=float))
            .to_dict()
        )
        shock_global = np.asarray(pd.to_numeric(df["shock"], errors="coerce").dropna(), dtype=float)
        shock_global_stats = {
            "mean": float(np.nanmean(shock_global)) if shock_global.size else 0.0,
            "std": float(np.nanstd(shock_global)) if shock_global.size else 0.0,
            "p05": float(np.nanquantile(shock_global, 0.05)) if shock_global.size else 0.0,
            "p25": float(np.nanquantile(shock_global, 0.25)) if shock_global.size else 0.0,
            "p50": float(np.nanquantile(shock_global, 0.50)) if shock_global.size else 0.0,
            "p75": float(np.nanquantile(shock_global, 0.75)) if shock_global.size else 0.0,
            "p95": float(np.nanquantile(shock_global, 0.95)) if shock_global.size else 0.0,
        }

        self.spdi_base_surface_ = {
            "full": base_group,
            "era_regime_month": base_by_era_regime_month,
            "era_regime": base_by_era_regime,
            "regime_month_hour": base_by_regime_month_hour,
            "month_hour": base_by_month_hour,
            "global": base_global,
        }
        self.cost_surface_ = {
            "full": cost_stats_full.to_dict(orient="index"),
            "era_regime_month": cost_stats_era_regime_month.to_dict(orient="index"),
            "era_regime": cost_stats_era_regime.to_dict(orient="index"),
            "regime_month_hour": cost_stats_regime_month_hour.to_dict(orient="index"),
            "month_hour": cost_stats_month_hour.to_dict(orient="index"),
            "global": cost_global,
        }
        self.shock_surface_ = {
            "stats_full": shock_stats.to_dict(orient="index"),
            "samples_full": shock_samples,
            "samples_era_regime_month": shock_by_era_regime_month,
            "samples_era_regime": shock_by_era_regime,
            "samples_regime_month_hour": shock_by_regime_month_hour,
            "samples_month_hour": shock_by_month_hour,
            "stats_global": shock_global_stats,
            "samples_global": shock_global,
        }

        df = self._build_spike_surfaces(df)
        return df

    def _build_climatology(self, prepared: pd.DataFrame) -> None:
        climate_cols = [
            "ena",
            "load",
            "solar",
            "wind",
            "curtailment",
            "avail_ren",
        ]
        climate_df = prepared[climate_cols + ["era_id", "month", "hour"]].copy()
        for column_name in ("solar", "wind", "curtailment", "avail_ren"):
            climate_df[column_name] = pd.to_numeric(climate_df[column_name], errors="coerce").fillna(0.0)
        climate_df["avail_ren"] = climate_df["avail_ren"].fillna(
            climate_df["solar"].fillna(0) + climate_df["wind"].fillna(0) + climate_df["curtailment"].fillna(0)
        )
        climate_df = climate_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ena", "load", "solar", "wind"])

        full = {
            key: grp[climate_cols].to_numpy(dtype=float)
            for key, grp in climate_df.groupby(["era_id", "month", "hour"])
        }
        era_month = {
            key: grp[climate_cols].to_numpy(dtype=float)
            for key, grp in climate_df.groupby(["era_id", "month"])
        }
        month_hour = {
            key: grp[climate_cols].to_numpy(dtype=float)
            for key, grp in climate_df.groupby(["month", "hour"])
        }
        month_only = {
            key: grp[climate_cols].to_numpy(dtype=float)
            for key, grp in climate_df.groupby(["month"])
        }
        global_pool = climate_df[climate_cols].to_numpy(dtype=float)

        self.climatology_ = {
            "cols": climate_cols,
            "full": full,
            "era_month": era_month,
            "month_hour": month_hour,
            "month_only": month_only,
            "global": global_pool,
        }

    def _build_recent_profile_surface(self, prepared: pd.DataFrame) -> None:
        window = prepared.tail(max(self.recent_profile_hours, 24)).copy()
        if window.empty:
            self.recent_profile_surface_ = {}
            return

        window["weekday"] = window.index.weekday.astype(int)
        window["hour"] = window.index.hour.astype(int)

        def _profile_from_ratio(ratio: pd.Series, default: float) -> Dict[str, Any]:
            clean = pd.to_numeric(ratio, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if clean.empty:
                return {"weekhour": {}, "hour": {}, "global": float(default)}
            grouped = clean.groupby([window.loc[clean.index, "weekday"], window.loc[clean.index, "hour"]]).median().to_dict()
            grouped_hour = clean.groupby(window.loc[clean.index, "hour"]).median().to_dict()
            return {
                "weekhour": grouped,
                "hour": grouped_hour,
                "global": float(np.nanmedian(clean)),
            }

        def _blend_profiles(base: Dict[str, Any], overlay: Dict[str, Any], weight: float) -> Dict[str, Any]:
            weight = float(np.clip(weight, 0.0, 1.0))
            if weight <= 0.0 or not overlay:
                return base
            if not base:
                return overlay

            out = {"weekhour": {}, "hour": {}, "global": float(base.get("global", overlay.get("global", 0.0)))}
            for level in ("weekhour", "hour"):
                keys = set(base.get(level, {}).keys()) | set(overlay.get(level, {}).keys())
                merged = {}
                for key in keys:
                    base_value = base.get(level, {}).get(key, base.get("global", 0.0))
                    overlay_value = overlay.get(level, {}).get(key, overlay.get("global", base_value))
                    merged[key] = float((1.0 - weight) * base_value + weight * overlay_value)
                out[level] = merged
            out["global"] = float(
                (1.0 - weight) * float(base.get("global", 0.0))
                + weight * float(overlay.get("global", base.get("global", 0.0)))
            )
            return out

        load_base = max(float(pd.to_numeric(window["load"], errors="coerce").median()), EPS)
        ena_base = max(float(pd.to_numeric(window["ena"], errors="coerce").median()), EPS)
        thermal_base = max(float(pd.to_numeric(window["thermal_share"], errors="coerce").median()), EPS)

        load_ratio = (pd.to_numeric(window["load"], errors="coerce") / load_base).clip(0.5, 1.6)
        ena_ratio = (pd.to_numeric(window["ena"], errors="coerce") / ena_base).clip(0.35, 1.8)
        thermal_ratio = (pd.to_numeric(window["thermal_share"], errors="coerce") / thermal_base).clip(0.3, 1.8)
        spdi_intraday = (
            pd.to_numeric(window["spdi"], errors="coerce")
            / pd.to_numeric(window["spdi_ma24"], errors="coerce").replace(0, np.nan)
        ).clip(0.55, 1.75)
        spike_prob = pd.to_numeric(window.get("positive_spike_flag", pd.Series(0, index=window.index)), errors="coerce").fillna(0.0)

        surface = {
            "load_ratio": _profile_from_ratio(load_ratio, default=1.0),
            "ena_ratio": _profile_from_ratio(ena_ratio, default=1.0),
            "thermal_ratio": _profile_from_ratio(thermal_ratio, default=1.0),
            "spdi_intraday": _profile_from_ratio(spdi_intraday, default=1.0),
            "spike_prob": {
                "weekhour": spike_prob.groupby([window["weekday"], window["hour"]]).mean().to_dict(),
                "hour": spike_prob.groupby(window["hour"]).mean().to_dict(),
                "global": float(np.nanmean(spike_prob)) if len(spike_prob) else 0.0,
            },
        }
        spectral_meta: Dict[str, Any] = {}
        try:
            from spectral_engine import SpectralEngine

            spectral_history = prepared.tail(max(self.recent_profile_hours * 12, 24 * 90)).copy()
            spectral_engine = SpectralEngine(self.duckdb_path.parent)
            spectral_prior = spectral_engine.build_intraday_prior(
                spectral_history,
                window_hours=max(min(self.recent_profile_hours, 24 * 14), 24 * 7),
                top_k=5,
            )
            if spectral_prior:
                match_confidence = float(np.clip(spectral_prior.get("match_confidence", 0.0), 0.0, 1.0))
                spdi_weight = float(np.clip(0.12 + 0.38 * match_confidence, 0.12, 0.5))
                spike_weight = float(np.clip(0.10 + 0.34 * match_confidence, 0.10, 0.44))
                surface["spdi_intraday"] = _blend_profiles(
                    surface["spdi_intraday"],
                    spectral_prior.get("spdi_intraday", {}),
                    spdi_weight,
                )
                surface["spike_prob"] = _blend_profiles(
                    surface["spike_prob"],
                    spectral_prior.get("spike_prob", {}),
                    spike_weight,
                )
                spectral_meta = {
                    "matched_regime": spectral_prior.get("matched_regime", "unknown"),
                    "match_confidence": match_confidence,
                    "matched_window_start": spectral_prior.get("matched_window_start", ""),
                    "matched_window_end": spectral_prior.get("matched_window_end", ""),
                }
        except Exception as exc:
            LOGGER.warning("Unable to build spectral intraday prior: %s", exc)

        if spectral_meta:
            surface["spectral_prior"] = spectral_meta
        self.recent_profile_surface_ = surface

    def _load_pmo_guidance(self, prepared: pd.DataFrame) -> None:
        self.pmo_guidance_ = {}
        if not self.pmo_xlsx_path.exists():
            return

        try:
            from pld_forecast_engine import load_pmo_features
        except Exception as exc:
            LOGGER.warning("Unable to import PMO loader for adaptive guidance: %s", exc)
            return

        try:
            pmo_df = load_pmo_features(self.pmo_xlsx_path)
        except Exception as exc:
            LOGGER.warning("Unable to load PMO guidance from %s: %s", self.pmo_xlsx_path, exc)
            return

        if pmo_df.empty:
            return

        latest = pmo_df.sort_index().iloc[-1].copy()
        recent = prepared.tail(max(self.recent_profile_hours, 24))

        load_mean = self._safe_scalar(latest.get("carga_sem1_sin"), default=float(pd.to_numeric(recent["load"], errors="coerce").median()))
        thermal_total = self._safe_scalar(latest.get("term_total_sin"), default=np.nan)
        thermal_share = (
            float(np.clip(thermal_total / max(load_mean, EPS), 0.0, 1.0))
            if np.isfinite(thermal_total) and load_mean > 0
            else float(pd.to_numeric(recent["thermal_share"], errors="coerce").median())
        )
        self.pmo_guidance_ = {
            "week_ref": pd.to_datetime(latest.name, errors="coerce"),
            "ena_mean": self._safe_scalar(
                latest.get("ena_prev_seco_mw"),
                default=float(pd.to_numeric(recent["ena"], errors="coerce").median()),
            ),
            "load_mean": load_mean,
            "thermal_share_mean": thermal_share,
            "cmo_mean": self._safe_scalar(latest.get("cmo_med_seco"), default=np.nan),
            "ear_init": self._safe_scalar(
                latest.get("ear_init_seco"),
                default=float(pd.to_numeric(recent["ear"], errors="coerce").median()),
            ),
        }

    def _guidance_weight_for_step(self, step: int, base_weight: float) -> float:
        if step >= self.pmo_guidance_hours:
            return 0.0
        progress = step / max(self.pmo_guidance_hours - 1, 1)
        floor = min(0.45, base_weight)
        return float(np.clip(base_weight - (base_weight - floor) * progress, 0.0, 1.0))

    def _lookup_recent_profile(self, profile_name: str, ts: pd.Timestamp, default: float) -> float:
        profile = self.recent_profile_surface_.get(profile_name, {})
        candidates = [
            profile.get("weekhour", {}).get((int(ts.weekday()), int(ts.hour))),
            profile.get("hour", {}).get(int(ts.hour)),
            profile.get("global", default),
        ]
        for value in candidates:
            if value is not None and not np.isnan(value):
                return float(value)
        return float(default)

    @staticmethod
    def _adjacent_month(month: int, direction: int) -> int:
        month0 = (int(month) - 1 + direction) % 12
        return month0 + 1

    def _seasonal_month_weights(self, ts: pd.Timestamp) -> Dict[int, float]:
        month = int(ts.month)
        transition_days = max(int(self.season_transition_days), 0)
        if transition_days <= 0:
            return {month: 1.0}

        day = int(ts.day)
        days_in_month = int(ts.days_in_month)
        weights: Dict[int, float] = {month: 1.0}

        if day <= transition_days:
            prev_month = self._adjacent_month(month, -1)
            prev_weight = (transition_days - day + 1) / (transition_days + 1)
            weights = {prev_month: prev_weight, month: 1.0 - prev_weight}
        elif day > days_in_month - transition_days:
            next_month = self._adjacent_month(month, 1)
            next_weight = (day - (days_in_month - transition_days)) / (transition_days + 1)
            weights = {month: 1.0 - next_weight, next_month: next_weight}

        total = sum(max(float(v), 0.0) for v in weights.values())
        if total <= EPS:
            return {month: 1.0}
        return {key: float(value) / total for key, value in weights.items()}

    def _get_climatology_pool(self, era_id: int, month: int, hour: int) -> np.ndarray:
        pool = self.climatology_["full"].get((era_id, month, hour))
        if pool is None or len(pool) == 0:
            pool = self.climatology_["era_month"].get((era_id, month))
        if pool is None or len(pool) == 0:
            pool = self.climatology_["month_hour"].get((month, hour))
        if pool is None or len(pool) == 0:
            pool = self.climatology_["month_only"].get(month)
        if pool is None or len(pool) == 0:
            pool = self.climatology_["global"]
        return np.asarray(pool, dtype=float)

    @staticmethod
    def _feedback_lead_bucket(horizon_hour: int) -> str:
        if horizon_hour <= 24:
            return "h001_024"
        if horizon_hour <= 72:
            return "h025_072"
        if horizon_hour <= 168:
            return "h073_168"
        if horizon_hour <= 336:
            return "h169_336"
        if horizon_hour <= 720:
            return "h337_720"
        if horizon_hour <= 1440:
            return "h721_1440"
        if horizon_hour <= 2160:
            return "h1441_2160"
        return "h2161_plus"

    def _normalize_archive_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()

        out = frame.copy()
        dt_cols = {"generated_at", "forecast_ts"}
        text_cols = {"run_id", "model_version", "history_source"}

        def _unwrap_scalar(value: Any) -> Any:
            while isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                if len(value) == 0:
                    return np.nan
                value = value[0]
            return value

        for col in out.columns:
            out[col] = out[col].map(_unwrap_scalar)

        for col in ("generated_at", "forecast_ts"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
                try:
                    if getattr(out[col].dt, "tz", None) is not None:
                        out[col] = out[col].dt.tz_localize(None)
                except Exception:
                    pass
        for col in out.columns:
            if col not in dt_cols and col not in text_cols:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        sort_cols = [col for col in ("forecast_ts", "generated_at") if col in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        return out

    def _load_forecast_archive_from_duckdb(self) -> pd.DataFrame:
        if duckdb is None or not self.duckdb_path.exists():
            return pd.DataFrame()

        con = duckdb.connect(str(self.duckdb_path), read_only=True)
        try:
            try:
                tables = set(con.execute("SHOW TABLES").df()["name"].str.lower().tolist())
            except Exception:
                tables = set()
            if LOCAL_TABLE_NAME.lower() not in tables:
                return pd.DataFrame()

            cutoff_ts = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=self.feedback_lookback_days)).to_pydatetime()
            query = f"""
                SELECT
                    run_id,
                    generated_at,
                    forecast_ts,
                    horizon_hour,
                    era_id,
                    regime_prob_low,
                    regime_prob_mid,
                    regime_prob_high,
                    expected_pld,
                    pld_p10,
                    pld_p50,
                    pld_p90,
                    expected_spdi,
                    expected_custo_fisico,
                    model_version
                FROM {LOCAL_TABLE_NAME}
                WHERE forecast_ts >= ?
                ORDER BY generated_at, forecast_ts
            """
            return self._normalize_archive_frame(con.execute(query, [cutoff_ts]).df())
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

    def _load_forecast_archive_from_auth(self) -> pd.DataFrame:
        auth_url = os.getenv("DATABASE_URL_AUTH", os.getenv("DATABASE_URL", ""))
        if not auth_url or psycopg2 is None:
            return pd.DataFrame()
        conn = psycopg2.connect(auth_url)
        try:
            cur = conn.cursor()
            cur.execute("SELECT to_regclass(%s)", (AUTH_TABLE_NAME,))
            exists = cur.fetchone()
            cur.close()
            if not exists or not exists[0]:
                return pd.DataFrame()
            cutoff_ts = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=self.feedback_lookback_days)
            query = f"""
                SELECT
                    run_id,
                    generated_at,
                    forecast_ts,
                    horizon_hour,
                    era_id,
                    regime_prob_low,
                    regime_prob_mid,
                    regime_prob_high,
                    expected_pld,
                    pld_p10,
                    pld_p50,
                    pld_p90,
                    expected_spdi,
                    expected_custo_fisico,
                    model_version
                FROM {AUTH_TABLE_NAME}
                WHERE forecast_ts >= %s
                ORDER BY generated_at, forecast_ts
            """
            return self._normalize_archive_frame(pd.read_sql_query(query, conn, params=[cutoff_ts]))
        except Exception:
            return pd.DataFrame()
        finally:
            conn.close()

    def _load_forecast_archive(self) -> pd.DataFrame:
        local = self._load_forecast_archive_from_duckdb()
        if not local.empty:
            return local
        return self._load_forecast_archive_from_auth()

    def _build_quality_observation_table(self, prepared: pd.DataFrame) -> pd.DataFrame:
        archive = self._load_forecast_archive()
        if archive.empty or prepared.empty or "pld" not in prepared.columns:
            return pd.DataFrame()

        actual_series = pd.to_numeric(prepared["pld"], errors="coerce").dropna()
        if actual_series.empty:
            return pd.DataFrame()

        actual_series.index = pd.to_datetime(actual_series.index, errors="coerce")
        actual_series = actual_series[~actual_series.index.isna()]
        if actual_series.empty:
            return pd.DataFrame()

        latest_actual_ts = pd.Timestamp(actual_series.index.max())
        actual_cutoff = latest_actual_ts - pd.Timedelta(days=self.quality_lookback_days)
        actual_map = actual_series.to_dict()

        df = archive.copy()
        for col in ("generated_at", "forecast_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                try:
                    if getattr(df[col].dt, "tz", None) is not None:
                        df[col] = df[col].dt.tz_localize(None)
                except Exception:
                    pass

        df = df[df["generated_at"].notna() & df["forecast_ts"].notna()]
        df = df[df["generated_at"] < df["forecast_ts"]]
        df = df[df["forecast_ts"] <= latest_actual_ts]
        df = df[df["forecast_ts"] >= actual_cutoff]
        if df.empty:
            return pd.DataFrame()

        df["actual_pld"] = df["forecast_ts"].map(actual_map)
        df = df.dropna(subset=["actual_pld", "pld_p50", "pld_p10", "pld_p90", "horizon_hour"])
        df = df[(pd.to_numeric(df["actual_pld"], errors="coerce") > 0) & (pd.to_numeric(df["pld_p50"], errors="coerce") > 0)]
        if df.empty:
            return pd.DataFrame()

        df["horizon_hour"] = pd.to_numeric(df["horizon_hour"], errors="coerce").astype(int)
        df["lead_bucket"] = df["horizon_hour"].map(self._feedback_lead_bucket)
        regime_probs = df[["regime_prob_low", "regime_prob_mid", "regime_prob_high"]].fillna(0.0).to_numpy(dtype=float)
        dominant_regime_id = np.argmax(regime_probs, axis=1).astype(int)
        df["dominant_regime_id"] = dominant_regime_id
        df["dominant_regime"] = [REGIME_ID_TO_LABEL.get(int(value), "mid") for value in dominant_regime_id]
        df["lead_days"] = df["horizon_hour"].div(24.0)
        df["actual_date"] = df["forecast_ts"].dt.normalize()
        df["generated_date"] = df["generated_at"].dt.normalize()
        df["forecast_error"] = pd.to_numeric(df["pld_p50"], errors="coerce") - pd.to_numeric(df["actual_pld"], errors="coerce")
        df["abs_error"] = df["forecast_error"].abs()
        df["ape"] = (
            df["abs_error"]
            / pd.to_numeric(df["actual_pld"], errors="coerce").clip(lower=1.0)
        ).clip(lower=0.0, upper=5.0)
        df["inside_band"] = (
            (pd.to_numeric(df["actual_pld"], errors="coerce") >= pd.to_numeric(df["pld_p10"], errors="coerce"))
            & (pd.to_numeric(df["actual_pld"], errors="coerce") <= pd.to_numeric(df["pld_p90"], errors="coerce"))
        ).astype(int)
        df["below_p10"] = (
            pd.to_numeric(df["actual_pld"], errors="coerce") < pd.to_numeric(df["pld_p10"], errors="coerce")
        ).astype(int)
        df["above_p90"] = (
            pd.to_numeric(df["actual_pld"], errors="coerce") > pd.to_numeric(df["pld_p90"], errors="coerce")
        ).astype(int)
        age_days = (
            latest_actual_ts - pd.to_datetime(df["forecast_ts"], errors="coerce")
        ).dt.total_seconds().div(24 * 3600).clip(lower=0.0)
        decay = np.log(2) / max(float(self.feedback_half_life_days), 1.0)
        df["quality_weight"] = np.exp(-decay * age_days.to_numpy(dtype=float))

        obs_cols = [
            "run_id",
            "generated_at",
            "generated_date",
            "forecast_ts",
            "actual_date",
            "horizon_hour",
            "lead_days",
            "lead_bucket",
            "era_id",
            "dominant_regime_id",
            "dominant_regime",
            "expected_pld",
            "pld_p10",
            "pld_p50",
            "pld_p90",
            "actual_pld",
            "forecast_error",
            "abs_error",
            "ape",
            "inside_band",
            "below_p10",
            "above_p90",
            "quality_weight",
            "model_version",
        ]
        out = df[obs_cols].copy()
        out.insert(0, "engine_name", "adaptive_pld_forward")
        out = out.sort_values(["forecast_ts", "generated_at", "run_id"]).reset_index(drop=True)
        return out

    def _build_quality_summary_table(self, quality_obs: pd.DataFrame) -> pd.DataFrame:
        if quality_obs is None or quality_obs.empty:
            return pd.DataFrame()

        obs = quality_obs.copy()
        obs["forecast_ts"] = pd.to_datetime(obs["forecast_ts"], errors="coerce")
        obs = obs[obs["forecast_ts"].notna()]
        if obs.empty:
            return pd.DataFrame()

        latest_actual_ts = pd.Timestamp(obs["forecast_ts"].max())
        lead_bucket_order = [
            "all",
            "h001_024",
            "h025_072",
            "h073_168",
            "h169_336",
            "h337_720",
            "h721_1440",
            "h1441_2160",
            "h2161_plus",
        ]

        rows: list[dict[str, Any]] = []
        updated_at = pd.Timestamp.utcnow().tz_localize(None)

        def _summarize(group: pd.DataFrame, window_days: int, lead_bucket: str) -> None:
            if group.empty:
                return
            error = pd.to_numeric(group["forecast_error"], errors="coerce")
            abs_error = pd.to_numeric(group["abs_error"], errors="coerce")
            ape = pd.to_numeric(group["ape"], errors="coerce")
            inside = pd.to_numeric(group["inside_band"], errors="coerce")
            below = pd.to_numeric(group["below_p10"], errors="coerce")
            above = pd.to_numeric(group["above_p90"], errors="coerce")
            weights = pd.to_numeric(group["quality_weight"], errors="coerce").fillna(0.0)
            weight_sum = float(weights.sum())
            if weight_sum > 0:
                weighted_mae = float(np.average(abs_error, weights=weights))
                weighted_mape = float(np.average(ape, weights=weights) * 100.0)
                weighted_bias = float(np.average(error, weights=weights))
            else:
                weighted_mae = float(abs_error.mean())
                weighted_mape = float(ape.mean() * 100.0)
                weighted_bias = float(error.mean())

            rows.append(
                {
                    "engine_name": "adaptive_pld_forward",
                    "updated_at": updated_at,
                    "window_days": int(window_days),
                    "lead_bucket": str(lead_bucket),
                    "n_obs": int(len(group)),
                    "actual_start_ts": pd.Timestamp(group["forecast_ts"].min()),
                    "actual_end_ts": pd.Timestamp(group["forecast_ts"].max()),
                    "mae": float(abs_error.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mape_pct": float(ape.mean() * 100.0),
                    "median_ape_pct": float(ape.median() * 100.0),
                    "bias": float(error.mean()),
                    "weighted_mae": weighted_mae,
                    "weighted_mape_pct": weighted_mape,
                    "weighted_bias": weighted_bias,
                    "band_coverage": float(inside.mean()),
                    "coverage_p10": float(1.0 - below.mean()),
                    "coverage_p90": float(1.0 - above.mean()),
                    "below_p10_rate": float(below.mean()),
                    "above_p90_rate": float(above.mean()),
                }
            )

        for window_days in QUALITY_SUMMARY_WINDOWS:
            cutoff = latest_actual_ts - pd.Timedelta(days=window_days)
            window_df = obs[obs["forecast_ts"] >= cutoff].copy()
            if window_df.empty:
                continue

            _summarize(window_df, window_days, "all")
            for lead_bucket in lead_bucket_order[1:]:
                _summarize(window_df[window_df["lead_bucket"] == lead_bucket], window_days, lead_bucket)

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["lead_bucket"] = pd.Categorical(out["lead_bucket"], categories=lead_bucket_order, ordered=True)
        return out.sort_values(["window_days", "lead_bucket"]).reset_index(drop=True)

    def _weighted_group_stats(
        self,
        frame: pd.DataFrame,
        group_cols: list[str],
        value_col: str,
        aux_col: str,
    ) -> Dict[Any, Dict[str, float]]:
        if frame.empty:
            return {}

        stats: Dict[Any, Dict[str, float]] = {}
        for keys, grp in frame.groupby(group_cols):
            weights = pd.to_numeric(grp["feedback_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            values = pd.to_numeric(grp[value_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            aux = pd.to_numeric(grp[aux_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if weights.sum() <= 0 or len(values) == 0:
                continue
            stats[keys] = {
                "value": float(np.average(values, weights=weights)),
                "error": float(np.average(aux, weights=weights)),
                "n": float(len(grp)),
                "weight": float(weights.sum()),
            }
        return stats

    def _fit_feedback_surface(self, prepared: pd.DataFrame) -> None:
        self.feedback_surface_ = {}
        archive = self._load_forecast_archive()
        if archive.empty:
            return

        actual_series = pd.to_numeric(prepared["pld"], errors="coerce").dropna()
        if actual_series.empty:
            return
        latest_actual_ts = pd.Timestamp(actual_series.index.max())
        actual_cutoff = latest_actual_ts - pd.Timedelta(days=self.feedback_lookback_days)
        actual_map = actual_series.to_dict()

        df = archive.copy()
        for col in ("generated_at", "forecast_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                try:
                    if getattr(df[col].dt, "tz", None) is not None:
                        df[col] = df[col].dt.tz_localize(None)
                except Exception:
                    pass
        df = df[df["generated_at"].notna() & df["forecast_ts"].notna()]
        df = df[df["generated_at"] < df["forecast_ts"]]
        df = df[df["forecast_ts"] <= latest_actual_ts]
        df = df[df["forecast_ts"] >= actual_cutoff]
        if df.empty:
            return

        df["actual_pld"] = df["forecast_ts"].map(actual_map)
        df = df.dropna(subset=["actual_pld", "pld_p50", "horizon_hour"])
        df = df[(df["actual_pld"] > 0) & (df["pld_p50"] > 0)]
        if df.empty:
            return

        df["lead_bucket"] = df["horizon_hour"].astype(int).map(self._feedback_lead_bucket)
        df["forecast_month"] = df["forecast_ts"].dt.month.astype(int)
        df["forecast_hour"] = df["forecast_ts"].dt.hour.astype(int)
        df["dominant_regime"] = np.argmax(
            df[["regime_prob_low", "regime_prob_mid", "regime_prob_high"]].fillna(0.0).to_numpy(dtype=float),
            axis=1,
        ).astype(int)
        df["ratio_log"] = np.log(
            np.clip(
                pd.to_numeric(df["actual_pld"], errors="coerce") / pd.to_numeric(df["pld_p50"], errors="coerce"),
                0.45,
                1.75,
            )
        ).clip(lower=-0.6, upper=0.6)
        df["ape"] = (
            (pd.to_numeric(df["actual_pld"], errors="coerce") - pd.to_numeric(df["pld_p50"], errors="coerce")).abs()
            / pd.to_numeric(df["pld_p50"], errors="coerce").clip(lower=1.0)
        ).clip(lower=0.0, upper=3.0)
        age_days = (
            latest_actual_ts - pd.to_datetime(df["forecast_ts"], errors="coerce")
        ).dt.total_seconds().div(24 * 3600).clip(lower=0.0)
        decay = np.log(2) / max(float(self.feedback_half_life_days), 1.0)
        df["feedback_weight"] = np.exp(-decay * age_days.to_numpy(dtype=float))
        df = df[df["feedback_weight"] > 0]
        if df.empty:
            return

        def _ratio_surface(group_cols: list[str]) -> Dict[Any, Dict[str, float]]:
            raw = self._weighted_group_stats(df, group_cols, "ratio_log", "ape")
            out: Dict[Any, Dict[str, float]] = {}
            for key, stats in raw.items():
                out[key] = {
                    "value": float(np.exp(stats["value"])),
                    "error": stats["error"],
                    "n": stats["n"],
                    "weight": stats["weight"],
                }
            return out

        global_log = float(np.average(df["ratio_log"], weights=df["feedback_weight"]))
        global_ape = float(np.average(df["ape"], weights=df["feedback_weight"]))
        self.feedback_surface_ = {
            "full": _ratio_surface(["lead_bucket", "forecast_month", "forecast_hour", "dominant_regime"]),
            "lead_hour_regime": _ratio_surface(["lead_bucket", "forecast_hour", "dominant_regime"]),
            "lead_hour": _ratio_surface(["lead_bucket", "forecast_hour"]),
            "lead": _ratio_surface(["lead_bucket"]),
            "month_hour": _ratio_surface(["forecast_month", "forecast_hour"]),
            "global": {
                "value": float(np.exp(global_log)),
                "error": global_ape,
                "n": float(len(df)),
                "weight": float(df["feedback_weight"].sum()),
            },
        }

    def _lookup_feedback_multiplier(self, horizon_hour: int, forecast_ts: pd.Timestamp, regime_id: int) -> tuple[float, float]:
        if not self.feedback_surface_:
            return 1.0, 1.0

        lead_bucket = self._feedback_lead_bucket(int(horizon_hour))
        month = int(forecast_ts.month)
        hour = int(forecast_ts.hour)
        candidates = [
            self.feedback_surface_.get("full", {}).get((lead_bucket, month, hour, regime_id)),
            self.feedback_surface_.get("lead_hour_regime", {}).get((lead_bucket, hour, regime_id)),
            self.feedback_surface_.get("lead_hour", {}).get((lead_bucket, hour)),
            self.feedback_surface_.get("lead", {}).get(lead_bucket),
            self.feedback_surface_.get("month_hour", {}).get((month, hour)),
            self.feedback_surface_.get("global"),
        ]
        chosen: Dict[str, float] | None = None
        for value in candidates:
            if value is not None:
                chosen = value
                break
        if not chosen:
            return 1.0, 1.0

        support = float(chosen.get("weight", 0.0))
        support_strength = np.clip(support / max(self.feedback_min_weight, EPS), 0.0, 1.0)
        horizon_strength = float(
            np.interp(
                float(horizon_hour),
                [1.0, 168.0, 720.0, 2160.0, 4320.0],
                [1.0, 0.95, 0.75, 0.55, 0.4],
            )
        )
        reliability = float(np.clip(1.0 / (1.0 + float(chosen.get("error", 0.0))), 0.25, 1.0))
        blend = self.feedback_strength_max * support_strength * horizon_strength * reliability
        raw_multiplier = float(chosen.get("value", 1.0))
        adjusted = 1.0 + (raw_multiplier - 1.0) * blend
        adjusted = float(np.clip(adjusted, 1.0 - self.feedback_ratio_clip, 1.0 + self.feedback_ratio_clip))
        return adjusted, reliability

    def _fit_physical_models(self, prepared: pd.DataFrame) -> None:
        df = prepared.copy()

        df["ear_delta"] = df["ear"].shift(-1) - df["ear"]
        df["thermal_share_next"] = df["thermal_share"].shift(-1)
        self.model_bundles_ = {
            "ear_delta": self._fit_regressor_bundle(df, self.dynamic_feature_cols, "ear_delta", model_type="gbr"),
            "thermal_share": self._fit_regressor_bundle(
                df,
                self.dynamic_feature_cols,
                "thermal_share_next",
                model_type="rf",
            ),
            "custo_fisico": self._fit_regressor_bundle(df, self.dynamic_feature_cols, "custo_fisico", model_type="gbr"),
        }

    def _fit_era_classifier(self, prepared: pd.DataFrame) -> None:
        self.era_classifier_ = self._fit_classifier_bundle(
            frame=prepared,
            feature_cols=self.regime_feature_cols,
            target_col="era_id",
            classes=sorted(int(v) for v in prepared["era_id"].dropna().unique()),
        )

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------
    def _simulate_paths(
        self,
        prepared: pd.DataFrame,
        current_state: pd.Series,
        current_era: int,
        horizon_hours: int,
        n_paths: int,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.random_state)
        start_ts = pd.Timestamp(current_state.name)

        pld_paths = np.zeros((n_paths, horizon_hours), dtype=float)
        spdi_paths = np.zeros((n_paths, horizon_hours), dtype=float)
        cost_paths = np.zeros((n_paths, horizon_hours), dtype=float)
        regime_probability_paths = np.zeros((horizon_hours, 3), dtype=float)
        feedback_reliability_paths = np.ones(horizon_hours, dtype=float)

        current_ear = np.full(n_paths, float(current_state["ear"]), dtype=float)
        current_thermal_share = np.full(
            n_paths,
            self._safe_scalar(current_state.get("thermal_share", 0.0), default=0.0),
            dtype=float,
        )
        prev_zscore = np.full(
            n_paths,
            self._safe_scalar(current_state.get("zscore", 0.0), default=0.0),
            dtype=float,
        )
        current_spike_flag = int(self._safe_scalar(current_state.get("positive_spike_flag", 0.0), default=0.0) > 0.0)
        prev_spike_state = np.full(n_paths, current_spike_flag, dtype=bool)
        prev_spike_shock = np.full(
            n_paths,
            max(self._safe_scalar(current_state.get("shock", 0.0), default=0.0), 0.0) if current_spike_flag else 0.0,
            dtype=float,
        )

        forecast_index: list[pd.Timestamp] = []

        for step in range(horizon_hours):
            forecast_ts = start_ts + pd.Timedelta(hours=step + 1)
            forecast_index.append(forecast_ts)
            sampled = self._sample_climatology_rows(
                rng=rng,
                era_id=current_era,
                ts=forecast_ts,
                n_samples=n_paths,
            )

            load = sampled["load"]
            solar = sampled["solar"]
            wind = sampled["wind"]
            ena = sampled["ena"]
            pmo_weight = (
                self._guidance_weight_for_step(step, self.pmo_guidance_weight)
                if self.pmo_guidance_
                else 0.0
            )
            spdi_recent_weight = (
                self._guidance_weight_for_step(step, self.spdi_recent_weight)
                if self.recent_profile_surface_
                else 0.0
            )
            spectral_confidence = self._safe_scalar(
                self.recent_profile_surface_.get("spectral_prior", {}).get("match_confidence"),
                default=0.0,
            )
            if spdi_recent_weight > 0 and spectral_confidence > 0:
                spdi_recent_weight = float(
                    np.clip(spdi_recent_weight + 0.10 * spectral_confidence, 0.0, 0.92)
                )
            if pmo_weight > 0:
                load_target = self._safe_scalar(self.pmo_guidance_.get("load_mean"), default=np.nan)
                ena_target = self._safe_scalar(self.pmo_guidance_.get("ena_mean"), default=np.nan)
                if np.isfinite(load_target) and load_target > 0:
                    load_shape = self._lookup_recent_profile("load_ratio", forecast_ts, default=1.0)
                    load = (1.0 - pmo_weight) * load + pmo_weight * (load_target * load_shape)
                if np.isfinite(ena_target) and ena_target > 0:
                    ena_shape = self._lookup_recent_profile("ena_ratio", forecast_ts, default=1.0)
                    ena = (1.0 - pmo_weight) * ena + pmo_weight * (ena_target * ena_shape)
            curtailment = np.maximum(sampled["curtailment"], 0.0)
            avail_ren = np.maximum(sampled["avail_ren"], solar + wind + curtailment)
            net_load = load - (solar + wind)
            curtailment_ratio = np.clip(
                curtailment / np.maximum(avail_ren, EPS),
                0.0,
                1.0,
            )
            isr = np.clip(avail_ren / np.maximum(np.abs(net_load), EPS), 0.0, 25.0)

            features_for_ear = self._build_dynamic_feature_frame(
                ear=current_ear,
                ena=ena,
                load=load,
                solar=solar,
                wind=wind,
                net_load=net_load,
                thermal_share=current_thermal_share,
                curtailment_ratio=curtailment_ratio,
                isr=isr,
                month=forecast_ts.month,
                hour=forecast_ts.hour,
            )
            ear_delta = self._predict_regressor("ear_delta", features_for_ear, rng=rng)
            current_ear = current_ear + ear_delta
            current_ear = np.clip(
                current_ear,
                self.bounds_["ear"][0],
                self.bounds_["ear"][1],
            )
            if pmo_weight > 0:
                ear_anchor = self._safe_scalar(self.pmo_guidance_.get("ear_init"), default=np.nan)
                if np.isfinite(ear_anchor):
                    ear_pull = 0.18 * pmo_weight
                    current_ear = (1.0 - ear_pull) * current_ear + ear_pull * ear_anchor

            features_for_thermal = self._build_dynamic_feature_frame(
                ear=current_ear,
                ena=ena,
                load=load,
                solar=solar,
                wind=wind,
                net_load=net_load,
                thermal_share=current_thermal_share,
                curtailment_ratio=curtailment_ratio,
                isr=isr,
                month=forecast_ts.month,
                hour=forecast_ts.hour,
            )
            thermal_share = self._predict_regressor("thermal_share", features_for_thermal, rng=rng)
            thermal_share = np.clip(
                thermal_share,
                self.bounds_["thermal_share"][0],
                self.bounds_["thermal_share"][1],
            )
            if pmo_weight > 0:
                thermal_anchor = self._safe_scalar(self.pmo_guidance_.get("thermal_share_mean"), default=np.nan)
                if np.isfinite(thermal_anchor):
                    thermal_shape = self._lookup_recent_profile("thermal_ratio", forecast_ts, default=1.0)
                    guided_thermal_share = np.clip(
                        thermal_anchor * thermal_shape,
                        self.bounds_["thermal_share"][0],
                        self.bounds_["thermal_share"][1],
                    )
                    thermal_share = (1.0 - pmo_weight) * thermal_share + pmo_weight * guided_thermal_share
            current_thermal_share = thermal_share

            features_for_cost = self._build_dynamic_feature_frame(
                ear=current_ear,
                ena=ena,
                load=load,
                solar=solar,
                wind=wind,
                net_load=net_load,
                thermal_share=thermal_share,
                curtailment_ratio=curtailment_ratio,
                isr=isr,
                month=forecast_ts.month,
                hour=forecast_ts.hour,
            )
            custo_fisico = self._predict_regressor("custo_fisico", features_for_cost, rng=rng)
            custo_fisico = np.clip(
                custo_fisico,
                self.bounds_["custo_fisico"][0],
                self.bounds_["custo_fisico"][1],
            )
            if pmo_weight > 0:
                cmo_anchor = self._safe_scalar(self.pmo_guidance_.get("cmo_mean"), default=np.nan)
                if np.isfinite(cmo_anchor) and cmo_anchor > 0:
                    cost_pull = 0.32 * pmo_weight
                    custo_fisico = (1.0 - cost_pull) * custo_fisico + cost_pull * cmo_anchor

            regime_features = pd.DataFrame(
                {
                    "ear": current_ear,
                    "ena": ena,
                    "net_load": net_load,
                    "thermal_share": thermal_share,
                    "curtailment": curtailment,
                    "isr": isr,
                }
            )
            regime_probs = self._predict_regime_probabilities(current_era, regime_features)
            regime_probability_paths[step] = regime_probs.mean(axis=0)
            regime_ids = np.array(
                [int(rng.choice([0, 1, 2], p=self._normalize_probs(prob))) for prob in regime_probs],
                dtype=int,
            )

            seasonal_cost_weight = float(
                np.interp(
                    float(step + 1),
                    [1.0, 168.0, 720.0, 2160.0, 4320.0],
                    [0.06, 0.08, 0.14, self.cost_anchor_weight, min(self.cost_anchor_weight + 0.08, 0.34)],
                )
            )
            seasonal_cost_weight *= 1.0 - 0.35 * pmo_weight
            for regime_id in (0, 1, 2):
                mask = regime_ids == regime_id
                if not np.any(mask):
                    continue
                cost_anchor = self._lookup_cost_stat(
                    era_id=current_era,
                    regime_id=regime_id,
                    ts=forecast_ts,
                    stat="p50",
                )
                cost_floor = self._lookup_cost_stat(
                    era_id=current_era,
                    regime_id=regime_id,
                    ts=forecast_ts,
                    stat="p10",
                )
                cost_cap = self._lookup_cost_stat(
                    era_id=current_era,
                    regime_id=regime_id,
                    ts=forecast_ts,
                    stat="p95",
                )
                if np.isfinite(cost_anchor) and cost_anchor > 0:
                    custo_fisico[mask] = (
                        (1.0 - seasonal_cost_weight) * custo_fisico[mask]
                        + seasonal_cost_weight * cost_anchor
                    )
                local_floor = max(cost_floor * 0.92, self.bounds_["custo_fisico"][0])
                # Teto absoluto: evitar divergência exponencial em horizontes longos
                # Limitar a ou_long_run_clip × mu_long_run (padrão: 3 × 150 = 450 R$/MWh)
                abs_cap = self.ou_mu_ * self.ou_long_run_clip
                local_cap = min(
                    max(cost_cap * self.cost_tail_clip, cost_anchor * 1.35, local_floor + 1.0),
                    abs_cap,
                )
                custo_fisico[mask] = np.clip(custo_fisico[mask], local_floor, local_cap)

            # ── Ornstein-Uhlenbeck mean-reversion no custo_fisico ───────────────
            # Aplica força restauradora em direção à média histórica de longo prazo
            # dX = κ(μ - X)dt + σdW  →  aplicado como correção suave por hora
            # Efeito crescente com o horizonte: previne divergência exponencial
            _ou_pull = self.ou_kappa * (self.ou_mu_ - custo_fisico)
            # Peso do OU aumenta com o horizonte (0% h=1 → 50% h=720 → 80% h=4320)
            _ou_weight = float(np.interp(
                float(step + 1),
                [1.0, 168.0, 720.0, 2160.0, 4320.0],
                [0.0, 0.15,  0.35,  0.55,   0.70],
            ))
            custo_fisico = custo_fisico + _ou_weight * _ou_pull
            # Re-aplicar piso após OU
            custo_fisico = np.clip(
                custo_fisico,
                self.bounds_["custo_fisico"][0],
                self.ou_mu_ * self.ou_long_run_clip,
            )

            spdi_trend = np.zeros(n_paths, dtype=float)
            normal_shock = np.zeros(n_paths, dtype=float)
            spike_shock = np.zeros(n_paths, dtype=float)
            shock_scale = np.ones(n_paths, dtype=float)
            current_spike_state = np.zeros(n_paths, dtype=bool)
            feedback_multiplier = np.ones(n_paths, dtype=float)
            feedback_reliability = np.ones(n_paths, dtype=float)

            for regime_id in (0, 1, 2):
                mask = regime_ids == regime_id
                if not np.any(mask):
                    continue
                regime_feedback_multiplier, regime_feedback_reliability = self._lookup_feedback_multiplier(
                    horizon_hour=step + 1,
                    forecast_ts=forecast_ts,
                    regime_id=regime_id,
                )
                feedback_multiplier[mask] = regime_feedback_multiplier
                feedback_reliability[mask] = regime_feedback_reliability
                base_spdi = self._lookup_spdi_base(
                    era_id=current_era,
                    regime_id=regime_id,
                    month=forecast_ts.month,
                    hour=forecast_ts.hour,
                )
                if spdi_recent_weight > 0:
                    intraday_factor = self._lookup_recent_profile("spdi_intraday", forecast_ts, default=1.0)
                    base_spdi = base_spdi * (
                        (1.0 - spdi_recent_weight) + spdi_recent_weight * intraday_factor
                    )
                spdi_trend[mask] = base_spdi
                sampled_shock = self._sample_normal_shock(
                    rng=rng,
                    era_id=current_era,
                    regime_id=regime_id,
                    month=forecast_ts.month,
                    hour=forecast_ts.hour,
                    n_samples=int(mask.sum()),
                )
                normal_shock[mask] = sampled_shock
                shock_scale[mask] = max(
                    self._lookup_shock_std(
                        era_id=current_era,
                        regime_id=regime_id,
                        month=forecast_ts.month,
                        hour=forecast_ts.hour,
                    ),
                    EPS,
                )

                mask_idx = np.flatnonzero(mask)
                prev_state_mask = prev_spike_state[mask_idx]
                prob_start = self._lookup_spike_probability(
                    era_id=current_era,
                    regime_id=regime_id,
                    month=forecast_ts.month,
                    hour=forecast_ts.hour,
                    prev_state=0,
                )
                prob_keep = self._lookup_spike_probability(
                    era_id=current_era,
                    regime_id=regime_id,
                    month=forecast_ts.month,
                    hour=forecast_ts.hour,
                    prev_state=1,
                )
                if spdi_recent_weight > 0:
                    recent_spike_prob = self._lookup_recent_profile("spike_prob", forecast_ts, default=0.0)
                    prob_start = np.clip(
                        (1.0 - 0.65 * spdi_recent_weight) * prob_start
                        + 0.65 * spdi_recent_weight * recent_spike_prob,
                        0.0,
                        1.0,
                    )
                    prob_keep = np.clip(
                        (1.0 - 0.45 * spdi_recent_weight) * prob_keep
                        + 0.45 * spdi_recent_weight * max(recent_spike_prob, prob_keep),
                        0.0,
                        1.0,
                    )
                spike_probs = np.where(prev_state_mask, prob_keep, prob_start)
                activated = rng.random(len(mask_idx)) < spike_probs
                current_spike_state[mask_idx] = activated

                if np.any(activated):
                    active_idx = mask_idx[activated]
                    spike_draw = self._sample_spike_shock(
                        rng=rng,
                        era_id=current_era,
                        regime_id=regime_id,
                        month=forecast_ts.month,
                        hour=forecast_ts.hour,
                        n_samples=len(active_idx),
                    )
                    persistence = np.where(
                        prev_spike_state[active_idx],
                        self.spike_persistence,
                        0.0,
                    )
                    spike_shock[active_idx] = (
                        persistence * prev_spike_shock[active_idx]
                        + (1.0 - persistence) * spike_draw
                    )

            mean_reversion_mask = (np.abs(prev_zscore) > self.zscore_reversion_threshold) & (~current_spike_state)
            normal_shock[mean_reversion_mask] *= self.shock_decay_factor
            spdi_shock = normal_shock + spike_shock

            spdi = np.clip(
                (spdi_trend + spdi_shock) * feedback_multiplier,
                self.bounds_["spdi"][0],
                self.bounds_["spdi"][1],
            )
            pld = custo_fisico * spdi

            pld_paths[:, step] = pld
            spdi_paths[:, step] = spdi
            cost_paths[:, step] = custo_fisico
            prev_zscore = spdi_shock / shock_scale
            prev_spike_state = current_spike_state
            prev_spike_shock = np.where(
                current_spike_state,
                np.maximum(spike_shock, 0.0),
                np.maximum(prev_spike_shock * 0.35, 0.0),
            )
            feedback_reliability_paths[step] = float(np.nanmean(feedback_reliability))

        return {
            "forecast_index": pd.DatetimeIndex(forecast_index),
            "pld_paths": pld_paths,
            "spdi_paths": spdi_paths,
            "cost_paths": cost_paths,
            "regime_probabilities_by_step": regime_probability_paths,
            "feedback_reliability_by_step": feedback_reliability_paths,
        }

    def _extract_scenarios(self, pld_paths: np.ndarray) -> Dict[str, ScenarioSummary]:
        if pld_paths.shape[0] < 3:
            median_path = np.median(pld_paths, axis=0)
            scenario = ScenarioSummary(
                trajectory=[round(float(v), 4) for v in median_path.tolist()],
                final_value=round(float(median_path[-1]), 4),
            )
            return {"abundance": scenario, "base": scenario, "stress": scenario}

        model = KMeans(n_clusters=3, random_state=self.random_state, n_init=20)
        model.fit(pld_paths)
        centers = model.cluster_centers_
        ordering = np.argsort(centers[:, -1])
        names = ["abundance", "base", "stress"]

        scenarios: Dict[str, ScenarioSummary] = {}
        for scenario_name, cluster_idx in zip(names, ordering):
            centroid = centers[cluster_idx]
            scenarios[scenario_name] = ScenarioSummary(
                trajectory=[round(float(v), 4) for v in centroid.tolist()],
                final_value=round(float(centroid[-1]), 4),
            )
        return scenarios

    def _build_hourly_table(
        self,
        generated_at: pd.Timestamp,
        run_id: str,
        current_era: int,
        sim: Dict[str, Any],
        scenarios: Dict[str, ScenarioSummary],
        confidence_score: float,
        n_paths: int,
    ) -> pd.DataFrame:
        forecast_index = sim["forecast_index"]
        pld_paths = sim["pld_paths"]
        spdi_paths = sim["spdi_paths"]
        cost_paths = sim["cost_paths"]
        regime_probabilities = sim["regime_probabilities_by_step"]

        pld_p10 = np.percentile(pld_paths, 10, axis=0)
        pld_p50 = np.percentile(pld_paths, 50, axis=0)
        pld_p90 = np.percentile(pld_paths, 90, axis=0)
        spdi_p10 = np.percentile(spdi_paths, 10, axis=0)
        spdi_p50 = np.percentile(spdi_paths, 50, axis=0)
        spdi_p90 = np.percentile(spdi_paths, 90, axis=0)
        cost_p10 = np.percentile(cost_paths, 10, axis=0)
        cost_p50 = np.percentile(cost_paths, 50, axis=0)
        cost_p90 = np.percentile(cost_paths, 90, axis=0)

        table = pd.DataFrame(
            {
                "run_id": run_id,
                "generated_at": generated_at,
                "forecast_ts": forecast_index,
                "horizon_hour": np.arange(1, len(forecast_index) + 1, dtype=int),
                "era_id": current_era,
                "regime_prob_low": regime_probabilities[:, 0],
                "regime_prob_mid": regime_probabilities[:, 1],
                "regime_prob_high": regime_probabilities[:, 2],
                "expected_pld": pld_paths.mean(axis=0),
                "pld_p10": pld_p10,
                "pld_p50": pld_p50,
                "pld_p90": pld_p90,
                "scenario_abundance": scenarios["abundance"].trajectory,
                "scenario_base": scenarios["base"].trajectory,
                "scenario_stress": scenarios["stress"].trajectory,
                "expected_spdi": spdi_paths.mean(axis=0),
                "spdi_p10": spdi_p10,
                "spdi_p50": spdi_p50,
                "spdi_p90": spdi_p90,
                "expected_custo_fisico": cost_paths.mean(axis=0),
                "custo_fisico_p10": cost_p10,
                "custo_fisico_p50": cost_p50,
                "custo_fisico_p90": cost_p90,
                "confidence_score": [
                    self._confidence_from_band(p10, p50, p90)
                    for p10, p50, p90 in zip(pld_p10, pld_p50, pld_p90)
                ],
                "confidence_score_global": confidence_score,
                "n_paths": n_paths,
                "model_version": MODEL_VERSION,
                "history_source": self.history_source_,
            }
        )
        return table

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_to_duckdb(self, hourly_table: pd.DataFrame) -> None:
        if duckdb is None:
            raise ImportError("duckdb is required to persist local results.")

        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.duckdb_path))
        try:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {LOCAL_TABLE_NAME} (
                    run_id VARCHAR,
                    generated_at TIMESTAMP,
                    forecast_ts TIMESTAMP,
                    horizon_hour INTEGER,
                    era_id INTEGER,
                    regime_prob_low DOUBLE,
                    regime_prob_mid DOUBLE,
                    regime_prob_high DOUBLE,
                    expected_pld DOUBLE,
                    pld_p10 DOUBLE,
                    pld_p50 DOUBLE,
                    pld_p90 DOUBLE,
                    scenario_abundance DOUBLE,
                    scenario_base DOUBLE,
                    scenario_stress DOUBLE,
                    expected_spdi DOUBLE,
                    spdi_p10 DOUBLE,
                    spdi_p50 DOUBLE,
                    spdi_p90 DOUBLE,
                    expected_custo_fisico DOUBLE,
                    custo_fisico_p10 DOUBLE,
                    custo_fisico_p50 DOUBLE,
                    custo_fisico_p90 DOUBLE,
                    confidence_score DOUBLE,
                    confidence_score_global DOUBLE,
                    n_paths INTEGER,
                    model_version VARCHAR,
                    history_source VARCHAR,
                    PRIMARY KEY (run_id, forecast_ts)
                )
                """
            )
            run_id = str(hourly_table["run_id"].iloc[0])
            con.execute(f"DELETE FROM {LOCAL_TABLE_NAME} WHERE run_id = ?", [run_id])
            con.register("adaptive_pld_forward_tmp", hourly_table)
            con.execute(
                f"""
                INSERT INTO {LOCAL_TABLE_NAME}
                SELECT *
                FROM adaptive_pld_forward_tmp
                """
            )
            LOGGER.info("Adaptive PLD forward table persisted to DuckDB: %s", self.duckdb_path)
        finally:
            con.close()

    def _persist_to_auth_neon(self, hourly_table: pd.DataFrame) -> None:
        auth_url = os.getenv("DATABASE_URL_AUTH", os.getenv("DATABASE_URL", ""))
        if not auth_url:
            LOGGER.warning(
                "DATABASE_URL_AUTH not configured; skipping Neon AUTH persistence for Adaptive PLD forward table."
            )
            return
        if psycopg2 is None:
            raise ImportError("psycopg2-binary is required to persist Neon AUTH results.")

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {AUTH_TABLE_NAME} (
            run_id TEXT NOT NULL,
            generated_at TIMESTAMP NOT NULL,
            forecast_ts TIMESTAMP NOT NULL,
            horizon_hour INTEGER NOT NULL,
            era_id INTEGER,
            regime_prob_low DOUBLE PRECISION,
            regime_prob_mid DOUBLE PRECISION,
            regime_prob_high DOUBLE PRECISION,
            expected_pld DOUBLE PRECISION,
            pld_p10 DOUBLE PRECISION,
            pld_p50 DOUBLE PRECISION,
            pld_p90 DOUBLE PRECISION,
            scenario_abundance DOUBLE PRECISION,
            scenario_base DOUBLE PRECISION,
            scenario_stress DOUBLE PRECISION,
            expected_spdi DOUBLE PRECISION,
            spdi_p10 DOUBLE PRECISION,
            spdi_p50 DOUBLE PRECISION,
            spdi_p90 DOUBLE PRECISION,
            expected_custo_fisico DOUBLE PRECISION,
            custo_fisico_p10 DOUBLE PRECISION,
            custo_fisico_p50 DOUBLE PRECISION,
            custo_fisico_p90 DOUBLE PRECISION,
            confidence_score DOUBLE PRECISION,
            confidence_score_global DOUBLE PRECISION,
            n_paths INTEGER,
            model_version TEXT,
            history_source TEXT,
            PRIMARY KEY (run_id, forecast_ts)
        );
        """

        columns = list(hourly_table.columns)
        column_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        update_cols = [col for col in columns if col not in {"run_id", "forecast_ts"}]
        update_sql = ", ".join(f"{col}=EXCLUDED.{col}" for col in update_cols)
        insert_sql = (
            f"INSERT INTO {AUTH_TABLE_NAME} ({column_list}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (run_id, forecast_ts) DO UPDATE SET {update_sql}"
        )

        records = self._sql_records(hourly_table)
        conn = psycopg2.connect(auth_url)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(create_sql)
            psycopg2.extras.execute_batch(cur, insert_sql, records, page_size=500)
            cur.close()
            LOGGER.info("Adaptive PLD forward table persisted to Neon AUTH.")
        finally:
            conn.close()

    def _persist_quality_to_duckdb(
        self,
        quality_obs: pd.DataFrame,
        quality_summary: pd.DataFrame,
    ) -> None:
        if duckdb is None:
            raise ImportError("duckdb is required to persist local quality results.")

        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.duckdb_path))
        try:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {QUALITY_LOCAL_TABLE_NAME} (
                    engine_name VARCHAR,
                    run_id VARCHAR,
                    generated_at TIMESTAMP,
                    generated_date TIMESTAMP,
                    forecast_ts TIMESTAMP,
                    actual_date TIMESTAMP,
                    horizon_hour INTEGER,
                    lead_days DOUBLE,
                    lead_bucket VARCHAR,
                    era_id INTEGER,
                    dominant_regime_id INTEGER,
                    dominant_regime VARCHAR,
                    expected_pld DOUBLE,
                    pld_p10 DOUBLE,
                    pld_p50 DOUBLE,
                    pld_p90 DOUBLE,
                    actual_pld DOUBLE,
                    forecast_error DOUBLE,
                    abs_error DOUBLE,
                    ape DOUBLE,
                    inside_band INTEGER,
                    below_p10 INTEGER,
                    above_p90 INTEGER,
                    quality_weight DOUBLE,
                    model_version VARCHAR,
                    PRIMARY KEY (run_id, forecast_ts)
                )
                """
            )
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {QUALITY_SUMMARY_LOCAL_TABLE_NAME} (
                    engine_name VARCHAR,
                    updated_at TIMESTAMP,
                    window_days INTEGER,
                    lead_bucket VARCHAR,
                    n_obs INTEGER,
                    actual_start_ts TIMESTAMP,
                    actual_end_ts TIMESTAMP,
                    mae DOUBLE,
                    rmse DOUBLE,
                    mape_pct DOUBLE,
                    median_ape_pct DOUBLE,
                    bias DOUBLE,
                    weighted_mae DOUBLE,
                    weighted_mape_pct DOUBLE,
                    weighted_bias DOUBLE,
                    band_coverage DOUBLE,
                    coverage_p10 DOUBLE,
                    coverage_p90 DOUBLE,
                    below_p10_rate DOUBLE,
                    above_p90_rate DOUBLE,
                    PRIMARY KEY (window_days, lead_bucket)
                )
                """
            )
            con.execute(f"DELETE FROM {QUALITY_LOCAL_TABLE_NAME}")
            con.execute(f"DELETE FROM {QUALITY_SUMMARY_LOCAL_TABLE_NAME}")
            if not quality_obs.empty:
                con.register("adaptive_pld_quality_obs_tmp", quality_obs)
                con.execute(
                    f"""
                    INSERT INTO {QUALITY_LOCAL_TABLE_NAME}
                    SELECT *
                    FROM adaptive_pld_quality_obs_tmp
                    """
                )
            if not quality_summary.empty:
                con.register("adaptive_pld_quality_summary_tmp", quality_summary)
                con.execute(
                    f"""
                    INSERT INTO {QUALITY_SUMMARY_LOCAL_TABLE_NAME}
                    SELECT *
                    FROM adaptive_pld_quality_summary_tmp
                    """
                )
            LOGGER.info(
                "Adaptive forecast quality persisted to DuckDB: obs=%s | summary=%s",
                len(quality_obs),
                len(quality_summary),
            )
        finally:
            con.close()

    def _persist_quality_to_auth_neon(
        self,
        quality_obs: pd.DataFrame,
        quality_summary: pd.DataFrame,
    ) -> None:
        auth_url = os.getenv("DATABASE_URL_AUTH", os.getenv("DATABASE_URL", ""))
        if not auth_url:
            LOGGER.warning(
                "DATABASE_URL_AUTH not configured; skipping Neon AUTH persistence for adaptive forecast quality."
            )
            return
        if psycopg2 is None:
            raise ImportError("psycopg2-binary is required to persist Neon AUTH quality results.")

        create_obs_sql = f"""
        CREATE TABLE IF NOT EXISTS {QUALITY_AUTH_TABLE_NAME} (
            engine_name TEXT,
            run_id TEXT NOT NULL,
            generated_at TIMESTAMP,
            generated_date TIMESTAMP,
            forecast_ts TIMESTAMP NOT NULL,
            actual_date TIMESTAMP,
            horizon_hour INTEGER,
            lead_days DOUBLE PRECISION,
            lead_bucket TEXT,
            era_id INTEGER,
            dominant_regime_id INTEGER,
            dominant_regime TEXT,
            expected_pld DOUBLE PRECISION,
            pld_p10 DOUBLE PRECISION,
            pld_p50 DOUBLE PRECISION,
            pld_p90 DOUBLE PRECISION,
            actual_pld DOUBLE PRECISION,
            forecast_error DOUBLE PRECISION,
            abs_error DOUBLE PRECISION,
            ape DOUBLE PRECISION,
            inside_band INTEGER,
            below_p10 INTEGER,
            above_p90 INTEGER,
            quality_weight DOUBLE PRECISION,
            model_version TEXT,
            PRIMARY KEY (run_id, forecast_ts)
        );
        """
        create_summary_sql = f"""
        CREATE TABLE IF NOT EXISTS {QUALITY_SUMMARY_AUTH_TABLE_NAME} (
            engine_name TEXT,
            updated_at TIMESTAMP,
            window_days INTEGER NOT NULL,
            lead_bucket TEXT NOT NULL,
            n_obs INTEGER,
            actual_start_ts TIMESTAMP,
            actual_end_ts TIMESTAMP,
            mae DOUBLE PRECISION,
            rmse DOUBLE PRECISION,
            mape_pct DOUBLE PRECISION,
            median_ape_pct DOUBLE PRECISION,
            bias DOUBLE PRECISION,
            weighted_mae DOUBLE PRECISION,
            weighted_mape_pct DOUBLE PRECISION,
            weighted_bias DOUBLE PRECISION,
            band_coverage DOUBLE PRECISION,
            coverage_p10 DOUBLE PRECISION,
            coverage_p90 DOUBLE PRECISION,
            below_p10_rate DOUBLE PRECISION,
            above_p90_rate DOUBLE PRECISION,
            PRIMARY KEY (window_days, lead_bucket)
        );
        """

        conn = psycopg2.connect(auth_url)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(create_obs_sql)
            cur.execute(create_summary_sql)
            cur.execute(f"DELETE FROM {QUALITY_AUTH_TABLE_NAME}")
            cur.execute(f"DELETE FROM {QUALITY_SUMMARY_AUTH_TABLE_NAME}")

            if not quality_obs.empty:
                obs_cols = list(quality_obs.columns)
                obs_column_list = ", ".join(obs_cols)
                obs_placeholders = ", ".join(["%s"] * len(obs_cols))
                obs_update_cols = [col for col in obs_cols if col not in {"run_id", "forecast_ts"}]
                obs_update_sql = ", ".join(f"{col}=EXCLUDED.{col}" for col in obs_update_cols)
                obs_insert_sql = (
                    f"INSERT INTO {QUALITY_AUTH_TABLE_NAME} ({obs_column_list}) "
                    f"VALUES ({obs_placeholders}) "
                    f"ON CONFLICT (run_id, forecast_ts) DO UPDATE SET {obs_update_sql}"
                )
                psycopg2.extras.execute_batch(
                    cur,
                    obs_insert_sql,
                    self._sql_records(quality_obs),
                    page_size=500,
                )

            if not quality_summary.empty:
                summary_cols = list(quality_summary.columns)
                summary_column_list = ", ".join(summary_cols)
                summary_placeholders = ", ".join(["%s"] * len(summary_cols))
                summary_update_cols = [col for col in summary_cols if col not in {"window_days", "lead_bucket"}]
                summary_update_sql = ", ".join(f"{col}=EXCLUDED.{col}" for col in summary_update_cols)
                summary_insert_sql = (
                    f"INSERT INTO {QUALITY_SUMMARY_AUTH_TABLE_NAME} ({summary_column_list}) "
                    f"VALUES ({summary_placeholders}) "
                    f"ON CONFLICT (window_days, lead_bucket) DO UPDATE SET {summary_update_sql}"
                )
                psycopg2.extras.execute_batch(
                    cur,
                    summary_insert_sql,
                    self._sql_records(quality_summary),
                    page_size=200,
                )

            cur.close()
            LOGGER.info(
                "Adaptive forecast quality persisted to Neon AUTH: obs=%s | summary=%s",
                len(quality_obs),
                len(quality_summary),
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    def _fit_regressor_bundle(
        self,
        frame: pd.DataFrame,
        feature_cols: Iterable[str],
        target_col: str,
        model_type: str = "rf",
    ) -> Dict[str, Any]:
        feature_list = list(dict.fromkeys(feature_cols))
        data = frame[feature_list].copy()
        data[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        if data.empty:
            return {"model": None, "resid_std": 0.0, "fallback": 0.0, "features": feature_list}

        X = data[feature_list]
        y = pd.to_numeric(data[target_col], errors="coerce").astype(float)

        if model_type == "gbr":
            model = GradientBoostingRegressor(
                random_state=self.random_state,
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
            )
        else:
            model = RandomForestRegressor(
                random_state=self.random_state,
                n_estimators=250,
                min_samples_leaf=6,
                n_jobs=-1,
            )

        if len(data) >= 24:
            model.fit(X, y)
            pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
            resid = y.to_numpy(dtype=float) - pred
            resid_std = float(np.nanstd(resid))
            fallback = float(np.nanmedian(y.to_numpy(dtype=float)))
        else:
            model = None
            resid_std = float(np.nanstd(y.to_numpy(dtype=float)))
            fallback = float(np.nanmedian(y.to_numpy(dtype=float)))

        return {
            "model": model,
            "resid_std": resid_std if not np.isnan(resid_std) else 0.0,
            "fallback": fallback if not np.isnan(fallback) else 0.0,
            "features": list(X.columns),
        }

    def _fit_classifier_bundle(
        self,
        frame: pd.DataFrame,
        feature_cols: Iterable[str],
        target_col: str,
        classes: list[int],
    ) -> Dict[str, Any]:
        data = frame[list(feature_cols) + [target_col]].replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        if data.empty:
            return self._constant_classifier_bundle(classes[0], classes=classes)

        X = data[list(feature_cols)]
        y = data[target_col].astype(int)
        observed_classes = sorted(int(v) for v in y.unique())
        priors = (
            y.value_counts(normalize=True)
            .reindex(classes, fill_value=0.0)
            .to_numpy(dtype=float)
        )

        if len(observed_classes) == 1:
            return self._constant_classifier_bundle(observed_classes[0], classes=classes)

        model = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=250,
            min_samples_leaf=4,
            n_jobs=-1,
        )
        model.fit(X, y)
        return {
            "model": model,
            "classes": classes,
            "observed_classes": observed_classes,
            "priors": priors,
            "features": list(X.columns),
        }

    def _constant_classifier_bundle(
        self,
        constant_class: int,
        classes: list[int],
    ) -> Dict[str, Any]:
        priors = np.zeros(len(classes), dtype=float)
        if constant_class in classes:
            priors[classes.index(constant_class)] = 1.0
        return {
            "model": None,
            "classes": classes,
            "observed_classes": [constant_class],
            "priors": priors,
            "features": [],
            "constant_class": constant_class,
        }

    def _predict_regressor(
        self,
        model_name: str,
        features: pd.DataFrame,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bundle = self.model_bundles_[model_name]
        model = bundle["model"]
        feature_order = list(
            getattr(model, "feature_names_in_", bundle["features"])
        ) if model is not None else list(bundle["features"])
        X = (
            features.copy()
            .reindex(columns=feature_order)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.feature_medians_)
        )
        expected = (
            model.predict(X)
            if model is not None
            else np.full(len(features), bundle["fallback"], dtype=float)
        )
        expected = np.asarray(expected, dtype=float)
        if expected.ndim > 1:
            if expected.shape[1] == 1:
                expected = expected[:, 0]
            else:
                LOGGER.warning(
                    "Regressor %s returned multi-output shape %s; collapsing with row median.",
                    model_name,
                    expected.shape,
                )
                expected = np.nanmedian(expected, axis=1)
        resid_std = self._safe_scalar(bundle.get("resid_std", 0.0), default=0.0)
        if resid_std > 0:
            expected = expected + rng.normal(0.0, resid_std, size=len(features))
        return np.asarray(expected, dtype=float)

    def _predict_current_era(self, current_state: pd.Series) -> int:
        if self.era_classifier_ is None:
            return self.latest_era_id_

        features = pd.DataFrame(
            {
                "ear": [float(current_state["ear"])],
                "ena": [float(current_state["ena"])],
                "net_load": [float(current_state["net_load"])],
                "thermal_share": [float(current_state["thermal_share"])],
                "curtailment": [float(current_state["curtailment"])],
                "isr": [float(current_state["isr"])],
            }
        )
        probs = self._predict_classifier_proba(self.era_classifier_, features)[0]
        classes = self.era_classifier_["classes"]
        predicted = int(classes[int(np.argmax(probs))])
        if np.max(probs) < 0.35:
            return self.latest_era_id_
        return predicted

    def _predict_regime_probabilities(
        self,
        era_id: int,
        features: pd.DataFrame,
    ) -> np.ndarray:
        bundle = self.regime_classifiers_.get(era_id)
        if bundle is None:
            bundle = self._constant_classifier_bundle(REGIME_LABEL_TO_ID["mid"], classes=[0, 1, 2])
        return self._predict_classifier_proba(bundle, features)

    def _predict_classifier_proba(self, bundle: Dict[str, Any], features: pd.DataFrame) -> np.ndarray:
        all_classes = bundle["classes"]
        model = bundle.get("model")
        if model is None:
            priors = np.asarray(bundle["priors"], dtype=float)
            return np.tile(self._normalize_probs(priors), (len(features), 1))

        feature_order = list(getattr(model, "feature_names_in_", bundle["features"]))
        X = features.copy()
        X = X.reindex(columns=feature_order)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(self.feature_medians_)
        raw_probs = model.predict_proba(X)
        aligned = np.zeros((len(X), len(all_classes)), dtype=float)
        model_classes = list(model.classes_)
        for idx, klass in enumerate(all_classes):
            if klass in model_classes:
                aligned[:, idx] = raw_probs[:, model_classes.index(klass)]
        return np.vstack([self._normalize_probs(row) for row in aligned])

    def _sample_climatology_rows(
        self,
        rng: np.random.Generator,
        era_id: int,
        ts: pd.Timestamp,
        n_samples: int,
    ) -> Dict[str, np.ndarray]:
        hour = int(ts.hour)
        month_weights = self._seasonal_month_weights(ts)
        months = list(month_weights.keys())
        weight_values = np.asarray([month_weights[m] for m in months], dtype=float)
        choices = rng.choice(len(months), size=n_samples, p=self._normalize_probs(weight_values))
        sampled = np.zeros((n_samples, len(self.climatology_["cols"])), dtype=float)
        for choice_idx, month in enumerate(months):
            mask = choices == choice_idx
            if not np.any(mask):
                continue
            pool = self._get_climatology_pool(era_id=era_id, month=month, hour=hour)
            idx = rng.integers(0, len(pool), size=int(mask.sum()))
            sampled[mask] = pool[idx]
        cols = self.climatology_["cols"]
        return {col: sampled[:, i].astype(float) for i, col in enumerate(cols)}

    def _lookup_cost_stat(self, era_id: int, regime_id: int, ts: pd.Timestamp, stat: str) -> float:
        month_weights = self._seasonal_month_weights(ts)
        hour = int(ts.hour)
        values: list[float] = []
        weights: list[float] = []

        for month, month_weight in month_weights.items():
            candidates = [
                self.cost_surface_.get("full", {}).get((era_id, regime_id, month, hour)),
                self.cost_surface_.get("era_regime_month", {}).get((era_id, regime_id, month)),
                self.cost_surface_.get("era_regime", {}).get((era_id, regime_id)),
                self.cost_surface_.get("regime_month_hour", {}).get((regime_id, month, hour)),
                self.cost_surface_.get("month_hour", {}).get((month, hour)),
                self.cost_surface_.get("global"),
            ]
            chosen: Dict[str, float] | None = None
            for value in candidates:
                if value is not None:
                    chosen = value
                    break
            if chosen is None:
                continue
            scalar = self._safe_scalar(chosen.get(stat), default=np.nan)
            if np.isfinite(scalar):
                values.append(float(scalar))
                weights.append(float(month_weight))

        if values:
            return float(np.average(values, weights=np.asarray(weights, dtype=float)))
        return self._safe_scalar(self.cost_surface_.get("global", {}).get(stat), default=0.0)

    def _lookup_spdi_base(self, era_id: int, regime_id: int, month: int, hour: int) -> float:
        candidates = [
            self.spdi_base_surface_["full"].get((era_id, regime_id, month, hour)),
            self.spdi_base_surface_["era_regime_month"].get((era_id, regime_id, month)),
            self.spdi_base_surface_["era_regime"].get((era_id, regime_id)),
            self.spdi_base_surface_["regime_month_hour"].get((regime_id, month, hour)),
            self.spdi_base_surface_["month_hour"].get((month, hour)),
            self.spdi_base_surface_["global"],
        ]
        for value in candidates:
            if value is not None and not np.isnan(value):
                return float(value)
        return float(self.spdi_base_surface_["global"])

    def _sample_shock(
        self,
        rng: np.random.Generator,
        era_id: int,
        regime_id: int,
        month: int,
        hour: int,
        n_samples: int,
    ) -> np.ndarray:
        candidates = [
            self.shock_surface_["samples_full"].get((era_id, regime_id, month, hour)),
            self.shock_surface_["samples_era_regime_month"].get((era_id, regime_id, month)),
            self.shock_surface_["samples_era_regime"].get((era_id, regime_id)),
            self.shock_surface_["samples_regime_month_hour"].get((regime_id, month, hour)),
            self.shock_surface_["samples_month_hour"].get((month, hour)),
            self.shock_surface_["samples_global"],
        ]
        sample_pool: np.ndarray | None = None
        for value in candidates:
            if value is not None and len(value) > 0:
                sample_pool = np.asarray(value, dtype=float)
                break

        if sample_pool is None or sample_pool.size == 0:
            stats = self.shock_surface_["stats_global"]
            return rng.normal(
                self._safe_scalar(stats.get("mean", 0.0), default=0.0),
                max(self._safe_scalar(stats.get("std", 0.0), default=0.0), EPS),
                size=n_samples,
            )

        idx = rng.integers(0, len(sample_pool), size=n_samples)
        return sample_pool[idx]

    def _sample_normal_shock(
        self,
        rng: np.random.Generator,
        era_id: int,
        regime_id: int,
        month: int,
        hour: int,
        n_samples: int,
    ) -> np.ndarray:
        candidates = [
            self.shock_surface_.get("samples_normal_full", {}).get((era_id, regime_id, month, hour)),
            self.shock_surface_.get("samples_normal_era_regime_month", {}).get((era_id, regime_id, month)),
            self.shock_surface_.get("samples_normal_era_regime", {}).get((era_id, regime_id)),
            self.shock_surface_.get("samples_normal_regime_month_hour", {}).get((regime_id, month, hour)),
            self.shock_surface_.get("samples_normal_month_hour", {}).get((month, hour)),
            self.shock_surface_.get("samples_normal_global"),
            self.shock_surface_.get("samples_global"),
        ]
        sample_pool: np.ndarray | None = None
        for value in candidates:
            if value is not None and len(value) > 0:
                sample_pool = np.asarray(value, dtype=float)
                break

        if sample_pool is None or sample_pool.size == 0:
            return self._sample_shock(
                rng=rng,
                era_id=era_id,
                regime_id=regime_id,
                month=month,
                hour=hour,
                n_samples=n_samples,
            )

        idx = rng.integers(0, len(sample_pool), size=n_samples)
        return sample_pool[idx]

    def _sample_spike_shock(
        self,
        rng: np.random.Generator,
        era_id: int,
        regime_id: int,
        month: int,
        hour: int,
        n_samples: int,
    ) -> np.ndarray:
        candidates = [
            self.spike_surface_.get("samples_full", {}).get((era_id, regime_id, month, hour)),
            self.spike_surface_.get("samples_era_regime_month", {}).get((era_id, regime_id, month)),
            self.spike_surface_.get("samples_era_regime", {}).get((era_id, regime_id)),
            self.spike_surface_.get("samples_regime_month_hour", {}).get((regime_id, month, hour)),
            self.spike_surface_.get("samples_month_hour", {}).get((month, hour)),
            self.spike_surface_.get("samples_global"),
        ]
        sample_pool: np.ndarray | None = None
        for value in candidates:
            if value is not None and len(value) > 0:
                sample_pool = np.asarray(value, dtype=float)
                break

        if sample_pool is None or sample_pool.size == 0:
            fallback = self._sample_shock(
                rng=rng,
                era_id=era_id,
                regime_id=regime_id,
                month=month,
                hour=hour,
                n_samples=n_samples,
            )
            return np.maximum(fallback, 0.0)

        idx = rng.integers(0, len(sample_pool), size=n_samples)
        return np.maximum(sample_pool[idx], 0.0)

    def _lookup_spike_probability(
        self,
        era_id: int,
        regime_id: int,
        month: int,
        hour: int,
        prev_state: int,
    ) -> float:
        candidates = [
            self.spike_surface_.get("transition_full", {}).get((era_id, regime_id, month, hour, prev_state)),
            self.spike_surface_.get("transition_era_regime_month", {}).get((era_id, regime_id, month, prev_state)),
            self.spike_surface_.get("transition_era_regime", {}).get((era_id, regime_id, prev_state)),
            self.spike_surface_.get("transition_regime_month_hour", {}).get((regime_id, month, hour, prev_state)),
            self.spike_surface_.get("transition_month_hour", {}).get((month, hour, prev_state)),
            self.spike_surface_.get("transition_global", {}).get(prev_state),
        ]
        for value in candidates:
            if value is not None and not np.isnan(value):
                return float(np.clip(value, 0.0, 1.0))
        return 0.05 if prev_state else 0.02

    def _lookup_shock_std(self, era_id: int, regime_id: int, month: int, hour: int) -> float:
        stats_full = self.shock_surface_["stats_full"]
        stats = stats_full.get((era_id, regime_id, month, hour))
        if stats:
            return self._safe_scalar(stats.get("std", 0.0), default=0.0)
        return self._safe_scalar(self.shock_surface_["stats_global"]["std"], default=0.0)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _build_dynamic_feature_frame(
        self,
        *,
        ear: np.ndarray,
        ena: np.ndarray,
        load: np.ndarray,
        solar: np.ndarray,
        wind: np.ndarray,
        net_load: np.ndarray,
        thermal_share: np.ndarray,
        curtailment_ratio: np.ndarray,
        isr: np.ndarray,
        month: int,
        hour: int,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "ear": ear,
                "ena": ena,
                "load": load,
                "solar": solar,
                "wind": wind,
                "net_load": net_load,
                "thermal_share": thermal_share,
                "curtailment_ratio": curtailment_ratio,
                "isr": isr,
                "month": np.full(len(ear), month, dtype=int),
                "hour": np.full(len(ear), hour, dtype=int),
            }
        )
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(self.feature_medians_)
        return frame

    def _coalesce_numeric(self, df: pd.DataFrame, aliases: list[str]) -> pd.Series:
        for alias in aliases:
            if alias in df.columns:
                return pd.to_numeric(df[alias], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    def _safe_scalar(self, value: Any, default: float = 0.0) -> float:
        scalar = pd.to_numeric(value, errors="coerce")
        return float(scalar) if pd.notna(scalar) else float(default)

    def _normalize_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        total = probs.sum()
        if total <= 0:
            return np.full(len(probs), 1.0 / len(probs), dtype=float)
        return probs / total

    def _select_current_state(self, prepared: pd.DataFrame) -> pd.Series:
        stable = prepared.copy()
        stable = stable.replace([np.inf, -np.inf], np.nan)

        spdi_upper_guard = min(self.bounds_.get("spdi", (0.05, 10.0))[1], 9.8)
        stable = stable[
            stable["pld"].notna()
            & stable["custo_fisico"].notna()
            & stable["spdi"].notna()
            & (stable["custo_fisico"] > 1.0)
            & (stable["spdi"] < spdi_upper_guard)
        ]

        if stable.empty:
            return prepared.iloc[-1].copy()
        return stable.iloc[-1].copy()

    def _series_bounds(
        self,
        series: pd.Series,
        lower: float | None = None,
        upper: float | None = None,
    ) -> tuple[float, float]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            low, high = 0.0, 1.0
        else:
            low = float(clean.quantile(0.01))
            high = float(clean.quantile(0.99))
            if np.isclose(low, high):
                high = low + 1.0
        if lower is not None:
            low = max(low, lower)
        if upper is not None:
            high = min(high, upper)
        return low, high

    def _confidence_from_band(self, p10: float, p50: float, p90: float) -> float:
        dispersion = max(float(p90) - float(p10), 0.0) / max(abs(float(p50)), 1.0)
        return float(np.clip(1.0 / (1.0 + dispersion), 0.0, 1.0))

    def _sql_records(self, df: pd.DataFrame) -> list[tuple[Any, ...]]:
        out = []
        object_df = df.astype(object).where(pd.notnull(df), None)
        for row in object_df.itertuples(index=False, name=None):
            normalized = []
            for value in row:
                if isinstance(value, pd.Timestamp):
                    normalized.append(value.to_pydatetime())
                elif isinstance(value, np.floating):
                    normalized.append(float(value))
                elif isinstance(value, np.integer):
                    normalized.append(int(value))
                else:
                    normalized.append(value)
            out.append(tuple(normalized))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaptivePLDForwardEngine.")
    parser.add_argument("--duckdb-path", default="data/kintuadi.duckdb", help="Local DuckDB path.")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24 * 30 * 6,
        help="Forecast horizon in hours (default: 6 months).",
    )
    parser.add_argument("--n-paths", type=int, default=1000, help="Number of Monte Carlo paths.")
    parser.add_argument("--no-persist", action="store_true", help="Do not persist outputs.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    engine = AdaptivePLDForwardEngine(
        duckdb_path=args.duckdb_path,
        horizon_hours=args.horizon_hours,
        n_paths=args.n_paths,
    )
    result = engine.run(persist=not args.no_persist)
    print(pd.Series(result.to_dict()).to_json(force_ascii=False))
    print(result.hourly_table.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
