from __future__ import annotations

"""
Adaptive PLD Backtest Engine.

Three complementary phases are evaluated:

1. Structural hindcast
2. Walk-forward real
3. Feedback enrichment for the adaptive forward
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - optional at import time
    duckdb = None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - optional at import time
    psycopg2 = None

from adaptive_pld_forward_engine import (
    BACKTEST_AUTH_TABLE_NAME,
    BACKTEST_LOCAL_TABLE_NAME,
    BACKTEST_QUALITY_AUTH_TABLE_NAME,
    BACKTEST_QUALITY_LOCAL_TABLE_NAME,
    BACKTEST_SUMMARY_AUTH_TABLE_NAME,
    BACKTEST_SUMMARY_LOCAL_TABLE_NAME,
    EPS,
    MODEL_VERSION,
    REGIME_ID_TO_LABEL,
    AdaptivePLDForwardEngine,
)


LOGGER = logging.getLogger(__name__)
BACKTEST_ENGINE_NAME = "adaptive_pld_backtest"
BACKTEST_SUMMARY_WINDOWS = (90, 180, 365, 99999)
PHASE1_NAME = "phase1_structural"
PHASE2_SHORT_NAME = "phase2_walkforward_short"
PHASE2_LONG_NAME = "phase2_walkforward_long"

ADAPTIVE_SUBMARKETS: dict[str, dict[str, str]] = {
    "SE": {"history_col": "pld_se", "label": "SE/CO"},
    "S": {"history_col": "pld_s", "label": "Sul"},
    "NE": {"history_col": "pld_ne", "label": "Nordeste"},
    "N": {"history_col": "pld_n", "label": "Norte"},
}

BACKTEST_MARKETS: dict[str, dict[str, str]] = {
    "SIN": {
        "label": "SIN",
        "actual_col": "actual_pld",
        "expected_col": "expected_pld",
        "p10_col": "pld_p10",
        "p50_col": "pld_p50",
        "p90_col": "pld_p90",
    },
    "SE": {
        "label": "SE/CO",
        "actual_col": "actual_pld_se",
        "expected_col": "expected_pld_se",
        "p10_col": "pld_se_p10",
        "p50_col": "pld_se_p50",
        "p90_col": "pld_se_p90",
    },
    "S": {
        "label": "Sul",
        "actual_col": "actual_pld_s",
        "expected_col": "expected_pld_s",
        "p10_col": "pld_s_p10",
        "p50_col": "pld_s_p50",
        "p90_col": "pld_s_p90",
    },
    "NE": {
        "label": "Nordeste",
        "actual_col": "actual_pld_ne",
        "expected_col": "expected_pld_ne",
        "p10_col": "pld_ne_p10",
        "p50_col": "pld_ne_p50",
        "p90_col": "pld_ne_p90",
    },
    "N": {
        "label": "Norte",
        "actual_col": "actual_pld_n",
        "expected_col": "expected_pld_n",
        "p10_col": "pld_n_p10",
        "p50_col": "pld_n_p50",
        "p90_col": "pld_n_p90",
    },
}


@dataclass
class AdaptivePLDBacktestResult:
    forecast_table: pd.DataFrame
    quality_observations: pd.DataFrame
    quality_summary: pd.DataFrame
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.metadata


class AdaptivePLDBacktestEngine:
    def __init__(
        self,
        duckdb_path: str | Path = "data/kintuadi.duckdb",
        pmo_xlsx_path: str | Path = "data/ons/PMOs/validacao_pmo.xlsx",
        start_date: str | pd.Timestamp = "2021-01-01",
        end_date: str | pd.Timestamp | None = None,
        short_stride_days: int = 1,
        long_stride_days: int = 7,
        phase1_horizon_hours: int = 24,
        phase2_short_horizon_hours: int = 24 * 7,
        phase2_long_horizon_hours: int = 24 * 30 * 6,
        phase2_n_paths: int = 250,
        physical_lag_hours: int = 48,
        weekly_lag_hours: int = 72,
        min_train_hours: int = 24 * 30,
    ) -> None:
        self.duckdb_path = Path(duckdb_path)
        self.pmo_xlsx_path = Path(pmo_xlsx_path)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None
        self.short_stride_days = max(int(short_stride_days), 1)
        self.long_stride_days = max(int(long_stride_days), 1)
        self.phase1_horizon_hours = int(phase1_horizon_hours)
        self.phase2_short_horizon_hours = int(phase2_short_horizon_hours)
        self.phase2_long_horizon_hours = int(phase2_long_horizon_hours)
        self.phase2_n_paths = int(phase2_n_paths)
        self.physical_lag_hours = int(physical_lag_hours)
        self.weekly_lag_hours = int(weekly_lag_hours)
        self.min_train_hours = int(min_train_hours)

        self.prototype_engine = AdaptivePLDForwardEngine(
            duckdb_path=self.duckdb_path,
            pmo_xlsx_path=self.pmo_xlsx_path,
            n_paths=max(self.phase2_n_paths, 100),
            horizon_hours=self.phase2_long_horizon_hours,
        )
        self.history_: pd.DataFrame | None = None
        self.prepared_history_: pd.DataFrame | None = None

    def run(
        self,
        *,
        persist: bool = True,
        include_phase1: bool = True,
        include_phase2: bool = True,
    ) -> AdaptivePLDBacktestResult:
        history = self._load_history()
        prepared = self.prepared_history_.copy()
        start_ts = max(pd.Timestamp(history.index.min()), self.start_date)
        end_ts = min(
            pd.Timestamp(history.index.max()),
            self.end_date if self.end_date is not None else pd.Timestamp(history.index.max()),
        )

        phase1_origins = self._build_origins(
            history.index,
            start_ts=start_ts,
            end_ts=end_ts,
            horizon_hours=self.phase1_horizon_hours,
            step_days=self.short_stride_days,
        )
        phase2_short_origins = self._build_origins(
            history.index,
            start_ts=start_ts,
            end_ts=end_ts,
            horizon_hours=self.phase2_short_horizon_hours,
            step_days=self.short_stride_days,
        )
        phase2_long_origins = self._build_origins(
            history.index,
            start_ts=start_ts,
            end_ts=end_ts,
            horizon_hours=self.phase2_long_horizon_hours,
            step_days=self.long_stride_days,
        )

        frames: list[pd.DataFrame] = []
        if include_phase1:
            frames.append(self._run_phase1_structural(history, prepared, phase1_origins, end_ts))
        if include_phase2:
            frames.append(
                self._run_phase2_walkforward(
                    history,
                    prepared,
                    phase2_short_origins,
                    end_ts=end_ts,
                    horizon_hours=self.phase2_short_horizon_hours,
                    n_paths=max(120, min(self.phase2_n_paths, 400)),
                    phase_name=PHASE2_SHORT_NAME,
                )
            )
            frames.append(
                self._run_phase2_walkforward(
                    history,
                    prepared,
                    phase2_long_origins,
                    end_ts=end_ts,
                    horizon_hours=self.phase2_long_horizon_hours,
                    n_paths=self.phase2_n_paths,
                    phase_name=PHASE2_LONG_NAME,
                )
            )

        forecast_table = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()
        quality_obs = self._build_quality_observations(forecast_table)
        quality_summary = self._build_quality_summary(quality_obs)

        metadata = {
            "engine_name": BACKTEST_ENGINE_NAME,
            "model_version": MODEL_VERSION,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "phase1_origins": len(phase1_origins),
            "phase2_short_origins": len(phase2_short_origins),
            "phase2_long_origins": len(phase2_long_origins),
            "forecast_rows": int(len(forecast_table)),
            "quality_rows": int(len(quality_obs)),
            "summary_rows": int(len(quality_summary)),
            "phase3_feedback_ready": bool(not forecast_table.empty),
        }

        if persist:
            self._persist_to_duckdb(forecast_table, quality_obs, quality_summary)
            try:
                self._persist_to_auth(forecast_table, quality_obs, quality_summary)
            except Exception as exc:  # pragma: no cover - auth optional at runtime
                LOGGER.warning("Adaptive backtest AUTH persistence failed: %s", exc)

        return AdaptivePLDBacktestResult(
            forecast_table=forecast_table,
            quality_observations=quality_obs,
            quality_summary=quality_summary,
            metadata=metadata,
        )

    def _load_history(self) -> pd.DataFrame:
        if self.history_ is None:
            history = self.prototype_engine.load_hourly_history()
            history = history.copy()
            history.index = pd.to_datetime(history.index, errors="coerce")
            history = history[~history.index.isna()]
            history = history[~history.index.duplicated(keep="last")].sort_index()
            self.history_ = history
            self.prepared_history_ = self.prototype_engine._prepare_history(history)
        return self.history_.copy()

    def _clone_forward_engine(
        self,
        *,
        horizon_hours: int,
        n_paths: int,
    ) -> AdaptivePLDForwardEngine:
        engine = self.prototype_engine
        return AdaptivePLDForwardEngine(
            duckdb_path=engine.duckdb_path,
            pmo_xlsx_path=engine.pmo_xlsx_path,
            n_paths=n_paths,
            horizon_hours=horizon_hours,
            random_state=engine.random_state,
            zscore_reversion_threshold=engine.zscore_reversion_threshold,
            shock_decay_factor=engine.shock_decay_factor,
            spike_zscore_threshold=engine.spike_zscore_threshold,
            spike_persistence=engine.spike_persistence,
            pmo_guidance_hours=engine.pmo_guidance_hours,
            recent_profile_hours=engine.recent_profile_hours,
            pmo_guidance_weight=engine.pmo_guidance_weight,
            spdi_recent_weight=engine.spdi_recent_weight,
            feedback_lookback_days=engine.feedback_lookback_days,
            feedback_half_life_days=engine.feedback_half_life_days,
            feedback_strength_max=engine.feedback_strength_max,
            feedback_min_weight=engine.feedback_min_weight,
            feedback_ratio_clip=engine.feedback_ratio_clip,
            quality_lookback_days=engine.quality_lookback_days,
            season_transition_days=engine.season_transition_days,
            cost_anchor_weight=engine.cost_anchor_weight,
            cost_tail_clip=engine.cost_tail_clip,
            ou_kappa=engine.ou_kappa,
            ou_mu_quantile=engine.ou_mu_quantile,
            ou_long_run_clip=engine.ou_long_run_clip,
            min_era_size_hours=engine.min_era_size_hours,
            max_eras=engine.max_eras,
        )

    def _build_origins(
        self,
        index: pd.DatetimeIndex,
        *,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        horizon_hours: int,
        step_days: int,
    ) -> list[pd.Timestamp]:
        idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
        idx = idx[~idx.isna()].sort_values()
        if idx.empty:
            return []

        min_origin = idx.min() + pd.Timedelta(hours=self.min_train_hours)
        max_origin = end_ts - pd.Timedelta(hours=horizon_hours)
        start_bound = max(pd.Timestamp(start_ts), pd.Timestamp(min_origin))
        max_bound = pd.Timestamp(max_origin)
        if start_bound > max_bound:
            return []

        daily_last = pd.Series(idx, index=idx).groupby(idx.normalize()).max().sort_values()
        origins = daily_last[(daily_last >= start_bound) & (daily_last <= max_bound)]
        if origins.empty:
            return []
        return [pd.Timestamp(ts) for ts in origins.iloc[::step_days].tolist()]

    def _apply_information_lags(self, history: pd.DataFrame, origin_ts: pd.Timestamp) -> pd.DataFrame:
        snapshot = history.loc[:origin_ts].copy()
        if snapshot.empty:
            return snapshot

        physical_cutoff = pd.Timestamp(origin_ts) - pd.Timedelta(hours=self.physical_lag_hours)
        weekly_cutoff = pd.Timestamp(origin_ts) - pd.Timedelta(hours=self.weekly_lag_hours)

        physical_cols = [
            "load",
            "load_se",
            "load_ne",
            "load_s",
            "load_n",
            "solar",
            "wind",
            "hydro",
            "thermal",
            "nuclear",
            "cmo",
            "cmo_dominante",
            "cmo_se",
            "cmo_ne",
            "cmo_s",
            "cmo_n",
            "disp_hydro",
            "disp_thermal",
            "disp_nuclear",
            "disp_solar",
            "disp_wind",
            "avail_solar",
            "avail_wind",
            "curtail_solar",
            "curtail_wind",
            "gfom_ger",
            "constrained_off",
            "thermal_inflex_gfom",
            "thermal_merit",
            "gfom",
            "ear_pct",
            "ena_bruta",
            "ena_arm",
        ]
        weekly_cols = ["cvu_semana"]

        physical_cols = [col for col in physical_cols if col in snapshot.columns]
        weekly_cols = [col for col in weekly_cols if col in snapshot.columns]

        if physical_cols:
            late_mask = snapshot.index > physical_cutoff
            snapshot.loc[late_mask, physical_cols] = np.nan
            snapshot[physical_cols] = snapshot[physical_cols].ffill()
        if weekly_cols:
            weekly_mask = snapshot.index > weekly_cutoff
            snapshot.loc[weekly_mask, weekly_cols] = np.nan
            snapshot[weekly_cols] = snapshot[weekly_cols].ffill()

        return snapshot

    def _run_phase1_structural(
        self,
        history: pd.DataFrame,
        prepared: pd.DataFrame,
        origins: list[pd.Timestamp],
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        total = len(origins)
        for idx, origin_ts in enumerate(origins, start=1):
            if idx == 1 or idx % 30 == 0 or idx == total:
                LOGGER.info("Phase 1 structural backtest: origin %s/%s (%s)", idx, total, origin_ts)

            train_history = self._apply_information_lags(history, origin_ts)
            if len(train_history) < self.min_train_hours:
                continue

            future_slice = prepared[
                (prepared.index > origin_ts)
                & (prepared.index <= origin_ts + pd.Timedelta(hours=self.phase1_horizon_hours))
                & (prepared.index <= end_ts)
            ].copy()
            if future_slice.empty:
                continue

            engine = self._clone_forward_engine(horizon_hours=self.phase1_horizon_hours, n_paths=64)
            try:
                engine.fit(train_history)
            except Exception as exc:
                LOGGER.debug("Phase 1 fit skipped at %s: %s", origin_ts, exc)
                continue

            phase_df = self._build_structural_hindcast_rows(
                engine=engine,
                origin_ts=origin_ts,
                actual_future=future_slice,
                spread_history=train_history,
            )
            if not phase_df.empty:
                rows.append(phase_df)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_structural_hindcast_rows(
        self,
        *,
        engine: AdaptivePLDForwardEngine,
        origin_ts: pd.Timestamp,
        actual_future: pd.DataFrame,
        spread_history: pd.DataFrame,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        run_id = f"{BACKTEST_ENGINE_NAME}_{PHASE1_NAME}_{origin_ts.strftime('%Y%m%dT%H%M%S')}"

        for forecast_ts, row in actual_future.iterrows():
            cost_actual = pd.to_numeric(row.get("custo_fisico"), errors="coerce")
            actual_pld = pd.to_numeric(row.get("pld"), errors="coerce")
            if pd.isna(cost_actual) or pd.isna(actual_pld) or float(cost_actual) <= 0:
                continue

            current_state = row.copy()
            era_id = int(engine._predict_current_era(current_state))
            feature_frame = pd.DataFrame(
                {
                    "ear": [engine._safe_scalar(current_state.get("ear"))],
                    "ena": [engine._safe_scalar(current_state.get("ena"))],
                    "net_load": [engine._safe_scalar(current_state.get("net_load"))],
                    "thermal_share": [engine._safe_scalar(current_state.get("thermal_share"))],
                    "curtailment": [engine._safe_scalar(current_state.get("curtailment"))],
                    "isr": [engine._safe_scalar(current_state.get("isr"))],
                }
            )
            regime_probs = engine._predict_regime_probabilities(era_id, feature_frame)[0]
            regime_id = int(np.argmax(regime_probs))
            shock_band = self._lookup_shock_band(engine, era_id, regime_id, int(forecast_ts.month), int(forecast_ts.hour))
            feedback_multiplier, feedback_reliability = engine._lookup_feedback_multiplier(
                int((forecast_ts - origin_ts).total_seconds() // 3600),
                pd.Timestamp(forecast_ts),
                regime_id,
            )

            spdi_base = engine._lookup_spdi_base(era_id, regime_id, int(forecast_ts.month), int(forecast_ts.hour))
            if engine.recent_profile_surface_:
                intraday_factor = engine._lookup_recent_profile("spdi_intraday", pd.Timestamp(forecast_ts), default=1.0)
                spectral_confidence = engine._lookup_recent_profile("spectral_confidence", pd.Timestamp(forecast_ts), default=0.0)
                recent_weight = float(np.clip(engine.spdi_recent_weight + 0.10 * spectral_confidence, 0.0, 0.92))
                spdi_base = spdi_base * ((1.0 - recent_weight) + recent_weight * intraday_factor)

            expected_spdi = float(
                np.clip(
                    (spdi_base + shock_band["mean"]) * feedback_multiplier,
                    engine.bounds_["spdi"][0],
                    engine.bounds_["spdi"][1],
                )
            )
            spdi_p10 = float(
                np.clip(
                    (spdi_base + shock_band["p10"]) * feedback_multiplier,
                    engine.bounds_["spdi"][0],
                    engine.bounds_["spdi"][1],
                )
            )
            spdi_p50 = float(
                np.clip(
                    (spdi_base + shock_band["p50"]) * feedback_multiplier,
                    engine.bounds_["spdi"][0],
                    engine.bounds_["spdi"][1],
                )
            )
            spdi_p90 = float(
                np.clip(
                    (spdi_base + shock_band["p90"]) * feedback_multiplier,
                    engine.bounds_["spdi"][0],
                    engine.bounds_["spdi"][1],
                )
            )
            pld_p10 = float(cost_actual) * spdi_p10
            pld_p50 = float(cost_actual) * spdi_p50
            pld_p90 = float(cost_actual) * spdi_p90

            confidence = float(engine._confidence_from_band(pld_p10, pld_p50, pld_p90))
            records.append(
                {
                    "engine_name": BACKTEST_ENGINE_NAME,
                    "phase_name": PHASE1_NAME,
                    "phase_group": "phase1",
                    "run_id": run_id,
                    "generated_at": pd.Timestamp(origin_ts),
                    "forecast_ts": pd.Timestamp(forecast_ts),
                    "horizon_hour": int((forecast_ts - origin_ts).total_seconds() // 3600),
                    "era_id": era_id,
                    "regime_prob_low": float(regime_probs[0]),
                    "regime_prob_mid": float(regime_probs[1]),
                    "regime_prob_high": float(regime_probs[2]),
                    "expected_pld": float(cost_actual) * expected_spdi,
                    "pld_p10": pld_p10,
                    "pld_p50": pld_p50,
                    "pld_p90": pld_p90,
                    "scenario_abundance": pld_p10,
                    "scenario_base": pld_p50,
                    "scenario_stress": pld_p90,
                    "expected_spdi": expected_spdi,
                    "spdi_p10": spdi_p10,
                    "spdi_p50": spdi_p50,
                    "spdi_p90": spdi_p90,
                    "expected_custo_fisico": float(cost_actual),
                    "custo_fisico_p10": float(cost_actual),
                    "custo_fisico_p50": float(cost_actual),
                    "custo_fisico_p90": float(cost_actual),
                    "confidence_score": confidence,
                    "confidence_score_global": confidence * (0.75 + 0.25 * feedback_reliability),
                    "n_paths": 0,
                    "uses_realized_physics": 1,
                    "model_version": MODEL_VERSION,
                    "history_source": engine.history_source_,
                }
            )

        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        frame = self._derive_submarket_forward(spread_history, frame)
        frame = self._attach_actuals(frame, actual_future)
        return frame

    def _run_phase2_walkforward(
        self,
        history: pd.DataFrame,
        prepared: pd.DataFrame,
        origins: list[pd.Timestamp],
        *,
        end_ts: pd.Timestamp,
        horizon_hours: int,
        n_paths: int,
        phase_name: str,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        total = len(origins)
        for idx, origin_ts in enumerate(origins, start=1):
            if idx == 1 or idx % 20 == 0 or idx == total:
                LOGGER.info("Phase 2 walk-forward (%s): origin %s/%s (%s)", phase_name, idx, total, origin_ts)

            train_history = self._apply_information_lags(history, origin_ts)
            if len(train_history) < self.min_train_hours:
                continue

            engine = self._clone_forward_engine(horizon_hours=horizon_hours, n_paths=n_paths)
            try:
                engine.fit(train_history)
                result = engine.forecast(horizon_hours=horizon_hours, n_paths=n_paths, persist=False)
            except Exception as exc:
                LOGGER.debug("Phase 2 forecast skipped at %s (%s): %s", origin_ts, phase_name, exc)
                continue

            frame = result.hourly_table.copy()
            if frame.empty:
                continue

            frame["engine_name"] = BACKTEST_ENGINE_NAME
            frame["phase_name"] = phase_name
            frame["phase_group"] = "phase2"
            frame["run_id"] = f"{BACKTEST_ENGINE_NAME}_{phase_name}_{origin_ts.strftime('%Y%m%dT%H%M%S')}"
            frame["generated_at"] = pd.Timestamp(origin_ts)
            frame["forecast_ts"] = pd.Timestamp(origin_ts) + pd.to_timedelta(
                pd.to_numeric(frame["horizon_hour"], errors="coerce").fillna(0).astype(int),
                unit="h",
            )
            frame["uses_realized_physics"] = 0
            frame["n_paths"] = int(n_paths)
            frame = frame[frame["forecast_ts"] <= end_ts].copy()
            if frame.empty:
                continue

            frame = self._derive_submarket_forward(train_history, frame)
            frame = self._attach_actuals(frame, prepared)
            rows.append(frame)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _lookup_shock_band(
        self,
        engine: AdaptivePLDForwardEngine,
        era_id: int,
        regime_id: int,
        month: int,
        hour: int,
    ) -> dict[str, float]:
        candidates = [
            engine.shock_surface_.get("samples_full", {}).get((era_id, regime_id, month, hour)),
            engine.shock_surface_.get("samples_era_regime_month", {}).get((era_id, regime_id, month)),
            engine.shock_surface_.get("samples_era_regime", {}).get((era_id, regime_id)),
            engine.shock_surface_.get("samples_regime_month_hour", {}).get((regime_id, month, hour)),
            engine.shock_surface_.get("samples_month_hour", {}).get((month, hour)),
            engine.shock_surface_.get("samples_global"),
        ]
        for sample_pool in candidates:
            if sample_pool is None:
                continue
            values = np.asarray(sample_pool, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            return {
                "mean": float(np.nanmean(values)),
                "p10": float(np.nanquantile(values, 0.10)),
                "p50": float(np.nanquantile(values, 0.50)),
                "p90": float(np.nanquantile(values, 0.90)),
            }

        stats = (
            engine.shock_surface_.get("stats_full", {}).get((era_id, regime_id, month, hour))
            or engine.shock_surface_.get("stats_global", {})
        )
        mean = engine._safe_scalar(stats.get("mean"), default=0.0)
        std = max(engine._safe_scalar(stats.get("std"), default=0.0), EPS)
        return {
            "mean": mean,
            "p10": mean - 1.2816 * std,
            "p50": mean,
            "p90": mean + 1.2816 * std,
        }

    def _derive_submarket_forward(
        self,
        history_df: pd.DataFrame,
        forward_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if forward_df.empty or "forecast_ts" not in forward_df.columns or "pld" not in history_df.columns:
            return forward_df.copy()

        out = forward_df.copy()
        history = history_df.copy()
        history.index = pd.to_datetime(history.index, errors="coerce")
        history = history[~history.index.isna()].sort_index()
        if history.empty:
            return out

        sin_hist = pd.to_numeric(history["pld"], errors="coerce")
        forecast_keys = pd.DataFrame({"forecast_ts": pd.to_datetime(out["forecast_ts"], errors="coerce")})
        forecast_keys["month"] = forecast_keys["forecast_ts"].dt.month.astype(int)
        forecast_keys["hour"] = forecast_keys["forecast_ts"].dt.hour.astype(int)
        recent_cutoff = forecast_keys["forecast_ts"].min() - pd.Timedelta(days=180)

        for submarket, spec in ADAPTIVE_SUBMARKETS.items():
            history_col = spec["history_col"]
            if history_col not in history.columns:
                continue

            spread = pd.to_numeric(history[history_col], errors="coerce") - sin_hist
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if len(spread) < 24:
                continue

            clip_low = float(spread.quantile(0.01))
            clip_high = float(spread.quantile(0.99))
            spread = spread.clip(lower=clip_low, upper=clip_high)
            spread_df = pd.DataFrame({"spread": spread})
            spread_df["month"] = spread_df.index.month.astype(int)
            spread_df["hour"] = spread_df.index.hour.astype(int)
            recent_df = spread_df[spread_df.index >= recent_cutoff]

            lookup = forecast_keys.copy()
            lookup = lookup.merge(
                recent_df.groupby(["month", "hour"])["spread"].median().rename("spread_recent_mh"),
                left_on=["month", "hour"],
                right_index=True,
                how="left",
            )
            lookup = lookup.merge(
                spread_df.groupby(["month", "hour"])["spread"].median().rename("spread_all_mh"),
                left_on=["month", "hour"],
                right_index=True,
                how="left",
            )
            lookup = lookup.merge(
                recent_df.groupby("hour")["spread"].median().rename("spread_recent_h"),
                left_on="hour",
                right_index=True,
                how="left",
            )
            lookup = lookup.merge(
                spread_df.groupby("hour")["spread"].median().rename("spread_all_h"),
                left_on="hour",
                right_index=True,
                how="left",
            )
            lookup = lookup.merge(
                spread_df.groupby("month")["spread"].median().rename("spread_month"),
                left_on="month",
                right_index=True,
                how="left",
            )

            weighted_sum = np.zeros(len(lookup), dtype=float)
            weight_sum = np.zeros(len(lookup), dtype=float)
            for col_name, weight in [
                ("spread_recent_mh", 0.40),
                ("spread_all_mh", 0.30),
                ("spread_recent_h", 0.15),
                ("spread_all_h", 0.10),
                ("spread_month", 0.05),
            ]:
                values = pd.to_numeric(lookup[col_name], errors="coerce")
                mask = values.notna().to_numpy(dtype=bool)
                if mask.any():
                    weighted_sum[mask] += values.to_numpy(dtype=float)[mask] * weight
                    weight_sum[mask] += weight

            spread_base = float(spread.median())
            spread_est = np.where(weight_sum > 0, weighted_sum / weight_sum, spread_base)
            spread_est = np.clip(spread_est, clip_low, clip_high)
            spread_col = f"spread_{submarket.lower()}"
            out[spread_col] = spread_est

            for source_col, target_col in [
                ("expected_pld", f"expected_pld_{submarket.lower()}"),
                ("pld_p10", f"pld_{submarket.lower()}_p10"),
                ("pld_p50", f"pld_{submarket.lower()}_p50"),
                ("pld_p90", f"pld_{submarket.lower()}_p90"),
                ("scenario_abundance", f"scenario_abundance_{submarket.lower()}"),
                ("scenario_base", f"scenario_base_{submarket.lower()}"),
                ("scenario_stress", f"scenario_stress_{submarket.lower()}"),
            ]:
                if source_col in out.columns:
                    out[target_col] = pd.to_numeric(out[source_col], errors="coerce") + spread_est

        return out

    def _attach_actuals(
        self,
        frame: pd.DataFrame,
        actual_history: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        out = frame.copy()
        lookup = actual_history.copy()
        lookup.index = pd.to_datetime(lookup.index, errors="coerce")
        lookup = lookup[~lookup.index.isna()].sort_index()
        if lookup.empty:
            return out

        actual_cols = {
            "pld": "actual_pld",
            "pld_se": "actual_pld_se",
            "pld_s": "actual_pld_s",
            "pld_ne": "actual_pld_ne",
            "pld_n": "actual_pld_n",
            "custo_fisico": "actual_custo_fisico",
            "spdi": "actual_spdi",
        }
        cols = [src for src in actual_cols if src in lookup.columns]
        if not cols:
            return out

        aligned = lookup.reindex(pd.to_datetime(out["forecast_ts"], errors="coerce"))[cols]
        aligned.index = out.index
        for src, target in actual_cols.items():
            if src in aligned.columns:
                out[target] = pd.to_numeric(aligned[src], errors="coerce")
        return out

    def _build_quality_observations(self, forecast_table: pd.DataFrame) -> pd.DataFrame:
        if forecast_table is None or forecast_table.empty:
            return pd.DataFrame()

        df = forecast_table.copy()
        df["forecast_ts"] = pd.to_datetime(df["forecast_ts"], errors="coerce")
        df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce")
        df = df[df["forecast_ts"].notna() & df["generated_at"].notna()]
        if df.empty:
            return pd.DataFrame()

        latest_actual_ts = pd.Timestamp(df["forecast_ts"].max())
        rows: list[pd.DataFrame] = []

        for market, spec in BACKTEST_MARKETS.items():
            required = [spec["actual_col"], spec["p50_col"], spec["p10_col"], spec["p90_col"]]
            if any(col not in df.columns for col in required):
                continue

            market_df = df[
                [
                    "engine_name",
                    "phase_name",
                    "phase_group",
                    "run_id",
                    "generated_at",
                    "forecast_ts",
                    "horizon_hour",
                    "era_id",
                    "regime_prob_low",
                    "regime_prob_mid",
                    "regime_prob_high",
                    "uses_realized_physics",
                    "n_paths",
                    "model_version",
                    spec["expected_col"],
                    spec["p10_col"],
                    spec["p50_col"],
                    spec["p90_col"],
                    spec["actual_col"],
                ]
            ].copy()
            market_df = market_df.dropna(subset=[spec["actual_col"], spec["p50_col"]])
            if market_df.empty:
                continue

            market_df["market"] = market
            market_df["market_label"] = spec["label"]
            market_df["expected_pld_market"] = pd.to_numeric(market_df[spec["expected_col"]], errors="coerce")
            market_df["pld_p10_market"] = pd.to_numeric(market_df[spec["p10_col"]], errors="coerce")
            market_df["pld_p50_market"] = pd.to_numeric(market_df[spec["p50_col"]], errors="coerce")
            market_df["pld_p90_market"] = pd.to_numeric(market_df[spec["p90_col"]], errors="coerce")
            market_df["actual_pld_market"] = pd.to_numeric(market_df[spec["actual_col"]], errors="coerce")
            market_df = market_df[
                (market_df["actual_pld_market"] > 0) & (market_df["pld_p50_market"] > 0)
            ].copy()
            if market_df.empty:
                continue

            market_df["lead_days"] = pd.to_numeric(market_df["horizon_hour"], errors="coerce").fillna(0).div(24.0)
            market_df["lead_bucket"] = (
                pd.to_numeric(market_df["horizon_hour"], errors="coerce")
                .fillna(0)
                .astype(int)
                .map(AdaptivePLDForwardEngine._feedback_lead_bucket)
            )
            regime_probs = market_df[["regime_prob_low", "regime_prob_mid", "regime_prob_high"]].fillna(0.0).to_numpy(dtype=float)
            dominant_regime_id = np.argmax(regime_probs, axis=1).astype(int)
            market_df["dominant_regime_id"] = dominant_regime_id
            market_df["dominant_regime"] = [REGIME_ID_TO_LABEL.get(int(v), "mid") for v in dominant_regime_id]
            market_df["forecast_error"] = market_df["pld_p50_market"] - market_df["actual_pld_market"]
            market_df["abs_error"] = market_df["forecast_error"].abs()
            market_df["ape"] = (
                market_df["abs_error"] / market_df["actual_pld_market"].clip(lower=1.0)
            ).clip(lower=0.0, upper=5.0)
            market_df["inside_band"] = (
                (market_df["actual_pld_market"] >= market_df["pld_p10_market"])
                & (market_df["actual_pld_market"] <= market_df["pld_p90_market"])
            ).astype(int)
            market_df["below_p10"] = (market_df["actual_pld_market"] < market_df["pld_p10_market"]).astype(int)
            market_df["above_p90"] = (market_df["actual_pld_market"] > market_df["pld_p90_market"]).astype(int)
            age_days = (
                latest_actual_ts - pd.to_datetime(market_df["forecast_ts"], errors="coerce")
            ).dt.total_seconds().div(24 * 3600).clip(lower=0.0)
            decay = np.log(2) / 21.0
            market_df["quality_weight"] = np.exp(-decay * age_days.to_numpy(dtype=float))

            rows.append(
                market_df[
                    [
                        "engine_name",
                        "phase_name",
                        "phase_group",
                        "market",
                        "market_label",
                        "run_id",
                        "generated_at",
                        "forecast_ts",
                        "horizon_hour",
                        "lead_days",
                        "lead_bucket",
                        "era_id",
                        "dominant_regime_id",
                        "dominant_regime",
                        "uses_realized_physics",
                        "n_paths",
                        "expected_pld_market",
                        "pld_p10_market",
                        "pld_p50_market",
                        "pld_p90_market",
                        "actual_pld_market",
                        "forecast_error",
                        "abs_error",
                        "ape",
                        "inside_band",
                        "below_p10",
                        "above_p90",
                        "quality_weight",
                        "model_version",
                    ]
                ]
            )

        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values(["phase_name", "market", "forecast_ts", "generated_at"]).reset_index(drop=True)

    def _build_quality_summary(self, quality_obs: pd.DataFrame) -> pd.DataFrame:
        if quality_obs is None or quality_obs.empty:
            return pd.DataFrame()

        obs = quality_obs.copy()
        obs["forecast_ts"] = pd.to_datetime(obs["forecast_ts"], errors="coerce")
        obs = obs[obs["forecast_ts"].notna()].copy()
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

        def summarize(group: pd.DataFrame, window_days: int, lead_bucket: str) -> None:
            if group.empty:
                return
            error = pd.to_numeric(group["forecast_error"], errors="coerce")
            abs_error = pd.to_numeric(group["abs_error"], errors="coerce")
            ape = pd.to_numeric(group["ape"], errors="coerce")
            inside = pd.to_numeric(group["inside_band"], errors="coerce")
            below = pd.to_numeric(group["below_p10"], errors="coerce")
            above = pd.to_numeric(group["above_p90"], errors="coerce")
            weights = pd.to_numeric(group["quality_weight"], errors="coerce").fillna(0.0)
            if float(weights.sum()) > 0:
                weighted_mae = float(np.average(abs_error, weights=weights))
                weighted_mape = float(np.average(ape, weights=weights) * 100.0)
                weighted_bias = float(np.average(error, weights=weights))
            else:
                weighted_mae = float(abs_error.mean())
                weighted_mape = float(ape.mean() * 100.0)
                weighted_bias = float(error.mean())

            base = group.iloc[0]
            rows.append(
                {
                    "engine_name": BACKTEST_ENGINE_NAME,
                    "updated_at": updated_at,
                    "phase_name": base["phase_name"],
                    "phase_group": base["phase_group"],
                    "market": base["market"],
                    "market_label": base["market_label"],
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

        for window_days in BACKTEST_SUMMARY_WINDOWS:
            if window_days >= 99999:
                window_df = obs.copy()
            else:
                cutoff = latest_actual_ts - pd.Timedelta(days=window_days)
                window_df = obs[obs["forecast_ts"] >= cutoff].copy()
            if window_df.empty:
                continue

            for _, scoped in window_df.groupby(["phase_name", "phase_group", "market"]):
                scoped = scoped.copy()
                scoped["market_label"] = scoped["market_label"].ffill().bfill()
                summarize(scoped, window_days, "all")
                for lead_bucket in lead_bucket_order[1:]:
                    summarize(scoped[scoped["lead_bucket"] == lead_bucket], window_days, lead_bucket)

        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out["lead_bucket"] = pd.Categorical(out["lead_bucket"], categories=lead_bucket_order, ordered=True)
        return out.sort_values(["phase_name", "market", "window_days", "lead_bucket"]).reset_index(drop=True)

    def _persist_to_duckdb(
        self,
        forecast_table: pd.DataFrame,
        quality_obs: pd.DataFrame,
        quality_summary: pd.DataFrame,
    ) -> None:
        if duckdb is None:
            raise ImportError("duckdb is required to persist adaptive backtest results.")

        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(self.duckdb_path))
        try:
            if not forecast_table.empty:
                con.register("adaptive_pld_backtest_tmp", forecast_table)
                con.execute(f"DROP TABLE IF EXISTS {BACKTEST_LOCAL_TABLE_NAME}")
                con.execute(
                    f"""
                    CREATE TABLE {BACKTEST_LOCAL_TABLE_NAME} AS
                    SELECT *
                    FROM adaptive_pld_backtest_tmp
                    """
                )
            if not quality_obs.empty:
                con.register("adaptive_pld_backtest_quality_tmp", quality_obs)
                con.execute(f"DROP TABLE IF EXISTS {BACKTEST_QUALITY_LOCAL_TABLE_NAME}")
                con.execute(
                    f"""
                    CREATE TABLE {BACKTEST_QUALITY_LOCAL_TABLE_NAME} AS
                    SELECT *
                    FROM adaptive_pld_backtest_quality_tmp
                    """
                )
            if not quality_summary.empty:
                con.register("adaptive_pld_backtest_summary_tmp", quality_summary)
                con.execute(f"DROP TABLE IF EXISTS {BACKTEST_SUMMARY_LOCAL_TABLE_NAME}")
                con.execute(
                    f"""
                    CREATE TABLE {BACKTEST_SUMMARY_LOCAL_TABLE_NAME} AS
                    SELECT *
                    FROM adaptive_pld_backtest_summary_tmp
                    """
                )
            LOGGER.info(
                "Adaptive PLD backtest persisted to DuckDB: forecast=%s | obs=%s | summary=%s",
                len(forecast_table),
                len(quality_obs),
                len(quality_summary),
            )
        finally:
            con.close()

    def _persist_to_auth(
        self,
        forecast_table: pd.DataFrame,
        quality_obs: pd.DataFrame,
        quality_summary: pd.DataFrame,
    ) -> None:
        auth_url = os.getenv("DATABASE_URL_AUTH", os.getenv("DATABASE_URL", ""))
        if not auth_url or psycopg2 is None:
            return

        conn = psycopg2.connect(auth_url)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            if not forecast_table.empty:
                self._execute_batch_insert(cur, BACKTEST_AUTH_TABLE_NAME, forecast_table)

            if not quality_obs.empty:
                self._execute_batch_insert(cur, BACKTEST_QUALITY_AUTH_TABLE_NAME, quality_obs)

            if not quality_summary.empty:
                self._execute_batch_insert(cur, BACKTEST_SUMMARY_AUTH_TABLE_NAME, quality_summary)
            cur.close()
        finally:
            conn.close()

    def _execute_batch_insert(self, cur: Any, table_name: str, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        dtype_map = {
            "object": "TEXT",
            "int64": "BIGINT",
            "Int64": "BIGINT",
            "float64": "DOUBLE PRECISION",
            "bool": "BOOLEAN",
            "datetime64[ns]": "TIMESTAMP",
        }
        col_defs = []
        for col in cols:
            dtype = str(df[col].dtype)
            sql_dtype = dtype_map.get(dtype, "DOUBLE PRECISION" if "float" in dtype else "TEXT")
            col_defs.append(f"{col} {sql_dtype}")
        cur.execute(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")
        column_list = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})"
        psycopg2.extras.execute_batch(cur, sql, self._sql_records(df), page_size=500)

    def _sql_records(self, df: pd.DataFrame) -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
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
    parser = argparse.ArgumentParser(description="Run adaptive PLD historical backtest.")
    parser.add_argument("--duckdb-path", default="data/kintuadi.duckdb", help="Local DuckDB path.")
    parser.add_argument("--start-date", default="2021-01-01", help="Backtest start date.")
    parser.add_argument("--end-date", default=None, help="Backtest end date.")
    parser.add_argument("--short-stride-days", type=int, default=1, help="Daily stride for phase 1 and short walk-forward.")
    parser.add_argument("--long-stride-days", type=int, default=7, help="Weekly stride for long walk-forward.")
    parser.add_argument("--phase2-n-paths", type=int, default=250, help="Monte Carlo paths for walk-forward phases.")
    parser.add_argument("--no-persist", action="store_true", help="Do not persist outputs.")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip structural hindcast.")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip walk-forward phases.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    engine = AdaptivePLDBacktestEngine(
        duckdb_path=args.duckdb_path,
        start_date=args.start_date,
        end_date=args.end_date,
        short_stride_days=args.short_stride_days,
        long_stride_days=args.long_stride_days,
        phase2_n_paths=args.phase2_n_paths,
    )
    result = engine.run(
        persist=not args.no_persist,
        include_phase1=not args.skip_phase1,
        include_phase2=not args.skip_phase2,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    if not result.quality_summary.empty:
        print(result.quality_summary.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
