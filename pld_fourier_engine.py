# -*- coding: utf-8 -*-
"""
pld_fourier_engine.py — MAÁTria Energia · Motor Fourier por Submercado
=======================================================================
Motor dedicado de previsão de curto prazo (7 dias) usando decomposição
espectral separada por submercado.

Arquitetura:
  PLD(t) = Componente_Estrutural(t)   ← Fourier (periodicidades fixas)
           + Componente_Fundamental(t) ← estado hidrológico (EAR, ENA, CMO)
           + Resíduo(t)                ← ML sobre erro do Fourier

Vantagem sobre modelo único:
  - SE/CO, Sul, NE e Norte têm dinâmicas espectrais distintas
  - Sul: forte componente semanal (menor interconexão)
  - NE: forte componente de geração eólica (período ~6h)
  - SE/CO: curva de pato dominante (período 24h)
  - Norte: regime hídrico diferente (sazonalidade oposta)

Uso:
  python pld_fourier_engine.py --analyze           # espectro por submercado
  python pld_fourier_engine.py --forecast          # previsão 7 dias
  python pld_fourier_engine.py --forecast --days 14
  python pld_fourier_engine.py --performance       # painel de performance
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

try:
    from scipy.signal import stft, welch
    from scipy.fft import rfft, rfftfreq, irfft
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    import duckdb as _ddb
    _DUCKDB_OK = True
except ImportError:
    _DUCKDB_OK = False

DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))
DUCKDB    = DATA_DIR / "kintuadi.duckdb"

SUBMERCADOS = {
    "seco": {"label": "SE/CO",     "db_sub": "SUDESTE"},
    "s":    {"label": "Sul",       "db_sub": "SUL"},
    "ne":   {"label": "Nordeste",  "db_sub": "NORDESTE"},
    "n":    {"label": "Norte",     "db_sub": "NORTE"},
}

# Frequências de interesse por submercado (ciclos/hora)
# Cada submercado tem seu perfil espectral característico
SPECTRAL_PROFILE = {
    "seco": {  # SE/CO: duck curve dominante
        "target_periods": [24, 12, 8, 168],  # horas
        "dominant": 24,
    },
    "s": {     # Sul: componente semanal forte
        "target_periods": [24, 168, 12],
        "dominant": 168,
    },
    "ne": {    # NE: vento (períodos ~6h) + diário
        "target_periods": [24, 6, 12, 168],
        "dominant": 24,
    },
    "n": {     # Norte: regime hídrico, menos intradiário
        "target_periods": [24, 168, 720],
        "dominant": 24,
    },
}

PLD_LIMITS_BY_YEAR = {
    2021: (49.77, 1141.85),
    2022: (55.70, 1326.50),
    2023: (69.04, 1391.56),
    2024: (61.07, 1470.57),
    2025: (58.60, 1542.23),
    2026: (57.31, 1217.65),
}


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUTURAS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubForecast:
    """Previsão horária de um submercado para os próximos N dias."""
    submercado:     str
    label:          str
    horizon_hours:  int
    timestamps:     List[str]
    pld_structural: List[float]  # componente Fourier
    pld_fundamental: List[float] # ajuste hidrológico
    pld_p10:        List[float]
    pld_p50:        List[float]
    pld_p90:        List[float]
    spectral_entropy: float
    dominant_period_h: float
    regime:         str
    confidence:     float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformancePanel:
    """Painel de performance do motor — previsto vs realizado."""
    periodo_avaliado: str
    n_semanas:        int
    subsistemas:      Dict[str, Dict]  # por sub: MAE, MAPE, cobertura, bias
    mae_global:       float
    mape_global:      float
    cobertura_p10:    float
    cobertura_p90:    float
    bias_direcional:  float
    melhor_sub:       str
    pior_sub:         str

    def to_dict(self) -> Dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR FOURIER POR SUBMERCADO
# ══════════════════════════════════════════════════════════════════════════════

class PLDFourierEngine:
    """
    Motor de previsão baseado em decomposição de Fourier por submercado.

    Para cada submercado:
      1. Extrai histórico horário do DuckDB (lookback adaptável)
      2. Aplica FFT para identificar componentes periódicas dominantes
      3. Reconstrói a componente estrutural (soma das N frequências mais energéticas)
      4. Ajusta com estado fundamental (CMO, EAR, ENA do PMO)
      5. Adiciona resíduo via ML simples (Ridge sobre features de estado)
      6. Projeta para os próximos N dias com incerteza crescente
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.duckdb_path = data_dir / "kintuadi.duckdb"
        self._history_cache: Optional[pd.DataFrame] = None

    @staticmethod
    def _pld_regulatory_bounds(ts: Optional[pd.Timestamp] = None) -> Tuple[float, float]:
        ref_ts = pd.Timestamp(ts or datetime.now())
        year = int(ref_ts.year)
        if year in PLD_LIMITS_BY_YEAR:
            return PLD_LIMITS_BY_YEAR[year]
        latest_year = max(PLD_LIMITS_BY_YEAR)
        earliest_year = min(PLD_LIMITS_BY_YEAR)
        return PLD_LIMITS_BY_YEAR[latest_year if year > latest_year else earliest_year]

    # ── Carga de dados ────────────────────────────────────────────────────────

    def load_history(self, days: int = 90) -> pd.DataFrame:
        """Carrega histórico horário por submercado do DuckDB."""
        if self._history_cache is not None and len(self._history_cache) > 0:
            return self._history_cache

        if not (self.duckdb_path.exists() and _DUCKDB_OK):
            return pd.DataFrame()

        con = _ddb.connect(str(self.duckdb_path), read_only=True)
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        results = {}
        try:
            for key, info in SUBMERCADOS.items():
                sub_db = info["db_sub"]
                try:
                    df = con.execute(f"""
                        SELECT
                            date_trunc('hour', data) AS ts,
                            AVG(pld) AS pld
                        FROM pld_historical
                        WHERE UPPER(TRIM(submercado)) = '{sub_db}'
                          AND data >= '{cutoff}' AND pld > 0
                        GROUP BY 1 ORDER BY 1
                    """).df()
                    if not df.empty:
                        df["ts"] = pd.to_datetime(df["ts"])
                        results[key] = df.set_index("ts")["pld"].rename(key)
                except Exception as e:
                    logger.warning(f"{key}: {e}")
        finally:
            con.close()

        if not results:
            return pd.DataFrame()

        hist = pd.DataFrame(results).sort_index()
        hist = hist.ffill(limit=3).dropna(how="all")
        self._history_cache = hist
        logger.info(f"Histórico: {len(hist)} horas | subs: {list(hist.columns)}")
        return hist

    # ── Decomposição Fourier por submercado ───────────────────────────────────

    def decompose(self, series: pd.Series, sub: str
                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Decompõe a série em componentes de Fourier.

        Retorna:
          structural: sinal reconstruído pelas N frequências dominantes
          residual:   série original − structural
          meta:       metadados espectrais
        """
        x = pd.to_numeric(series, errors="coerce").ffill().bfill().values
        n = len(x)
        if n < 48:
            return x.copy(), np.zeros(n), {}

        # FFT
        X = rfft(x)
        freqs = rfftfreq(n, d=1.0)  # ciclos/hora

        # Amplitudes
        amp = np.abs(X) / n

        # Frequências de interesse para este submercado
        profile = SPECTRAL_PROFILE.get(sub, SPECTRAL_PROFILE["seco"])
        target_periods = profile["target_periods"]

        # Selecionar os índices de frequência mais relevantes
        # Estratégia: top-K por amplitude + frequências dos períodos-alvo
        n_harmonics = 8  # até 8 harmônicos
        top_idx = np.argsort(-amp)[1:n_harmonics+1]  # ignorar DC (idx=0)

        # Também incluir índices próximos aos períodos-alvo
        target_idx = set()
        for period in target_periods:
            if period > 0:
                target_freq = 1.0 / period
                idx = np.argmin(np.abs(freqs - target_freq))
                # incluir ±1 para capturar pequenas variações
                target_idx.update([max(0, idx-1), idx, min(len(freqs)-1, idx+1)])

        selected_idx = sorted(set(list(top_idx)) | target_idx)

        # Reconstruir apenas com frequências selecionadas
        X_filtered = np.zeros_like(X)
        X_filtered[0] = X[0]  # DC (média)
        for i in selected_idx:
            X_filtered[i] = X[i]

        structural = irfft(X_filtered, n=n)
        residual   = x - structural

        # Metadados
        dom_idx = int(np.argmax(amp[1:])) + 1  # ignorar DC
        dom_freq = float(freqs[dom_idx])
        dom_period = (1.0 / dom_freq) if dom_freq > 1e-9 else float('inf')

        # Entropia espectral normalizada
        p = amp / (amp.sum() + 1e-12)
        entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0] + 1e-12))) / np.log2(len(p))

        meta = {
            "dominant_period_h":   round(dom_period, 1),
            "dominant_freq":       round(dom_freq, 6),
            "spectral_entropy":    round(entropy, 4),
            "n_components":        len(selected_idx),
            "structural_r2":       float(1 - np.var(residual) / (np.var(x) + 1e-12)),
            "amplitude_at_24h":    float(amp[np.argmin(np.abs(freqs - 1/24))]) if len(freqs) > 0 else 0.0,
            "amplitude_at_168h":   float(amp[np.argmin(np.abs(freqs - 1/168))]) if len(freqs) > 0 else 0.0,
        }
        return structural, residual, meta

    # ── Previsão por extrapolação Fourier ─────────────────────────────────────

    def forecast_structural(self, series: pd.Series, sub: str,
                             horizon: int = 168) -> np.ndarray:
        """
        Projeta o componente estrutural para os próximos `horizon` horas.
        Usa os coeficientes de Fourier estimados no histórico recente.
        """
        x  = pd.to_numeric(series, errors="coerce").ffill().bfill().values
        n  = len(x)
        if n < 48:
            return np.full(horizon, float(np.nanmean(x)))

        X = rfft(x)
        freqs = rfftfreq(n, d=1.0)
        amp   = np.abs(X) / n

        # Manter apenas as frequências mais energéticas (top-8)
        top_k = 8
        top_idx = np.argsort(-amp)[1:top_k+1]

        # Extrapolação: recalcular a série para n + horizon pontos
        t_future = np.arange(n, n + horizon, dtype=float)
        forecast  = np.full(horizon, float(X[0].real / n))  # inicia com média

        for i in top_idx:
            freq  = float(freqs[i])
            coef  = X[i] / n
            phase = float(np.angle(coef))
            amp_i = float(np.abs(coef))
            forecast += 2 * amp_i * np.cos(2 * np.pi * freq * t_future + phase)

        return forecast

    # ── Ajuste fundamental (estado hidrológico) ───────────────────────────────

    def get_fundamental_adjustment(self, sub: str) -> float:
        """
        Retorna ajuste de nível baseado no estado atual do sistema.
        Usa CMO, EAR e ENA do PMO ou do DuckDB como referência.
        """
        try:
            from pld_forecast_engine import get_latest_pmo_state, PMO_XLSX
            pmo = get_latest_pmo_state(PMO_XLSX)

            sub_map = {"seco": "SE/CO", "s": "Sul", "ne": "NE", "n": "Norte"}
            lbl = sub_map.get(sub, "SE/CO")

            # CMO atual como proxy do nível fundamental
            cmo = pmo.get(f"{lbl} Med.Sem. (R$/MWh)")
            if cmo is not None:
                return float(cmo)
        except Exception:
            pass

        # Fallback: usar média dos últimos 7 dias do DuckDB
        hist = self.load_history(days=14)
        if not hist.empty and sub in hist.columns:
            return float(hist[sub].iloc[-168:].mean())
        return 250.0  # fallback neutro

    # ── Incerteza por horizonte ────────────────────────────────────────────────

    def compute_uncertainty(self, residuals: np.ndarray, horizon: int,
                             meta: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula bandas de incerteza P10/P90 para o horizonte.
        A incerteza cresce com o horizonte e é calibrada pelo resíduo histórico.
        """
        sigma_base = float(np.std(residuals)) if len(residuals) > 1 else 30.0
        sigma_min  = sigma_base * 0.5   # incerteza mínima (h=1)
        sigma_max  = sigma_base * 3.0   # incerteza máxima (h=168)

        # Cresce como raiz quadrada do horizonte (difusão browniana)
        h   = np.arange(1, horizon + 1, dtype=float)
        sig = sigma_min + (sigma_max - sigma_min) * np.sqrt(h / horizon)

        # Percentil ~10 e ~90 (±1.28σ para distribuição normal)
        p10 = -1.28 * sig
        p90 = +1.28 * sig
        return p10, p90

    # ── Pipeline completo de previsão ─────────────────────────────────────────

    def forecast(self, days: int = 7) -> Dict[str, SubForecast]:
        """
        Gera previsão de PLD para os próximos `days` dias por submercado.
        """
        horizon = days * 24
        hist    = self.load_history(days=90)

        if hist.empty:
            logger.warning("Sem dados históricos disponíveis")
            return {}

        results = {}
        start_ts = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        timestamps = [(start_ts + timedelta(hours=i)).isoformat() for i in range(horizon)]
        pld_floor, pld_cap = self._pld_regulatory_bounds(pd.Timestamp(start_ts))

        for sub in SUBMERCADOS:
            if sub not in hist.columns:
                continue
            series = hist[sub].dropna()
            if len(series) < 48:
                continue

            # 1. Decomposição Fourier
            structural, residuals, meta = self.decompose(series, sub)

            # 2. Previsão estrutural (extrapolação dos coeficientes)
            fc_structural = self.forecast_structural(series, sub, horizon)

            # 3. Ajuste fundamental (nível hidrológico atual)
            fc_mean  = float(np.mean(fc_structural))
            fund_lvl = self.get_fundamental_adjustment(sub)
            # Blend: 60% Fourier shape + 40% nível fundamental
            fc_adjusted = fc_structural + 0.40 * (fund_lvl - fc_mean)

            # 4. Clip regulatório
            fc_adjusted = np.clip(fc_adjusted, pld_floor, pld_cap)

            # 5. Bandas de incerteza
            p10_delta, p90_delta = self.compute_uncertainty(residuals, horizon, meta)
            fc_p10 = np.clip(fc_adjusted + p10_delta, pld_floor, pld_cap)
            fc_p90 = np.clip(fc_adjusted + p90_delta, pld_floor, pld_cap)

            # 6. Detectar regime
            entropy = meta.get("spectral_entropy", 0.5)
            mean_pld = float(series.iloc[-168:].mean()) if len(series) >= 168 else float(series.mean())
            regime = ("stressed"   if mean_pld > 350 else
                      "duck_curve" if entropy < 0.55 else
                      "flat"       if mean_pld < 150 else "normal")

            results[sub] = SubForecast(
                submercado      = sub,
                label           = SUBMERCADOS[sub]["label"],
                horizon_hours   = horizon,
                timestamps      = timestamps,
                pld_structural  = [round(float(v), 2) for v in fc_structural.tolist()],
                pld_fundamental = [round(float(v), 2) for v in fc_adjusted.tolist()],
                pld_p10         = [round(float(v), 2) for v in fc_p10.tolist()],
                pld_p50         = [round(float(v), 2) for v in fc_adjusted.tolist()],
                pld_p90         = [round(float(v), 2) for v in fc_p90.tolist()],
                spectral_entropy   = round(entropy, 4),
                dominant_period_h  = meta.get("dominant_period_h", 24.0),
                regime          = regime,
                confidence      = round(1.0 - entropy, 4),
            )
            logger.info(f"{sub}: estrutural R²={meta.get('structural_r2',0):.3f} | "
                        f"período dom.={meta.get('dominant_period_h',0):.0f}h | regime={regime}")

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL DE PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════

    def compute_performance(self, lookback_weeks: int = 4) -> PerformancePanel:
        """
        Compara previsão vs realizado para as últimas `lookback_weeks` semanas.

        Metodologia:
          Para cada semana W no passado:
            - Usa dados até domingo da semana W-1 como base
            - Prevê PLD horário da semana W com o motor Fourier
            - Compara com PLD realizado da semana W no DuckDB

        Métricas:
          MAE, MAPE, cobertura P10/P90, bias direcional
        """
        hist = self.load_history(days=(lookback_weeks + 2) * 7)
        if hist.empty:
            return self._empty_performance()

        sub_metrics = {}
        for sub in SUBMERCADOS:
            if sub not in hist.columns:
                continue
            series = hist[sub].dropna()
            if len(series) < 336:  # mínimo 2 semanas
                continue

            mae_list = mape_list = bias_list = []
            cov_p10  = cov_p90  = 0
            n_hours  = 0

            # Rolling: para cada semana completa disponível
            idx = pd.to_datetime(series.index)
            week_starts = pd.date_range(
                idx[0].normalize() + timedelta(days=(7-idx[0].weekday()) % 7),
                idx[-1].normalize() - timedelta(days=7),
                freq="W-MON"
            )

            for ws in week_starts[-lookback_weeks:]:
                we = ws + timedelta(days=7)
                # Histórico até o início da semana
                base = series[series.index < ws]
                # Realizado da semana
                real = series[(series.index >= ws) & (series.index < we)]
                if len(base) < 168 or len(real) < 24:
                    continue

                # Previsão para a semana
                fc_struct = self.forecast_structural(base, sub, len(real))
                fc_mean   = float(np.mean(fc_struct))
                fund      = float(base.iloc[-168:].mean())
                fc_adj    = fc_struct + 0.40 * (fund - fc_mean)
                pld_floor, pld_cap = self._pld_regulatory_bounds(pd.Timestamp(ws))
                fc_adj    = np.clip(fc_adj, pld_floor, pld_cap)

                # Resíduo para bandas
                _, resid, _ = self.decompose(base, sub)
                p10d, p90d = self.compute_uncertainty(resid, len(real), {})
                fc_p10 = np.clip(fc_adj + p10d, pld_floor, pld_cap)
                fc_p90 = np.clip(fc_adj + p90d, pld_floor, pld_cap)

                y_real = real.values
                y_pred = fc_adj[:len(y_real)]
                fc_p10 = fc_p10[:len(y_real)]
                fc_p90 = fc_p90[:len(y_real)]

                # Métricas
                mae_list.append(float(np.mean(np.abs(y_pred - y_real))))
                mape_ok = y_real > 10
                if mape_ok.any():
                    mape_list.append(float(np.mean(np.abs(y_pred[mape_ok] - y_real[mape_ok]) / y_real[mape_ok]) * 100))
                bias_list.append(float(np.mean(y_pred - y_real)))
                cov_p10 += int(np.sum(y_real >= fc_p10))
                cov_p90 += int(np.sum(y_real <= fc_p90))
                n_hours += len(y_real)

            if not mae_list:
                continue

            sub_metrics[sub] = {
                "label":        SUBMERCADOS[sub]["label"],
                "mae":          round(float(np.mean(mae_list)), 2),
                "mape_pct":     round(float(np.mean(mape_list)), 1) if mape_list else None,
                "bias":         round(float(np.mean(bias_list)), 2),
                "cobertura_p10": round(cov_p10 / max(n_hours, 1), 3),
                "cobertura_p90": round(cov_p90 / max(n_hours, 1), 3),
                "n_semanas":    len(mae_list),
                "n_horas":      n_hours,
            }

        if not sub_metrics:
            return self._empty_performance()

        mae_global  = float(np.mean([m["mae"] for m in sub_metrics.values()]))
        mape_global = float(np.mean([m["mape_pct"] for m in sub_metrics.values()
                                      if m["mape_pct"] is not None]))
        cov10_global = float(np.mean([m["cobertura_p10"] for m in sub_metrics.values()]))
        cov90_global = float(np.mean([m["cobertura_p90"] for m in sub_metrics.values()]))
        bias_global  = float(np.mean([m["bias"] for m in sub_metrics.values()]))

        melhor = min(sub_metrics, key=lambda s: sub_metrics[s]["mae"])
        pior   = max(sub_metrics, key=lambda s: sub_metrics[s]["mae"])

        return PerformancePanel(
            periodo_avaliado = f"últimas {lookback_weeks} semanas",
            n_semanas        = lookback_weeks,
            subsistemas      = sub_metrics,
            mae_global       = round(mae_global, 2),
            mape_global      = round(mape_global, 1),
            cobertura_p10    = round(cov10_global, 3),
            cobertura_p90    = round(cov90_global, 3),
            bias_direcional  = round(bias_global, 2),
            melhor_sub       = SUBMERCADOS[melhor]["label"],
            pior_sub         = SUBMERCADOS[pior]["label"],
        )

    def _empty_performance(self) -> PerformancePanel:
        return PerformancePanel(
            periodo_avaliado="sem dados", n_semanas=0, subsistemas={},
            mae_global=0.0, mape_global=0.0, cobertura_p10=0.0,
            cobertura_p90=0.0, bias_direcional=0.0,
            melhor_sub="N/A", pior_sub="N/A",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE ESPECTRAL POR SUBMERCADO
# ══════════════════════════════════════════════════════════════════════════════

def analyze_spectra(data_dir: Path = DATA_DIR) -> Dict:
    """Análise espectral comparativa dos 4 submercados."""
    engine = PLDFourierEngine(data_dir)
    hist   = engine.load_history(days=90)

    if hist.empty:
        return {}

    results = {}
    for sub in SUBMERCADOS:
        if sub not in hist.columns:
            continue
        series = hist[sub].dropna()
        if len(series) < 168:
            continue

        _, _, meta = engine.decompose(series, sub)
        profile    = SPECTRAL_PROFILE.get(sub, {})

        results[sub] = {
            "label":    SUBMERCADOS[sub]["label"],
            "n_horas":  len(series),
            "media":    round(float(series.mean()), 2),
            "std":      round(float(series.std()), 2),
            "periodo_dominante_h": meta.get("dominant_period_h"),
            "entropia_espectral":  meta.get("spectral_entropy"),
            "r2_estrutural":       meta.get("structural_r2"),
            "amp_24h":             round(meta.get("amplitude_at_24h", 0), 4),
            "amp_168h":            round(meta.get("amplitude_at_168h", 0), 4),
            "target_periods":      profile.get("target_periods", [24]),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="MAÁTria · PLD Fourier Engine")
    ap.add_argument("--analyze",     action="store_true", help="Análise espectral por submercado")
    ap.add_argument("--forecast",    action="store_true", help="Previsão 7 dias por submercado")
    ap.add_argument("--performance", action="store_true", help="Painel de performance previsto vs realizado")
    ap.add_argument("--days",        type=int, default=7, help="Horizonte de previsão em dias")
    ap.add_argument("--weeks",       type=int, default=4, help="Semanas de avaliação de performance")
    ap.add_argument("--data",        default="data")
    args = ap.parse_args()

    engine = PLDFourierEngine(Path(args.data))

    if args.analyze or (not args.forecast and not args.performance):
        print("\n=== Análise espectral por submercado ===")
        spectra = analyze_spectra(Path(args.data))
        for sub, info in spectra.items():
            print(f"\n  [{info['label']}]")
            print(f"    Horas:               {info['n_horas']}")
            print(f"    PLD médio:           R${info['media']:.0f}/MWh")
            print(f"    Desvio padrão:       R${info['std']:.0f}/MWh")
            print(f"    Período dominante:   {info['periodo_dominante_h']:.0f}h")
            print(f"    Entropia espectral:  {info['entropia_espectral']:.3f} "
                  f"({'previsível' if info['entropia_espectral'] < 0.5 else 'complexo'})")
            print(f"    R² componente est.:  {info['r2_estrutural']:.3f}")
            print(f"    Amplitude 24h:       {info['amp_24h']:.2f}")
            print(f"    Amplitude semanal:   {info['amp_168h']:.2f}")

    if args.performance:
        print(f"\n=== Painel de performance (últimas {args.weeks} semanas) ===")
        panel = engine.compute_performance(lookback_weeks=args.weeks)
        print(f"\n  Período avaliado: {panel.periodo_avaliado}")
        print(f"  MAE global:       R${panel.mae_global:.2f}/MWh")
        print(f"  MAPE global:      {panel.mape_global:.1f}%")
        print(f"  Cobertura P10:    {panel.cobertura_p10:.1%}  (ideal: 90%)")
        print(f"  Cobertura P90:    {panel.cobertura_p90:.1%}  (ideal: 90%)")
        print(f"  Bias direcional:  R${panel.bias_direcional:+.2f}/MWh "
              f"({'subestimando' if panel.bias_direcional < 0 else 'superestimando'})")
        print(f"  Melhor sub:       {panel.melhor_sub}")
        print(f"  Pior sub:         {panel.pior_sub}")

        if panel.subsistemas:
            print(f"\n  Por submercado:")
            for sub, m in panel.subsistemas.items():
                print(f"    {m['label']:<12} MAE=R${m['mae']:>7.1f}  "
                      f"MAPE={m['mape_pct']:>5.1f}%  "
                      f"Bias={m['bias']:>+8.1f}  "
                      f"Cob.P90={m['cobertura_p90']:.1%}")

    if args.forecast:
        print(f"\n=== Previsão Fourier — próximos {args.days} dias ===")
        forecasts = engine.forecast(days=args.days)
        for sub, fc in forecasts.items():
            print(f"\n  [{fc.label}]  regime={fc.regime}  "
                  f"confiança={fc.confidence:.2f}  "
                  f"período_dom={fc.dominant_period_h:.0f}h")
            # Mostrar por dia
            for day in range(min(args.days, 7)):
                h_start = day * 24
                h_end   = h_start + 24
                day_p50 = fc.pld_p50[h_start:h_end]
                day_p10 = fc.pld_p10[h_start:h_end]
                day_p90 = fc.pld_p90[h_start:h_end]
                if not day_p50:
                    break
                dt = datetime.fromisoformat(fc.timestamps[h_start]).strftime("%d/%m")
                print(f"    {dt}:  P10={min(day_p10):.0f}–{max(day_p10):.0f}  "
                      f"P50={min(day_p50):.0f}–{max(day_p50):.0f}  "
                      f"P90={min(day_p90):.0f}–{max(day_p90):.0f}")
