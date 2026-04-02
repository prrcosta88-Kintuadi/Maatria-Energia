# -*- coding: utf-8 -*-
"""
spectral_engine.py — MAÁTria Energia · Spectral Analysis Engine
================================================================
Análise espectral do PLD e net_load para identificação de regimes
operacionais, fingerprints do sistema e features para modelos preditivos.

Inspirado na arquitetura do Shazam (Avery Wang, 2003):
  áudio → espectrograma → picos → hash → matching
  PLD   → STFT          → picos → fingerprint → regime

Pipeline:
  1. STFT (Short-Time Fourier Transform) do PLD e net_load horário
  2. Extração de picos dominantes por banda de frequência
  3. Fingerprint temporal do sistema (hash comparável)
  4. Detecção de regime via coerência espectral PLD × net_load
  5. Features espectrais para modelos ML (pld_forecast_engine)
  6. Previsão da curva intra-diária (duck curve forecasting)

Uso standalone:
  python spectral_engine.py --analyze           # analisa dados disponíveis
  python spectral_engine.py --features          # extrai features para ML
  python spectral_engine.py --fingerprint       # gera fingerprint atual
  python spectral_engine.py --regime            # detecta regime atual

Integração:
  from spectral_engine import SpectralEngine, get_spectral_features
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

# ── Dependências opcionais ─────────────────────────────────────────────────────
try:
    from scipy.signal import stft, coherence, welch
    from scipy.fft import fft, fftfreq
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    logger.warning("scipy não instalado. pip install scipy")

try:
    import duckdb as _ddb
    _DUCKDB_OK = True
except ImportError:
    _DUCKDB_OK = False

# ── Constantes ────────────────────────────────────────────────────────────────
DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/models"))

# Frequências de interesse no sistema elétrico brasileiro (em ciclos/hora)
# Período 24h = freq 1/24 ≈ 0.042 cph
# Período 12h (harmônico) = freq 1/12 ≈ 0.083 cph
# Período semanal = freq 1/168 ≈ 0.006 cph
FREQ_DIARIA    = 1 / 24     # ciclos/hora — ciclo solar/demanda
FREQ_SEMIDIARIA = 1 / 12    # harmônico — pico manha/tarde
FREQ_SEMANAL   = 1 / 168    # ciclo semanal
FREQ_MENSAL    = 1 / 720    # ciclo mensal

# Janelas de análise
WINDOW_STFT   = 24          # horas por janela STFT (1 dia)
OVERLAP_STFT  = 12          # sobreposição (50%)
LOOKBACK_DAYS = 30          # dias de histórico para análise
LIBRARY_LOOKBACK_DAYS = 180
LIBRARY_DAILY_STRIDE_H = 24
LIBRARY_WEEKLY_STRIDE_H = 24 * 7
DEFAULT_MATCH_WINDOW_H = 24 * 14
CONSTELLATION_BANDS = 6
CONSTELLATION_FANOUT = 4
CONSTELLATION_MAX_DT = 8
CONSTELLATION_QUANTILE = 0.78

PLD_LIMITS_BY_YEAR = {
    2021: (49.77, 1141.85),
    2022: (55.70, 1326.50),
    2023: (69.04, 1391.56),
    2024: (61.07, 1470.57),
    2025: (58.60, 1542.23),
    2026: (57.31, 1611.04),
}


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralFingerprint:
    """
    Fingerprint espectral do sistema — análogo ao hash do Shazam.
    Representa a assinatura comparável do estado operacional atual.
    """
    timestamp:           str
    # Frequências dominantes (top-3) e suas amplitudes
    freq_dom_1:          float   # frequência dominante (ciclos/hora)
    amp_dom_1:           float   # amplitude (R$/MWh ou MW)
    freq_dom_2:          float
    amp_dom_2:           float
    freq_dom_3:          float
    amp_dom_3:           float
    # Hora de pico da curva diária
    peak_hour:           float   # hora do pico de preço (0-23)
    valley_hour:         float   # hora do vale (0-23)
    peak_valley_spread:  float   # diferença pico-vale em R$/MWh
    # Coerência espectral PLD × net_load
    coherence_mean:      float   # 0-1: 1 = preço segue físico perfeitamente
    coherence_at_daily:  float   # coerência na frequência diária
    # Entropia espectral (medida de complexidade/previsibilidade)
    spectral_entropy:    float   # 0-1: 0 = dominado por 1 freq, 1 = ruído puro
    # Regime detectado
    regime:              str     # "duck_curve" | "flat" | "stressed" | "distorted"
    regime_confidence:   float   # 0-1
    # Duck curve metrics
    duck_depth:          float   # profundidade do vale solar (MW ou R$)
    evening_ramp:        float   # rampa noturna (delta MW ou R$/h)
    spread_normalized:   float   # spread normalizado pela faixa regulatória / escala do sinal
    duck_depth_z:        float   # profundidade do duck curve em desvios-padrão
    ramp_ratio:          float   # rampa / spread
    signal_basis:        str     # "spdi" | "spdi_from_cost" | "pld_reg_norm"
    n_hashes:            int     # quantidade de anchor-target hashes
    hash_density:        float   # hashes por frame STFT
    matched_regime:      str     # regime mais parecido na biblioteca histórica
    matched_similarity:  float   # score combinado do matching
    matched_hash_overlap: float  # overlap dos hashes com o melhor match
    matched_window_start: str    # início da janela histórica mais próxima
    matched_window_end:   str    # fim da janela histórica mais próxima
    matched_library_kind: str    # "daily_like" | "weekly_like" | "none"

    def to_dict(self) -> Dict:
        return asdict(self)

    def similarity(self, other: "SpectralFingerprint") -> float:
        """
        Calcula similaridade entre dois fingerprints (0-1).
        Análogo ao matching do Shazam — quanto mais próximo de 1,
        mais parecido é o estado operacional do sistema.
        """
        scores = []
        # Similaridade de frequências dominantes
        for attr in ["freq_dom_1", "freq_dom_2", "peak_hour", "valley_hour"]:
            a, b = getattr(self, attr), getattr(other, attr)
            diff  = abs(a - b) / (max(abs(a), abs(b), 1e-9))
            scores.append(max(0, 1 - diff))
        for attr in ["spread_normalized", "duck_depth_z", "ramp_ratio", "hash_density"]:
            a, b = getattr(self, attr), getattr(other, attr)
            scores.append(max(0, 1 - abs(a - b)))
        # Similaridade de regime
        scores.append(1.0 if self.regime == other.regime else 0.0)
        # Similaridade de coerência
        scores.append(1 - abs(self.coherence_mean - other.coherence_mean))
        return float(np.mean(scores))


@dataclass
class SpectralFeatures:
    """
    Features espectrais para uso nos modelos ML (pld_forecast_engine).
    Cada feature captura um aspecto da estrutura temporal do sistema.
    """
    # Features de frequência
    dominant_freq:       float   # frequência mais energética
    dominant_period_h:   float   # período correspondente em horas
    spectral_entropy:    float   # previsibilidade espectral
    power_at_daily:      float   # energia na frequência diária
    power_at_weekly:     float   # energia na frequência semanal
    power_ratio_dw:      float   # razão diária/semanal

    # Features de forma da curva diária (duck curve)
    peak_hour:           float   # hora do pico médio
    valley_hour:         float   # hora do vale médio
    peak_valley_spread:  float   # spread pico-vale
    spread_normalized:   float   # spread normalizado
    morning_ramp_rate:   float   # taxa de subida matinal (R$/MWh por hora)
    evening_ramp_rate:   float   # taxa de subida noturna
    duck_depth:          float   # profundidade do vale solar
    duck_depth_z:        float   # profundidade do vale em desvio-padrão
    ramp_ratio:          float   # rampa / spread

    # Features de regime e coerência
    coherence_pld_load:  float   # coerência PLD × carga
    regime_encoded:      float   # regime codificado numericamente
    days_since_regime_change: float  # dias desde última mudança de regime
    matched_similarity:  float   # similaridade com regimes passados
    matched_hash_overlap: float  # overlap de hashes com melhor match
    hash_density:        float   # densidade de hashes tempo-frequência

    # Features de volatilidade espectral
    spectral_volatility: float   # variação do espectro ao longo do tempo
    phase_stability:     float   # estabilidade da fase (0=instável, 1=estável)

    def to_series(self) -> pd.Series:
        return pd.Series(asdict(self))


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SpectralEngine:
    """
    Motor de análise espectral do sistema elétrico brasileiro.

    Implementa o paradigma Shazam aplicado ao PLD:
      - identifica padrões robustos sob ruído hidrológico/despacho
      - cria fingerprints comparáveis do estado operacional
      - detecta regimes (duck curve, estresse, distorção)
      - fornece features para modelos preditivos
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.duckdb_path = data_dir / "kintuadi.duckdb"
        self._fingerprint_cache: Optional[SpectralFingerprint] = None
        self._features_cache: Optional[SpectralFeatures] = None
        self._library_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    @staticmethod
    def _pld_regulatory_bounds(ts: Optional[pd.Timestamp] = None) -> Tuple[float, float]:
        ref_ts = pd.Timestamp(ts or datetime.now())
        year = int(ref_ts.year)
        if year in PLD_LIMITS_BY_YEAR:
            return PLD_LIMITS_BY_YEAR[year]
        latest_year = max(PLD_LIMITS_BY_YEAR)
        earliest_year = min(PLD_LIMITS_BY_YEAR)
        return PLD_LIMITS_BY_YEAR[latest_year if year > latest_year else earliest_year]

    def _analysis_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"basis": "empty", "raw": pd.Series(dtype=float), "centered": pd.Series(dtype=float), "scale": 1.0}

        frame = df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        if frame.empty:
            return {"basis": "empty", "raw": pd.Series(dtype=float), "centered": pd.Series(dtype=float), "scale": 1.0}

        if "spdi" in frame.columns:
            raw = pd.to_numeric(frame["spdi"], errors="coerce").clip(lower=0.05, upper=10.0)
            basis = "spdi"
            scale = max(float(raw.quantile(0.95) - raw.quantile(0.05)), 0.15)
        elif {"pld", "custo_fisico"}.issubset(frame.columns):
            raw = (
                pd.to_numeric(frame["pld"], errors="coerce")
                / pd.to_numeric(frame["custo_fisico"], errors="coerce").replace(0, np.nan)
            ).clip(lower=0.05, upper=10.0)
            basis = "spdi_from_cost"
            scale = max(float(raw.quantile(0.95) - raw.quantile(0.05)), 0.15)
        else:
            pld = pd.to_numeric(frame.get("pld", pd.Series(index=frame.index, dtype=float)), errors="coerce")
            floors: list[float] = []
            caps: list[float] = []
            for ts in frame.index:
                floor, cap = self._pld_regulatory_bounds(ts)
                floors.append(floor)
                caps.append(cap)
            floors_arr = np.asarray(floors, dtype=float)
            caps_arr = np.asarray(caps, dtype=float)
            denom = np.maximum(caps_arr - floors_arr, 1.0)
            raw = pd.Series((pld.to_numpy(dtype=float) - floors_arr) / denom, index=frame.index).clip(lower=0.0, upper=1.0)
            basis = "pld_reg_norm"
            scale = 1.0

        raw = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return {"basis": basis, "raw": raw, "centered": raw, "scale": max(scale, 1e-6)}

        centered = raw - raw.rolling(168, min_periods=24).median().fillna(raw.median())
        centered = centered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return {
            "basis": basis,
            "raw": raw,
            "centered": centered,
            "scale": max(float(scale), 1e-6),
        }

    @staticmethod
    def _group_profile(series: pd.Series, default: float) -> Dict[str, Any]:
        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return {"weekhour": {}, "hour": {}, "global": float(default)}
        weekhour = clean.groupby([clean.index.weekday, clean.index.hour]).median().to_dict()
        hour = clean.groupby(clean.index.hour).median().to_dict()
        return {
            "weekhour": weekhour,
            "hour": hour,
            "global": float(np.nanmedian(clean)),
        }

    @staticmethod
    def _blend_profile_maps(base: Dict[str, Any], overlay: Dict[str, Any], weight: float) -> Dict[str, Any]:
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 0:
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
        base_global = float(base.get("global", 0.0))
        overlay_global = float(overlay.get("global", base_global))
        out["global"] = float((1.0 - weight) * base_global + weight * overlay_global)
        return out

    # ── Carga de dados ────────────────────────────────────────────────────────

    def load_hourly_data(self, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
        """
        Carrega dados horários do DuckDB (local) ou Neon (fallback).
        Retorna DataFrame com colunas: pld, net_load, load, hydro, solar, wind, thermal, cmo
        """
        if self.duckdb_path.exists() and _DUCKDB_OK:
            return self._load_from_duckdb(days)
        return self._load_from_neon(days)

    def _load_from_duckdb(self, days: int) -> pd.DataFrame:
        con = _ddb.connect(str(self.duckdb_path), read_only=True)
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = pd.DataFrame()
            try:
                tables = set(con.execute("SHOW TABLES").df()["name"].astype(str).str.lower().tolist())
            except Exception:
                tables = set()

            def _table_columns(table_name: str) -> set[str]:
                try:
                    info = con.execute(f"PRAGMA table_info('{table_name}')").df()
                    return set(info["name"].astype(str).str.lower().tolist())
                except Exception:
                    return set()

            def _first_existing(candidates: tuple[str, ...], available: set[str]) -> Optional[str]:
                for candidate in candidates:
                    if candidate.lower() in available:
                        return candidate
                return None

            # PLD por hora (média SE/CO)
            try:
                pld = con.execute(f"""
                    SELECT date_trunc('hour', data) AS ts,
                           AVG(pld) AS pld
                    FROM pld_historical
                    WHERE UPPER(TRIM(submercado)) IN ('SUDESTE','SE')
                      AND data >= '{cutoff}' AND pld > 0
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not pld.empty:
                    pld["ts"] = pd.to_datetime(pld["ts"])
                    df = pld.set_index("ts")[["pld"]]
            except Exception as e:
                logger.warning(f"PLD load: {e}")

            # Carga horária SIN
            try:
                load = con.execute(f"""
                    SELECT date_trunc('hour', din_instante) AS ts,
                           SUM(val_cargaenergiahomwmed) AS load
                    FROM curva_carga
                    WHERE din_instante >= '{cutoff}'
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not load.empty:
                    load["ts"] = pd.to_datetime(load["ts"])
                    df = df.join(load.set_index("ts"), how="outer")
            except Exception as e:
                logger.warning(f"Carga load: {e}")

            # Geração por tipo
            try:
                gen_df = pd.DataFrame()
                if "geracao_tipo_hora" in tables:
                    gen_df = con.execute(f"""
                        SELECT
                            date_trunc('hour', din_instante) AS ts,
                            LOWER(TRIM(tipo_geracao)) AS tipo,
                            SUM(val_geracao_mw) AS valor
                        FROM geracao_tipo_hora
                        WHERE din_instante >= '{cutoff}' AND val_geracao_mw IS NOT NULL
                        GROUP BY 1, 2
                        ORDER BY 1, 2
                    """).df()
                elif "geracao_usina_horaria" in tables:
                    gen_df = con.execute(f"""
                        SELECT
                            date_trunc('hour', din_instante) AS ts,
                            CASE
                                WHEN UPPER(nom_tipousina) IN ('FOTOVOLTAICA', 'SOLAR', 'FOTOVOLT') THEN 'solar'
                                WHEN UPPER(nom_tipousina) IN ('EOLIELÉTRICA', 'EÓLICA', 'EOLICA', 'EOLIELETRICA', 'EOLIELÉTRICO', 'EOL') THEN 'wind'
                                WHEN UPPER(nom_tipousina) IN ('HIDROELÉTRICA', 'HIDROELETRICA', 'HIDRÁULICA', 'HIDRAULICA', 'UHE', 'PCH', 'CGH', 'HIDRO') THEN 'hydro'
                                WHEN UPPER(nom_tipousina) IN ('TÉRMICA', 'TERMICA', 'UTE', 'BIOMASSA', 'GAS', 'GÁS', 'OLEO', 'ÓLEO', 'CARVAO', 'CARVÃO', 'DERIVADOS') THEN 'thermal'
                                ELSE 'other'
                            END AS tipo,
                            SUM(val_geracao) AS valor
                        FROM geracao_usina_horaria
                        WHERE din_instante >= '{cutoff}' AND val_geracao IS NOT NULL
                        GROUP BY 1, 2
                        ORDER BY 1, 2
                    """).df()

                if not gen_df.empty:
                    gen_df["ts"] = pd.to_datetime(gen_df["ts"], errors="coerce")
                    gen_df = gen_df.dropna(subset=["ts"])
                    gen_df["tipo"] = gen_df["tipo"].astype(str).str.lower().str.strip()
                    gen_df["valor"] = pd.to_numeric(gen_df["valor"], errors="coerce")
                    for generation_type, column_name in (
                        ("hydro", "hydro"),
                        ("solar", "solar"),
                        ("wind", "wind"),
                        ("thermal", "thermal"),
                    ):
                        sub = gen_df[gen_df["tipo"] == generation_type]
                        if not sub.empty:
                            df[column_name] = sub.groupby("ts")["valor"].sum()
                else:
                    for table_name, column_name in (
                        ("geracao_sin_hidraulica", "hydro"),
                        ("geracao_sin_solar", "solar"),
                        ("geracao_sin_eolica", "wind"),
                        ("geracao_sin_termica", "thermal"),
                    ):
                        if table_name not in tables:
                            continue
                        available = _table_columns(table_name)
                        ts_col = _first_existing(("din_instante", "instante", "data"), available)
                        value_col = _first_existing(("val_geracao", "val_geracao_mw", "valor"), available)
                        if ts_col is None or value_col is None:
                            continue
                        gen = con.execute(f"""
                            SELECT date_trunc('hour', {ts_col}) AS ts,
                                   SUM({value_col}) AS valor
                            FROM {table_name}
                            WHERE {ts_col} >= '{cutoff}' AND {value_col} IS NOT NULL
                            GROUP BY 1 ORDER BY 1
                        """).df()
                        if not gen.empty:
                            gen["ts"] = pd.to_datetime(gen["ts"], errors="coerce")
                            gen = gen.dropna(subset=["ts"])
                            df[column_name] = pd.to_numeric(gen.set_index("ts")["valor"], errors="coerce")
            except Exception as e:
                logger.warning(f"Geração load: {e}")

            # CMO SE/CO
            try:
                cmo = con.execute(f"""
                    SELECT date_trunc('hour', din_instante) AS ts,
                           AVG(val_cmo) AS cmo
                    FROM cmo
                    WHERE UPPER(TRIM(id_subsistema)) IN ('SUDESTE','SE')
                      AND din_instante >= '{cutoff}' AND val_cmo > 0
                    GROUP BY 1 ORDER BY 1
                """).df()
                if not cmo.empty:
                    cmo["ts"] = pd.to_datetime(cmo["ts"])
                    df["cmo"] = cmo.set_index("ts")["cmo"]
            except Exception as e:
                logger.warning(f"CMO load: {e}")

        finally:
            con.close()

        # Net load = carga - solar - wind
        if "load" in df.columns and ("solar" in df.columns or "wind" in df.columns):
            solar = pd.to_numeric(df.get("solar", pd.Series(np.nan, index=df.index)), errors="coerce")
            wind = pd.to_numeric(df.get("wind", pd.Series(np.nan, index=df.index)), errors="coerce")
            renewable_sum = solar.fillna(0.0) + wind.fillna(0.0)
            renewable_available = solar.notna() | wind.notna()
            df["net_load"] = pd.to_numeric(df["load"], errors="coerce") - renewable_sum
            df.loc[~renewable_available, "net_load"] = np.nan
        elif "load" in df.columns:
            df["net_load"] = df["load"]

        df = df.sort_index().ffill(limit=3)
        logger.info(f"DuckDB: {len(df)} horas | colunas: {list(df.columns)}")
        return df

    def _load_from_neon(self, days: int) -> pd.DataFrame:
        """Fallback: carrega do Neon via db_neon."""
        try:
            import db_neon
            if not db_neon.is_configured():
                return pd.DataFrame()
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            pld_df = db_neon.fetchdf(f"""
                SELECT date_trunc('hour',
                    MAKE_DATE(
                        CAST(SUBSTR(CAST(mes_referencia AS TEXT),1,4) AS INTEGER),
                        CAST(SUBSTR(CAST(mes_referencia AS TEXT),5,2) AS INTEGER), dia)
                    + hora * INTERVAL '1 hour') AS ts,
                    AVG(pld_hora) AS pld
                FROM pld_historical
                WHERE mes_referencia IS NOT NULL AND pld_hora > 0
                GROUP BY 1 ORDER BY 1
            """)
            if not pld_df.empty:
                pld_df["ts"] = pd.to_datetime(pld_df["ts"])
                return pld_df.set_index("ts")
        except Exception as e:
            logger.warning(f"Neon fallback: {e}")
        return pd.DataFrame()

    # ── STFT e espectrograma ──────────────────────────────────────────────────

    def compute_stft(self, series: pd.Series,
                     window: int = WINDOW_STFT,
                     overlap: int = OVERLAP_STFT
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT da série temporal.

        Retorna: (freqs, times, magnitudes)
        - freqs: array de frequências em ciclos/hora
        - times: array de timestamps
        - magnitudes: matriz de amplitudes |STFT|
        """
        if not _SCIPY_OK:
            # Implementação básica sem scipy
            return self._compute_stft_basic(series, window, overlap)

        x = pd.to_numeric(series, errors="coerce").fillna(series.median()).values
        fs = 1.0  # 1 amostra/hora
        hop = window - overlap

        freqs, times, Zxx = stft(x, fs=fs, window="hann",
                                  nperseg=window, noverlap=overlap)
        magnitudes = np.abs(Zxx)
        return freqs, times, magnitudes

    def _compute_stft_basic(self, series: pd.Series,
                             window: int, overlap: int
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STFT manual sem scipy."""
        x   = pd.to_numeric(series, errors="coerce").fillna(series.median()).values
        hop = window - overlap
        n_frames = (len(x) - window) // hop + 1
        freqs  = np.fft.rfftfreq(window, d=1.0)
        times  = np.array([i * hop for i in range(n_frames)], dtype=float)
        mags   = np.zeros((len(freqs), n_frames))
        hann   = np.hanning(window)
        for i in range(n_frames):
            seg = x[i * hop: i * hop + window] * hann
            mags[:, i] = np.abs(np.fft.rfft(seg))
        return freqs, times, mags

    # ── Extração de picos (Shazam-style) ─────────────────────────────────────

    def extract_peaks(self, freqs: np.ndarray, magnitudes: np.ndarray,
                      n_peaks: int = 5) -> List[Tuple[float, float]]:
        """
        Extrai os N picos mais energéticos do espectro médio.
        Análogo à extração de 'constellation points' do Shazam.

        Retorna lista de (frequência, amplitude) ordenada por amplitude.
        """
        spectrum = magnitudes.mean(axis=1)  # média temporal
        # Normalizar
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        # Encontrar picos locais
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                peaks.append((freqs[i], float(spectrum[i])))

        # Ordenar por amplitude e retornar top-N
        peaks.sort(key=lambda x: -x[1])
        return peaks[:n_peaks]

    def extract_constellation_points(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        magnitudes: np.ndarray,
        n_bands: int = CONSTELLATION_BANDS,
        threshold_quantile: float = CONSTELLATION_QUANTILE,
    ) -> List[Dict[str, float]]:
        """
        Extrai pontos de constelação tempo-frequência no espírito do Shazam.

        Para cada frame STFT, mantém os picos mais fortes por banda de frequência.
        """
        if magnitudes.size == 0 or len(freqs) == 0 or len(times) == 0:
            return []

        positive = magnitudes[magnitudes > 0]
        threshold = float(np.quantile(positive, threshold_quantile)) if positive.size else 0.0
        band_edges = np.linspace(0, len(freqs), num=n_bands + 1, dtype=int)
        points: List[Dict[str, float]] = []

        for time_idx in range(magnitudes.shape[1]):
            column = magnitudes[:, time_idx]
            for band_idx in range(n_bands):
                start = int(band_edges[band_idx])
                end = int(band_edges[band_idx + 1])
                if end - start <= 0:
                    continue
                segment = column[start:end]
                if segment.size == 0:
                    continue
                local_idx = int(np.argmax(segment))
                amplitude = float(segment[local_idx])
                if amplitude < threshold:
                    continue
                global_idx = start + local_idx
                points.append(
                    {
                        "time_idx": float(time_idx),
                        "time": float(times[time_idx]),
                        "freq_idx": float(global_idx),
                        "freq": float(freqs[global_idx]),
                        "amp": amplitude,
                    }
                )

        points.sort(key=lambda p: (p["time_idx"], -p["amp"]))
        return points

    def build_anchor_hashes(
        self,
        points: List[Dict[str, float]],
        fanout: int = CONSTELLATION_FANOUT,
        max_time_delta: int = CONSTELLATION_MAX_DT,
    ) -> list[str]:
        """
        Gera hashes (f1, f2, delta_t) a partir de anchor-target pairs.
        """
        if not points:
            return []

        hashes: list[str] = []
        sorted_points = sorted(points, key=lambda p: (p["time_idx"], -p["amp"]))
        for anchor_idx, anchor in enumerate(sorted_points):
            emitted = 0
            for target in sorted_points[anchor_idx + 1:]:
                delta_t = int(round(target["time_idx"] - anchor["time_idx"]))
                if delta_t <= 0:
                    continue
                if delta_t > max_time_delta:
                    break
                freq_a = int(round(anchor["freq"] * 1000))
                freq_b = int(round(target["freq"] * 1000))
                hashes.append(f"{freq_a}|{freq_b}|{delta_t}")
                emitted += 1
                if emitted >= fanout:
                    break
        return hashes

    def compute_spectral_entropy(self, magnitudes: np.ndarray) -> float:
        """
        Entropia espectral — mede complexidade/previsibilidade.
        0 = dominado por 1 frequência (ciclo perfeito, muito previsível)
        1 = distribuição uniforme (ruído puro, imprevisível)
        """
        spectrum = magnitudes.mean(axis=1)
        spectrum = spectrum / (spectrum.sum() + 1e-12)
        spectrum = spectrum[spectrum > 0]
        entropy  = float(-np.sum(spectrum * np.log2(spectrum + 1e-12)))
        max_entr = np.log2(len(spectrum))
        return float(entropy / max_entr) if max_entr > 0 else 0.5

    # ── Análise da curva diária (Duck Curve) ──────────────────────────────────

    def compute_daily_profile(self, series: pd.Series) -> pd.Series:
        """
        Perfil médio diário (0-23h) dos últimos LOOKBACK_DAYS.
        Normalizado pelo valor médio (0 = média, positivo = acima).
        """
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return pd.Series(0.0, index=range(24))
        profile = s.groupby(s.index.hour).mean()
        profile = profile.reindex(range(24), fill_value=profile.mean())
        return profile

    def detect_duck_curve(self, pld_profile: pd.Series,
                           load_profile: pd.Series
                           ) -> Dict[str, float]:
        """
        Detecta características da curva de pato (duck curve).
        Retorna métricas de profundidade do vale solar e rampa noturna.
        """
        metrics = {}

        # Análise do PLD
        if len(pld_profile) == 24:
            p = pld_profile.values
            metrics["peak_hour"]  = float(np.argmax(p))
            metrics["valley_hour"] = float(np.argmin(p))
            metrics["peak_value"]  = float(p.max())
            metrics["valley_value"] = float(p.min())
            metrics["spread"]      = float(p.max() - p.min())

            # Rampa matinal (6-10h) e noturna (16-21h)
            morning = p[6:10]
            evening = p[16:22]
            metrics["morning_ramp"] = float(np.diff(morning).mean()) if len(morning) > 1 else 0.0
            metrics["evening_ramp"] = float(np.diff(evening).mean()) if len(evening) > 1 else 0.0

        # Análise do net_load (duck curve clássica)
        if len(load_profile) == 24:
            nl = load_profile.values
            # Vale solar: mínimo nas horas 10-15h
            solar_hours = nl[10:15]
            off_peak    = np.concatenate([nl[:8], nl[20:]])
            duck_depth  = float(max(off_peak.mean() - solar_hours.min(), 0.0)) if len(solar_hours) > 0 else 0.0
            metrics["duck_depth"]  = duck_depth
            metrics["net_peak_h"]  = float(np.argmax(nl))
            metrics["net_valley_h"] = float(np.argmin(nl[8:18]) + 8)

        return metrics

    # ── Coerência espectral ───────────────────────────────────────────────────

    def compute_coherence(self, pld: pd.Series, net_load: pd.Series
                          ) -> Dict[str, float]:
        """
        Calcula coerência espectral entre PLD e net_load.

        Alta coerência na freq. diária → preço segue físico (modelo fundamental confiável)
        Baixa coerência → preço dominado por restrições/despacho (usar ML)

        Retorna dict com:
          - mean: coerência média 0-1
          - at_daily: coerência na frequência diária (1/24 cph)
          - regime: "coupled" | "decoupled"
        """
        if not _SCIPY_OK:
            return {"mean": 0.5, "at_daily": 0.5, "regime": "unknown"}

        # Alinhar e limpar
        aligned = pd.concat(
            [
                pd.to_numeric(pld, errors="coerce").rename("pld"),
                pd.to_numeric(net_load, errors="coerce").rename("net_load"),
            ],
            axis=1,
        ).dropna()
        if len(aligned) < 48:
            return {"mean": 0.5, "at_daily": 0.5, "regime": "unknown"}

        x = aligned["pld"].to_numpy(dtype=float)
        y = aligned["net_load"].to_numpy(dtype=float)

        try:
            freqs, Cxy = coherence(x, y, fs=1.0, nperseg=min(168, len(x)//4))
            coh_mean = float(Cxy.mean())

            # Coerência na frequência diária
            daily_idx = np.argmin(np.abs(freqs - FREQ_DIARIA))
            coh_daily = float(Cxy[daily_idx])

            regime = "coupled" if coh_daily > 0.6 else (
                     "partial"  if coh_daily > 0.3 else "decoupled")

            return {
                "mean":     round(coh_mean,  3),
                "at_daily": round(coh_daily, 3),
                "regime":   regime,
                "freqs":    freqs.tolist(),
                "Cxy":      Cxy.tolist(),
            }
        except Exception as e:
            logger.warning(f"Coherence calc: {e}")
            return {"mean": 0.5, "at_daily": 0.5, "regime": "unknown"}

    # ── Fingerprint e regime ──────────────────────────────────────────────────

    def _fingerprint_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty or "pld" not in df.columns:
            empty = self._empty_fingerprint()
            return {"fingerprint": empty, "hash_counter": Counter(), "points": [], "analysis": {"basis": "empty"}}

        frame = df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        if frame.empty:
            empty = self._empty_fingerprint()
            return {"fingerprint": empty, "hash_counter": Counter(), "points": [], "analysis": {"basis": "empty"}}

        analysis = self._analysis_signal(frame)
        signal_raw = analysis["raw"]
        signal_centered = analysis["centered"]
        signal_basis = str(analysis["basis"])
        signal_scale = float(analysis["scale"])
        if signal_raw.empty:
            empty = self._empty_fingerprint()
            return {"fingerprint": empty, "hash_counter": Counter(), "points": [], "analysis": analysis}

        net_load = frame.get("net_load", frame.get("load", pd.Series(index=frame.index, dtype=float)))
        net_load = pd.to_numeric(net_load, errors="coerce")
        latest_ts = pd.Timestamp(frame.index.max())
        recent_cutoff = latest_ts - pd.Timedelta(days=7)
        recent_net = net_load[net_load.index >= recent_cutoff].dropna()
        net_load_fresh = (
            not recent_net.empty
            and (latest_ts - pd.Timestamp(recent_net.index.max())) <= pd.Timedelta(hours=72)
            and len(recent_net) >= 24 * 3
        )
        freqs, times, mags = self.compute_stft(signal_centered)
        peaks = self.extract_peaks(freqs, mags)
        points = self.extract_constellation_points(freqs, times, mags)
        hashes = self.build_anchor_hashes(points)
        hash_counter = Counter(hashes)
        entropy = self.compute_spectral_entropy(mags)

        signal_profile = self.compute_daily_profile(signal_raw)
        load_profile = self.compute_daily_profile(net_load) if net_load_fresh else pd.Series(dtype=float)
        duck = self.detect_duck_curve(signal_profile, load_profile)
        coh = (
            self.compute_coherence(signal_raw, net_load)
            if net_load_fresh else
            {"mean": 0.5, "at_daily": 0.5, "regime": "unknown"}
        )

        spread = float(duck.get("spread", 0.0))
        load_std = float(recent_net.std()) if net_load_fresh else 0.0
        spread_normalized = float(np.clip(spread / max(signal_scale, 1e-6), 0.0, 5.0))
        duck_depth_z = float(np.clip(duck.get("duck_depth", 0.0) / max(load_std, 1e-6), 0.0, 8.0))
        ramp_ratio = float(np.clip(duck.get("evening_ramp", 0.0) / max(abs(spread), 1e-6), -3.0, 3.0))

        regime, confidence = self._detect_regime(
            duck=duck,
            coh=coh,
            entropy=entropy,
            pld=signal_raw,
            spread_normalized=spread_normalized,
            duck_depth_z=duck_depth_z,
            ramp_ratio=ramp_ratio,
        )

        fp = SpectralFingerprint(
            timestamp=str(frame.index.max().isoformat()),
            freq_dom_1=float(peaks[0][0]) if len(peaks) > 0 else FREQ_DIARIA,
            amp_dom_1=float(peaks[0][1]) if len(peaks) > 0 else 0.0,
            freq_dom_2=float(peaks[1][0]) if len(peaks) > 1 else FREQ_SEMIDIARIA,
            amp_dom_2=float(peaks[1][1]) if len(peaks) > 1 else 0.0,
            freq_dom_3=float(peaks[2][0]) if len(peaks) > 2 else FREQ_SEMANAL,
            amp_dom_3=float(peaks[2][1]) if len(peaks) > 2 else 0.0,
            peak_hour=duck.get("peak_hour", 18.0),
            valley_hour=duck.get("valley_hour", 4.0),
            peak_valley_spread=spread,
            coherence_mean=coh["mean"],
            coherence_at_daily=coh["at_daily"],
            spectral_entropy=entropy,
            regime=regime,
            regime_confidence=confidence,
            duck_depth=duck.get("duck_depth", 0.0),
            evening_ramp=duck.get("evening_ramp", 0.0),
            spread_normalized=spread_normalized,
            duck_depth_z=duck_depth_z,
            ramp_ratio=ramp_ratio,
            signal_basis=signal_basis,
            n_hashes=int(len(hashes)),
            hash_density=float(len(hashes) / max(len(times), 1)),
            matched_regime="unknown",
            matched_similarity=0.0,
            matched_hash_overlap=0.0,
            matched_window_start="",
            matched_window_end="",
            matched_library_kind="none",
        )
        return {"fingerprint": fp, "hash_counter": hash_counter, "points": points, "analysis": analysis}

    @staticmethod
    def _hash_overlap_score(left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        common = float(sum(min(count, right.get(key, 0)) for key, count in left.items()))
        base = max(float(sum(left.values())), float(sum(right.values())), 1.0)
        return common / base

    def build_fingerprint_library(
        self,
        history_df: Optional[pd.DataFrame] = None,
        *,
        window_hours: int = DEFAULT_MATCH_WINDOW_H,
        lookback_days: int = LIBRARY_LOOKBACK_DAYS,
        end_before: Optional[pd.Timestamp] = None,
    ) -> List[Dict[str, Any]]:
        history = history_df.copy() if history_df is not None else self.load_hourly_data(days=lookback_days)
        if history.empty or len(history) < max(window_hours * 2, 24 * 21):
            return []
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index, errors="coerce")
        history = history[~history.index.isna()].sort_index()
        if end_before is not None:
            history = history[history.index < pd.Timestamp(end_before)]
        history = history.tail(max(lookback_days * 24, window_hours * 2))
        if len(history) < window_hours:
            return []

        latest_ts = pd.Timestamp(history.index.max())
        cache_key = (latest_ts.isoformat(), int(len(history)), int(window_hours), str(end_before) if end_before is not None else "none")
        if cache_key in self._library_cache:
            return self._library_cache[cache_key]

        library: List[Dict[str, Any]] = []
        max_start = len(history) - window_hours
        for kind, stride in (("daily_like", LIBRARY_DAILY_STRIDE_H), ("weekly_like", LIBRARY_WEEKLY_STRIDE_H)):
            for start in range(0, max_start + 1, stride):
                window = history.iloc[start:start + window_hours].copy()
                if len(window) < window_hours * 0.9:
                    continue
                payload = self._fingerprint_payload(window)
                fp = payload["fingerprint"]
                if fp.n_hashes <= 0:
                    continue
                library.append(
                    {
                        "kind": kind,
                        "window_start": pd.Timestamp(window.index.min()),
                        "window_end": pd.Timestamp(window.index.max()),
                        "fingerprint": fp,
                        "hash_counter": payload["hash_counter"],
                    }
                )

        self._library_cache[cache_key] = library
        return library

    def match_against_library(
        self,
        fingerprint: SpectralFingerprint,
        hash_counter: Counter,
        library: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not library:
            return []

        matches: List[Dict[str, Any]] = []
        for candidate in library:
            candidate_fp = candidate["fingerprint"]
            hash_overlap = self._hash_overlap_score(hash_counter, candidate.get("hash_counter", Counter()))
            scalar_similarity = fingerprint.similarity(candidate_fp)
            score = 0.62 * hash_overlap + 0.38 * scalar_similarity
            matches.append(
                {
                    "kind": candidate["kind"],
                    "window_start": candidate["window_start"],
                    "window_end": candidate["window_end"],
                    "fingerprint": candidate_fp,
                    "hash_overlap": float(hash_overlap),
                    "scalar_similarity": float(scalar_similarity),
                    "score": float(score),
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    @staticmethod
    def _apply_match_regime_hint(
        fingerprint: SpectralFingerprint,
        best_match: Optional[Dict[str, Any]],
    ) -> SpectralFingerprint:
        if not best_match:
            return fingerprint

        matched_fp = best_match.get("fingerprint")
        match_score = float(best_match.get("score", 0.0))
        hash_overlap = float(best_match.get("hash_overlap", 0.0))
        if matched_fp is None:
            return fingerprint

        pld_shape_duck = (
            17.0 <= float(fingerprint.peak_hour) <= 21.0
            and 8.0 <= float(fingerprint.valley_hour) <= 13.0
            and float(fingerprint.spread_normalized) > 0.16
            and float(fingerprint.ramp_ratio) > 0.05
        )
        physical_confirmation_missing = (
            float(fingerprint.coherence_mean) == 0.5
            and float(fingerprint.coherence_at_daily) == 0.5
            and float(fingerprint.duck_depth_z) == 0.0
        )
        if (
            physical_confirmation_missing
            and pld_shape_duck
            and str(matched_fp.regime) == "duck_curve"
            and match_score >= 0.62
            and hash_overlap >= 0.45
        ):
            fingerprint.regime = "duck_curve"
            fingerprint.regime_confidence = float(
                max(fingerprint.regime_confidence, min(0.82, 0.42 + 0.38 * match_score))
            )
        return fingerprint

    def build_intraday_prior(
        self,
        history_df: pd.DataFrame,
        *,
        window_hours: int = DEFAULT_MATCH_WINDOW_H,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        history = history_df.copy()
        if history.empty:
            return {}
        if not isinstance(history.index, pd.DatetimeIndex):
            history.index = pd.to_datetime(history.index, errors="coerce")
        history = history[~history.index.isna()].sort_index()
        if len(history) < max(window_hours * 2, 24 * 21):
            return {}

        current = history.tail(window_hours).copy()
        library = self.build_fingerprint_library(
            history_df=history,
            window_hours=min(window_hours, len(current)),
            end_before=current.index.min(),
        )
        if not library:
            return {}

        payload = self._fingerprint_payload(current)
        current_fp = payload["fingerprint"]
        matches = self.match_against_library(current_fp, payload["hash_counter"], library, top_k=top_k)
        if not matches:
            return {}

        weights = np.asarray([match["score"] for match in matches], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(matches), dtype=float)
        weights = weights / weights.sum()

        spdi_profile: Optional[Dict[str, Any]] = None
        spike_profile: Optional[Dict[str, Any]] = None
        spdi_cum_weight = 0.0
        spike_cum_weight = 0.0
        for weight, match in zip(weights, matches):
            window = history.loc[match["window_start"]:match["window_end"]].copy()
            if window.empty:
                continue
            if "spdi" in window.columns and "spdi_ma24" in window.columns:
                ratio = (
                    pd.to_numeric(window["spdi"], errors="coerce")
                    / pd.to_numeric(window["spdi_ma24"], errors="coerce").replace(0, np.nan)
                ).clip(0.55, 1.75)
            else:
                analysis = self._analysis_signal(window)
                raw = pd.to_numeric(analysis["raw"], errors="coerce")
                ratio = (raw / raw.rolling(24, min_periods=6).median().replace(0, np.nan)).clip(0.55, 1.75)
            spdi_local = self._group_profile(ratio, default=1.0)
            spdi_mix_weight = float(weight / max(spdi_cum_weight + weight, 1e-9))
            spdi_profile = self._blend_profile_maps(
                spdi_profile or {"weekhour": {}, "hour": {}, "global": 1.0},
                spdi_local,
                spdi_mix_weight,
            )
            spdi_cum_weight += float(weight)

            if "positive_spike_flag" in window.columns:
                spike_series = pd.to_numeric(window["positive_spike_flag"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            else:
                analysis = self._analysis_signal(window)
                centered = pd.to_numeric(analysis["centered"], errors="coerce")
                threshold = float(centered.quantile(0.85)) if not centered.dropna().empty else 0.0
                spike_series = (centered >= threshold).astype(float)
            spike_local = self._group_profile(spike_series, default=0.0)
            spike_mix_weight = float(weight / max(spike_cum_weight + weight, 1e-9))
            spike_profile = self._blend_profile_maps(
                spike_profile or {"weekhour": {}, "hour": {}, "global": 0.0},
                spike_local,
                spike_mix_weight,
            )
            spike_cum_weight += float(weight)

        best = matches[0]
        current_fp.matched_regime = str(best["fingerprint"].regime)
        current_fp.matched_similarity = float(best["score"])
        current_fp.matched_hash_overlap = float(best["hash_overlap"])
        current_fp.matched_window_start = str(best["window_start"])
        current_fp.matched_window_end = str(best["window_end"])
        current_fp.matched_library_kind = str(best["kind"])
        return {
            "current_fingerprint": current_fp,
            "matches": matches,
            "match_confidence": float(best["score"]),
            "matched_regime": str(best["fingerprint"].regime),
            "matched_window_start": str(best["window_start"]),
            "matched_window_end": str(best["window_end"]),
            "spdi_intraday": spdi_profile or {"weekhour": {}, "hour": {}, "global": 1.0},
            "spike_prob": spike_profile or {"weekhour": {}, "hour": {}, "global": 0.0},
        }

    def generate_fingerprint(self, df: Optional[pd.DataFrame] = None,
                             enable_matching: bool = True
                             ) -> SpectralFingerprint:
        """
        Gera fingerprint espectral do estado atual do sistema.
        Análogo ao hash do Shazam — comparável com estados históricos.
        """
        if df is None:
            df = self.load_hourly_data()

        frame = df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        if len(frame) > DEFAULT_MATCH_WINDOW_H:
            frame = frame.tail(DEFAULT_MATCH_WINDOW_H).copy()

        payload = self._fingerprint_payload(frame)
        fp = payload["fingerprint"]
        if fp.signal_basis == "empty":
            return fp

        if enable_matching:
            history_source = (
                df
                if len(df) >= max(DEFAULT_MATCH_WINDOW_H * 2, 24 * 21)
                else self.load_hourly_data(days=LIBRARY_LOOKBACK_DAYS)
            )
            window_hours = max(min(len(frame), DEFAULT_MATCH_WINDOW_H), 24 * 7)
            library = self.build_fingerprint_library(
                history_df=history_source,
                window_hours=window_hours,
                end_before=frame.index.min(),
            )
            matches = self.match_against_library(fp, payload["hash_counter"], library, top_k=1)
            if matches:
                best = matches[0]
                fp.matched_regime = str(best["fingerprint"].regime)
                fp.matched_similarity = float(best["score"])
                fp.matched_hash_overlap = float(best["hash_overlap"])
                fp.matched_window_start = str(best["window_start"])
                fp.matched_window_end = str(best["window_end"])
                fp.matched_library_kind = str(best["kind"])
                fp = self._apply_match_regime_hint(fp, best)

        self._fingerprint_cache = fp
        return fp

    def _detect_regime(self, duck: Dict, coh: Dict,
                        entropy: float, pld: pd.Series,
                        spread_normalized: float,
                        duck_depth_z: float,
                        ramp_ratio: float,
                        ) -> Tuple[str, float]:
        """
        Detecta o regime operacional atual.

        Regimes:
          duck_curve  — curva de pato pronunciada (solar forte)
          flat        — preço relativamente uniforme (baixo solar)
          stressed    — preços altos com volatilidade (stress hídrico)
          distorted   — descoupling PLD × físico (restrições/despacho)
        """
        scores = {"duck_curve": 0.0, "flat": 0.0, "stressed": 0.0, "distorted": 0.0}

        pld_med = float(pld.median()) if not pld.empty else 0.5
        pld_shape_duck = (
            17.0 <= float(duck.get("peak_hour", -1.0)) <= 21.0
            and 8.0 <= float(duck.get("valley_hour", -1.0)) <= 13.0
            and spread_normalized > 0.16
            and ramp_ratio > 0.05
        )

        if spread_normalized > 0.18 and duck_depth_z > 0.55:
            scores["duck_curve"] += 0.45
        elif pld_shape_duck:
            scores["duck_curve"] += 0.30
        if ramp_ratio > 0.12:
            scores["duck_curve"] += 0.25
        if coh["at_daily"] > 0.7:
            scores["duck_curve"] += 0.2
        if entropy < 0.72:
            scores["duck_curve"] += 0.1

        # Flat: baixo spread, baixa volatilidade
        if spread_normalized < 0.08:
            scores["flat"] += 0.6
        if entropy < 0.45:
            scores["flat"] += 0.2
        if coh["at_daily"] < 0.3:
            scores["flat"] += 0.2

        # Stressed: prêmio econômico persistente + forma aberta
        if pld_med > 1.35:
            scores["stressed"] += 0.4
        if entropy > 0.7:
            scores["stressed"] += 0.3
        if spread_normalized > 0.28:
            scores["stressed"] += 0.3

        # Distorted: baixa coerência (PLD descolado do físico)
        if coh["mean"] < 0.3:
            scores["distorted"] += 0.5
        if coh["at_daily"] < 0.2:
            scores["distorted"] += 0.3
        if entropy > 0.8:
            scores["distorted"] += 0.2

        regime = max(scores, key=scores.__getitem__)
        confidence = min(1.0, float(scores[regime]))
        return regime, confidence

    def _empty_fingerprint(self) -> SpectralFingerprint:
        return SpectralFingerprint(
            timestamp=datetime.now().isoformat(),
            freq_dom_1=FREQ_DIARIA, amp_dom_1=0.0,
            freq_dom_2=FREQ_SEMIDIARIA, amp_dom_2=0.0,
            freq_dom_3=FREQ_SEMANAL, amp_dom_3=0.0,
            peak_hour=18.0, valley_hour=4.0, peak_valley_spread=0.0,
            coherence_mean=0.5, coherence_at_daily=0.5,
            spectral_entropy=0.5, regime="unknown", regime_confidence=0.0,
            duck_depth=0.0, evening_ramp=0.0,
            spread_normalized=0.0,
            duck_depth_z=0.0,
            ramp_ratio=0.0,
            signal_basis="empty",
            n_hashes=0,
            hash_density=0.0,
            matched_regime="unknown",
            matched_similarity=0.0,
            matched_hash_overlap=0.0,
            matched_window_start="",
            matched_window_end="",
            matched_library_kind="none",
        )

    # ── Features para ML ─────────────────────────────────────────────────────

    def extract_features(self, df: Optional[pd.DataFrame] = None,
                          fingerprint: Optional[SpectralFingerprint] = None
                          ) -> SpectralFeatures:
        """
        Extrai features espectrais para uso nos modelos ML.
        Retorna SpectralFeatures com todas as features computadas.
        """
        if fingerprint is None:
            fingerprint = self.generate_fingerprint(df)
        if df is None:
            df = self.load_hourly_data()
        analysis = self._analysis_signal(df)

        # Regime encoding
        regime_map = {"duck_curve": 1.0, "flat": 0.0,
                       "stressed": 2.0, "distorted": -1.0, "unknown": 0.0}
        regime_num = regime_map.get(fingerprint.regime, 0.0)

        # Calcular volatilidade espectral (variância do espectro ao longo do tempo)
        spectral_vol = 0.0
        phase_stab   = 0.5
        power_daily  = fingerprint.amp_dom_1 if abs(fingerprint.freq_dom_1 - FREQ_DIARIA) < 0.01 else 0.0
        power_weekly = fingerprint.amp_dom_3 if abs(fingerprint.freq_dom_3 - FREQ_SEMANAL) < 0.01 else 0.0
        morning_ramp = 0.0

        raw_signal = pd.to_numeric(analysis.get("raw", pd.Series(dtype=float)), errors="coerce").dropna()
        centered_signal = pd.to_numeric(analysis.get("centered", pd.Series(dtype=float)), errors="coerce").dropna()
        stft_signal = centered_signal if len(centered_signal) >= 24 else raw_signal

        if len(stft_signal) >= 24:
            freqs, _, mags = self.compute_stft(stft_signal)
            spectral_vol = float(mags.std(axis=1).mean())
            phase_stab   = 1.0 - float(np.clip(spectral_vol / (mags.mean() + 1e-9), 0, 1))

            # Power em frequências específicas
            if len(freqs) > 0:
                daily_idx  = np.argmin(np.abs(freqs - FREQ_DIARIA))
                weekly_idx = np.argmin(np.abs(freqs - FREQ_SEMANAL))
                power_daily  = float(mags[daily_idx].mean())
                power_weekly = float(mags[weekly_idx].mean()) if weekly_idx < len(mags) else 0.0
        if len(raw_signal) >= 24:
            daily_profile = self.compute_daily_profile(raw_signal)
            morning = daily_profile.iloc[6:10].to_numpy(dtype=float)
            if len(morning) > 1:
                morning_ramp = float(np.diff(morning).mean())

        power_dw_ratio = (power_daily / (power_weekly + 1e-9))

        # Dominant period
        dom_freq   = fingerprint.freq_dom_1
        dom_period = (1.0 / dom_freq) if dom_freq > 1e-9 else 24.0

        feat = SpectralFeatures(
            dominant_freq         = round(dom_freq, 6),
            dominant_period_h     = round(dom_period, 2),
            spectral_entropy      = round(fingerprint.spectral_entropy, 4),
            power_at_daily        = round(power_daily, 4),
            power_at_weekly       = round(power_weekly, 4),
            power_ratio_dw        = round(float(np.clip(power_dw_ratio, 0, 100)), 4),
            peak_hour             = round(fingerprint.peak_hour, 1),
            valley_hour           = round(fingerprint.valley_hour, 1),
            peak_valley_spread    = round(fingerprint.peak_valley_spread, 2),
            spread_normalized     = round(fingerprint.spread_normalized, 4),
            morning_ramp_rate     = round(morning_ramp, 3),
            evening_ramp_rate     = round(fingerprint.evening_ramp, 3),
            duck_depth            = round(fingerprint.duck_depth, 2),
            duck_depth_z          = round(fingerprint.duck_depth_z, 4),
            ramp_ratio            = round(fingerprint.ramp_ratio, 4),
            coherence_pld_load    = round(fingerprint.coherence_at_daily, 4),
            regime_encoded        = regime_num,
            days_since_regime_change = 0.0,  # calculado em build_weekly_features
            matched_similarity    = round(fingerprint.matched_similarity, 4),
            matched_hash_overlap  = round(fingerprint.matched_hash_overlap, 4),
            hash_density          = round(fingerprint.hash_density, 4),
            spectral_volatility   = round(spectral_vol, 4),
            phase_stability       = round(phase_stab, 4),
        )
        self._features_cache = feat
        return feat

    # ── Interface semanal para pld_forecast_engine ────────────────────────────

    def build_weekly_features(self, weeks: int = 52) -> pd.DataFrame:
        """
        Constrói dataset de features espectrais semanais.
        Para uso como features adicionais no pld_forecast_engine.
        """
        df_full = self.load_hourly_data(days=weeks * 7)
        if df_full.empty:
            return pd.DataFrame()

        # Agrupar por semana e calcular features para cada janela
        records = []
        df_full.index = pd.to_datetime(df_full.index)
        weeks_idx = pd.date_range(
            df_full.index[0].normalize(),
            df_full.index[-1].normalize(),
            freq="W-MON"
        )

        prev_regime = None
        days_since  = 0

        for week_start in weeks_idx:
            week_end = week_start + timedelta(days=7)
            window   = df_full.loc[week_start:week_end]
            if len(window) < 24:
                continue

            try:
                payload = self._fingerprint_payload(window)
                fp = payload["fingerprint"]
                window_hours = max(min(len(window), DEFAULT_MATCH_WINDOW_H), 24 * 7)
                history_prior = df_full.loc[df_full.index < window.index.min()].copy()
                if len(history_prior) >= max(window_hours * 2, 24 * 21):
                    library = self.build_fingerprint_library(
                        history_df=history_prior,
                        window_hours=window_hours,
                    )
                    matches = self.match_against_library(fp, payload["hash_counter"], library, top_k=1)
                    if matches:
                        best = matches[0]
                        fp.matched_regime = str(best["fingerprint"].regime)
                        fp.matched_similarity = float(best["score"])
                        fp.matched_hash_overlap = float(best["hash_overlap"])
                        fp.matched_window_start = str(best["window_start"])
                        fp.matched_window_end = str(best["window_end"])
                        fp.matched_library_kind = str(best["kind"])
                        fp = self._apply_match_regime_hint(fp, best)
                feat = self.extract_features(window, fp)

                if prev_regime != fp.regime:
                    days_since = 0
                    prev_regime = fp.regime
                else:
                    days_since += 7

                rec = feat.to_series().to_dict()
                rec["semana"]             = week_start
                rec["days_since_regime_change"] = float(days_since)
                records.append(rec)
            except Exception as e:
                logger.warning(f"Semana {week_start}: {e}")

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records).set_index("semana")
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    # ── Previsão da curva intra-diária ────────────────────────────────────────

    def forecast_daily_curve(self, days_ahead: int = 7
                              ) -> pd.DataFrame:
        """
        Prevê a forma da curva diária de PLD para os próximos dias.
        Baseado no fingerprint atual + sazonalidade histórica.

        Retorna DataFrame com colunas: hora, pld_p10, pld_p50, pld_p90, regime
        """
        df = self.load_hourly_data()
        if df.empty or "pld" not in df.columns:
            return pd.DataFrame()

        fp = self.generate_fingerprint(df)
        pld_profile = self.compute_daily_profile(df["pld"])
        pld_last    = float(df["pld"].iloc[-24:].mean())
        profile_mean = max(float(pld_profile.mean()), 1.0)
        hourly_shape = (pld_profile / profile_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        prior = self.build_intraday_prior(
            df,
            window_hours=max(min(len(df) // 2, DEFAULT_MATCH_WINDOW_H), 24 * 7),
        )
        prior_shape = prior.get("spdi_intraday", {}) if prior else {}
        spike_profile = prior.get("spike_prob", {}) if prior else {}
        match_weight = float(np.clip(prior.get("match_confidence", fp.matched_similarity), 0.0, 1.0)) * 0.45 if prior else float(np.clip(fp.matched_similarity, 0.0, 1.0)) * 0.35
        base_level = max(pld_last, 1.0)

        records = []
        for day in range(days_ahead):
            for hour in range(24):
                target_ts = pd.Timestamp(datetime.now().replace(minute=0, second=0, microsecond=0)) + pd.Timedelta(days=day, hours=hour)
                base_factor = float(hourly_shape.iloc[hour] if hour < len(hourly_shape) else 1.0)
                prior_factor = float(
                    prior_shape.get("hour", {}).get(hour, prior_shape.get("global", 1.0))
                ) if prior_shape else 1.0
                shape_factor = (1.0 - match_weight) * base_factor + match_weight * prior_factor
                base = float(base_level * np.clip(shape_factor, 0.35, 2.2))
                spike_bias = float(
                    spike_profile.get("hour", {}).get(hour, spike_profile.get("global", 0.0))
                ) if spike_profile else 0.0
                uncertainty = base_level * (0.05 * (1 + day * 0.1) + 0.12 * np.clip(spike_bias, 0.0, 1.0))
                pld_floor, pld_cap = self._pld_regulatory_bounds(target_ts)
                records.append({
                    "dia":     day + 1,
                    "hora":    hour,
                    "pld_p10": round(float(np.clip(base - uncertainty, pld_floor, pld_cap)), 2),
                    "pld_p50": round(float(np.clip(base, pld_floor, pld_cap)), 2),
                    "pld_p90": round(float(np.clip(base + uncertainty, pld_floor, pld_cap)), 2),
                    "regime":  fp.regime,
                    "matched_regime": prior.get("matched_regime", fp.matched_regime) if prior else fp.matched_regime,
                    "match_confidence": round(float(prior.get("match_confidence", fp.matched_similarity)) if prior else float(fp.matched_similarity), 4),
                })

        return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA — usada pelo pld_forecast_engine
# ══════════════════════════════════════════════════════════════════════════════

_engine_singleton: Optional[SpectralEngine] = None


def get_spectral_engine(data_dir: Path = DATA_DIR) -> SpectralEngine:
    """Retorna instância singleton do SpectralEngine."""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SpectralEngine(data_dir)
    return _engine_singleton


def get_spectral_features(data_dir: Path = DATA_DIR) -> Optional[SpectralFeatures]:
    """
    Interface rápida para obter features espectrais atuais.
    Usado pelo pld_forecast_engine para enriquecer features_now.
    """
    try:
        engine = get_spectral_engine(data_dir)
        df = engine.load_hourly_data(days=30)
        if df.empty:
            return None
        return engine.extract_features(df)
    except Exception as e:
        logger.warning(f"get_spectral_features: {e}")
        return None


def get_current_fingerprint(data_dir: Path = DATA_DIR) -> Optional[SpectralFingerprint]:
    """Retorna fingerprint espectral atual do sistema."""
    try:
        engine = get_spectral_engine(data_dir)
        return engine.generate_fingerprint()
    except Exception as e:
        logger.warning(f"get_current_fingerprint: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRAÇÃO COM pld_forecast_engine
# ══════════════════════════════════════════════════════════════════════════════

SPECTRAL_FEATURE_COLS = [
    "dominant_freq",
    "dominant_period_h",
    "spectral_entropy",
    "power_at_daily",
    "power_at_weekly",
    "power_ratio_dw",
    "peak_hour",
    "valley_hour",
    "peak_valley_spread",
    "spread_normalized",
    "morning_ramp_rate",
    "evening_ramp_rate",
    "duck_depth",
    "duck_depth_z",
    "ramp_ratio",
    "coherence_pld_load",
    "regime_encoded",
    "matched_similarity",
    "matched_hash_overlap",
    "hash_density",
    "spectral_volatility",
    "phase_stability",
]


def enrich_features_now(features_now: "pd.Series",
                         data_dir: Path = DATA_DIR) -> "pd.Series":
    """
    Enriquece features_now do pld_forecast_engine com features espectrais.
    Chamado em run_forecast() antes de entrar no modelo.
    """
    feat = get_spectral_features(data_dir)
    if feat is None:
        return features_now
    feat_series = feat.to_series()
    for col in SPECTRAL_FEATURE_COLS:
        if col in feat_series.index:
            features_now[f"spec_{col}"] = feat_series[col]
    return features_now


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="MAÁTria · Spectral Engine")
    ap.add_argument("--analyze",     action="store_true",
                    help="Análise espectral completa dos dados disponíveis")
    ap.add_argument("--features",    action="store_true",
                    help="Extrai features espectrais para ML")
    ap.add_argument("--fingerprint", action="store_true",
                    help="Gera fingerprint atual do sistema")
    ap.add_argument("--regime",      action="store_true",
                    help="Detecta e exibe regime operacional atual")
    ap.add_argument("--weekly",      action="store_true",
                    help="Constrói dataset semanal de features espectrais")
    ap.add_argument("--data",        default="data")
    args = ap.parse_args()

    engine = SpectralEngine(Path(args.data))

    if args.fingerprint or args.analyze or args.regime or (not any(vars(args).values())):
        print("\n=== Carregando dados ===")
        df = engine.load_hourly_data()
        print(f"  {len(df)} horas | colunas: {list(df.columns)}")

        print("\n=== Fingerprint do sistema ===")
        fp = engine.generate_fingerprint(df)
        print(json.dumps(fp.to_dict(), indent=2, default=str))

    if args.features or args.analyze:
        print("\n=== Features espectrais ===")
        feat = engine.extract_features()
        print(json.dumps(feat.to_series().to_dict(), indent=2, default=str))

    if args.weekly:
        print("\n=== Dataset semanal ===")
        weekly = engine.build_weekly_features(weeks=52)
        if not weekly.empty:
            print(f"  {len(weekly)} semanas | {weekly.shape[1]} features")
            print(weekly.tail(4).to_string())
        else:
            print("  Dados insuficientes para dataset semanal")

    if args.regime or args.analyze:
        if "fp" not in dir():
            df = engine.load_hourly_data()
            fp = engine.generate_fingerprint(df)
        if fp.signal_basis in {"spdi", "spdi_from_cost"}:
            spread_label = f"{fp.peak_valley_spread:.2f}x"
        elif fp.signal_basis == "pld_reg_norm":
            spread_label = f"{fp.peak_valley_spread * 100:.1f}% da faixa regulatória"
        else:
            spread_label = f"R${fp.peak_valley_spread:.0f}/MWh"
        print(f"\n=== Regime atual ===")
        print(f"  Base sinal: {fp.signal_basis}")
        print(f"  Regime:     {fp.regime} (confiança: {fp.regime_confidence:.0%})")
        print(f"  Coerência:  {fp.coherence_at_daily:.2f} (PLD × net_load na freq. diária)")
        print(f"  Entropia:   {fp.spectral_entropy:.2f} ({'previsível' if fp.spectral_entropy < 0.5 else 'complexo'})")
        print(f"  Pico:       {fp.peak_hour:.0f}h | Vale: {fp.valley_hour:.0f}h")
        print(f"  Spread:     {spread_label}")
        if fp.regime == "duck_curve":
            print("  → Curva de pato ativa: pico solar reduzindo preço nas 10-15h")
            print("    Estratégia BESS: carregar 10-14h, descarregar 18-21h")
        elif fp.regime == "stressed":
            print("  → Sistema estressado: PLD elevado e volátil")
            print("    Estratégia BESS: maximizar descarga nos picos, conservar energia")
        elif fp.regime == "distorted":
            print("  → Distorção: PLD descolado do custo físico (restrições/despacho)")
            print("    Usar modelo ML sobre modelo físico para previsão")
        elif fp.regime == "flat":
            print("  → Regime plano: baixa diferença ponta/fora-ponta")
            print("    Arbitragem BESS menos atrativa; focar em peak-shaving")
