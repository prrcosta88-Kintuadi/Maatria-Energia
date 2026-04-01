# -*- coding: utf-8 -*-
"""
report_engine.py — MAÁTria Energia · Periodic Report Engine
═══════════════════════════════════════════════════════════════
Gera automaticamente papers de inteligência de mercado com:

  • Análise econômica completa do período (diária/semanal/mensal)
  • Narrativa gerada via Claude API (claude-sonnet-4-20250514)
  • PDF corporativo com paleta MAÁTria
  • Entrega por email via Resend API
  • Armazenamento no banco AUTH (maat_subscriptions / log)
  • Integração com update_neon.py (scheduler existente)

Uso:
    python report_engine.py --period monthly   # relatório mensal
    python report_engine.py --period weekly    # relatório semanal
    python report_engine.py --period daily     # flash diário
    python report_engine.py --preview          # gera PDF sem enviar
    python report_engine.py --send-to email@x  # enviar para email específico
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── ReportLab ─────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Image as RLImage, KeepTogether, PageBreak,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import Flowable

# ── Cores MAÁTria ─────────────────────────────────────────────────────────────
GOLD        = colors.HexColor("#c8a44d")
GOLD_LIGHT  = colors.HexColor("#e3bf4c")
BG_DARK     = colors.HexColor("#0b0f14")
BG_PANEL    = colors.HexColor("#111827")
BG_PANEL2   = colors.HexColor("#0d1b2a")
BORDER      = colors.HexColor("#1e3a5f")
TEXT_WHITE  = colors.HexColor("#f3f4f6")
TEXT_GRAY   = colors.HexColor("#9ca3af")
TEXT_LIGHT  = colors.HexColor("#e5e7eb")
RED         = colors.HexColor("#f87171")
GREEN       = colors.HexColor("#34d399")
BLUE        = colors.HexColor("#60a5fa")
PURPLE      = colors.HexColor("#a78bfa")

W, H   = A4
ML, MR = 18*mm, 18*mm
MT, MB = 26*mm, 20*mm
CW     = W - ML - MR

LOGO_PATH    = Path("data/MAATria_logo.png")
EMBLEMA_PATH = Path("data/emblema_maatria.png")

# ══════════════════════════════════════════════════════════════════════════════
# PERÍODO E CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

PERIOD_CONFIG = {
    "daily":   {"label": "Flash Diário",    "days": 1,   "freq": "D"},
    "weekly":  {"label": "Relatório Semanal","days": 7,   "freq": "W"},
    "monthly": {"label": "Relatório Mensal", "days": 30,  "freq": "ME"},
}

# Lista de destinatários por plano (mínimo Professional)
REPORT_PLANS = ["professional", "institutional"]


# ══════════════════════════════════════════════════════════════════════════════
# COLETA DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

def _build_df_standalone() -> pd.DataFrame:
    """
    Constrói DataFrame horário completo SEM dependência do Streamlit.
    Replica a lógica essencial do _build_hourly_df_cached do app_premium,
    acessando o Neon diretamente via psycopg2 + pandas.
    Também chama ccee_charges.enrich_df para calcular os indicadores econômicos.
    """
    import psycopg2

    url = os.getenv("DATABASE_URL", "")
    if not url:
        print("⚠️  DATABASE_URL não configurada.")
        return pd.DataFrame()

    try:
        conn = psycopg2.connect(url)
    except Exception as e:
        print(f"⚠️  Conexão Neon falhou: {e}")
        return pd.DataFrame()

    def _q(sql: str, params=None) -> pd.DataFrame:
        try:
            return pd.read_sql(sql, conn, params=params)
        except Exception as ex:
            print(f"  ⚠️  Query falhou: {ex}")
            return pd.DataFrame()

    print("  Carregando dados do Neon...")
    frames = {}

    # PLD histórico
    pld_df = _q("""
        SELECT mes_referencia::text AS mes, dia, hora,
               pld_se, pld_ne, pld_s, pld_n
        FROM pld_historical
        ORDER BY mes_referencia, dia, hora
    """)
    if not pld_df.empty:
        pld_df["ts"] = pd.to_datetime(
            pld_df["mes"].str[:4] + "-" + pld_df["mes"].str[4:6] +
            "-" + pld_df["dia"].astype(str).str.zfill(2) +
            " " + pld_df["hora"].astype(str).str.zfill(2) + ":00",
            errors="coerce")
        pld_df = pld_df.dropna(subset=["ts"]).set_index("ts")
        for col in ["pld_se","pld_ne","pld_s","pld_n"]:
            if col in pld_df.columns:
                pld_df[col] = pd.to_numeric(pld_df[col], errors="coerce")
        pld_df["pld"] = pld_df[["pld_se","pld_ne","pld_s","pld_n"]].mean(axis=1)
        frames["pld"] = pld_df[["pld","pld_se","pld_ne","pld_s","pld_n"]]
        print(f"    PLD: {len(pld_df):,} horas")

    # Geração por tipo
    gen_df = _q("""
        SELECT din_instante, tipo_geracao, val_geracao
        FROM geracao_tipo_hora
        WHERE din_instante IS NOT NULL
        ORDER BY din_instante
    """)
    if not gen_df.empty:
        gen_df["din_instante"] = pd.to_datetime(gen_df["din_instante"], errors="coerce")
        gen_df = gen_df.dropna(subset=["din_instante"])
        for tipo in ["solar","wind","hydro","thermal","nuclear"]:
            s = (gen_df[gen_df["tipo_geracao"]==tipo]
                 .set_index("din_instante")["val_geracao"]
                 .apply(pd.to_numeric, errors="coerce")
                 .rename(tipo))
            frames[tipo] = s.to_frame()
        print(f"    Geração: {len(gen_df):,} registros")

    # Carga SIN
    carga_df = _q("""
        SELECT instante, SUM(valor) AS load
        FROM curva_carga
        WHERE instante IS NOT NULL
        GROUP BY instante ORDER BY instante
    """)
    if not carga_df.empty:
        carga_df["instante"] = pd.to_datetime(carga_df["instante"], errors="coerce")
        carga_df = carga_df.dropna(subset=["instante"]).set_index("instante")
        carga_df["load"] = pd.to_numeric(carga_df["load"], errors="coerce")
        frames["load"] = carga_df[["load"]]
        print(f"    Carga: {len(carga_df):,} horas")

    # CMO
    cmo_df = _q("""
        SELECT din_instante, id_subsistema, val_cmo
        FROM cmo
        WHERE din_instante IS NOT NULL AND id_subsistema ILIKE '%SUL%'
           OR id_subsistema ILIKE '%SECO%' OR id_subsistema ILIKE '%SE%'
        ORDER BY din_instante LIMIT 500000
    """)
    if not cmo_df.empty:
        cmo_df["din_instante"] = pd.to_datetime(cmo_df["din_instante"], errors="coerce")
        # usar SE como dominante
        se_mask = cmo_df["id_subsistema"].str.upper().isin(["SE","SUDESTE","SECO"])
        cmo_se = (cmo_df[se_mask]
                  .set_index("din_instante")["val_cmo"]
                  .apply(pd.to_numeric, errors="coerce")
                  .rename("cmo_dominante"))
        frames["cmo"] = cmo_se.to_frame()
        print(f"    CMO: {len(cmo_se):,} horas")

    # EAR
    ear_df = _q("""
        SELECT instante, SUM(ear) AS ear, SUM(earmaxp) AS earmaxp
        FROM ear_diario_subsistema
        WHERE instante IS NOT NULL
        GROUP BY instante ORDER BY instante
    """)
    if not ear_df.empty:
        ear_df["instante"] = pd.to_datetime(ear_df["instante"], errors="coerce")
        ear_df = ear_df.dropna(subset=["instante"]).set_index("instante")
        ear_df["ear_pct"] = (ear_df["ear"] / ear_df["earmaxp"].replace(0, np.nan) * 100).clip(0, 100)
        # Expandir diário → horário
        ear_h = ear_df[["ear_pct"]].resample("h").ffill()
        frames["ear"] = ear_h
        print(f"    EAR: {len(ear_df):,} dias → {len(ear_h):,} horas")

    # CVU térmico
    cvu_df = _q("""
        SELECT instante, cvu_medio
        FROM cvu_usina_termica
        WHERE instante IS NOT NULL
        ORDER BY instante
    """)
    if not cvu_df.empty:
        cvu_df["instante"] = pd.to_datetime(cvu_df["instante"], errors="coerce")
        cvu_df = cvu_df.dropna(subset=["instante"]).set_index("instante")
        cvu_df["cvu_semana"] = pd.to_numeric(cvu_df["cvu_medio"], errors="coerce")
        cvu_h = cvu_df[["cvu_semana"]].resample("h").ffill()
        frames["cvu"] = cvu_h
        print(f"    CVU: {len(cvu_h):,} horas")

    # Despacho GFOM (constrained_off)
    gfom_df = _q("""
        SELECT din_instante,
               SUM(val_verifconstrainedoff) AS constrained_off,
               SUM(val_verifgfom)           AS gfom
        FROM despacho_gfom
        WHERE din_instante IS NOT NULL
        GROUP BY din_instante ORDER BY din_instante
    """)
    if not gfom_df.empty:
        gfom_df["din_instante"] = pd.to_datetime(gfom_df["din_instante"], errors="coerce")
        gfom_df = gfom_df.dropna(subset=["din_instante"]).set_index("din_instante")
        for col in ["constrained_off","gfom"]:
            gfom_df[col] = pd.to_numeric(gfom_df[col], errors="coerce").fillna(0)
        frames["gfom"] = gfom_df[["constrained_off","gfom"]]
        print(f"    GFOM: {len(gfom_df):,} horas")

    conn.close()

    if not frames:
        return pd.DataFrame()

    # Consolidar
    dfs = []
    for key, fdf in frames.items():
        fdf.index = pd.to_datetime(fdf.index, errors="coerce")
        fdf = fdf[~fdf.index.isna()].sort_index()
        # remover tz
        if fdf.index.tz is not None:
            fdf.index = fdf.index.tz_localize(None)
        dfs.append(fdf)

    df = dfs[0]
    for fdf in dfs[1:]:
        df = df.join(fdf, how="outer")

    df = df.sort_index()
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "instante"

    # Colunas derivadas essenciais
    z = pd.Series(0.0, index=df.index)
    def _c(n): return pd.to_numeric(df[n], errors="coerce") if n in df.columns else z.copy()

    load_s    = _c("load")
    solar_s   = _c("solar")
    wind_s    = _c("wind")
    hydro_s   = _c("hydro")
    thermal_s = _c("thermal")
    pld_s     = _c("pld")
    cmo_s     = _c("cmo_dominante")
    cvu_s     = _c("cvu_semana")

    df["load"]         = load_s
    df["net_load"]     = load_s - solar_s.fillna(0) - wind_s.fillna(0)
    df["geracao_total"]= (solar_s.fillna(0) + wind_s.fillna(0) +
                          hydro_s.fillna(0) + thermal_s.fillna(0))
    df["thermal_real_cost"] = thermal_s.fillna(0) * cvu_s.fillna(0)

    # sin_cost = carga × PLD (proxy SE)
    df["sin_cost"]     = (load_s.fillna(0) * pld_s.fillna(0)).where(load_s > 0, np.nan)

    # t_total simplificado (sem sub-mercados individuais)
    _thermal_cost = thermal_s.fillna(0) * cvu_s.fillna(0)
    _hydro_cost   = hydro_s.fillna(0) * cmo_s.fillna(0)
    _water_val    = (_c("hydro") - _c("hydro")).clip(lower=0) * cmo_s  # Hydro preserved proxy
    _t_prud       = np.where(cmo_s > pld_s, _water_val.fillna(0), 0.0)
    _t_sist       = (pld_s - cmo_s).abs() * df["net_load"].fillna(0)
    df["t_total"]  = (_thermal_cost + _water_val.fillna(0) +
                      pd.Series(_t_prud, index=df.index).fillna(0) +
                      _t_sist.fillna(0))

    # Encargos via ccee_charges
    try:
        from ccee_charges import enrich_df as _enrich_df
        print("  Aplicando ccee_charges.enrich_df...")
        df = _enrich_df(df)
        print(f"  ✅ Indicadores econômicos calculados")
    except Exception as ex:
        print(f"  ⚠️  ccee_charges.enrich_df falhou: {ex}")

    print(f"  DataFrame final: {len(df):,} horas | {len(df.columns)} colunas")
    return df


def collect_report_data(period: str = "monthly") -> Dict[str, Any]:
    """
    Coleta métricas do DataFrame horário do período solicitado.
    Usa _build_df_standalone — sem dependência do Streamlit.
    Retorna dict com todas as métricas necessárias para o paper.
    """
    days = PERIOD_CONFIG[period]["days"]

    df = _build_df_standalone()

    if df.empty:
        return _empty_metrics(period)

    # Filtrar período
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    dff = df[df.index >= cutoff].copy()
    if dff.empty:
        dff = df.iloc[-days*24:].copy() if len(df) >= days*24 else df.copy()

    def _stat(col: str) -> Dict[str, float]:
        if col not in dff.columns:
            return {"median": float("nan"), "mean": float("nan"),
                    "max": float("nan"), "min": float("nan"),
                    "last": float("nan"), "pct_change": float("nan")}
        s = pd.to_numeric(dff[col], errors="coerce").dropna()
        if s.empty:
            return {"median": float("nan"), "mean": float("nan"),
                    "max": float("nan"), "min": float("nan"),
                    "last": float("nan"), "pct_change": float("nan")}
        # comparação com período anterior
        cutoff_prev = cutoff - pd.Timedelta(days=days)
        prev = pd.to_numeric(
            df[(df.index >= cutoff_prev) & (df.index < cutoff)].get(col,
            pd.Series(dtype=float)), errors="coerce").dropna()
        prev_med = float(prev.median()) if not prev.empty else float("nan")
        curr_med = float(s.median())
        pct = ((curr_med - prev_med) / abs(prev_med) * 100
               if prev_med and not np.isnan(prev_med) else float("nan"))
        return {
            "median": curr_med, "mean": float(s.mean()),
            "max":    float(s.max()), "min": float(s.min()),
            "last":   float(s.iloc[-1]),
            "pct_change": round(pct, 1),
        }

    # Coleta de todos os indicadores
    metrics = {
        "period":        period,
        "period_label":  PERIOD_CONFIG[period]["label"],
        "date_start":    str(dff.index.min().date()) if not dff.empty else "",
        "date_end":      str(dff.index.max().date()) if not dff.empty else "",
        "generated_at":  datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_hours":       len(dff),

        # Métricas econômicas principais
        "market_cost":         _stat("market_cost_Rh"),
        "physical_cost":       _stat("physical_generation_cost_Rh"),
        "encargos":            _stat("Encargos_total_R$/h"),
        "custo_real":          _stat("Custo_real_R$/h"),
        "custo_real_mwh":      _stat("Custo_real_R$/MWh"),
        "hidden_cost":         _stat("Hidden_system_cost_R$/MWh"),
        "imr_corrigido":       _stat("infra_marginal_rent_corrigido"),
        "imr_fisico":          _stat("infra_marginal_physical"),
        "structural_gap":      _stat("Structural_gap_R$/MWh"),
        "spdi":                _stat("spdi"),
        "eii":                 _stat("encargo_intensity_index"),
        "structural_drift":    _stat("structural_drift"),

        # Físicos
        "pld":                 _stat("pld"),
        "cmo":                 _stat("cmo_dominante"),
        "ear_pct":             _stat("ear_pct"),
        "thermal_ratio":       _stat("Thermal_inflex_ratio"),
        "curtail":             _stat("curtail_total"),
        "load":                _stat("load"),

        # Padrão horário do EII (valor diagnóstico operacional)
        # Identifica em quais horas o operador intervém mais
        "eii_hourly_pattern":  _eii_hourly_pattern(dff),
        "eii_phases":          _eii_phase_narrative(dff),
    }

    # Classificar regime atual
    spdi_now = metrics["spdi"]["last"]
    if np.isnan(spdi_now):
        spdi_now = metrics["spdi"]["median"]
    metrics["regime"] = (
        "Prêmio Estrutural"   if spdi_now > 1.3 else
        "Transição"           if spdi_now > 1.0 else
        "Déficit Sistêmico"
    )
    metrics["regime_color"] = (
        "#34d399" if spdi_now > 1.3 else
        "#c8a44d" if spdi_now > 1.0 else
        "#f87171"
    )

    return metrics


def _empty_metrics(period: str) -> Dict[str, Any]:
    return {
        "period": period,
        "period_label": PERIOD_CONFIG[period]["label"],
        "date_start": "", "date_end": "",
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_hours": 0, "regime": "N/D", "regime_color": "#9ca3af",
    }


# ══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DE NARRATIVA VIA CLAUDE API
# ══════════════════════════════════════════════════════════════════════════════

def _eii_hourly_pattern(dff: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa o padrão horário do EII — valor diagnóstico operacional.
    Identifica quais horas concentram intervenções do operador.
    """
    if "encargo_intensity_index" not in dff.columns or dff.empty:
        return {}
    eii = pd.to_numeric(dff["encargo_intensity_index"], errors="coerce").dropna()
    if eii.empty:
        return {}
    hourly_med = eii.groupby(eii.index.hour).median()
    peak_hour  = int(hourly_med.idxmax())
    off_hour   = int(hourly_med.idxmin())
    solar_hours= hourly_med.loc[7:16].mean()
    night_hours= hourly_med.loc[list(range(0,6))+list(range(19,24))].mean()
    ratio      = float(solar_hours / night_hours) if night_hours > 0 else 1.0
    pattern = (
        "curtailment solar"   if ratio > 1.5 else
        "ponta noturna"       if ratio < 0.7 else
        "difuso"
    )
    return {
        "peak_hour":     peak_hour,
        "off_hour":      off_hour,
        "solar_vs_night":round(ratio, 2),
        "pattern":       pattern,
        "hourly_median": {h: round(float(v)*100, 3) for h, v in hourly_med.items()},
    }


def _eii_phase_narrative(dff: pd.DataFrame) -> str:
    """
    Gera frase descritiva do padrão de intervenção operacional do período.
    """
    pat = _eii_hourly_pattern(dff)
    if not pat:
        return ""
    p = pat.get("pattern", "difuso")
    ph = pat.get("peak_hour", 0)
    ratio = pat.get("solar_vs_night", 1.0)
    if p == "curtailment solar":
        return (f"O padrão horário do EII revela concentração de intervenções entre "
                f"7h e 16h (pico às {ph:02d}h), com intensidade {ratio:.1f}× maior que "
                f"o período noturno — assinatura de curtailment solar: o operador "
                f"pagou para reduzir geração renovável excedente nas horas de insolação.")
    elif p == "ponta noturna":
        return (f"O EII apresenta pico às {ph:02d}h, concentrado no período noturno "
                f"— indicando acionamento de reservas e serviços ancilares na ponta, "
                f"quando a geração solar já cessou e a carga permanece elevada.")
    else:
        return (f"O EII distribui-se de forma difusa ao longo do dia (pico às {ph:02d}h), "
                f"sugerindo intervenções operacionais não associadas a um único regime "
                f"de geração — possivelmente restrições de transmissão ou unit commitment.")


def generate_narrative(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Chama a API da Claude para gerar narrativa analítica do período.
    Retorna dict com seções: executive, regime, signals, outlook.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    # Fallback sem API key
    if not api_key:
        return _static_narrative(metrics)

    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "N/D"
        if isinstance(v, float) and abs(v) >= 1_000_000:
            return f"R${v/1_000_000:.1f}M"
        if isinstance(v, float) and abs(v) >= 1_000:
            return f"R${v/1_000:.0f}K"
        return str(round(v, 2)) if isinstance(v, float) else str(v)

    prompt = f"""Você é o analista-chefe da MAÁTria Energia, plataforma de inteligência de mercado elétrico brasileiro.
Gere uma análise econômica profissional e objetiva do período indicado, em português brasileiro.

DADOS DO PERÍODO ({metrics.get('date_start')} a {metrics.get('date_end')}):
- Horas analisadas: {metrics.get('n_hours')}
- SPDI (mediana): {_fmt(metrics.get('spdi', {}).get('median', float('nan')))}× | variação: {_fmt(metrics.get('spdi', {}).get('pct_change', float('nan')))}%
- Structural Gap (mediana): {_fmt(metrics.get('structural_gap', {}).get('median', float('nan')))}/MWh
- IMR Corrigido (mediana): {_fmt(metrics.get('imr_corrigido', {}).get('median', float('nan')))}/h
- Custo Real Unitário (mediana): {_fmt(metrics.get('custo_real_mwh', {}).get('median', float('nan')))}/MWh
- PLD (mediana): {_fmt(metrics.get('pld', {}).get('median', float('nan')))}/MWh | máx: {_fmt(metrics.get('pld', {}).get('max', float('nan')))}/MWh
- EAR% (mediana): {_fmt(metrics.get('ear_pct', {}).get('median', float('nan')))}%
- Regime atual identificado: {metrics.get('regime')}
- EII (mediana): {_fmt(metrics.get('eii', {}).get('median', float('nan')))}%
- Structural Drift: {_fmt(metrics.get('structural_drift', {}).get('last', float('nan')))}×

Escreva EXATAMENTE 4 seções, cada uma iniciada com o marcador indicado:
[EXECUTIVE] 2-3 frases objetivas resumindo o período. Inclua os 3 números mais relevantes.
[REGIME] 2-3 frases sobre o regime de mercado atual e sua evolução no período.
[SIGNALS] 2-3 frases sobre os principais sinais físicos e econômicos observados.
[OUTLOOK] 2-3 frases sobre perspectivas para o próximo período, sem especulação excessiva.

Escreva de forma direta, técnica e corporativa. Sem bullet points. Sem subtítulos extras."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 900,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json()["content"][0]["text"]
            return _parse_narrative(text)
    except Exception as e:
        print(f"⚠️  API Claude: {e}")

    return _static_narrative(metrics)


def _parse_narrative(text: str) -> Dict[str, str]:
    sections = {"executive": "", "regime": "", "signals": "", "outlook": ""}
    markers  = {"[EXECUTIVE]": "executive", "[REGIME]": "regime",
                "[SIGNALS]": "signals", "[OUTLOOK]": "outlook"}
    current = None
    for line in text.split("\n"):
        stripped = line.strip()
        matched = False
        for marker, key in markers.items():
            if stripped.startswith(marker):
                current = key
                rest = stripped[len(marker):].strip()
                if rest:
                    sections[current] = rest
                matched = True
                break
        if not matched and current and stripped:
            sep = " " if sections[current] else ""
            sections[current] += sep + stripped
    return sections


def _static_narrative(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Narrativa estática para quando a API não está disponível."""
    regime = metrics.get("regime", "N/D")
    spdi   = metrics.get("spdi",   {}).get("median", float("nan"))
    gap    = metrics.get("structural_gap", {}).get("median", float("nan"))
    pld    = metrics.get("pld",    {}).get("median", float("nan"))

    def _f(v, prefix="R$", suffix=""):
        if isinstance(v, float) and np.isnan(v):
            return "N/D"
        if abs(v) >= 1_000_000:
            return f"{prefix}{v/1_000_000:.1f}M{suffix}"
        return f"{prefix}{v:,.0f}{suffix}"

    return {
        "executive": (
            f"O mercado elétrico brasileiro operou no regime de {regime} durante o período analisado. "
            f"O SPDI mediano de {spdi:.2f}× e o Structural Gap de {_f(gap)}/MWh "
            f"caracterizam a relação entre preço de mercado e custo físico de operação do sistema."
        ),
        "regime": (
            f"Com PLD mediano de {_f(pld)}/MWh, o sistema manteve coerência entre "
            f"as variáveis físicas e econômicas. O Structural Drift indica o posicionamento "
            f"do mercado em relação ao nível estrutural de referência de 2021."
        ),
        "signals": (
            "Os indicadores de custo físico e encargos ESS apresentaram comportamento "
            "consistente com o histórico recente. O Encargo Intensity Index reflete "
            "o nível de intervenção sistêmica no período."
        ),
        "outlook": (
            "As condições hidrológicas e o despacho térmico serão os fatores determinantes "
            "para a evolução dos indicadores no próximo período. "
            "Monitorar o Structural Gap como termômetro da relação preço-custo."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FLOWABLES PDF
# ══════════════════════════════════════════════════════════════════════════════

class GoldLine(Flowable):
    def __init__(self, w=CW, thick=0.8, alpha=0.7):
        super().__init__()
        self.width = w; self.thick = thick; self.alpha = alpha
        self.height = thick + 2

    def draw(self):
        self.canv.setStrokeColor(GOLD, alpha=self.alpha)
        self.canv.setLineWidth(self.thick)
        self.canv.line(0, self.thick/2, self.width, self.thick/2)


class KPIRow(Flowable):
    def __init__(self, kpis, w=CW, h=42):
        super().__init__()
        self.kpis = kpis  # (label, value, delta, color)
        self.width = w; self.height = h

    def draw(self):
        c = self.canv
        n = len(self.kpis)
        cw = self.width / n
        for i, (label, value, delta, col) in enumerate(self.kpis):
            x = i * cw
            c.setFillColor(BG_PANEL2)
            c.setStrokeColor(BORDER, alpha=0.6)
            c.setLineWidth(0.4)
            c.rect(x+1, 0, cw-2, self.height, fill=1, stroke=1)
            c.setStrokeColor(col, alpha=0.9)
            c.setLineWidth(1.8)
            c.line(x+1, self.height, x+cw-1, self.height)
            c.setFont("Helvetica", 6.2)
            c.setFillColor(TEXT_GRAY)
            c.drawCentredString(x+cw/2, self.height-10, label)
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(col)
            c.drawCentredString(x+cw/2, self.height/2-2, value)
            if delta:
                is_pos = delta.startswith("+")
                dc = GREEN if is_pos else RED
                c.setFont("Helvetica", 6.2)
                c.setFillColor(dc)
                c.drawCentredString(x+cw/2, 4, delta)


class RegimeBadge(Flowable):
    def __init__(self, text, color_hex, w=CW):
        super().__init__()
        self.text = text
        self.col  = colors.HexColor(color_hex)
        self.width = w
        self.height = 22

    def draw(self):
        c = self.canv
        c.setFillColor(BG_PANEL2)
        c.setStrokeColor(self.col, alpha=0.5)
        c.setLineWidth(0.6)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        c.setStrokeColor(self.col)
        c.setLineWidth(2.5)
        c.line(0, 0, 0, self.height)
        c.setFont("Helvetica-Bold", 8.5)
        c.setFillColor(self.col)
        c.drawString(10, self.height/2 - 4, f"REGIME ATUAL: {self.text.upper()}")


def make_styles() -> Dict[str, ParagraphStyle]:
    return {
        "h1":    ParagraphStyle("h1", fontName="Helvetica-Bold",
                    fontSize=13, textColor=GOLD, leading=17,
                    spaceBefore=12, spaceAfter=5),
        "h2":    ParagraphStyle("h2", fontName="Helvetica-Bold",
                    fontSize=10, textColor=TEXT_WHITE, leading=14,
                    spaceBefore=8, spaceAfter=3),
        "body":  ParagraphStyle("body", fontName="Helvetica",
                    fontSize=8.8, textColor=TEXT_LIGHT, leading=13.5,
                    alignment=TA_JUSTIFY, spaceAfter=4),
        "bold":  ParagraphStyle("bold", fontName="Helvetica-Bold",
                    fontSize=8.8, textColor=TEXT_WHITE, leading=13,
                    spaceAfter=3),
        "caption":ParagraphStyle("caption", fontName="Helvetica-Oblique",
                    fontSize=7.2, textColor=TEXT_GRAY, leading=10,
                    alignment=TA_CENTER, spaceAfter=3),
        "cover": ParagraphStyle("cover", fontName="Helvetica-Bold",
                    fontSize=22, textColor=GOLD, leading=28,
                    alignment=TA_CENTER, spaceAfter=3),
        "cover2":ParagraphStyle("cover2", fontName="Helvetica",
                    fontSize=11, textColor=TEXT_LIGHT, leading=16,
                    alignment=TA_CENTER),
        "meta":  ParagraphStyle("meta", fontName="Helvetica",
                    fontSize=8, textColor=TEXT_GRAY, leading=12,
                    alignment=TA_CENTER),
        "th":    ParagraphStyle("th", fontName="Helvetica-Bold",
                    fontSize=7.5, textColor=BG_DARK, alignment=TA_CENTER),
        "td":    ParagraphStyle("td", fontName="Helvetica",
                    fontSize=7.8, textColor=TEXT_LIGHT, alignment=TA_RIGHT),
        "td_l":  ParagraphStyle("td_l", fontName="Helvetica-Bold",
                    fontSize=7.8, textColor=TEXT_WHITE, alignment=TA_LEFT),
    }


def _page_cb(c_obj, doc, logo_path, page_num_ref, label, period_str):
    """Callback de cabeçalho/rodapé por página."""
    c_obj.saveState()
    # Fundo
    c_obj.setFillColor(BG_DARK)
    c_obj.rect(0, 0, W, H, fill=1, stroke=0)
    # Cabeçalho
    if page_num_ref[0] > 1:
        c_obj.setFillColor(GOLD, alpha=0.1)
        c_obj.rect(0, H-22*mm, W, 22*mm, fill=1, stroke=0)
        c_obj.setStrokeColor(GOLD, alpha=0.5)
        c_obj.setLineWidth(0.7)
        c_obj.line(ML, H-22*mm, W-MR, H-22*mm)
        if logo_path and logo_path.exists():
            try:
                c_obj.drawImage(str(logo_path), ML, H-19*mm,
                               width=35*mm, height=11*mm,
                               preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
        c_obj.setFont("Helvetica", 7.2)
        c_obj.setFillColor(TEXT_GRAY)
        c_obj.drawRightString(W-MR, H-11*mm, f"MAÁTria Energia · {label} · {period_str}")
        c_obj.setFont("Helvetica-Bold", 7)
        c_obj.setFillColor(GOLD, alpha=0.8)
        c_obj.drawRightString(W-MR, H-17*mm, "CONFIDENCIAL")
    # Rodapé
    c_obj.setStrokeColor(GOLD, alpha=0.35)
    c_obj.setLineWidth(0.4)
    c_obj.line(ML, MB-4*mm, W-MR, MB-4*mm)
    c_obj.setFont("Helvetica", 6.2)
    c_obj.setFillColor(TEXT_GRAY)
    c_obj.drawString(ML, MB-9*mm, "MAÁTria Energia  ·  Inteligência de Mercado  ·  Confidencial")
    c_obj.drawRightString(W-MR, MB-9*mm, f"Página {page_num_ref[0]}")
    c_obj.restoreState()
    page_num_ref[0] += 1


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO PDF
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf(metrics: Dict[str, Any],
              narrative: Dict[str, str],
              output_path: str) -> str:
    """Constrói o PDF e retorna o caminho."""
    S = make_styles()
    period    = metrics.get("period", "monthly")
    label     = PERIOD_CONFIG[period]["label"]
    d_start   = metrics.get("date_start", "")
    d_end     = metrics.get("date_end",   "")
    gen_at    = metrics.get("generated_at", "")
    period_str= f"{d_start} — {d_end}"
    regime    = metrics.get("regime",       "N/D")
    regime_col= metrics.get("regime_color", "#9ca3af")
    n_hours   = metrics.get("n_hours", 0)

    page_num = [1]
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=ML, rightMargin=MR, topMargin=MT+4*mm, bottomMargin=MB,
        title=f"MAÁTria Energia — {label}",
        author="MAÁTria Energia",
    )

    def on_page(c_obj, d):
        _page_cb(c_obj, d, LOGO_PATH, page_num, label, period_str)

    story = []

    # ── CAPA ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12*mm))
    if LOGO_PATH.exists():
        img = RLImage(str(LOGO_PATH), width=80*mm, height=60*mm, kind='proportional')
        img.hAlign = 'CENTER'
        story.append(img)
    story.append(Spacer(1, 6*mm))
    story.append(GoldLine())
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(label.upper(), S["cover"]))
    story.append(Paragraph("MERCADO ELÉTRICO BRASILEIRO", S["cover2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(f"Período: {period_str}", S["meta"]))
    story.append(Paragraph(f"Gerado em: {gen_at}  ·  {n_hours:,} horas analisadas", S["meta"]))
    story.append(Spacer(1, 4*mm))
    story.append(GoldLine(alpha=0.3))
    story.append(Spacer(1, 5*mm))
    story.append(RegimeBadge(regime, regime_col))
    story.append(Spacer(1, 8*mm))
    if EMBLEMA_PATH.exists():
        emb = RLImage(str(EMBLEMA_PATH), width=45*mm, height=26*mm, kind='proportional')
        emb.hAlign = 'CENTER'
        story.append(emb)
    story.append(PageBreak())

    # ── SUMÁRIO EXECUTIVO ─────────────────────────────────────────────────────
    story.append(Paragraph("SUMÁRIO EXECUTIVO", S["h1"]))
    story.append(GoldLine())
    story.append(Spacer(1, 3*mm))

    exec_text = narrative.get("executive", "")
    if exec_text:
        story.append(Paragraph(exec_text, S["body"]))
    story.append(Spacer(1, 3*mm))

    def _kv(m, key, prefix="R$", suffix="", mult=1, digits=2):
        v = m.get(key, {}).get("median", float("nan"))
        if isinstance(v, float) and np.isnan(v):
            return "N/D"
        v *= mult
        if abs(v) >= 1_000_000:
            return f"{prefix}{v/1_000_000:.{digits}f}M{suffix}"
        if abs(v) >= 1_000:
            return f"{prefix}{v/1_000:.{digits}f}K{suffix}"
        return f"{prefix}{v:.{digits}f}{suffix}"

    def _delta(m, key):
        p = m.get(key, {}).get("pct_change", float("nan"))
        if isinstance(p, float) and np.isnan(p):
            return ""
        sign = "+" if p >= 0 else ""
        return f"{sign}{p:.1f}% vs. período anterior"

    story.append(KPIRow([
        ("Custo Mercado",     _kv(metrics,"market_cost"),      _delta(metrics,"market_cost"),     GOLD),
        ("Custo Físico",      _kv(metrics,"physical_cost"),    _delta(metrics,"physical_cost"),   BLUE),
        ("Custo Real Unitário",_kv(metrics,"custo_real_mwh","R$","/MWh",1,0),
                                                               _delta(metrics,"custo_real_mwh"),  RED),
        ("Structural Gap",    _kv(metrics,"structural_gap","R$","/MWh",1,0),
                                                               _delta(metrics,"structural_gap"),  GREEN),
        ("SPDI",              _kv(metrics,"spdi","","×",1,2),  _delta(metrics,"spdi"),            PURPLE),
    ], CW))
    story.append(Spacer(1, 5*mm))

    story.append(KPIRow([
        ("IMR Corrigido",     _kv(metrics,"imr_corrigido"),    _delta(metrics,"imr_corrigido"),   GREEN),
        ("PLD Mediana",       _kv(metrics,"pld","R$","/MWh",1,0), _delta(metrics,"pld"),         GOLD),
        ("EAR%",              _kv(metrics,"ear_pct","","% EAR",1,1), _delta(metrics,"ear_pct"),   BLUE),
        ("Encargos ESS",      _kv(metrics,"encargos"),         _delta(metrics,"encargos"),        TEXT_GRAY),
        ("Structural Drift",  _kv(metrics,"structural_drift","","×",1,2), "",                    GOLD),
    ], CW))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"Período: {period_str}  ·  {n_hours:,} horas  ·  "
        "Fonte: MAÁTria Energia (ONS/CCEE). Deltas vs. período anterior equivalente.",
        S["caption"]))

    story.append(PageBreak())

    # ── ANÁLISE DO REGIME ─────────────────────────────────────────────────────
    story.append(Paragraph("ANÁLISE DE REGIME E SINAIS DE MERCADO", S["h1"]))
    story.append(GoldLine())
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Dinâmica de Regime", S["h2"]))
    story.append(Paragraph(narrative.get("regime", ""), S["body"]))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Sinais Físicos e Econômicos", S["h2"]))
    story.append(Paragraph(narrative.get("signals", ""), S["body"]))
    story.append(Spacer(1, 3*mm))

    # Tabela de indicadores do período
    story.append(Paragraph("Indicadores do Período", S["h2"]))

    def _row(label, key, fmt="M", prefix="R$", suffix=""):
        stat = metrics.get(key, {})
        def _f(v):
            if isinstance(v, float) and np.isnan(v):
                return "—"
            if fmt == "M" and abs(v) >= 1_000_000:
                return f"{prefix}{v/1_000_000:.2f}M{suffix}"
            if fmt == "K" and abs(v) >= 1_000:
                return f"{prefix}{v/1_000:.1f}K{suffix}"
            return f"{prefix}{v:.2f}{suffix}"
        delta = stat.get("pct_change", float("nan"))
        delta_str = ("—" if np.isnan(delta) else
                     f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%")
        return [
            Paragraph(label, S["td_l"]),
            Paragraph(_f(stat.get("min",float("nan"))), S["td"]),
            Paragraph(_f(stat.get("median",float("nan"))), S["td"]),
            Paragraph(_f(stat.get("max",float("nan"))), S["td"]),
            Paragraph(delta_str, S["td"]),
        ]

    tbl_data = [
        [Paragraph(h, S["th"]) for h in
         ["Indicador", "Mínimo", "Mediana", "Máximo", "Δ Período ant."]],
        _row("Custo Mercado (R$/h)",    "market_cost"),
        _row("Custo Físico (R$/h)",     "physical_cost"),
        _row("Encargos ESS (R$/h)",     "encargos",    "K"),
        _row("Custo Real (R$/h)",       "custo_real"),
        _row("Custo Real (R$/MWh)",     "custo_real_mwh","","R$","/MWh"),
        _row("Hidden Cost (R$/MWh)",    "hidden_cost","","R$","/MWh"),
        _row("IMR Corrigido (R$/h)",    "imr_corrigido"),
        _row("IMR Físico (R$/h)",       "imr_fisico"),
        _row("Structural Gap (R$/MWh)", "structural_gap","","R$","/MWh"),
        _row("SPDI (×)",                "spdi","","","×"),
        _row("EII (%)",                 "eii","","","%"),
    ]

    cws = [55*mm, 25*mm, 25*mm, 25*mm, 28*mm]
    tbl = Table(tbl_data, colWidths=cws, rowHeights=[9*mm]+[7*mm]*11)
    ts  = TableStyle([
        ('BACKGROUND', (0,0),(-1,0),  GOLD),
        ('GRID',       (0,0),(-1,-1), 0.3, BORDER),
        ('LINEBELOW',  (0,0),(-1,0),  1.0, GOLD),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0),(-1,-1), 4),
        ('RIGHTPADDING',(0,0),(-1,-1), 4),
        ('TOPPADDING',  (0,0),(-1,-1), 1),
        ('BOTTOMPADDING',(0,0),(-1,-1),1),
    ])
    for i in range(1, 12):
        ts.add('BACKGROUND', (0,i),(-1,i),
               BG_PANEL2 if i % 2 == 1 else BG_PANEL)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Fonte: MAÁTria Energia. Processamento proprietário ONS/CCEE. "
        "Custo Real = T_total + Encargos ESS. "
        "Structural Gap = PLD − Custo Real Unitário.",
        S["caption"]))

    story.append(PageBreak())

    # ── PERSPECTIVAS ─────────────────────────────────────────────────────────
    story.append(Paragraph("PERSPECTIVAS", S["h1"]))
    story.append(GoldLine())
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(narrative.get("outlook", ""), S["body"]))
    story.append(Spacer(1, 5*mm))

    # Nota metodológica
    story.append(Paragraph("NOTA METODOLÓGICA", S["h2"]))
    story.append(Paragraph(
        "Este relatório é gerado automaticamente pela plataforma MAÁTria Energia, "
        "combinando dados horários do Operador Nacional do Sistema Elétrico (ONS) com "
        "encargos ESS da Câmara de Comercialização de Energia Elétrica (CCEE). "
        "Os indicadores proprietários — SPDI, Structural Gap, IMR Corrigido e Hidden "
        "System Cost — são calculados sobre o DataFrame horário consolidado e não "
        "constituem recomendação de investimento. O Custo Físico utiliza CMO como "
        "shadow price da água (custo de oportunidade hídrica). Encargos ESS conforme "
        "dataset encargo_horario_submercado (CCEE), processados em memória.",
        S["body"]))
    story.append(Spacer(1, 4*mm))
    story.append(GoldLine(alpha=0.4))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"MAÁTria Energia  ·  Gerado em {gen_at}  ·  Confidencial",
        S["meta"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ENVIO POR EMAIL
# ══════════════════════════════════════════════════════════════════════════════

def get_report_recipients() -> List[str]:
    """
    Busca emails de usuários Professional e Institutional no banco AUTH.
    """
    import psycopg2
    url = os.getenv("DATABASE_URL_AUTH", "")
    if not url:
        return []
    try:
        conn = psycopg2.connect(url)
        cur  = conn.cursor()
        cur.execute(
            "SELECT email FROM maat_users WHERE plan_type = ANY(%s) AND is_active = TRUE",
            (REPORT_PLANS,)
        )
        emails = [r[0] for r in cur.fetchall()]
        cur.close(); conn.close()
        return emails
    except Exception as e:
        print(f"⚠️  Erro ao buscar destinatários: {e}")
        return []


def send_report_email(pdf_path: str, metrics: Dict[str, Any],
                      recipients: List[str]) -> Tuple[int, int]:
    """
    Envia o PDF como anexo via Resend API.
    Retorna (enviados, falhas).
    """
    api_key  = os.getenv("RESEND_API_KEY",  "")
    from_addr= os.getenv("RESEND_FROM",     "MAÁTria Energia <onboarding@resend.dev>")
    if not api_key:
        print("⚠️  RESEND_API_KEY não configurada.")
        return 0, len(recipients)

    import base64
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    label   = metrics.get("period_label", "Relatório")
    d_start = metrics.get("date_start", "")
    d_end   = metrics.get("date_end",   "")
    regime  = metrics.get("regime",     "N/D")
    spdi    = metrics.get("spdi",       {}).get("median", float("nan"))
    gap     = metrics.get("structural_gap", {}).get("median", float("nan"))
    spdi_s  = f"{spdi:.2f}×" if not np.isnan(spdi) else "N/D"
    gap_s   = (f"R${gap/1_000_000:.1f}M/MWh" if not np.isnan(gap) and abs(gap) >= 1_000_000
               else f"R${gap:.0f}/MWh" if not np.isnan(gap) else "N/D")

    subject = f"MAÁTria Energia · {label} · {d_end}"

    html_body = f"""
    <html>
    <body style="font-family:sans-serif;background:#0b0f14;color:#f3f4f6;padding:24px">
      <div style="max-width:580px;margin:0 auto">
        <div style="border-bottom:2px solid #c8a44d;padding-bottom:16px;margin-bottom:20px">
          <h1 style="color:#c8a44d;font-size:1.4rem;margin:0">{label}</h1>
          <p style="color:#9ca3af;font-size:.85rem;margin:4px 0 0">
            Mercado Elétrico Brasileiro · {d_start} a {d_end}
          </p>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
          <div style="background:#111827;border-top:2px solid #c8a44d;border-radius:6px;padding:12px;text-align:center">
            <div style="color:#9ca3af;font-size:.7rem">REGIME</div>
            <div style="color:#c8a44d;font-weight:700;font-size:1rem">{regime}</div>
          </div>
          <div style="background:#111827;border-top:2px solid #60a5fa;border-radius:6px;padding:12px;text-align:center">
            <div style="color:#9ca3af;font-size:.7rem">SPDI</div>
            <div style="color:#60a5fa;font-weight:700;font-size:1rem">{spdi_s}</div>
          </div>
          <div style="background:#111827;border-top:2px solid #34d399;border-radius:6px;padding:12px;text-align:center">
            <div style="color:#9ca3af;font-size:.7rem">STRUCTURAL GAP</div>
            <div style="color:#34d399;font-weight:700;font-size:1rem">{gap_s}</div>
          </div>
        </div>

        <p style="color:#e5e7eb;font-size:.88rem;line-height:1.6">
          O relatório completo com análise de {metrics.get('n_hours', 0):,} horas de operação
          do SIN está disponível em anexo.<br><br>
          Para visualização interativa dos indicadores, acesse a plataforma.
        </p>

        <div style="border-top:1px solid #1e3a5f;margin-top:20px;padding-top:12px">
          <p style="color:#6b7280;font-size:.75rem;margin:0">
            MAÁTria Energia · Inteligência de Mercado Elétrico Brasileiro<br>
            Este relatório é gerado automaticamente e de caráter confidencial.
          </p>
        </div>
      </div>
    </body>
    </html>
    """

    ok_count = err_count = 0
    for email in recipients:
        try:
            payload = {
                "from":     from_addr,
                "to":       [email],
                "subject":  subject,
                "html":     html_body,
                "attachments": [{
                    "filename":    f"maatria_{metrics['period']}_{d_end}.pdf",
                    "content":     pdf_b64,
                    "content_type":"application/pdf",
                }],
            }
            resp = requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type":  "application/json"},
                json=payload, timeout=30,
            )
            if resp.status_code in (200, 201):
                ok_count += 1
                print(f"  ✅ Enviado: {email}")
            else:
                err_count += 1
                print(f"  ❌ Falha {email}: {resp.status_code}")
        except Exception as e:
            err_count += 1
            print(f"  ❌ Erro {email}: {e}")

    return ok_count, err_count


def log_report(period: str, pdf_path: str, n_sent: int) -> None:
    """Registra geração no maat_usage_log."""
    import psycopg2
    url = os.getenv("DATABASE_URL_AUTH", "")
    if not url:
        return
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO maat_usage_log (user_id, action_type, metadata)
            VALUES (1, 'report_generated', %s)
            """,
            (json.dumps({
                "period":   period,
                "pdf_path": str(pdf_path),
                "n_sent":   n_sent,
                "ts":       datetime.now().isoformat(),
            }),)
        )
        cur.close(); conn.close()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def run(period: str = "monthly",
        preview: bool = False,
        send_to: Optional[str] = None,
        output_dir: str = "reports") -> str:
    """
    Pipeline completo:
    1. Coleta métricas
    2. Gera narrativa via Claude API
    3. Constrói PDF
    4. Envia por email (se não preview)
    5. Registra no log

    Retorna caminho do PDF gerado.
    """
    print(f"\n{'='*55}")
    print(f"MAÁTria Energia · Report Engine · {period.upper()}")
    print(f"{'='*55}")

    # 1. Métricas
    print("\n[1/4] Coletando métricas...")
    metrics = collect_report_data(period)
    print(f"  Período: {metrics.get('date_start')} → {metrics.get('date_end')}")
    print(f"  Horas:   {metrics.get('n_hours', 0):,}")
    print(f"  Regime:  {metrics.get('regime')}")

    # 2. Narrativa
    print("\n[2/4] Gerando narrativa (Claude API)...")
    narrative = generate_narrative(metrics)
    has_api   = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"  {'✅ API Claude' if has_api else '⚠️  Narrativa estática (sem ANTHROPIC_API_KEY)'}")

    # 3. PDF
    print("\n[3/4] Construindo PDF...")
    Path(output_dir).mkdir(exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    pdf_name = f"maatria_{period}_{ts}.pdf"
    pdf_path = str(Path(output_dir) / pdf_name)
    build_pdf(metrics, narrative, pdf_path)
    size_kb  = Path(pdf_path).stat().st_size // 1024
    print(f"  ✅ {pdf_path} ({size_kb} KB)")

    # 4. Envio
    n_sent = 0
    if not preview:
        print("\n[4/4] Enviando emails...")
        if send_to:
            recipients = [send_to]
        else:
            recipients = get_report_recipients()
        print(f"  Destinatários: {len(recipients)}")
        if recipients:
            n_sent, n_err = send_report_email(pdf_path, metrics, recipients)
            print(f"  ✅ Enviados: {n_sent}  ❌ Falhas: {n_err}")
        else:
            print("  ⚠️  Nenhum destinatário encontrado.")
        log_report(period, pdf_path, n_sent)
    else:
        print("\n[4/4] Preview — email não enviado.")

    print(f"\n✅ Concluído: {pdf_path}\n")
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAÁTria Energia · Report Engine")
    parser.add_argument("--period",   choices=["daily","weekly","monthly"],
                        default="monthly", help="Periodicidade do relatório")
    parser.add_argument("--preview",  action="store_true",
                        help="Gera PDF sem enviar emails")
    parser.add_argument("--send-to",  type=str, default=None,
                        help="Enviar para email específico (teste)")
    parser.add_argument("--output-dir", default="reports",
                        help="Diretório de saída dos PDFs")
    args = parser.parse_args()
    run(period=args.period, preview=args.preview,
        send_to=args.send_to, output_dir=args.output_dir)
