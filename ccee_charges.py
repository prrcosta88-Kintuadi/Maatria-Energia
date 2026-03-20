# -*- coding: utf-8 -*-
"""
ccee_charges.py — ESS Charges & Physical Cost Module
══════════════════════════════════════════════════════
Módulo de encargos do sistema e custo físico de geração para a
plataforma MAÁTria Energia.

Dados obtidos via API pública da CCEE (dadosabertos.ccee.org.br)
e processados APENAS em memória — nunca persistidos no Neon.

Métricas calculadas:
    ess_cost_Rh               : custo ESS horário agregado SIN (R$/h)
    physical_generation_cost_Rh: custo físico de geração (R$/h)
    system_cost_Rh            : custo total do sistema (físico + ESS)
    market_cost_Rh            : custo de mercado (carga × PLD)
    infra_marginal_market     : IMR pelo modelo econômico
    infra_marginal_physical   : IMR pelo custo físico ← métrica principal
    infra_marginal_system     : IMR pelo custo sistêmico (c/ ESS)
    spdi                      : Structural Price Distortion Index
    encargo_intensity_index   : participação ESS no custo de mercado
    hidden_system_cost        : custos ocultos (ESS + GFOM)
    structural_drift          : drift estrutural via IMR físico
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional
from datetime import datetime as _dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Resource IDs — encargo_horario_submercado ────────────────────────────────
_ESS_HORARIO_RID = {
    2023: "41a519bf-67b0-469c-8356-3fcbe5f8484b",
    2024: "7ffaa369-e3b0-43c2-bf88-37002afc0bf8",
    2025: "a6ccc7b0-d85a-4586-a4cb-eec8eb8c0827",
    2026: "bf1fb5ee-bff1-4ff7-a791-83dd892438f5",
}

# ─── Resource IDs — encargo_ess_ancilar ──────────────────────────────────────
_ESS_ANCILAR_RID = {
    2023: "2a7994d6-cf6e-4f7f-af0e-3d8f0ef7cf5f",
    2024: "b06431bc-e509-4d65-94cc-6d81b0f1db5c",
    2025: "d93904d1-6a35-400b-927d-2dc52dfef4d8",
    2026: "c7aa8371-ef19-48b7-a6dc-93f857e4e094",
}

# ─── Resource IDs — rd_encargos_contab_horario ───────────────────────────────
_ESS_CONTAB_RID = {
    2024: "ff96fbe9-fd82-4f06-ba39-56e5d1035365",
    2025: "54b886a7-82a9-4d4f-bf81-49d13d93a4c3",
    2026: "edfc76ba-9779-4eda-83c1-98a7c3f6900a",
}

# ─── Resource ID — rd_disp_encargo_horario_sandbox ───────────────────────────
_ESS_SANDBOX_RID = "7ce79b02-ec6d-413d-8c8c-a52ad78c1858"

_CCEE_BASE = "https://dadosabertos.ccee.org.br/api/3/action"

_SUBMERCADOS_FULL = ["SUDESTE", "NORDESTE", "SUL", "NORTE"]


# ══════════════════════════════════════════════════════════════════════════════
# ACESSO À API CCEE
# ══════════════════════════════════════════════════════════════════════════════

def _ccee_headers() -> dict:
    """Headers que simulam browser — necessários para passar pelo Cloudflare."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
        ),
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":           "pt-BR,pt;q=0.9,en;q=0.8",
        "sec-fetch-dest":            "document",
        "sec-fetch-mode":            "navigate",
        "sec-fetch-site":            "none",
        "upgrade-insecure-requests": "1",
    }
    cookie = os.getenv("CCEE_COOKIE", "")
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _ccee_get(rid: str, filters: Optional[dict] = None,
              limit: int = 1000, offset: int = 0,
              timeout: int = 15) -> Optional[pd.DataFrame]:
    """Fetch de uma página do datastore_search. Retorna DataFrame ou None."""
    import json as _json
    params: dict = {"resource_id": rid, "limit": limit, "offset": offset}
    if filters:
        params["filters"] = _json.dumps(filters)
    try:
        resp = requests.get(
            f"{_CCEE_BASE}/datastore_search",
            params=params, headers=_ccee_headers(), timeout=timeout,
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


def _ccee_fetch_all(rid: str, filters: Optional[dict] = None,
                    max_records: int = 50_000,
                    timeout: int = 20) -> pd.DataFrame:
    """
    Busca completa com paginação automática.
    Limita a max_records para proteger RAM do Render.
    Dados processados em memória — sem persistência.
    """
    PAGE, frames = 1000, []
    for offset in range(0, max_records, PAGE):
        df = _ccee_get(rid, filters=filters, limit=PAGE,
                       offset=offset, timeout=timeout)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — ENCARGOS ESS HORÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ess_periodo(df: pd.DataFrame,
                       mes_ref_col: str = "MES_REFERENCIA",
                       periodo_col: str = "PERIODO_COMERCIALIZACAO") -> pd.DataFrame:
    """
    Converte MES_REFERENCIA (YYYYMM) + PERIODO_COMERCIALIZACAO (int sequencial)
    em timestamp horário real.

    PERIODO_COMERCIALIZACAO = 1 → primeira hora do mês (hora 0h).
    """
    if df.empty:
        return df

    df = df.copy()
    df[mes_ref_col]  = df[mes_ref_col].astype(str).str[:6]
    df[periodo_col]  = pd.to_numeric(df[periodo_col], errors="coerce").fillna(1).astype(int)

    # Base = primeiro dia do mês às 00h
    base = pd.to_datetime(df[mes_ref_col], format="%Y%m", errors="coerce")
    # Cada período = 1 hora; período 1 = hora 0
    df["timestamp"] = base + pd.to_timedelta(df[periodo_col] - 1, unit="h")
    return df


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_ess_horario(years: Optional[list] = None,
                      timeout: int = 20) -> pd.DataFrame:
    """
    Busca encargos horários por submercado (encargo_horario_submercado).

    Campos relevantes:
        MES_REFERENCIA, SUBMERCADO, PERIODO_COMERCIALIZACAO,
        TOTAL_NAO_AJUSTADO_ESS, VALOR_AJUSTADO_DEMAIS_ESS

    Agrega ao nível SIN (soma dos 4 submercados por período).
    Retorna DataFrame com colunas:
        timestamp, ess_total_sin, ess_ajustado_sin, ess_consumo_ref,
        ess_on_local, ess_on_multi, ess_off_local, ess_off_multi, ess_cs

    DADOS EM MEMÓRIA — não persistidos.
    """
    if years is None:
        cur = _dt.now().year
        years = [cur - 1, cur] if cur > 2023 else [cur]

    frames = []
    for yr in years:
        rid = _ESS_HORARIO_RID.get(yr)
        if not rid:
            continue
        df = _ccee_fetch_all(rid, max_records=50_000, timeout=timeout)
        if df.empty:
            continue

        # Normalizar colunas numéricas
        for col in ["TOTAL_NAO_AJUSTADO_ESS", "VALOR_AJUSTADO_DEMAIS_ESS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = _parse_ess_periodo(df)
        if "timestamp" not in df.columns:
            continue

        # Normalizar todos os campos numéricos ESS
        ess_cols = [
            "TOTAL_NAO_AJUSTADO_ESS",
            "VALOR_AJUSTADO_DEMAIS_ESS",
            "CONSUMO_REFERENCIA_ESS",
            "ENCARGO_CONST_ON_REST_OP_LOCAL",
            "ENCARGO_CONST_ON_REST_MULTI",
            "ENCARGO_CONST_OFF_REST_OP_LOCAL",
            "ENCARGO_CONST_OFF_REST_MULTI",
            "ENCARGO_CS",
        ]
        for col in ess_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Agregar SIN — soma dos 4 submercados por período
        agg_dict = {}
        if "TOTAL_NAO_AJUSTADO_ESS" in df.columns:
            agg_dict["ess_total_sin"]    = ("TOTAL_NAO_AJUSTADO_ESS", "sum")
        if "VALOR_AJUSTADO_DEMAIS_ESS" in df.columns:
            agg_dict["ess_ajustado_sin"] = ("VALOR_AJUSTADO_DEMAIS_ESS", "sum")
        if "CONSUMO_REFERENCIA_ESS" in df.columns:
            agg_dict["ess_consumo_ref"]  = ("CONSUMO_REFERENCIA_ESS", "sum")
        # Encargos de restrição operacional (on/off + multi + CS)
        for raw, agg_name in [
            ("ENCARGO_CONST_ON_REST_OP_LOCAL", "ess_on_local"),
            ("ENCARGO_CONST_ON_REST_MULTI",    "ess_on_multi"),
            ("ENCARGO_CONST_OFF_REST_OP_LOCAL","ess_off_local"),
            ("ENCARGO_CONST_OFF_REST_MULTI",   "ess_off_multi"),
            ("ENCARGO_CS",                     "ess_cs"),
        ]:
            if raw in df.columns:
                agg_dict[agg_name] = (raw, "sum")

        if not agg_dict:
            continue

        grp = df.groupby("timestamp").agg(**agg_dict).reset_index()

        # ESS total real = soma de todos os componentes disponíveis
        # Usar TOTAL_NAO_AJUSTADO_ESS como principal; calcular soma de componentes como fallback
        if "ess_total_sin" not in grp.columns:
            comp_cols = [c for c in ["ess_on_local","ess_on_multi",
                                     "ess_off_local","ess_off_multi","ess_cs"]
                         if c in grp.columns]
            grp["ess_total_sin"] = grp[comp_cols].sum(axis=1) if comp_cols else 0.0

        frames.append(grp)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "ess_total_sin", "ess_ajustado_sin"])

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp").drop_duplicates("timestamp")
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.set_index("timestamp")
    return result



@st.cache_data(show_spinner=False, ttl=1800)
def fetch_ess_ancilar(years: Optional[list] = None,
                      timeout: int = 20) -> pd.DataFrame:
    """
    Busca encargos ancilares (ESS ancilar) da CCEE.
    Contém: constrained-on, constrained-off, serviços ancilares, CS.
    Agrega ao nível SIN por período horário.
    DADOS EM MEMÓRIA — não persistidos.
    """
    if years is None:
        cur = _dt.now().year
        years = [cur - 1, cur] if cur > 2023 else [cur]

    frames = []
    for yr in years:
        rid = _ESS_ANCILAR_RID.get(yr)
        if not rid:
            continue
        df = _ccee_fetch_all(rid, max_records=50_000, timeout=timeout)
        if df.empty:
            continue
        df = _parse_ess_periodo(df)
        if "timestamp" not in df.columns:
            continue

        # Normalizar todos os campos disponíveis
        num_cols = [col for col in df.columns
                    if col not in ("timestamp","MES_REFERENCIA","SUBMERCADO",
                                   "PERIODO_COMERCIALIZACAO","_id")]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Tentar mapear campos conhecidos do dataset ancilar
        # O schema exato varia — mapeamos o que estiver disponível
        field_map = {}
        for col in df.columns:
            cu = col.upper()
            if "CONSTRAINED_ON" in cu or "CONST_ON" in cu:
                field_map.setdefault("constrained_on_total", []).append(col)
            elif "CONSTRAINED_OFF" in cu or "CONST_OFF" in cu:
                field_map.setdefault("constrained_off_total", []).append(col)
            elif "SEGURANCA" in cu or "SECURITY" in cu or "ENER" in cu:
                field_map.setdefault("seguranca_energetica", []).append(col)
            elif "RESERVA" in cu or "OPERATIVA" in cu or "RESERVE" in cu:
                field_map.setdefault("reserva_operativa", []).append(col)
            elif "ANCILAR" in cu or "ANCILIAR" in cu or "ANCIL" in cu:
                field_map.setdefault("servicos_ancilares", []).append(col)
            elif "TOTAL" in cu and "ESS" in cu:
                field_map.setdefault("ess_ancilar_total", []).append(col)

        # Agregar SIN
        grp = df.groupby("timestamp")[num_cols].sum().reset_index()
        # Mapear para nomes padronizados
        out = grp[["timestamp"]].copy()
        for std_name, raw_cols in field_map.items():
            out[std_name] = grp[[c for c in raw_cols if c in grp.columns]].sum(axis=1)
        # Se não mapeou nada, usar soma total
        if len(out.columns) == 1:
            out["ess_ancilar_total"] = grp[num_cols].sum(axis=1)

        frames.append(out)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp").drop_duplicates("timestamp")
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result.set_index("timestamp")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2–10 — ENRIQUECIMENTO DO DF HORÁRIO
# ══════════════════════════════════════════════════════════════════════════════

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona ao DataFrame horário do app.py as seguintes colunas:

    Seção 1 : ess_cost_Rh
    Seção 2 : physical_generation_cost_Rh
    Seção 3 : system_cost_Rh
    Seção 4 : market_cost_Rh
    Seção 5 : infra_marginal_market, infra_marginal_physical, infra_marginal_system
    Seção 6 : spdi
    Seção 7 : encargo_intensity_index
    Seção 8 : hidden_system_cost
    Seção 9 : t_sistemica (versão corrigida)
    Seção 10: structural_drift

    Não modifica o DataFrame original — retorna cópia enriquecida.
    Operações vetorizadas — sem loops por linha.
    """
    if df.empty:
        return df

    df = df.copy()

    def _c(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
               else pd.Series(np.nan, index=df.index)

    pld_s    = _c("pld")
    cmo_s    = _c("cmo_dominante")
    load_s   = _c("load")
    hydro_s  = _c("hydro")
    thermal_s= _c("thermal")
    cvu_s    = _c("cvu_semana")
    t_total_s= _c("t_total")
    gfom_s   = _c("gfom") if "gfom" in df.columns else pd.Series(0.0, index=df.index)
    net_load_s = _c("net_load")

    # ── Seção 1: ESS cost ─────────────────────────────────────────────────────
    # Busca anos cobrindo o intervalo do df
    years_needed = sorted(df.index.year.unique().tolist())
    ess_df = _fetch_ess_for_df(years_needed)

    if not ess_df.empty:
        # Alinhar ao índice do df por hora mais próxima
        ess_aligned = ess_df["ess_total_sin"].reindex(df.index, method="ffill").ffill().bfill()
        df["ess_cost_Rh"] = ess_aligned.fillna(0.0)
    else:
        df["ess_cost_Rh"] = 0.0

    ess_s = _c("ess_cost_Rh")

    # ── Seção 2: Custo físico de geração ─────────────────────────────────────
    # C_fisico = Σ(thermal × CVU) + (hydro × CMO)
    # thermal_real_cost já existe no df = thermal × CVU
    _thermal_cost = _c("thermal_real_cost")
    if _thermal_cost.isna().all():
        _thermal_cost = thermal_s.fillna(0) * cvu_s.fillna(0)

    # Custo hidráulico: hydro × CMO (shadow price da água)
    # ⚠️ CMO ≠ custo contábil da hidráulica — é custo marginal do sistema.
    # Interpretação correta: "custo de oportunidade da água" (não custo operacional).
    _hydro_cost = hydro_s.fillna(0) * cmo_s.fillna(0)
    df["physical_generation_cost_Rh"] = (_thermal_cost + _hydro_cost).where(
        load_s > 0, np.nan)

    phys_s = _c("physical_generation_cost_Rh")

    # ── Seção 3: Custo sistêmico (físico + ESS) ───────────────────────────────
    df["system_cost_Rh"] = (phys_s.fillna(0) + ess_s.fillna(0)).where(
        load_s > 0, np.nan)

    sys_s = _c("system_cost_Rh")

    # ── Seção 4: Custo de mercado = carga × PLD ───────────────────────────────
    sin_cost_s = _c("sin_cost")
    if sin_cost_s.isna().all():
        sin_cost_s = load_s.fillna(0) * pld_s.fillna(0)
    df["market_cost_Rh"] = sin_cost_s.where(load_s > 0, np.nan)

    mkt_s = _c("market_cost_Rh")

    # ── Seção 5: Três indicadores de renda infra-marginal ─────────────────────
    # 5a) Market IMR = mercado - decomposição econômica
    df["infra_marginal_market"] = (mkt_s - t_total_s).where(
        mkt_s.notna() & t_total_s.notna(), np.nan)

    # 5b) Physical IMR = mercado - custo físico  ← MÉTRICA PRINCIPAL
    df["infra_marginal_physical"] = (mkt_s - phys_s).where(
        mkt_s.notna() & phys_s.notna(), np.nan)

    # 5c) System IMR = mercado - custo sistêmico (c/ ESS)
    df["infra_marginal_system"] = (mkt_s - sys_s).where(
        mkt_s.notna() & sys_s.notna(), np.nan)

    # ── Seção 6: SPDI = mercado / custo físico ────────────────────────────────
    df["spdi"] = (mkt_s / phys_s.replace(0, np.nan)).clip(lower=0, upper=10).where(
        mkt_s.notna() & phys_s.notna(), np.nan)

    # ── Seção 7: Encargo Intensity Index = ESS / mercado ─────────────────────
    df["encargo_intensity_index"] = (ess_s / mkt_s.replace(0, np.nan)).clip(
        lower=0, upper=1).where(mkt_s.notna() & ess_s.notna(), np.nan)

    # ── Seção 8: Hidden System Cost = ESS + GFOM ─────────────────────────────
    # GFOM já está no df como coluna "gfom" (MWmed de despacho fora do mérito)
    # Estimar custo GFOM = gfom_despacho × CVU (proxy)
    _gfom_cost = gfom_s.fillna(0) * cvu_s.fillna(0)
    df["hidden_system_cost"] = (ess_s.fillna(0) + _gfom_cost).where(
        load_s > 0, np.nan)

    # ── Seção 9: t_sistemica corrigida ────────────────────────────────────────
    # Nova fórmula: |PLD - CMO| × net_load (distorção aplicada à carga flexível)
    df["t_sistemica"] = (
        (pld_s - cmo_s).abs() * net_load_s.fillna(0)
    ).where(pld_s.notna() & cmo_s.notna(), np.nan)

    # ── Seção 10: Structural Drift via IMR físico ──────────────────────────────
    imr_phys = _c("infra_marginal_physical")
    rolling_imr = imr_phys.rolling(720, min_periods=24).mean()
    _2021_mask = df.index.year == 2021
    baseline = float(imr_phys[_2021_mask].mean()) if _2021_mask.sum() >= 24 else np.nan
    if np.isnan(baseline) or baseline == 0:
        baseline = float(imr_phys.quantile(0.10)) if not imr_phys.dropna().empty else 1.0
    if baseline and not np.isnan(baseline) and baseline != 0:
        df["structural_drift"] = (rolling_imr / abs(baseline)).clip(lower=0.1, upper=10.0)
    else:
        df["structural_drift"] = np.nan

    # ══════════════════════════════════════════════════════════════════════════
    # SEÇÕES 11–17 — FECHAMENTO ECONÔMICO COMPLETO (prompt.txt)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Seção 11: Decomposição detalhada dos encargos ─────────────────────────
    # Fonte 1: ESS horário já carregado (ess_cost_Rh = TOTAL_NAO_AJUSTADO_ESS)
    # Fonte 2: componentes on/off do gfom_comp (Neon — já no df se disponível)
    # Fonte 3: encargos ancilares via API CCEE (fetch_ess_ancilar)

    # constrained_off — preferir coluna Neon (gfom/despacho_gfom) se existir
    _constrained_off = _c("constrained_off")     # do Neon via gfom_comp
    _constrained_on  = _c("constrained_on")      # do Neon se disponível

    # Tentar enriquecer com dataset ancilar da CCEE (em memória)
    try:
        years_needed = sorted(df.index.year.unique().tolist())
        _ancilar_df = fetch_ess_ancilar(years=years_needed)
        if not _ancilar_df.empty:
            for _acol in ["constrained_on_total", "constrained_off_total",
                          "seguranca_energetica", "reserva_operativa",
                          "servicos_ancilares", "ess_ancilar_total"]:
                if _acol in _ancilar_df.columns:
                    _s = _ancilar_df[_acol].reindex(df.index, method="ffill").ffill().bfill()
                    df[f"ess_{_acol}"] = _s.fillna(0.0)
                    # Atualizar constrained_on/off se vier do ancilar
                    if _acol == "constrained_on_total" and _constrained_on.isna().all():
                        _constrained_on = _s.fillna(0.0)
                    if _acol == "constrained_off_total" and _constrained_off.isna().all():
                        _constrained_off = _s.fillna(0.0)
    except Exception:
        pass

    # ── Colunas de encargos padronizadas (R$/h) ───────────────────────────────
    # ESS_total_R$ = encargo ESS horário total (já calculado como ess_cost_Rh)
    df["ESS_total_R$"]            = ess_s.fillna(0.0)

    # constrained_on/off: despacho forçado por restrição de rede
    # Fonte: gfom_comp do Neon (constrained_off = val_verifconstrainedoff × CVU)
    # Se vier em MWmed, multiplicar por CVU para obter R$/h
    # constrained_off/on: proxy via Neon (MWmed) × CVU
    # ⚠️ APROXIMAÇÃO: unidade do Neon é MWmed — pode já ser custo em R$
    # Mantido como breakdown analítico; NÃO entra no Encargos_total (evitar dupla contagem)
    _has_constrained_off_mw = "constrained_off" in df.columns and _constrained_off.notna().any()
    if _has_constrained_off_mw:
        # Proxy: MWmed × CVU → R$/h (aprox.)
        df["constrained_off_R$"] = (_constrained_off.fillna(0) * cvu_s.fillna(0))
    elif "ess_constrained_off_total" in df.columns:
        df["constrained_off_R$"] = _c("ess_constrained_off_total")
    else:
        df["constrained_off_R$"] = np.nan   # NaN = sem dado, não zero

    _has_constrained_on_mw = "constrained_on" in df.columns and _constrained_on.notna().any()
    if _has_constrained_on_mw:
        df["constrained_on_R$"] = (_constrained_on.fillna(0) * cvu_s.fillna(0))
    elif "ess_constrained_on_total" in df.columns:
        df["constrained_on_R$"] = _c("ess_constrained_on_total")
    else:
        df["constrained_on_R$"] = np.nan

    # Segurança energética e reserva operativa (do dataset ancilar se disponível)
    df["seguranca_energetica_R$"] = _c("ess_seguranca_energetica").fillna(0.0)         if "ess_seguranca_energetica" in df.columns else pd.Series(0.0, index=df.index)
    df["reserva_operativa_R$"]    = _c("ess_reserva_operativa").fillna(0.0)         if "ess_reserva_operativa" in df.columns else pd.Series(0.0, index=df.index)

    # ── Seção 12: Encargos_total_R$/h ────────────────────────────────────────
    # OPÇÃO A (implementada): usar ESS_total como base única.
    # TOTAL_NAO_AJUSTADO_ESS já inclui constrained-on/off, segurança e reserva.
    # Somá-los separadamente geraria DUPLA CONTAGEM.
    # Os componentes individuais são mantidos apenas para BREAKDOWN ANALÍTICO.
    # constrained_off do Neon (MWmed×CVU) é proxy adicional — não entra na soma.
    df["Encargos_total_R$/h"] = ess_s.where(load_s > 0, np.nan)

    # Breakdown analítico (não somados ao total — apenas informativos):
    # constrained_off_R$, constrained_on_R$, seguranca_energetica_R$, reserva_operativa_R$

    _encargos_s = _c("Encargos_total_R$/h")

    # ── Seção 13: Custo real do sistema ───────────────────────────────────────
    # Custo_real = T_total (custo físico modelado) + Encargos (custo oculto)
    # T_total aqui é o já calculado no enrich_df via sin_cost e t_total
    _t_total_s = _c("t_total")
    if _t_total_s.isna().all():
        # Fallback: usar custo físico + encargos
        _t_total_s = phys_s.fillna(0)

    df["Custo_real_R$/h"] = (_t_total_s.fillna(0) + _encargos_s.fillna(0)).where(
        load_s > 0, np.nan)

    # ── Seção 14: Custo unitário real (R$/MWh) ────────────────────────────────
    df["Custo_real_R$/MWh"] = (
        df["Custo_real_R$/h"] / load_s.replace(0, np.nan)
    ).where(load_s > 0, np.nan)

    # ── Seção 15: Hidden System Cost (custo oculto unitário) ──────────────────
    # Representa o custo que o PLD não captura diretamente
    df["Hidden_system_cost_R$/MWh"] = (
        _encargos_s / load_s.replace(0, np.nan)
    ).where(load_s > 0, np.nan)

    # ── Seção 16: Infra-marginal rent corrigido ────────────────────────────────
    # IMR_corrigido = Receita (sin_cost) − Custo real (físico + encargos)
    # Fechamento econômico completo: quanto sobra além do custo total real
    df["infra_marginal_rent_corrigido"] = (
        mkt_s - _c("Custo_real_R$/h")
    ).where(mkt_s.notna() & _c("Custo_real_R$/h").notna(), np.nan)

    # ── Seção 17: Structural Gap (gap preço × custo real) ─────────────────────
    # Structural_gap = PLD - Custo_real_unitário
    # >0: preço acima do custo real → prêmio estrutural de mercado
    # <0: preço abaixo do custo real → transferência implícita
    df["Structural_gap_R$/MWh"] = (
        pld_s - _c("Custo_real_R$/MWh")
    ).where(pld_s.notna() & _c("Custo_real_R$/MWh").notna(), np.nan)

    # ── IMR sistêmico: manter a versão original (market − physical − ESS) ──────
    # infra_marginal_rent_corrigido = IMR econômico completo (nova métrica)
    # infra_marginal_system         = IMR sistêmico original (market − sys_cost)
    # NÃO sobrescrever: são métricas conceitualmente distintas.
    # infra_marginal_system já foi calculado na Seção 5c acima.
    # Nota: "IMR sistêmico foi redefinido como IMR econômico completo" — FALSO.
    # Mantemos as duas métricas separadas.

    return df


def _fetch_ess_for_df(years: list) -> pd.DataFrame:
    """
    Busca ESS para os anos necessários — em memória, sem persistência.
    Usa cache de sessão (@st.cache_data) para evitar re-fetch na mesma sessão.
    """
    try:
        return fetch_ess_horario(years=years)
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS PARA VISUALIZAÇÃO (Seção 12)
# ══════════════════════════════════════════════════════════════════════════════

def imr_selector_label(key: str) -> str:
    """Retorna rótulo legível para seleção de IMR."""
    return {
        "infra_marginal_physical": "IMR Físico (mercado − custo físico) ← recomendado",
        "infra_marginal_market":   "IMR Mercado (mercado − decomposição econômica)",
        "infra_marginal_system":   "IMR Sistêmico (mercado − custo físico + ESS)",
    }.get(key, key)


def spdi_label(val: float) -> tuple[str, str]:
    """Retorna (label, cor) para o SPDI."""
    if np.isnan(val):
        return "N/D", "#6b7280"
    if val >= 1.6:
        return f"{val:.2f}× — distorção forte", "#f87171"
    if val >= 1.3:
        return f"{val:.2f}× — prêmio estrutural", "#c8a44d"
    return f"{val:.2f}× — alinhado", "#34d399"


def eii_label(val: float) -> tuple[str, str]:
    """Retorna (label, cor) para o Encargo Intensity Index."""
    if np.isnan(val):
        return "N/D", "#6b7280"
    if val >= 0.10:
        return f"{val:.1%} — intervenção elevada", "#f87171"
    if val >= 0.05:
        return f"{val:.1%} — estresse operacional", "#c8a44d"
    return f"{val:.1%} — normal", "#34d399"


def summary_metrics(df: pd.DataFrame,
                    imr_col: str = "infra_marginal_physical") -> dict:
    """
    Calcula sumário das novas métricas para exibição em KPIs.
    Usa últimos 30 dias disponíveis como janela de referência.
    """
    if df.empty:
        return {}

    def _last30(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        cutoff = s.index.max() - pd.Timedelta(days=30) if not s.empty else s.index.min()
        return s[s.index >= cutoff]

    imr_s   = _last30(imr_col)
    spdi_s  = _last30("spdi")
    eii_s   = _last30("encargo_intensity_index")
    hsc_s   = _last30("hidden_system_cost")
    ess_s   = _last30("ess_cost_Rh")
    mkt_s   = _last30("market_cost_Rh")
    phys_s  = _last30("physical_generation_cost_Rh")
    drift_s = _last30("structural_drift")

    return {
        "imr_median":        float(imr_s.median())   if not imr_s.empty   else np.nan,
        "imr_p90":           float(imr_s.quantile(.9)) if not imr_s.empty else np.nan,
        "spdi_median":       float(spdi_s.median())  if not spdi_s.empty  else np.nan,
        "eii_median":        float(eii_s.median())   if not eii_s.empty   else np.nan,
        "hidden_cost_median":float(hsc_s.median())   if not hsc_s.empty   else np.nan,
        "ess_total_30d":     float(ess_s.sum())       if not ess_s.empty   else np.nan,
        "mkt_total_30d":     float(mkt_s.sum())       if not mkt_s.empty   else np.nan,
        "phys_total_30d":    float(phys_s.sum())      if not phys_s.empty  else np.nan,
        "drift_current":     float(drift_s.iloc[-1])  if not drift_s.empty else np.nan,
    }
