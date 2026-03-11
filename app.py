import json
import os
from datetime import datetime, date, time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import duckdb


def _core_cache_token() -> str:
    watched = [
        Path("data/core_analysis_latest.parquet"),
        Path("core_analysis_latest.parquet"),
        Path("data/core_analysis_latest.json"),
        Path("core_analysis_latest.json"),
    ]
    parts = []
    for p in watched:
        if p.exists():
            stt = p.stat()
            parts.append(f"{p}:{int(stt.st_mtime)}:{stt.st_size}")
        else:
            parts.append(f"{p}:missing")
    return "|".join(parts)


def _core_file_diagnostics() -> list[str]:
    msgs = []
    for p in [Path("data/core_analysis_latest.parquet"), Path("core_analysis_latest.parquet")]:
        if not p.exists():
            msgs.append(f"{p}: ausente")
            continue
        size = p.stat().st_size
        pointer = False
        try:
            with p.open("rb") as f:
                head = f.read(200)
            pointer = b"git-lfs.github.com/spec/v1" in head
        except Exception:
            pass
        if pointer:
            msgs.append(f"{p}: presente ({size} bytes), mas é ponteiro Git LFS (objeto real não baixado)")
        else:
            msgs.append(f"{p}: presente ({size} bytes)")
    return msgs


@st.cache_data
def _load_core(_token: str) -> Dict[str, Any]:

    parquet_paths = [
        Path("data/core_analysis_latest.parquet"),
        Path("core_analysis_latest.parquet"),
    ]

    for p in parquet_paths:
        if not p.exists():
            continue
        try:
            con = duckdb.connect()
            try:
                row = con.execute(
                    "SELECT core_json FROM read_parquet(?) LIMIT 1",
                    [str(p)],
                ).fetchone()
            finally:
                con.close()

            if row and row[0]:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
        except Exception:
            continue

    json_paths = [
        Path("data/core_analysis_latest.json"),
        Path("core_analysis_latest.json"),
    ]
    for p in json_paths:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)

    return {}

#@st.cache_data
#def _load_core() -> Dict[str, Any]:
#    for p in [Path("data/core_analysis_latest.json"), Path("core_analysis_latest.json")]:
#        if p.exists():
#            with p.open("r", encoding="utf-8") as f:
#                return json.load(f)
#    return {}


def _series_from_hourly(d: Dict[str, Any], name: str) -> pd.Series:
    if not isinstance(d, dict) or not d:
        return pd.Series(dtype=float, name=name)
    s = pd.Series(d, name=name)
    s.index = pd.to_datetime(s.index, errors="coerce")
    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    except Exception:
        pass
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()


def _series_from_operacao(records, value_key: str, name: str) -> pd.Series:
    if not isinstance(records, list) or not records:
        return pd.Series(dtype=float, name=name)
    df = pd.DataFrame(records)
    if "instante" not in df.columns or value_key not in df.columns:
        return pd.Series(dtype=float, name=name)
    df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
    df[value_key] = pd.to_numeric(df[value_key], errors="coerce")
    df = df.dropna(subset=["instante", value_key])
    s = df.groupby("instante")[value_key].sum().sort_index().rename(name)
    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
    except Exception:
        pass
    return s


def _build_hourly_df(core: Dict[str, Any]) -> pd.DataFrame:
    econ = core.get("economic", {}) or core.get("advanced_metrics", {}).get("economic", {}) or {}
    adv = core.get("advanced_metrics", {})

    df = pd.DataFrame()
    # economic-driven
    for k, col in [
        ("sin_cost_hourly", "sin_cost"),
        ("T_prudencia_hourly", "t_prudencia"),
        ("T_hidro_hourly", "t_hidro"),
        ("T_eletric_hourly", "t_eletric"),
        ("T_sistemica_hourly", "t_sistemica"),
        ("CVaR_implicit_hourly", "cvar_implicit"),
        ("Risk_Aversion_Gap_hourly", "risk_gap"),
        ("curtailment_loss_hourly", "curtailment_loss"),
        ("hydro_gap_hourly", "hydro_gap"),
        ("required_hydro_hourly", "required_hydro"),
        ("mandatory_generation_hourly", "mandatory_generation"),
        ("thermal_prudential_dispatch_hourly", "thermal_prudential_dispatch"),
        ("infra_marginal_rent_hourly", "infra_marginal_rent"),
    ]:
        s = _series_from_hourly(econ.get(k, {}), col)
        if not s.empty:
            df = df.join(s, how="outer") if not df.empty else s.to_frame()

    # system state
    st_h = econ.get("system_state_hourly", {})
    if isinstance(st_h, dict) and st_h:
        s = pd.Series(st_h, name="system_state")
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.dropna()
        df = df.join(s, how="outer") if not df.empty else s.to_frame()

    # from operacao
    oper = core.get("operacao", {})
    gen = oper.get("generation", {})
    load = oper.get("load", {})

    load_sin = _series_from_operacao((load.get("sin") or {}).get("serie", []), "carga", "load")
    if not load_sin.empty:
        df = df.join(load_sin, how="outer") if not df.empty else load_sin.to_frame()

    for source_keys, col in [
        (["fotovoltaica"], "solar"),
        (["eolielétrica"], "wind"),
        (["térmica"], "thermal"),
        (["hidroelétrica"], "hydro"),
        (["nuclear"], "nuclear"),
    ]:
        serie_records = []
        for key in source_keys:
            if (gen.get(key) or {}).get("serie"):
                serie_records = (gen.get(key) or {}).get("serie", [])
                break
        s = _series_from_operacao(serie_records, "geracao", col)
        if not s.empty:
            df = df.join(s, how="outer") if not df.empty else s.to_frame()

    # pld from ccee records
    ccee = core.get("ccee", {}).get("data", [])
    if ccee:
        cdf = pd.DataFrame(ccee)
        if {"mes_referencia", "dia", "hora", "pld_hora"}.issubset(cdf.columns):
            cdf["mr"] = cdf["mes_referencia"].astype(str).str.zfill(6)
            cdf["ts"] = pd.to_datetime(cdf["mr"].str[:4] + "-" + cdf["mr"].str[4:6] + "-" + cdf["dia"].astype(str).str.zfill(2) + " " + cdf["hora"].astype(str).str.zfill(2) + ":00:00", errors="coerce")
            cdf["pld_hora"] = pd.to_numeric(cdf["pld_hora"], errors="coerce")
            cdf = cdf.dropna(subset=["ts", "pld_hora"])
            if not cdf.empty:
                pld = cdf.groupby("ts")["pld_hora"].mean().rename("pld")
                try:
                    if getattr(pld.index, "tz", None) is not None:
                        pld.index = pld.index.tz_localize(None)
                except Exception:
                    pass
                df = df.join(pld, how="outer") if not df.empty else pld.to_frame()

    # panel from advanced metrics
    panel = pd.DataFrame(adv.get("painel_horario_renovavel", []))
    if not panel.empty and "instante" in panel.columns:
        panel["instante"] = pd.to_datetime(panel["instante"], errors="coerce")
        panel = panel.dropna(subset=["instante"]).set_index("instante")
        for src, dst in [("gfom_pct", "gfom_pct"), ("ipr", "ipr"), ("isr", "isr"), ("ear", "ear"), ("ena", "ena")]:
            if src in panel.columns:
                s = pd.to_numeric(panel[src], errors="coerce").rename(dst)
                df = df.join(s, how="outer") if not df.empty else s.to_frame()

    # curtailment series
    for key, col in [("solar", "curtail_solar"), ("eolica", "curtail_wind")]:
        ser = pd.DataFrame(((core.get("renewables", {}).get("curtailment", {}).get(key, {}) or {}).get("serie", [])))
        if not ser.empty and {"instante", "valor"}.issubset(ser.columns):
            ser["instante"] = pd.to_datetime(ser["instante"], errors="coerce")
            ser["valor"] = pd.to_numeric(ser["valor"], errors="coerce")
            s = ser.dropna().set_index("instante")["valor"].groupby(level=0).sum().rename(col)
            try:
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
            except Exception:
                pass
            df = df.join(s, how="outer") if not df.empty else s.to_frame()

    if not df.empty:
        df = df.sort_index()
        z = pd.Series(0.0, index=df.index)
        load_s = pd.to_numeric(df["load"], errors="coerce") if "load" in df.columns else pd.Series(np.nan, index=df.index)
        solar_s = pd.to_numeric(df["solar"], errors="coerce") if "solar" in df.columns else z
        wind_s = pd.to_numeric(df["wind"], errors="coerce") if "wind" in df.columns else z
        hydro_s = pd.to_numeric(df["hydro"], errors="coerce") if "hydro" in df.columns else z
        thermal_s = pd.to_numeric(df["thermal"], errors="coerce") if "thermal" in df.columns else z
        nuclear_s = pd.to_numeric(df["nuclear"], errors="coerce") if "nuclear" in df.columns else z
        cur_solar = pd.to_numeric(df["curtail_solar"], errors="coerce") if "curtail_solar" in df.columns else z
        cur_wind = pd.to_numeric(df["curtail_wind"], errors="coerce") if "curtail_wind" in df.columns else z
        df["net_load"] = load_s - solar_s.fillna(0) - wind_s.fillna(0)
        # auditoria operacional solicitada: carga total e carga líquida por soma de fontes
        df["carga_total"] = load_s
        df["curtail_total"] = cur_solar.fillna(0) + cur_wind.fillna(0)
        if "cmo_dominante" not in df.columns:
            cmo_sm = ((adv.get("aderencia_fisico_economica", {}) or {}).get("cmo_horario_por_submercado", {}) or {})
            if isinstance(cmo_sm, dict) and cmo_sm:
                # prefer SUDESTE
                first_key = "SUDESTE" if "SUDESTE" in cmo_sm else list(cmo_sm.keys())[0]
                s = _series_from_hourly(cmo_sm.get(first_key, {}), "cmo_dominante")
                if not s.empty:
                    df = df.join(s, how="left")

    return _ensure_hourly(df)


def _fmt_ptbr(value: Any, decimals: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        s = f"{float(value):,.{decimals}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def _fmt_money_compact(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000:
        return f"R$ {_fmt_ptbr(v/1_000_000, 2)} MM"
    if av >= 1_000:
        return f"R$ {_fmt_ptbr(v/1_000, 2)} k"
    return f"R$ {_fmt_ptbr(v, 2)}"


def _prepare_logo(path: Path) -> Optional[Path]:
    """Recorta bordas escuras do PNG para reduzir fundo/preenchimento visual."""
    if not path.exists():
        return None
    try:
        from PIL import Image

        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        # pixels não quase-pretos e não transparentes
        mask = (arr[:, :, 3] > 5) & ((arr[:, :, 0] > 20) | (arr[:, :, 1] > 20) | (arr[:, :, 2] > 20))
        if not mask.any():
            return path
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cropped = img.crop((max(0, x0 - 4), max(0, y0 - 4), min(img.width, x1 + 5), min(img.height, y1 + 5)))
        out = Path("data") / "emblema_maatria_trimmed.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out)
        return out
    except Exception:
        return path


def _kpi_card(label: str, value: str, border_color: str):
    st.markdown(
        f"""
        <div style='background:#131722;border:1px solid #2a2f3a;border-top:3px solid {border_color};
                    border-radius:12px;padding:10px 12px;height:95px;'>
          <div style='font-size:12px;color:#9ba3af;'>{label}</div>
          <div style='font-size:23px;color:#f3f4f6;font-weight:700;line-height:1.2'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _system_text(row: pd.Series) -> str:
    s = row.get("system_state")
    if isinstance(s, str) and s:
        return f"Regime {s} com carga líquida de {_fmt_ptbr(row.get('net_load', np.nan), 0)} MWmed."
    return "Sem dados suficientes para diagnóstico automático da hora selecionada."


def _plot_df(dff: pd.DataFrame) -> pd.DataFrame:
    out = dff.copy().reset_index()
    first_col = out.columns[0]
    if first_col != "instante":
        out = out.rename(columns={first_col: "instante"})
    return out


def _ensure_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida qualquer série semihorária em base horária (média de :00 e :30)."""
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if out.empty:
        return out
    out["hora_ref"] = out.index.floor("h")
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in out.columns if c not in num_cols + ["hora_ref"]]
    agg_map = {c: "mean" for c in num_cols}
    agg_map.update({c: "last" for c in other_cols})
    out = out.groupby("hora_ref", as_index=True).agg(agg_map).sort_index()
    out.index.name = "instante"
    return out


def _latest_operational_dates(core: Dict[str, Any]) -> Dict[str, Optional[date]]:
    oper = core.get("operacao", {}) if isinstance(core, dict) else {}

    def _max_date_from_records(records: Any) -> Optional[date]:
        if not isinstance(records, list) or not records:
            return None
        df = pd.DataFrame(records)
        if "instante" not in df.columns:
            return None
        ts = pd.to_datetime(df["instante"], errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            return None
        return ts.max().date()

    load_records = ((oper.get("load", {}) or {}).get("sin", {}) or {}).get("serie", [])
    load_day = _max_date_from_records(load_records)

    generation = oper.get("generation", {}) or {}
    gen_days = []
    for payload in generation.values():
        day = _max_date_from_records((payload or {}).get("serie", []))
        if day is not None:
            gen_days.append(day)
    generation_day = max(gen_days) if gen_days else None

    return {"load": load_day, "generation": generation_day}


def main():
    st.set_page_config(page_title="MAÁTria Energia", layout="wide", initial_sidebar_state="collapsed")

    st.markdown(
        """
        <style>
          .stApp { background-color:#0b0f14; color:#f3f4f6; }
          [data-testid="stSidebar"] { display:none !important; }
          .block-container { padding-top: 40px; }
          .fixed-header { position: fixed; top: 0; left:0; right:0; z-index:999; background:#0b0f14; }
          .full-bleed-line { height:0.1px; background:#c8a44d; width:100vw; margin-left:calc(50% - 50vw); }
          .tabs-layer { background: linear-gradient(180deg, #0b1222 0%, #070d1a 100%); padding:0.01rem 0.01rem 0.01rem 0.01rem; }
          label { color:#ffffff !important; font-weight:700 !important; }
          .stTabs [data-baseweb="tab-list"] { gap: 0.15rem; flex-wrap: nowrap !important; overflow-x: auto !important; scrollbar-width: thin; }
          .stTabs [data-baseweb="tab"] { color:#e5e7eb; border-radius:6px; padding:0.25rem 0.45rem; font-size:0.78rem; white-space:nowrap; }
          .stTabs [aria-selected="true"] { background:#152238 !important; color:#f8fafc !important; border:1px solid #c8a44d !important; }
          div[data-testid="stFormSubmitButton"] > button {
            background:#d4af37 !important; color:#111827 !important; font-weight:800 !important; border:1px solid #b38f2b !important;
          }
          div[data-testid="stFormSubmitButton"] > button:hover { background:#e3bf4c !important; color:#000 !important; }
          .cards-row { margin-bottom: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    core = _load_core(_core_cache_token())
    if not core:
        st.warning("⚠️ core_analysis_latest.parquet/json não encontrado ou inválido. Gerando nova análise...")
        for _msg in _core_file_diagnostics():
            st.caption(f"🔎 {_msg}")
        
        # IMPORTANTE: Você precisa ter os dados brutos em algum lugar!
        # Opção 1: Se os dados brutos estão em um arquivo
        raw_data_path = Path("data/kintuadi_latest.json")  # ou o caminho correto
        if raw_data_path.exists():
            with open(raw_data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            st.error("❌ Arquivo de dados brutos não encontrado. Não é possível gerar nova análise.")
            return
        
        try:
            # Importar a função build_core_analysis
            from scripts.core_analysis import build_core_analysis
            
            # PASSAR OS DADOS BRUTOS, não o core vazio!
            new_core = build_core_analysis(raw_data)
            
            if new_core:
                # Verificar estrutura básica
                required = ["timestamp", "hydrology", "prices"]
                missing = [key for key in required if key not in new_core]
                
                if missing:
                    st.warning(f"⚠️ Análise gerada, mas faltam campos obrigatórios: {missing}")
                
                # Salvar o arquivo
                os.makedirs("data", exist_ok=True)
                with open("data/core_analysis_latest.json", "w", encoding="utf-8") as f:
                    json.dump(new_core, f, indent=2, ensure_ascii=False, default=str)
                
                st.success(f"✅ Nova análise gerada e salva! Timestamp: {new_core.get('timestamp', 'N/A')}")
                
                # Atualizar a variável core com o novo conteúdo
                core = new_core
            else:
                st.error("❌ build_core_analysis retornou None")
                return
                
        except ImportError:
            st.error("❌ Não foi possível importar build_core_analysis de scripts.core_analysis")
            return
        except Exception as e:
            st.error(f"❌ Erro ao gerar nova análise: {e}")
            import traceback
            traceback.print_exc()
            return

    df = _build_hourly_df(core)
    if df.empty:
        st.warning("Sem séries horárias suficientes no core para renderizar o painel.")
        return

    min_d, max_d = df.index.min().date(), df.index.max().date()
    default_day = date.today() - pd.Timedelta(days=1)
    if default_day < min_d or default_day > max_d:
        default_day = max_d

    if "date_start" not in st.session_state:
        st.session_state["date_start"] = default_day
    if "date_end" not in st.session_state:
        st.session_state["date_end"] = default_day

    st.markdown("<div class='fixed-header'>", unsafe_allow_html=True)
    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    colc1, colc2, colc3 = st.columns([1, 1, 1])
    with colc2:
        logo = _prepare_logo(Path("streamlit/img/emblema_maatria.png"))
        if logo and logo.exists():
            st.image(str(logo), width=200)
        else:
            st.markdown("## MAÁTria Energia")

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    analyze_clicked = False
    form_col, _ = st.columns([0.4, 0.6])
    with form_col:
        with st.form("period_form", clear_on_submit=False):
            # CSS específico para este formulário
            st.markdown("""
            <style>
            div[data-testid="stForm"] .stDateInput label {
                font-size: 0.7rem !important;
                margin-bottom: 2px !important;
            }
            div[data-testid="stForm"] .stDateInput input {
                font-size: 0.75rem !important;
                padding: 0.2rem 0.5rem !important;
            }
            div[data-testid="stForm"] .stFormSubmitButton button {
                font-size: 0.7rem !important;
                padding: 0.2rem 0.5rem !important;
            }
            </style>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.05, 1.05, 0.8])
            with c1:
                dt_start = st.date_input("DE", value=st.session_state["date_start"], min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
            with c2:
                dt_end = st.date_input("ATÉ", value=st.session_state["date_end"], min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
            with c3:
                st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
                analyze_clicked = st.form_submit_button("ANALISAR", use_container_width=True)

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    tabs = st.tabs([
        "📸 Fotografia Operativa",
        "💰 Decomposição Econômica",
        "⚡ Curtailment & Restrições",
        "🧠 Coerência Operativa",
        "🔋 Simulação BESS",
        "📊 Matriz Horária do SIN",
        "📘 Metodologia & Glossário",
    ])

    st.markdown("<div class='full-bleed-line'></div>", unsafe_allow_html=True)

    if analyze_clicked:
        if dt_start > dt_end:
            st.error("Período inválido: DE deve ser menor ou igual a ATÉ.")
        else:
            st.session_state["date_start"] = dt_start
            st.session_state["date_end"] = dt_end
            st.rerun()

    selected_start = st.session_state.get("date_start", default_day)
    selected_end = st.session_state.get("date_end", default_day)

    dff = df[(df.index.date >= selected_start) & (df.index.date <= selected_end)].copy()
    if dff.empty:
        st.warning("Não há dados para o período selecionado.")
        return

    latest_operational = _latest_operational_dates(core)
    available_reference_days = [d for d in [latest_operational.get("load"), latest_operational.get("generation")] if d is not None]
    photo_day = max(available_reference_days) if available_reference_days else max_d
    if photo_day < selected_start or photo_day > selected_end:
        photo_day = selected_end
    dff_photo = dff[dff.index.date == photo_day].copy()
    if dff_photo.empty:
        dff_photo = dff.copy()

    dff = _ensure_hourly(dff)
    dff_photo = _ensure_hourly(dff_photo)

    current = dff.mean(numeric_only=True)
    current_state = dff["system_state"].dropna().iloc[-1] if "system_state" in dff.columns and not dff["system_state"].dropna().empty else "-"

    # Totais do período selecionado (soma hora a hora)
    total_sin_cost = pd.to_numeric(dff.get("sin_cost", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_prud = pd.to_numeric(dff.get("t_prudencia", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_agua = pd.to_numeric(dff.get("t_hidro", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_sistemica = pd.to_numeric(dff.get("t_sistemica", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_eletrica = pd.to_numeric(dff.get("t_eletric", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_infra_marginal = pd.to_numeric(dff.get("infra_marginal_rent", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_curt_loss = pd.to_numeric(dff.get("curtailment_loss", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_gfom = pd.to_numeric(dff.get("gfom_pct", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_isr = pd.to_numeric(dff.get("isr", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    total_ipr = pd.to_numeric(dff.get("ipr", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)

    kpis = [
        ("PLD médio", f"R$ {_fmt_ptbr(current.get('pld', np.nan),2)}", "#22c55e"),
        ("CMO dominante", f"R$ {_fmt_ptbr(current.get('cmo_dominante', np.nan),2)}", "#3b82f6"),
        ("Custo Total SIN", _fmt_money_compact(total_sin_cost), "#f59e0b"),
        ("Custo Prudência", _fmt_money_compact(total_prud), "#ef4444"),
        ("Custo Hídrico", _fmt_money_compact(total_agua), "#14b8a6"),
        ("Custo Elétrico", _fmt_money_compact(total_eletrica), "#FFEE00"),
        ("Custo Sistêmico", _fmt_money_compact(total_sistemica), "#004918"),
        ("Custo Infra-marginal", _fmt_money_compact(total_infra_marginal), "#3D0049"),
        ("Curtailment", f"{_fmt_ptbr(current.get('curtail_total', np.nan),2)} MWmed", "#a78bfa"),
        ("Valor (R$) Curtailment", _fmt_money_compact(total_curt_loss), "#eab308"),
        ("GFOM", _fmt_ptbr(total_gfom,2), "#38bdf8"),
        ("ISR", _fmt_ptbr(total_isr,2), "#f97316"),
        ("IPR", _fmt_ptbr(total_ipr,2), "#84cc16"),
        ("Risk Gap", _fmt_ptbr(current.get("risk_gap", np.nan),2), "#fb7185"),
        ("CVaR Implícito", f"R$ {_fmt_ptbr(current.get('cvar_implicit', np.nan),2)}", "#60a5fa"),
    ]

    for base in (0, 5, 10):
        cols = st.columns(5)
        for i in range(5):
            idx = base + i
            if idx < len(kpis):
                lab, val, color = kpis[idx]
                with cols[i]:
                    _kpi_card(lab, val, color)
        st.markdown("<div class='cards-row'></div>", unsafe_allow_html=True)

    st.info(f"Estado Operativo do SIN: **{current_state}** | Período: **{selected_start}** até **{selected_end}**")

    with st.expander("Ver tabela dos cards (hora a hora)", expanded=False):
        card_cols = [
            c
            for c in [
                "pld",
                "cmo_dominante",
                "sin_cost",
                "t_prudencia",
                "t_hidro",
                "t_eletric",
                "t_sistemica",
                "infra_marginal_rent",
                "curtailment_loss",
                "curtail_total"
                "gfom_pct",
                "isr",
                "ipr",
                "risk_gap",
                "cvar_implicit",
            ]
            if c in dff.columns
        ]
        st.dataframe(_plot_df(dff[card_cols]), width="stretch", height=320)

    with tabs[0]:
        load_ref = latest_operational.get("load")
        gen_ref = latest_operational.get("generation")
        load_txt = load_ref.strftime("%d/%m/%Y") if load_ref else "N/D"
        gen_txt = gen_ref.strftime("%d/%m/%Y") if gen_ref else "N/D"
        st.caption(f"Dados extraídos até o dia **{load_txt}** (load) e **{gen_txt}** (generation).")
        st.caption("Montagem: séries horárias observadas de geração por fonte + carga e carga líquida (`Carga - (Solar + Eólica)`).")
        st.write(_system_text(current))
        fig = go.Figure()
        labels = {
            "hydro": "Hidro", "thermal": "Térmica", "nuclear": "Nuclear", "solar": "Solar", "wind": "Eólica"
        }
        for src in ["hydro", "thermal", "nuclear", "solar", "wind"]:
            if src in dff_photo.columns:
                fig.add_bar(x=dff_photo.index, y=dff_photo[src], name=labels[src])
        if "carga_total" in dff_photo.columns:
            fig.add_scatter(x=dff_photo.index, y=dff_photo["carga_total"], name="Carga Total", mode="lines")
        if "net_load" in dff_photo.columns:
            fig.add_scatter(x=dff_photo.index, y=dff_photo["net_load"], name="Carga Líquida", mode="lines")
        fig.update_layout(template="plotly_dark", barmode="stack", height=420)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Ver dados do gráfico (hora a hora)"):
            plot_cols = [c for c in ["carga_total", "net_load", "solar", "wind", "hydro", "thermal", "nuclear"] if c in dff_photo.columns]
            st.dataframe(_plot_df(dff_photo[plot_cols]), width="stretch", height=280)

    with tabs[1]:
        pdf = _plot_df(dff)
        decomp_cols = [c for c in ["t_hidro", "t_sistemica", "t_prudencia", "t_eletric"] if c in pdf.columns]
        if decomp_cols:
            st.caption("Montagem: decomposição econômica horária `T_total = T_hidro + T_sistêmica + T_prudência`.")
            label_map = {
                "t_hidro": "Custo Hídrico", "t_sistemica": "Custo Sistêmico", "t_prudencia": "Custo Prudencial", "t_eletric": "Custo elétrico"
            }
            fig = go.Figure()
            fig = px.bar(
                pdf,
                x="instante",
                y=decomp_cols,
                template="plotly_dark",
                barmode="stack"
            )
            # adicionar linha
            fig.add_scatter(
                x=pdf["instante"],
                y=pdf["sin_cost"],
                mode="lines",
                name="Custo Total SIN",
                line=dict(width=3)
            ) 
            fig.update_layout(title="Decomposição horária empilhada (R$/h)")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver dados do gráfico (hora a hora)"):
                st.dataframe(pdf[["instante"] + decomp_cols], width="stretch", height=280)

        if {"thermal", "thermal_prudential_dispatch"}.issubset(dff.columns):
            g2 = _plot_df(dff[["thermal", "thermal_prudential_dispatch"]])
            thermal_labels = {
            "thermal": "Geração Térmica Total", "thermal_prudential_dispatch": "Geração Térmica Prudencial"
            }
            fig2 = px.line(g2, x="instante", y=["thermal", "thermal_prudential_dispatch"], template="plotly_dark", labels=thermal_labels)
            fig2.update_layout(title="Despacho térmico total vs despacho prudencial (MWmed)")
            st.plotly_chart(fig2, width="stretch")
            st.caption("A segunda curva mostra a parcela térmica associada à prudência operativa.")
            with st.expander("Ver dados do gráfico térmico (hora a hora)"):
                st.dataframe(g2, width="stretch", height=260)

        # ===============================
        # HEATMAP – INFRA MARGINAL RENT
        # ===============================

        if "infra_marginal_rent" in df.columns:

            st.subheader("Mapa estrutural — Renda Infra-Marginal do SIN")
            st.caption(
                "Mapa calculado sobre toda a base histórica disponível. "
                "Os dados são diários, mas o eixo Y marca as mudanças de mês."
            )

            heat_df = df.copy()

            heat_df["data"] = heat_df.index.date
            heat_df["hora"] = heat_df.index.hour

            pivot = heat_df.pivot_table(
                index="data",
                columns="hora",
                values="infra_marginal_rent",
                aggfunc="mean"
            )

            # converter índice para datetime para manipular meses
            pivot.index = pd.to_datetime(pivot.index)

            # localizar início de cada mês
            month_starts = pivot.index.to_series().groupby(
                [pivot.index.year, pivot.index.month]
            ).first()

            y_ticks = pd.to_datetime(month_starts.values)
            y_labels = [d.strftime("%m-%Y") for d in y_ticks]

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdBu_r",
                    colorbar=dict(title="R$/h"),
                    xgap=2,   # espaço entre horas
                    ygap=0    # pequeno espaço entre dias
                )
            )

            fig_heat.update_layout(
                template="plotly_dark",
                height=5000,
                xaxis=dict(
                    title="Hora do dia",
                    tickmode="linear",
                    dtick=1
                ),
                yaxis=dict(
                    title="Data",
                    tickmode="array",
                    tickvals=y_ticks,
                    ticktext=y_labels
                )
            )

            st.plotly_chart(fig_heat, width="stretch")

            with st.expander("Ver dados do mapa de calor (base completa)"):
                st.dataframe(pivot, width="stretch", height=300)

    with tabs[2]:
        cdf = _plot_df(dff)
        cols = [c for c in ["curtail_solar", "curtail_wind", "curtail_total"] if c in cdf.columns]
        if cols:
            st.caption("Montagem: curtailment horário por fonte (solar/eólica) e total agregado.")
            fig = go.Figure()
            fig = px.bar(
                pdf,
                x="instante",
                y=["curtail_solar", "curtail_wind"],
                template="plotly_dark",
                barmode="stack"
            )
            #fig = px.bar(cdf, x="instante", y=cols, template="plotly_dark", barmode="group")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Ver dados do gráfico (hora a hora)"):
                st.dataframe(cdf[["instante"] + cols], width="stretch", height=280)
        st.caption("Distribuição por tipo de restrição disponível no painel horário do core quando fornecido pelo ONS.")

    with tabs[3]:
        metrics = {
            "Risk Gap": current.get("risk_gap", np.nan),
            "CVaR": current.get("cvar_implicit", np.nan),
            "EAR_norm": np.nan,
            "ENA_norm": np.nan,
            "Load pressure": np.nan,
        }
        metrics_norm = {
            "risk": np.tanh(metrics["Risk Gap"]/300),
            "cvar": metrics["CVaR"]/100,
            "ear": 1-metrics["EAR_norm"],
            "ena": 1-metrics["ENA_norm"],
            "load": abs(metrics["Load pressure"]-1),
        }

        norm = ((core.get("economic") or {}).get("normalization_hourly") or {})
        if norm and not dff.empty:
            tkey = dff.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            metrics["EAR_norm"] = (norm.get("EAR_norm") or {}).get(tkey, np.nan)
            metrics["ENA_norm"] = (norm.get("ENA_norm") or {}).get(tkey, np.nan)
            metrics["Load pressure"] = (norm.get("Load_norm") or {}).get(tkey, np.nan)
        score_vals = [abs(v) for v in metrics_norm.values() if pd.notna(v)]
        coherence = 100*(1-np.mean(score_vals))
        coherence = np.clip(coherence,0,100)
        color = "🟢" if coherence >= 70 else ("🟡" if coherence >= 40 else "🔴")
        st.metric("Métrica de Coerência do SIN", "-" if pd.isna(coherence) else f"{coherence:.1f}")
        st.markdown(f"Classificação: {color}")
        st.json(metrics)

    with tabs[4]:

        sim_shift = st.slider(
            "Percentual de deslocamento do curtailment solar para 19h–23h",
            0, 100, 0
        )
        sim = dff.copy()
        if sim.empty:
            st.warning("Sem dados suficientes para simulação.")
            st.stop()
        frac = sim_shift / 100.0
        solar_curt = sim.get("curtail_solar", pd.Series(0, index=sim.index)).fillna(0)
        # energia total deslocada
        energy_shift = solar_curt * frac
        night_hours = [19, 20, 21, 22, 23]
        night_mask = sim.index.hour.isin(night_hours)
        if night_mask.sum() == 0:
            st.warning("Não há horas noturnas no período selecionado.")
            st.stop()
        # distribuição uniforme
        per_hour = energy_shift.sum() / night_mask.sum()
        thermal = sim.get("thermal", pd.Series(0, index=sim.index)).fillna(0)
        hydro = sim.get("hydro", pd.Series(0, index=sim.index)).fillna(0)
        thermal_after = thermal.copy()
        hydro_after = hydro.copy()
        for ts in sim.index[night_mask]:
            remove_thermal = min(thermal_after.loc[ts], per_hour)
            thermal_after.loc[ts] -= remove_thermal
            remainder = per_hour - remove_thermal
            if remainder > 0:
                remove_hydro = min(hydro_after.loc[ts], remainder)
                hydro_after.loc[ts] -= remove_hydro
        # --------------------------
        # cálculo da geração BESS
        # --------------------------
        bess = (thermal - thermal_after) + (hydro - hydro_after)
        # dataframe antes
        before = pd.DataFrame({
            "hydro": hydro,
            "thermal": thermal,
            "nuclear": sim.get("nuclear", 0),
            "solar": sim.get("solar", 0),
            "wind": sim.get("wind", 0)
        })
        # dataframe depois
        after = pd.DataFrame({
            "hydro": hydro_after,
            "thermal": thermal_after,
            "nuclear": sim.get("nuclear", 0),
            "solar": sim.get("solar", 0),
            "wind": sim.get("wind", 0),
            "bess": bess
        })
        labels = {
            "hydro": "Hidro",
            "thermal": "Térmica",
            "nuclear": "Nuclear",
            "solar": "Solar",
            "wind": "Eólica",
            "bess": "BESS"
        }
        # --------------------------
        # gráfico antes
        # --------------------------
        st.subheader("Situação observada")
        fig_before = go.Figure()
        for src in ["hydro", "thermal", "nuclear", "solar", "wind"]:
            if src in before.columns:
                fig_before.add_bar(
                    x=before.index,
                    y=before[src],
                    name=labels[src]
                )
        fig_before.update_layout(
            template="plotly_dark",
            barmode="stack",
            height=400
        )
        st.plotly_chart(
            fig_before,
            width="stretch",
            key="bess_generation_before"
        )
        # --------------------------
        # gráfico depois
        # --------------------------
        st.subheader("Situação simulada (curtailment redistribuído)")
        fig_after = go.Figure()
        for src in ["hydro", "thermal", "nuclear", "solar", "wind", "bess"]:
            if src in after.columns:
                fig_after.add_bar(
                    x=after.index,
                    y=after[src],
                    name=labels[src]
                )
        fig_after.update_layout(
            template="plotly_dark",
            barmode="stack",
            height=400
        )
        st.plotly_chart(
            fig_after,
            width="stretch",
            key="bess_generation_after"
        )

    with tabs[5]:
        matrix_cols = [
            c for c in ["pld","cmo_dominante","load","net_load","hydro","thermal","nuclear","solar","wind","gfom_pct","curtail_total","ear","ena","risk_gap","system_state"] if c in dff.columns
        ]
        m = dff[matrix_cols].copy()
        if not m.empty:
            m["interpretacao"] = m.apply(_system_text, axis=1)
            st.dataframe(m, width="stretch", height=420)
            st.download_button("Exportar CSV", data=m.reset_index().to_csv(index=False).encode("utf-8"), file_name="matriz_horaria_sin.csv", mime="text/csv")

    with tabs[6]:

        st.markdown("### 📘 Metodologia & Glossário")
        st.caption("Guia conceitual da plataforma: como os indicadores são calculados e como interpretar a operação do SIN.")

        with st.expander("🎯 1) Propósito da Plataforma", expanded=False):
            st.markdown("""
A plataforma **não tem como objetivo prever o PLD**.

O foco é analisar a **coerência entre condições físicas do sistema e os sinais econômicos observados**, permitindo interpretar a operação do SIN em base **hora a hora**.

A análise cruza informações de:

- hidrologia
- disponibilidade de geração
- despacho térmico
- penetração de renováveis
- curtailment
- preços marginais (PLD e CMO)

**Fotografia Operativa do SIN**

é um diagnóstico instantâneo da condição **física, energética e econômica** do sistema.
""")
            st.info("Interprete esta aba como um guia de leitura do sistema elétrico, não como um modelo de previsão de preços.")

        with st.expander("⚙️ 2) Conceitos Fundamentais do SIN", expanded=False):
            st.markdown("""
**Carga (Demanda)**  
Energia total consumida pelo sistema em uma determinada hora.

**Geração**  
Energia efetivamente produzida pelas diferentes fontes do sistema.

**Carga Líquida**  
Parcela da demanda que precisa ser atendida por fontes **flexíveis** (hidrelétricas e térmicas).

**Fórmula**

`Carga Líquida = Carga − (Solar + Eólica)`

**Valor da Água**

Representa o custo de oportunidade de utilizar água armazenada nos reservatórios agora em vez de preservá-la para uso futuro.

Como aproximação operacional, utilizamos o **CMO (Custo Marginal de Operação)** como proxy desse valor.
""")

        with st.expander("📊 3) Métricas Principais", expanded=False):
            st.markdown("""
### GFOM (Geração Fora da Ordem de Mérito)

Indica a parcela da geração térmica despachada **fora da lógica econômica do mérito de custo**.

**Fórmula**

`GFOM = Térmica_GFOM / Térmica_Total`

**Interpretação típica**

- `< 5%` → despacho majoritariamente econômico  
- `5–15%` → despacho misto  
- `> 15%` → presença relevante de decisão operativa

---

### Curtailment

Energia renovável **disponível mas não utilizada pelo sistema**.

Principais causas:

- restrições de transmissão
- estabilidade elétrica
- saturação de geração
- inflexibilidade de usinas térmicas ou nucleares

**Leitura econômica**

energia de baixo custo que deixa de ser utilizada.

---

### IPR — Índice de Pressão Renovável

Mede o peso das renováveis sobre a demanda.

`IPR = Renovável Disponível / Carga`

---

### ISR — Índice de Saturação Renovável

Avalia a pressão renovável sobre a parcela flexível da demanda.

`ISR = Renovável Disponível / Carga Líquida`

Quando:

`ISR > 1`

há risco de **saturação estrutural de geração renovável**.

---

### EAR — Energia Armazenada

Estoque de energia contido nos reservatórios hidráulicos.

Representa a **segurança energética futura do sistema**.

---

### ENA — Energia Natural Afluente

Energia hidrológica que entra naturalmente no sistema por meio das vazões.

Interpretação:

- **ENA alta** → tendência de alívio hidrológico  
- **ENA baixa** → aumento do risco de escassez
""")

            st.warning("Interprete sempre IPR e ISR junto com o curtailment para distinguir excesso renovável de restrições elétricas.")

        with st.expander("💰 4) Decomposição Econômica do Sistema", expanded=False):
            st.markdown("""
A plataforma separa o custo horário do sistema em componentes econômicos.

### Estrutura central

`T_total = T_elétrico + T_hidro + T_prudência + T_sistêmica`

onde:

**T_elétrico — Custo Estrutural de Geração**

Representa o custo mínimo necessário para atender a carga considerando o despacho por mérito econômico.

Principal componente:

`Térmica por mérito × CVU médio`

---

**T_hidro — Custo Hidrológico**

Valor econômico associado ao uso da água armazenada.

A água funciona como um **ativo energético armazenável**, cujo valor é aproximado pelo **CMO**.

---

**T_prudência — Custo de Decisão Operativa**

Representa o custo adicional associado a decisões conservadoras do operador, como:

- preservação de reservatórios
- despacho térmico preventivo
- restrições operativas

---

**T_sistêmica — Ajuste Estrutural do Sistema**

Captura diferenças entre o valor econômico do mercado e o custo físico da geração.

Pode assumir valores positivos ou negativos dependendo da condição estrutural do sistema.
""")

        with st.expander("🛡️ 5) CVaR e Aversão ao Risco", expanded=False):
            st.markdown("""
O planejamento da operação considera cenários hidrológicos adversos.

Para isso são utilizados mecanismos de **aversão ao risco**, como o **CVaR (Conditional Value at Risk)**.

Exemplo de parametrização usada no setor:

`(15%, 40%)`

significando:

- análise dos **15% piores cenários hidrológicos**
- com **peso de 40% na decisão operativa**

Maior aversão ao risco tende a produzir:

- maior preservação hídrica
- maior despacho térmico
- maior pressão sobre o PLD

### CVaR Implícito Observado

A plataforma estima uma aproximação do valor implícito da aversão ao risco:

`CVaR_implícito = max(PLD − CMO, 0)`

Quando o **PLD atinge o teto regulatório**, o valor implícito não é observável diretamente.
""")

        with st.expander("📉 6) Risk Aversion Gap", expanded=False):
            st.markdown("""
O **Risk Aversion Gap** compara o nível de aversão ao risco observado com o custo médio da geração térmica.

**Definição**

`Risk Gap = CVaR_implícito − CVU_médio`

**Interpretação**

- **positivo** → operação conservadora  
- **próximo de zero** → operação neutra  
- **negativo** → sistema em regime de abundância energética
""")

        with st.expander("💧 7) Teste de Necessidade Hidráulica", expanded=False):
            st.markdown("""
Esse teste avalia quanto da geração hidráulica é **estruturalmente necessária** para atender a demanda.

### Passo 1 — Geração mandatória

`Renováveis + Nuclear + Térmica Inflexível`

### Passo 2 — Hidro necessária

`Hidro_necessária = Carga − Geração_mandatória`

### Passo 3 — Comparação com a geração observada

- `Hidro observada > Hidro necessária`  
→ sistema com forte presença hidráulica

- `Hidro observada < Hidro necessária`  
→ maior dependência térmica
""")

        with st.expander("🧾 8) Custo Econômico do SIN (R$/h)", expanded=False):
            st.markdown("""
A exposição econômica total do sistema pode ser aproximada por:

`Custo SIN = Carga × PLD`

Esse valor representa o **valor econômico da energia liquidada no mercado** naquela hora.

Ele não corresponde diretamente ao custo físico de geração.
""")

        with st.expander("📊 9) Infra-Marginal Rent", expanded=False):
            st.markdown("""
Chamamos de **Infra-Marginal Rent** a diferença entre:

`Valor econômico da energia`  
e  
`Custo físico estimado de produção`.

Na plataforma:

`Infra_marginal = Custo SIN − T_total`

### Interpretação

**Infra-marginal positivo**

O valor pago pelo mercado é maior que o custo físico de geração.

Isso ocorre tipicamente quando:

- o PLD é definido por usinas marginais caras
- grande parte da geração vem de fontes mais baratas (hidro, renováveis)

Nesse caso, **geradores recebem renda infra-marginal**.

---

**Infra-marginal negativo**

O valor econômico do mercado é inferior ao custo físico estimado.

Isso ocorre em situações como:

- excesso estrutural de oferta
- saturação renovável
- PLD muito baixo

Nesse caso, o sistema está operando com **sinais de preço comprimidos**.
""")

        with st.expander("🌡️ 10) Heatmap de Infra-Marginal ao Longo do Tempo", expanded=False):
            st.markdown("""
O mapa de calor apresentado no dashboard mostra a evolução histórica da diferença entre:

`Custo SIN` e `T_total`.

Cada célula representa **uma hora de operação do sistema**.

Esse gráfico permite identificar **regimes estruturais de mercado**, como:

**Regime de escassez**

PLD elevado e forte renda infra-marginal.

**Regime de abundância**

PLD baixo e compressão de receitas.

**Transições estruturais**

Mudanças de comportamento ao longo das estações hidrológicas ou da expansão renovável.

Por utilizar toda a base histórica disponível, o heatmap permite visualizar **padrões operativos recorrentes do SIN**.
""")

        with st.expander("🏷️ 11) Classificação Operativa do SIN", expanded=False):
            st.markdown("""
A plataforma classifica o estado do sistema em regimes operativos.

**Escassez Hidrológica**

Baixo armazenamento e maior dependência térmica.

---

**Preservação Hídrica**

Estratégia deliberada de poupar reservatórios.

---

**Saturação Renovável**

Excesso instantâneo de geração renovável.

---

**Stress Operativo**

Sinais simultâneos de risco físico e pressão econômica.

---

**Equilíbrio Estrutural**

Operação estável sem pressões relevantes.
""")

        with st.expander("🧭 12) Como Interpretar o Dashboard", expanded=False):
            st.markdown("""
### Roteiro sugerido de leitura

1️⃣ selecione o período de análise  
2️⃣ observe os indicadores principais  
3️⃣ verifique o score de coerência operativa  
4️⃣ analise a decomposição econômica  
5️⃣ identifique causas de curtailment  
6️⃣ observe o heatmap histórico  
7️⃣ utilize a simulação BESS

### Objetivo final

Avaliar se o comportamento observado do **PLD** está **coerente com as condições físicas do SIN**.
""")

            st.success("Combine sempre sinais físicos (carga, geração, reservatórios) com sinais econômicos (PLD, CMO e custos).")

if __name__ == "__main__":

    main()
