# -*- coding: utf-8 -*-
"""
profile.py — User Profile Page (MAÁTria Energia)
═════════════════════════════════════════════════
Renderiza a página de perfil com:
  • Status do plano e uso mensal
  • Upgrade de plano
  • Compra de créditos
  • Alteração de senha
  • Histórico de uso
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from monetization import (
    User, PLANS, CREDIT_PACKAGES,
    change_password, upgrade_plan, add_credits,
    get_usage_stats, create_subscription,
)

_COLORS = {
    "free":         "#9ca3af",
    "analyst":      "#60a5fa",
    "professional": "#34d399",
    "institutional":"#c8a44d",
}
_PLAN_ICONS = {
    "free": "🆓", "analyst": "📊",
    "professional": "🚀", "institutional": "🏛",
}


def render_profile_page(user: User) -> None:
    """Renderiza a página de perfil completa."""

    st.markdown("""
    <style>
    .profile-header{background:#111827;border-left:4px solid #c8a44d;
        border-radius:8px;padding:16px 20px;margin-bottom:20px}
    .plan-badge{display:inline-block;font-size:.72rem;font-weight:700;
        letter-spacing:.1em;padding:3px 10px;border-radius:4px;
        text-transform:uppercase;margin-bottom:6px}
    .plan-free        {background:#9ca3af22;border:1px solid #9ca3af55;color:#9ca3af}
    .plan-analyst     {background:#60a5fa22;border:1px solid #60a5fa55;color:#60a5fa}
    .plan-professional{background:#34d39922;border:1px solid #34d39955;color:#34d399}
    .plan-institutional{background:#c8a44d22;border:1px solid #c8a44d55;color:#c8a44d}
    .metric-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
        padding:14px;text-align:center;margin:4px}
    .metric-val{font-size:1.5rem;font-weight:700;color:#e5e7eb}
    .metric-lbl{font-size:.72rem;color:#6b7280;margin-top:2px}
    .upgrade-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
        padding:16px;margin:6px 0;transition:.2s}
    .credit-card{background:#111827;border:1px solid #c8a44d44;border-radius:8px;
        padding:12px 16px;margin:4px 0}
    </style>
    """, unsafe_allow_html=True)

    plan      = PLANS.get(user.plan_type, PLANS["free"])
    pcolor    = _COLORS.get(user.plan_type, "#9ca3af")
    picon     = _PLAN_ICONS.get(user.plan_type, "📋")
    pct_used  = (user.simulations_used_current_month / user.simulation_limit * 100
                 if user.simulation_limit > 0 else 0)
    bar_color = "#f87171" if pct_used >= 90 else "#c8a44d" if pct_used >= 70 else "#34d399"

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='profile-header'>
      <div class='plan-badge plan-{user.plan_type}'>{picon} {plan['label']}</div>
      <div style='font-size:1.1rem;font-weight:700;color:#e5e7eb'>{user.email}</div>
      <div style='font-size:.8rem;color:#6b7280;margin-top:4px'>
        Conta criada em {user.created_at.strftime('%d/%m/%Y') if user.created_at else 'N/D'}
        &nbsp;·&nbsp; Último reset: {user.last_reset_at.strftime('%d/%m/%Y') if user.last_reset_at else 'N/D'}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs da página de perfil ───────────────────────────────────────────────
    ptabs = st.tabs([
        "📊 Status do Plano",
        "⬆️ Upgrade / Créditos",
        "🔑 Segurança",
        "📈 Histórico de Uso",
    ])

    # ════════════════════════════════════════════════════════════════════════════
    with ptabs[0]:  # Status do Plano
    # ════════════════════════════════════════════════════════════════════════════
        st.markdown("#### Uso mensal")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-val' style='color:{bar_color}'>
                {user.simulations_used_current_month}
              </div>
              <div class='metric-lbl'>Simulações usadas</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-val'>{user.simulation_limit}</div>
              <div class='metric-lbl'>Limite mensal</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-val' style='color:#a78bfa'>
                {user.credits_balance}
              </div>
              <div class='metric-lbl'>Créditos extras</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            hist = plan.get("history_days")
            hist_str = f"{hist} dias" if hist else "Completo"
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-val' style='color:{pcolor}'>{hist_str}</div>
              <div class='metric-lbl'>Histórico disponível</div>
            </div>""", unsafe_allow_html=True)

        # Barra de progresso
        st.markdown(f"""
        <div style='margin:16px 0 4px'>
          <div style='display:flex;justify-content:space-between;
               font-size:.75rem;color:#9ca3af;margin-bottom:4px'>
            <span>Simulações usadas este mês</span>
            <span>{pct_used:.0f}%</span>
          </div>
          <div style='background:#1f2937;border-radius:6px;height:10px'>
            <div style='background:{bar_color};width:{min(pct_used,100):.0f}%;
                 height:10px;border-radius:6px;transition:.3s'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if pct_used >= 90:
            st.warning("⚠️ Você está quase no limite mensal. Considere adquirir créditos ou fazer upgrade.")

        st.markdown("---")
        st.markdown("#### Detalhes do plano")

        modules_str = " · ".join(plan.get("modules", []))
        ccee_str    = "✅ Incluído" if plan.get("ccee_access") else "❌ Não incluído"
        price_str   = f"R$ {plan['price_brl']}/mês" if plan["price_brl"] > 0 else "**Gratuito**"

        st.markdown(f"""
        | Item | Detalhe |
        |---|---|
        | Plano | {picon} **{plan['label']}** |
        | Valor | {price_str} |
        | Simulações/mês | {plan['simulation_limit']} |
        | Histórico | {hist_str} |
        | API CCEE | {ccee_str} |
        | Módulos | {modules_str} |
        """)

        if user.is_free:
            st.info(
                "Você está no plano gratuito. "
                "Faça upgrade para acessar histórico completo, mais simulações e API CCEE."
            )

    # ════════════════════════════════════════════════════════════════════════════
    with ptabs[1]:  # Upgrade / Créditos
    # ════════════════════════════════════════════════════════════════════════════
        col_up, col_cr = st.columns(2)

        with col_up:
            st.markdown("#### Upgrade de plano")
            next_plans = {
                "free":         ["analyst", "professional", "institutional"],
                "analyst":      ["professional", "institutional"],
                "professional": ["institutional"],
                "institutional": [],
            }.get(user.plan_type, [])

            if not next_plans:
                st.success("🏆 Você está no plano mais completo!")
            else:
                for p in next_plans:
                    info   = PLANS[p]
                    pcolor2 = _COLORS.get(p, "#9ca3af")
                    icon   = _PLAN_ICONS.get(p, "📋")
                    st.markdown(f"""
                    <div class='upgrade-card'>
                      <div style='color:{pcolor2};font-weight:700'>{icon} {info['label']}</div>
                      <div style='font-size:.85rem;color:#e5e7eb;margin:4px 0'>
                        R$ {info['price_brl']}/mês
                      </div>
                      <div style='font-size:.75rem;color:#9ca3af'>{info['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(
                    "Para fazer upgrade, entre em contato via "
                    "**maatriaenergia@gmail.com** informando o plano desejado. "
                    "O acesso é liberado após confirmação do pagamento."
                )

        with col_cr:
            st.markdown("#### Créditos adicionais")
            st.caption("Use após esgotar o limite mensal do plano.")

            for pkg in CREDIT_PACKAGES:
                st.markdown(f"""
                <div class='credit-card'>
                  <span style='color:#a78bfa;font-weight:700'>
                    💎 {pkg['credits']} créditos
                  </span>
                  <span style='color:#c8a44d;font-weight:700;float:right'>
                    R$ {pkg['price_brl']}
                  </span>
                  <br>
                  <span style='font-size:.72rem;color:#6b7280'>
                    R$ {pkg['price_brl']/pkg['credits']:.2f} por crédito
                  </span>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "Para adquirir créditos, entre em contato: "
                "**maatriaenergia@gmail.com**"
            )

        # Painel admin — apenas institutional pode adicionar créditos manualmente
        if user.plan_type == "institutional":
            st.markdown("---")
            st.markdown("#### 🔧 Painel admin — adicionar créditos (uso interno)")
            adm_email  = st.text_input("Email do usuário", key="adm_email")
            adm_n      = st.number_input("Quantidade de créditos", min_value=1,
                                         max_value=1000, value=10, key="adm_credits")
            adm_reason = st.text_input("Motivo", value="manual", key="adm_reason")
            if st.button("Adicionar créditos", key="btn_adm_credits"):
                from monetization import _db_fetchone
                row = _db_fetchone("SELECT id FROM maat_users WHERE email = %s",
                                   (adm_email.strip().lower(),))
                if row:
                    ok, msg = add_credits(row[0], int(adm_n), adm_reason)
                    st.success(msg) if ok else st.error(msg)
                else:
                    st.error("Usuário não encontrado.")

    # ════════════════════════════════════════════════════════════════════════════
    with ptabs[2]:  # Segurança
    # ════════════════════════════════════════════════════════════════════════════
        st.markdown("#### Alterar senha")
        with st.form("form_change_pwd"):
            old_pwd  = st.text_input("Senha atual", type="password")
            new_pwd  = st.text_input("Nova senha (mín. 8 caracteres)", type="password")
            new_pwd2 = st.text_input("Confirmar nova senha", type="password")
            submit   = st.form_submit_button("Alterar senha", type="primary")

        if submit:
            if not old_pwd or not new_pwd or not new_pwd2:
                st.error("Preencha todos os campos.")
            elif new_pwd != new_pwd2:
                st.error("As senhas não conferem.")
            elif len(new_pwd) < 8:
                st.error("Nova senha deve ter ao menos 8 caracteres.")
            else:
                ok, msg = change_password(user.id, old_pwd, new_pwd)
                st.success(msg) if ok else st.error(msg)

        st.markdown("---")
        st.markdown("#### Informações de segurança")
        st.markdown(f"""
        - **Email:** {user.email}
        - **Conta criada em:** {user.created_at.strftime('%d/%m/%Y %H:%M') if user.created_at else 'N/D'}
        - **Plano atual:** {PLANS.get(user.plan_type, {}).get('label', user.plan_type)}
        - **Status:** {'✅ Ativa' if user.is_active else '❌ Inativa'}
        """)

    # ════════════════════════════════════════════════════════════════════════════
    with ptabs[3]:  # Histórico de Uso
    # ════════════════════════════════════════════════════════════════════════════
        st.markdown("#### Atividade recente")

        period = st.radio("Período", ["7 dias", "30 dias", "90 dias"],
                          horizontal=True, index=1, key="hist_period")
        days   = {"7 dias": 7, "30 dias": 30, "90 dias": 90}[period]

        stats = get_usage_stats(user.id, days=days)
        if stats:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Simulações", stats.get("total_simulations", 0))
            k2.metric("Via crédito", stats.get("simulations_credit", 0))
            k3.metric("Via plano",   stats.get("simulations_plan", 0))
            avg = stats.get("avg_processing_ms", 0)
            k4.metric("Tempo médio", f"{avg:.0f} ms" if avg else "N/D")

            last = stats.get("last_activity")
            if last:
                st.caption(f"Última atividade: {last}")
        else:
            st.info(f"Sem atividade registrada nos últimos {days} dias.")

        # Log detalhado
        from monetization import _db_fetchone as _dbo
        try:
            import psycopg2, os
            url = os.getenv("DATABASE_URL_AUTH", "")
            if url:
                conn = psycopg2.connect(url)
                cur  = conn.cursor()
                cur.execute(
                    """
                    SELECT ts, action_type, used_credit, api_calls_made, processing_ms
                    FROM maat_usage_log
                    WHERE user_id = %s AND ts >= NOW() - INTERVAL '%s days'
                    ORDER BY ts DESC LIMIT 50
                    """,
                    (user.id, days),
                )
                rows = cur.fetchall()
                cur.close(); conn.close()

                if rows:
                    log_df = pd.DataFrame(rows, columns=[
                        "Data/Hora", "Ação", "Via Crédito", "API Calls", "Tempo (ms)"])
                    log_df["Via Crédito"] = log_df["Via Crédito"].map(
                        {True: "💎 crédito", False: "📋 plano"})
                    log_df["Data/Hora"] = pd.to_datetime(
                        log_df["Data/Hora"]).dt.strftime("%d/%m/%Y %H:%M")
                    st.dataframe(log_df, use_container_width=True, height=320)
                else:
                    st.caption("Sem registros no período.")
        except Exception:
            st.caption("Log detalhado indisponível.")
