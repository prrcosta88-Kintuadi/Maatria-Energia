# -*- coding: utf-8 -*-
"""
auth.py — Authentication & Session Management (MAÁTria Energia)
════════════════════════════════════════════════════════════════
Substitui o sistema de PREMIUM_USERS (env var) por autenticação
via banco Neon com controle de plano e uso.

Integração com monetization.py — render_login() retorna User completo
com plano, créditos e limites.
"""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from monetization import (
    User, PLANS, CREDIT_PACKAGES,
    authenticate, create_user, change_password,
    can_run_simulation, consume_simulation,
    add_credits, upgrade_plan,
    check_ccee_access, get_usage_stats,
    get_simulation_priority,
)

# ─── CSS compartilhado ────────────────────────────────────────────────────────
_LOGIN_CSS = """
<style>
.maat-login-wrap{max-width:440px;margin:60px auto 0;background:#111827;
    border:1px solid #c8a44d44;border-radius:16px;padding:40px 36px}
.maat-login-title{text-align:center;font-size:1.25rem;font-weight:700;
    color:#c8a44d;letter-spacing:.07em;text-transform:uppercase;margin:6px 0 4px}
.maat-login-sub{text-align:center;font-size:.78rem;color:#6b7280;margin-bottom:24px}
.maat-badge{display:inline-block;background:#c8a44d22;border:1px solid #c8a44d55;
    color:#c8a44d;font-size:.65rem;font-weight:700;letter-spacing:.14em;
    padding:2px 10px;border-radius:4px;text-transform:uppercase}
.plan-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
    padding:12px 14px;margin:4px 0}
.plan-name{color:#60a5fa;font-weight:700;font-size:.95rem}
.plan-price{color:#c8a44d;font-size:1.1rem;font-weight:700}
.plan-desc{color:#9ca3af;font-size:.78rem;margin-top:2px}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# RENDER LOGIN — ponto de entrada do app
# ══════════════════════════════════════════════════════════════════════════════

def render_login() -> bool:
    """
    Renderiza tela de login/cadastro.
    Retorna True se usuário está autenticado, False caso contrário.

    Substituição direta de _render_login() no app_premium.py.
    """
    if "maat_user" not in st.session_state:
        st.session_state["maat_user"] = None
    if "maat_auth_tab" not in st.session_state:
        st.session_state["maat_auth_tab"] = "login"

    user: Optional[User] = st.session_state.get("maat_user")
    if user is not None and user.is_active:
        return True

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.7, 1])
    with col:
        st.markdown("""
        <div class='maat-login-wrap'>
          <div style='text-align:center;margin-bottom:10px'>
            <span class='maat-badge'>✦ MAÁTria Energia</span>
          </div>
          <div class='maat-login-title'>Plataforma Premium</div>
          <div class='maat-login-sub'>Análise do Mercado Elétrico Brasileiro</div>
        </div>
        """, unsafe_allow_html=True)

        auth_tab = st.radio("", ["Entrar", "Criar conta"],
                            horizontal=True, label_visibility="collapsed",
                            key="auth_tab_radio")

        if auth_tab == "Entrar":
            _render_login_form()
        else:
            _render_register_form()

    return False


def _render_login_form() -> None:
    if "login_mode" not in st.session_state:
        st.session_state["login_mode"] = "login"
    if "reset_step" not in st.session_state:
        st.session_state["reset_step"] = 1

    if st.session_state["login_mode"] == "forgot":
        _render_forgot_password()
        return

    email    = st.text_input("Email", placeholder="seu@email.com",
                             key="login_email", label_visibility="collapsed")
    password = st.text_input("Senha", type="password", placeholder="senha",
                             key="login_pwd", label_visibility="collapsed")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Entrar →", type="primary", use_container_width=True, key="btn_login"):
        if not email or not password:
            st.error("Preencha email e senha.")
            return
        with st.spinner("Verificando..."):
            user = authenticate(email, password)
        if user:
            st.session_state["maat_user"] = user
            st.rerun()
        else:
            st.error("Email ou senha incorretos, ou conta inativa.")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("Esqueci minha senha", use_container_width=True,
                 key="btn_forgot", type="secondary"):
        st.session_state["login_mode"] = "forgot"
        st.session_state["reset_step"] = 1
        st.rerun()

    st.markdown(
        "<div style='text-align:center;margin-top:12px;font-size:.72rem;color:#4b5563'>"
        "Dúvidas: maatriaenergia@gmail.com</div>",
        unsafe_allow_html=True,
    )


def _render_forgot_password() -> None:
    step = st.session_state.get("reset_step", 1)
    st.markdown(
        "<div style='text-align:center;color:#c8a44d;font-weight:700;"
        "margin-bottom:12px'>🔑 Recuperar senha</div>",
        unsafe_allow_html=True,
    )
    if step == 1:
        st.caption("Insira seu email. Enviaremos um código de 6 dígitos válido por 30 min.")
        reset_email = st.text_input("Email", placeholder="seu@email.com",
                                    key="reset_email", label_visibility="collapsed")
        if st.button("Enviar código →", type="primary",
                     use_container_width=True, key="btn_send_code"):
            if not reset_email:
                st.error("Informe seu email.")
            else:
                with st.spinner("Enviando..."):
                    ok, msg = request_password_reset(reset_email.strip())
                st.success(msg)
                if ok:
                    st.session_state["reset_email_used"] = reset_email.strip()
                    st.session_state["reset_step"] = 2
                    st.rerun()
    elif step == 2:
        saved_email = st.session_state.get("reset_email_used", "")
        st.caption(f"Código enviado para **{saved_email}**.")
        token    = st.text_input("Código de 6 dígitos", placeholder="000000",
                                 max_chars=6, key="reset_token",
                                 label_visibility="collapsed")
        new_pwd  = st.text_input("Nova senha", type="password",
                                 placeholder="nova senha (mín. 8 chars)",
                                 key="reset_new_pwd", label_visibility="collapsed")
        new_pwd2 = st.text_input("Confirmar nova senha", type="password",
                                 placeholder="confirme",
                                 key="reset_new_pwd2", label_visibility="collapsed")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Redefinir senha →", type="primary",
                     use_container_width=True, key="btn_reset_pwd"):
            if not token or not new_pwd or not new_pwd2:
                st.error("Preencha todos os campos.")
            elif new_pwd != new_pwd2:
                st.error("As senhas não conferem.")
            elif len(new_pwd) < 8:
                st.error("Senha deve ter ao menos 8 caracteres.")
            else:
                with st.spinner("Verificando código..."):
                    ok, msg = reset_password_with_token(saved_email, token, new_pwd)
                if ok:
                    st.success(f"✅ {msg} Faça login com a nova senha.")
                    st.session_state["login_mode"] = "login"
                    st.session_state["reset_step"] = 1
                    st.rerun()
                else:
                    st.error(msg)
        if st.button("← Reenviar código", key="btn_resend"):
            st.session_state["reset_step"] = 1
            st.rerun()

    if st.button("← Voltar ao login", key="btn_back_login", use_container_width=True):
        st.session_state["login_mode"] = "login"
        st.session_state["reset_step"] = 1
        st.rerun()


def _render_register_form() -> None:
    email  = st.text_input("Email", placeholder="seu@email.com",
                           key="reg_email", label_visibility="collapsed")
    pwd1   = st.text_input("Senha", type="password", placeholder="crie uma senha",
                           key="reg_pwd1", label_visibility="collapsed")
    pwd2   = st.text_input("Confirmar senha", type="password",
                           placeholder="confirme a senha",
                           key="reg_pwd2", label_visibility="collapsed")

    # Seletor de plano
    _plan_order  = ["free", "analyst", "professional", "institutional"]
    plan_options = {k: f"{PLANS[k]['label']} — R${PLANS[k]['price_brl']}/mês — {PLANS[k]['description']}"
                    for k in _plan_order if k in PLANS}
    plan_key = st.selectbox("Plano", list(plan_options.keys()),
                            format_func=lambda x: plan_options[x],
                            key="reg_plan")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Criar conta →", type="primary",
                 use_container_width=True, key="btn_register"):
        if not email or not pwd1 or not pwd2:
            st.error("Preencha todos os campos.")
        elif pwd1 != pwd2:
            st.error("Senhas não conferem.")
        elif len(pwd1) < 8:
            st.error("Senha deve ter ao menos 8 caracteres.")
        else:
            ok, msg = create_user(email, pwd1, plan_key)
            if ok:
                st.success(f"{msg} Faça login para continuar.")
            else:
                st.error(msg)

    st.caption("⚠️ Conta criada no plano selecionado. "
               "Pagamento confirmado manualmente após contato.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — perfil e uso
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Sidebar com perfil, uso e botão de logout."""
    user: Optional[User] = st.session_state.get("maat_user")
    if not user:
        return

    with st.sidebar:
        plan = user.plan
        plan_color = {"analyst": "#9ca3af",
                      "professional": "#60a5fa",
                      "institutional": "#c8a44d"}.get(user.plan_type, "#9ca3af")

        st.markdown(f"""
        <div style='background:#111827;border-left:3px solid {plan_color};
        border-radius:6px;padding:10px 12px;margin-bottom:12px'>
        <div style='color:{plan_color};font-weight:700;font-size:.85rem'>
            {plan['label'].upper()}
        </div>
        <div style='color:#9ca3af;font-size:.75rem'>{user.email}</div>
        </div>
        """, unsafe_allow_html=True)

        # Uso mensal
        remaining = user.simulations_remaining
        pct = (user.simulations_used_current_month / user.simulation_limit * 100
               if user.simulation_limit > 0 else 0)
        bar_color = "#f87171" if pct >= 90 else "#c8a44d" if pct >= 70 else "#34d399"

        st.markdown(f"""
        <div style='font-size:.75rem;color:#9ca3af;margin-bottom:4px'>
            Simulações este mês
        </div>
        <div style='background:#1f2937;border-radius:4px;height:6px;margin-bottom:4px'>
            <div style='background:{bar_color};width:{min(pct,100):.0f}%;height:6px;border-radius:4px'></div>
        </div>
        <div style='font-size:.75rem;color:#e5e7eb'>
            {user.simulations_used_current_month}/{user.simulation_limit} usadas
            &nbsp;·&nbsp; {remaining} restantes
        </div>
        """, unsafe_allow_html=True)

        if user.credits_balance > 0:
            st.markdown(
                f"<div style='font-size:.75rem;color:#a78bfa;margin-top:6px'>"
                f"💎 {user.credits_balance} crédito(s) disponível(is)</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Estatísticas rápidas
        with st.expander("📊 Meu uso (30 dias)"):
            stats = get_usage_stats(user.id, days=30)
            if stats:
                st.metric("Simulações", stats.get("total_simulations", 0))
                st.metric("Via crédito", stats.get("simulations_credit", 0))
                st.metric("Calls API CCEE", stats.get("total_api_calls", 0))
                avg_ms = stats.get("avg_processing_ms", 0)
                if avg_ms:
                    st.metric("Tempo médio", f"{avg_ms:.0f} ms")
            else:
                st.caption("Sem dados de uso.")

        # Créditos
        with st.expander("💳 Adquirir créditos"):
            st.caption("Créditos extras após esgotar o plano mensal.")
            for pkg in CREDIT_PACKAGES:
                st.markdown(f"**{pkg['label']}**")
            st.caption("Entre em contato: maatriaenergia@gmail.com")

        # Upgrade
        if user.plan_type != "institutional":
            with st.expander("⬆️ Upgrade de plano"):
                next_plans = {
                    "analyst":      ["professional", "institutional"],
                    "professional": ["institutional"],
                }.get(user.plan_type, [])
                for p in next_plans:
                    info = PLANS[p]
                    st.markdown(f"**{info['label']}** — R${info['price_brl']}/mês")
                    st.caption(info["description"])
                st.caption("Solicite upgrade: maatriaenergia@gmail.com")

        # Mudar senha
        with st.expander("🔑 Alterar senha"):
            old_pwd  = st.text_input("Senha atual", type="password", key="chg_old")
            new_pwd  = st.text_input("Nova senha",  type="password", key="chg_new")
            new_pwd2 = st.text_input("Confirmar",   type="password", key="chg_new2")
            if st.button("Alterar", key="btn_chg_pwd"):
                if new_pwd != new_pwd2:
                    st.error("Senhas não conferem.")
                else:
                    ok, msg = change_password(user.id, old_pwd, new_pwd)
                    st.success(msg) if ok else st.error(msg)

        st.markdown("---")
        if st.button("Sair", use_container_width=True, key="btn_logout"):
            st.session_state["maat_user"] = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# GUARD DE SIMULAÇÃO — decorador para o engine premium
# ══════════════════════════════════════════════════════════════════════════════

def simulation_guard(sim_type: str = "price_justification"):
    """
    Verifica se o usuário pode rodar uma simulação.
    Retorna (pode_rodar, user, mensagem).
    Chame antes de executar run_justification_engine().
    """
    user: Optional[User] = st.session_state.get("maat_user")
    if not user:
        return False, None, "Não autenticado."

    ok, reason = can_run_simulation(user)
    return ok, user, reason


def consume_simulation_guard(user: User,
                              processing_ms: Optional[int] = None,
                              api_calls: int = 0) -> Tuple[bool, str]:
    """
    Consome simulação e atualiza objeto User na sessão.
    Chamar APÓS execução bem-sucedida da simulação.
    """
    from typing import Tuple as _T
    ok, result_type = consume_simulation(
        user.id, sim_type="price_justification",
        api_calls=api_calls, processing_ms=processing_ms,
    )
    if ok:
        # Atualizar objeto na sessão
        if result_type == "credit":
            user.credits_balance = max(0, user.credits_balance - 1)
        else:
            user.simulations_used_current_month += 1
        st.session_state["maat_user"] = user
    return ok, result_type


# ══════════════════════════════════════════════════════════════════════════════
# GUARD DE CCEE — verificação de acesso
# ══════════════════════════════════════════════════════════════════════════════

def ccee_access_guard() -> tuple:
    """
    Verifica se usuário pode acessar API CCEE.
    Retorna (pode, user, mensagem).
    """
    user: Optional[User] = st.session_state.get("maat_user")
    if not user:
        return False, None, "Não autenticado."
    ok, msg = check_ccee_access(user)
    return ok, user, msg


def get_current_user() -> Optional[User]:
    """Retorna usuário da sessão atual."""
    return st.session_state.get("maat_user")
