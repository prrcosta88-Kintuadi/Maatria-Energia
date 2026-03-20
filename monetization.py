# -*- coding: utf-8 -*-
"""
monetization.py — Hybrid Monetization Engine (MAÁTria Energia)
════════════════════════════════════════════════════════════════
Controla planos, uso de simulações, créditos e logging.

Três camadas:
    SaaS         : assinatura mensal (Analyst / Professional / Institutional)
    Usage-based  : créditos adicionais após esgotar o plano
    Institucional: limite alto, todos os módulos

Banco: Neon PostgreSQL — apenas tabelas maat_users, maat_subscriptions,
       maat_usage_log. Storage mínimo (sem dados operacionais).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# ─── Conexão direta ao banco AUTH (DATABASE_URL_AUTH) ────────────────────────
# Banco separado: maatriaenergia-ccee (500MB livre)
# Não usa db_neon para não misturar com dados operacionais ONS
_DB_OK = True  # sempre tenta; falha silenciosa em _db_fetchone/_db_execute


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DOS PLANOS
# ══════════════════════════════════════════════════════════════════════════════

PLANS: Dict[str, Dict[str, Any]] = {
    "free": {
        "label":            "Free",
        "price_brl":        0,
        "simulation_limit": 2,
        "ccee_access":      False,
        "history_days":     30,        # restrição: apenas 30 dias de histórico
        "modules":          ["dashboard"],
        "description":      "Dashboard + 2 simulações/mês + histórico 30 dias",
    },
    "analyst": {
        "label":            "Analyst",
        "price_brl":        99,
        "simulation_limit": 5,
        "ccee_access":      False,
        "history_days":     None,      # sem restrição
        "modules":          ["dashboard"],
        "description":      "Dashboard + 5 simulações/mês",
    },
    "professional": {
        "label":            "Professional",
        "price_brl":        400,
        "simulation_limit": 30,
        "ccee_access":      True,
        "modules":          ["dashboard", "premium", "ccee"],
        "description":      "Acesso completo + 30 simulações/mês + API CCEE",
    },
    "institutional": {
        "label":            "Institutional",
        "price_brl":        2000,
        "simulation_limit": 500,
        "ccee_access":      True,
        "modules":          ["dashboard", "premium", "ccee", "custom"],
        "description":      "Simulações ilimitadas + prioridade + todos os módulos",
    },
}

CREDIT_PACKAGES = [
    {"credits": 10,  "price_brl": 50,  "label": "10 créditos — R$ 50"},
    {"credits": 50,  "price_brl": 200, "label": "50 créditos — R$ 200"},
    {"credits": 100, "price_brl": 350, "label": "100 créditos — R$ 350"},
]

# Rate limit: max chamadas API CCEE por usuário por hora
_CCEE_RATE_LIMIT_PER_HOUR = 20

# Cache em memória por sessão (evita re-fetch dentro de 5 minutos)
_CCEE_CACHE_TTL_SECONDS = 300


# ══════════════════════════════════════════════════════════════════════════════
# MODELO DE USUÁRIO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class User:
    id:                             int
    email:                          str
    plan_type:                      str
    simulations_used_current_month: int
    simulation_limit:               int
    credits_balance:                int
    is_active:                      bool
    created_at:                     datetime
    last_reset_at:                  datetime
    # Não persistido — apenas na sessão
    _api_calls_this_hour:           int = field(default=0, repr=False)
    _api_hour_start:                float = field(default_factory=time.time, repr=False)

    @property
    def plan(self) -> Dict[str, Any]:
        return PLANS.get(self.plan_type, PLANS["analyst"])

    @property
    def simulations_remaining(self) -> int:
        return max(0, self.simulation_limit - self.simulations_used_current_month)

    @property
    def has_ccee_access(self) -> bool:
        return self.plan.get("ccee_access", False)

    @property
    def history_days(self) -> Optional[int]:
        """None = sem restrição; int = limite em dias."""
        return self.plan.get("history_days", None)

    @property
    def is_free(self) -> bool:
        return self.plan_type == "free"

    @property
    def is_institutional(self) -> bool:
        return self.plan_type == "institutional"


# ══════════════════════════════════════════════════════════════════════════════
# ACESSO AO BANCO
# ══════════════════════════════════════════════════════════════════════════════

def _db_fetchone(sql: str, params: tuple = ()) -> Optional[tuple]:
    """Busca no banco AUTH (maatriaenergia-ccee)."""
    try:
        import psycopg2
        url = _get_auth_url()
        if not url:
            return None
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row
    except Exception:
        return None


def _get_auth_url() -> str:
    """Retorna a URL do banco de autenticação/monetização (maatriaenergia-ccee)."""
    return os.getenv("DATABASE_URL_AUTH", os.getenv("DATABASE_URL", ""))


def _db_execute(sql: str, params: tuple = ()) -> bool:
    """Executa DDL/DML no banco AUTH. Retorna True se sucesso."""
    try:
        import psycopg2
        url = _get_auth_url()
        if not url:
            return False
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(sql, params)
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# AUTENTICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate(email: str, password: str) -> Optional[User]:
    """
    Verifica credenciais contra maat_users no Neon.
    Retorna User se válido e ativo, None caso contrário.
    """
    if not email or not password:
        return None

    pwd_hash = _hash_password(password)
    row = _db_fetchone(
        """
        SELECT id, email, plan_type,
               simulations_used_current_month, simulation_limit,
               credits_balance, is_active, created_at, last_reset_at
        FROM maat_users
        WHERE email = %s AND password_hash = %s AND is_active = TRUE
        """,
        (email.strip().lower(), pwd_hash),
    )
    if not row:
        return None

    _reset_if_needed(row[0], row[8])  # reset mensal automático

    # Re-fetch após possível reset
    row = _db_fetchone(
        """
        SELECT id, email, plan_type,
               simulations_used_current_month, simulation_limit,
               credits_balance, is_active, created_at, last_reset_at
        FROM maat_users WHERE id = %s
        """,
        (row[0],),
    )
    if not row:
        return None

    return User(
        id=row[0], email=row[1], plan_type=row[2],
        simulations_used_current_month=row[3],
        simulation_limit=row[4], credits_balance=row[5],
        is_active=row[6], created_at=row[7], last_reset_at=row[8],
    )


def create_user(email: str, password: str,
                plan_type: str = "free") -> Tuple[bool, str]:
    """
    Cria novo usuário no Neon.
    Retorna (sucesso, mensagem).
    """
    if not email or not password:
        return False, "Email e senha obrigatórios."
    if plan_type not in PLANS:
        return False, f"Plano inválido: {plan_type}"

    pwd_hash = _hash_password(password)
    sim_limit = PLANS[plan_type]["simulation_limit"]

    existing = _db_fetchone("SELECT id FROM maat_users WHERE email = %s",
                            (email.strip().lower(),))
    if existing:
        return False, "Email já cadastrado."

    ok = _db_execute(
        """
        INSERT INTO maat_users
            (email, password_hash, plan_type, simulation_limit)
        VALUES (%s, %s, %s, %s)
        """,
        (email.strip().lower(), pwd_hash, plan_type, sim_limit),
    )
    return (True, "Usuário criado com sucesso.") if ok \
           else (False, "Erro ao criar usuário. Tente novamente.")


def change_password(user_id: int, old_password: str,
                    new_password: str) -> Tuple[bool, str]:
    """Troca senha verificando a senha atual."""
    row = _db_fetchone("SELECT password_hash FROM maat_users WHERE id = %s",
                       (user_id,))
    if not row:
        return False, "Usuário não encontrado."
    if row[0] != _hash_password(old_password):
        return False, "Senha atual incorreta."
    ok = _db_execute("UPDATE maat_users SET password_hash = %s WHERE id = %s",
                     (_hash_password(new_password), user_id))
    return (True, "Senha alterada.") if ok else (False, "Erro ao alterar senha.")


# ══════════════════════════════════════════════════════════════════════════════
# CONTROLE DE SIMULAÇÕES (USAGE-BASED)
# ══════════════════════════════════════════════════════════════════════════════

def can_run_simulation(user: User) -> Tuple[bool, str]:
    """
    Verifica se o usuário pode executar uma simulação.
    Retorna (pode, motivo).
    """
    if not user.is_active:
        return False, "Conta inativa."
    if user.simulations_used_current_month < user.simulation_limit:
        remaining = user.simulation_limit - user.simulations_used_current_month
        return True, f"{remaining} simulação(ões) restante(s) no plano"
    if user.credits_balance > 0:
        return True, f"{user.credits_balance} crédito(s) disponível(is)"
    return False, (
        f"Limite do plano atingido ({user.simulation_limit}/mês) e sem créditos. "
        "Faça upgrade ou adquira créditos adicionais."
    )


def consume_simulation(user_id: int, sim_type: str = "price_justification",
                       api_calls: int = 0,
                       processing_ms: Optional[int] = None) -> Tuple[bool, str]:
    """
    Consome 1 unidade de simulação (plano ou crédito).
    Registra log de uso. Retorna (sucesso, tipo_consumido).
    """
    row = _db_fetchone(
        "SELECT simulations_used_current_month, simulation_limit, credits_balance "
        "FROM maat_users WHERE id = %s AND is_active = TRUE",
        (user_id,),
    )
    if not row:
        return False, "Usuário não encontrado."

    used, limit, credits = row
    used_credit = False

    if used < limit:
        ok = _db_execute(
            "UPDATE maat_users SET simulations_used_current_month = simulations_used_current_month + 1 "
            "WHERE id = %s",
            (user_id,),
        )
    elif credits > 0:
        ok = _db_execute(
            "UPDATE maat_users SET credits_balance = credits_balance - 1 WHERE id = %s",
            (user_id,),
        )
        used_credit = True
    else:
        return False, "Sem simulações disponíveis."

    if ok:
        _log_usage(user_id, sim_type, used_credit, api_calls, processing_ms)
        return True, "credit" if used_credit else "plan"
    return False, "Erro ao registrar consumo."


def _reset_if_needed(user_id: int, last_reset_at: datetime) -> None:
    """Reset mensal automático — verifica se passou um mês desde o último reset."""
    now = datetime.utcnow()
    if last_reset_at is None:
        last_reset_at = now - timedelta(days=32)

    # Normalizar timezone
    if hasattr(last_reset_at, 'tzinfo') and last_reset_at.tzinfo is not None:
        import pytz
        last_reset_at = last_reset_at.replace(tzinfo=None)

    # Passou de mês calendário?
    if (now.year, now.month) > (last_reset_at.year, last_reset_at.month):
        _db_execute(
            """
            UPDATE maat_users
            SET simulations_used_current_month = 0,
                last_reset_at = NOW()
            WHERE id = %s
            """,
            (user_id,),
        )


# ══════════════════════════════════════════════════════════════════════════════
# RECUPERAÇÃO DE SENHA VIA EMAIL (Resend API)
# ══════════════════════════════════════════════════════════════════════════════

import secrets as _secrets

def generate_reset_token(email: str) -> Optional[str]:
    """
    Gera token de recuperação de senha (6 dígitos) e armazena no banco.
    Válido por 30 minutos.
    Retorna o token ou None se email não encontrado.
    """
    email = email.strip().lower()
    row = _db_fetchone("SELECT id FROM maat_users WHERE email = %s AND is_active = TRUE",
                       (email,))
    if not row:
        return None  # não revelar se email existe

    token = str(_secrets.randbelow(900000) + 100000)  # 6 dígitos
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Salvar token com expiração (30 min) — reutilizando maat_usage_log como store
    _db_execute(
        """
        INSERT INTO maat_usage_log (user_id, action_type, metadata)
        VALUES (%s, 'password_reset_token', %s)
        """,
        (row[0], json.dumps({"token_hash": token_hash,
                              "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()})),
    )
    return token


def verify_reset_token(email: str, token: str) -> Optional[int]:
    """
    Verifica token de recuperação.
    Retorna user_id se válido e não expirado, None caso contrário.
    """
    email = email.strip().lower()
    token_hash = hashlib.sha256(token.strip().encode()).hexdigest()

    row = _db_fetchone("SELECT id FROM maat_users WHERE email = %s", (email,))
    if not row:
        return None
    user_id = row[0]

    # Buscar token mais recente não usado
    log_row = _db_fetchone(
        """
        SELECT id, metadata FROM maat_usage_log
        WHERE user_id = %s AND action_type = 'password_reset_token'
        ORDER BY ts DESC LIMIT 1
        """,
        (user_id,),
    )
    if not log_row:
        return None

    try:
        meta = json.loads(log_row[1] or "{}")
        if meta.get("token_hash") != token_hash:
            return None
        expires_at = datetime.fromisoformat(meta.get("expires_at", "2000-01-01"))
        if datetime.utcnow() > expires_at:
            return None
        # Invalidar token após uso
        _db_execute(
            "DELETE FROM maat_usage_log WHERE id = %s", (log_row[0],))
        return user_id
    except Exception:
        return None


def reset_password_with_token(email: str, token: str,
                               new_password: str) -> Tuple[bool, str]:
    """Redefine senha após verificar token."""
    if len(new_password) < 8:
        return False, "Senha deve ter ao menos 8 caracteres."
    user_id = verify_reset_token(email, token)
    if not user_id:
        return False, "Token inválido ou expirado. Solicite um novo código."
    ok = _db_execute(
        "UPDATE maat_users SET password_hash = %s WHERE id = %s",
        (hashlib.sha256(new_password.encode()).hexdigest(), user_id),
    )
    return (True, "Senha redefinida com sucesso.") if ok            else (False, "Erro ao redefinir senha.")


def send_reset_email(email: str, token: str) -> Tuple[bool, str]:
    """
    Envia email com token de recuperação via Resend API.
    Requer RESEND_API_KEY no ambiente.
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    from_addr = os.getenv("RESEND_FROM", "MAÁTria Energia <onboarding@resend.dev>")
    if not api_key:
        return False, "RESEND_API_KEY não configurada."

    import requests as _req
    body = {
        "from":    from_addr,
        "to":      [email],
        "subject": "MAÁTria Energia — Código de recuperação de senha",
        "html": f"""
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto">
          <h2 style="color:#c8a44d">MAÁTria Energia</h2>
          <p>Recebemos uma solicitação para redefinir sua senha.</p>
          <p>Seu código de verificação é:</p>
          <div style="background:#111827;border:2px solid #c8a44d;border-radius:8px;
               padding:20px;text-align:center;margin:20px 0">
            <span style="font-size:2.5rem;font-weight:700;color:#c8a44d;
                  letter-spacing:.3em">{token}</span>
          </div>
          <p style="color:#6b7280;font-size:.85rem">
            Este código é válido por <strong>30 minutos</strong>.<br>
            Se você não solicitou a redefinição, ignore este email.
          </p>
        </div>
        """,
    }
    try:
        resp = _req.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=10,
        )
        if resp.status_code in (200, 201):
            return True, "Email enviado."
        return False, f"Erro Resend: {resp.status_code}"
    except Exception as e:
        return False, str(e)


def request_password_reset(email: str) -> Tuple[bool, str]:
    """
    Fluxo completo: gerar token + enviar email.
    Sempre retorna mensagem genérica (não revela se email existe).
    """
    token = generate_reset_token(email)
    if token:
        ok, msg = send_reset_email(email.strip().lower(), token)
        if not ok:
            return False, f"Erro ao enviar email: {msg}"
    # Mensagem genérica independente de o email existir
    return True, "Se o email estiver cadastrado, você receberá o código em breve."


def reset_all_monthly(dry_run: bool = False) -> int:
    """
    Rotina de reset mensal para todos os usuários.
    Chamar via scheduler (ex: update_neon.py às 00h00 do dia 1).
    Retorna número de usuários resetados.
    """
    row = _db_fetchone(
        "SELECT COUNT(*) FROM maat_users WHERE is_active = TRUE "
        "AND date_trunc('month', last_reset_at) < date_trunc('month', NOW())",
        (),
    )
    n = row[0] if row else 0
    if dry_run:
        return n
    if n > 0:
        _db_execute(
            "UPDATE maat_users SET simulations_used_current_month = 0, "
            "last_reset_at = NOW() "
            "WHERE is_active = TRUE "
            "AND date_trunc('month', last_reset_at) < date_trunc('month', NOW())",
            (),
        )
    return n


# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE CRÉDITOS
# ══════════════════════════════════════════════════════════════════════════════

def add_credits(user_id: int, n_credits: int,
                reason: str = "purchase") -> Tuple[bool, str]:
    """Adiciona créditos ao usuário (após confirmação de pagamento)."""
    if n_credits <= 0:
        return False, "Quantidade inválida."
    ok = _db_execute(
        "UPDATE maat_users SET credits_balance = credits_balance + %s WHERE id = %s",
        (n_credits, user_id),
    )
    if ok:
        _log_usage(user_id, f"credit_add:{reason}", False, 0, None)
        return True, f"{n_credits} crédito(s) adicionado(s)."
    return False, "Erro ao adicionar créditos."


def upgrade_plan(user_id: int, new_plan: str) -> Tuple[bool, str]:
    """Muda plano do usuário e atualiza limite de simulações."""
    if new_plan not in PLANS:
        return False, f"Plano inválido: {new_plan}"
    new_limit = PLANS[new_plan]["simulation_limit"]
    ok = _db_execute(
        "UPDATE maat_users SET plan_type = %s, simulation_limit = %s WHERE id = %s",
        (new_plan, new_limit, user_id),
    )
    return (True, f"Plano atualizado para {PLANS[new_plan]['label']}.") if ok \
           else (False, "Erro ao atualizar plano.")


# ══════════════════════════════════════════════════════════════════════════════
# CONTROLE DE CUSTO — API CCEE
# ══════════════════════════════════════════════════════════════════════════════

def check_ccee_access(user: User) -> Tuple[bool, str]:
    """Verifica se usuário pode acessar API CCEE e se não atingiu rate limit."""
    if not user.has_ccee_access:
        return False, (f"Acesso à API CCEE disponível apenas nos planos "
                       f"Professional e Institutional.")

    # Rate limit por hora — controle em session_state
    now = time.time()
    calls_key  = f"ccee_calls_{user.id}"
    start_key  = f"ccee_hour_start_{user.id}"

    hour_start = st.session_state.get(start_key, now)
    calls      = st.session_state.get(calls_key, 0)

    # Reset se passou 1 hora
    if now - hour_start > 3600:
        st.session_state[start_key] = now
        st.session_state[calls_key] = 0
        calls = 0

    if calls >= _CCEE_RATE_LIMIT_PER_HOUR:
        remaining = int(3600 - (now - hour_start))
        return False, (f"Rate limit atingido ({_CCEE_RATE_LIMIT_PER_HOUR} calls/h). "
                       f"Aguarde {remaining//60}min {remaining%60}s.")

    st.session_state[calls_key] = calls + 1
    return True, f"OK ({calls + 1}/{_CCEE_RATE_LIMIT_PER_HOUR} calls/h)"


def get_ccee_cache(cache_key: str) -> Optional[Any]:
    """Cache em memória (session_state) com TTL de 5 minutos."""
    entry = st.session_state.get(f"ccee_cache_{cache_key}")
    if not entry:
        return None
    data, ts = entry
    if time.time() - ts > _CCEE_CACHE_TTL_SECONDS:
        del st.session_state[f"ccee_cache_{cache_key}"]
        return None
    return data


def set_ccee_cache(cache_key: str, data: Any) -> None:
    """Armazena no cache de sessão (em memória — sem persistência)."""
    st.session_state[f"ccee_cache_{cache_key}"] = (data, time.time())


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _log_usage(user_id: int, action_type: str, used_credit: bool,
               api_calls: int, processing_ms: Optional[int],
               metadata: Optional[dict] = None) -> None:
    """Registra uso no maat_usage_log. Falha silenciosa."""
    meta_str = json.dumps(metadata) if metadata else None
    _db_execute(
        """
        INSERT INTO maat_usage_log
            (user_id, action_type, used_credit, api_calls_made,
             processing_ms, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (user_id, action_type, used_credit, api_calls,
         processing_ms, meta_str),
    )
    # Purge automático: remover logs > 90 dias (a cada 1000 chamadas, aleatório)
    import random
    if random.randint(1, 1000) == 1:
        _db_execute(
            "DELETE FROM maat_usage_log WHERE ts < NOW() - INTERVAL '90 days'",
            (),
        )


def get_usage_stats(user_id: int, days: int = 30) -> Dict[str, Any]:
    """Retorna estatísticas de uso do usuário nos últimos N dias."""
    row = _db_fetchone(
        """
        SELECT
            COUNT(*) FILTER (WHERE action_type LIKE '%simulation%')      AS total_sims,
            COUNT(*) FILTER (WHERE used_credit = TRUE)                    AS sims_via_credit,
            SUM(api_calls_made)                                           AS total_api_calls,
            AVG(processing_ms) FILTER (WHERE processing_ms IS NOT NULL)   AS avg_ms,
            MAX(ts)                                                        AS last_activity
        FROM maat_usage_log
        WHERE user_id = %s AND ts >= NOW() - INTERVAL '%s days'
        """,
        (user_id, days),
    )
    if not row:
        return {}
    return {
        "total_simulations":  int(row[0] or 0),
        "simulations_credit": int(row[1] or 0),
        "simulations_plan":   int(row[0] or 0) - int(row[1] or 0),
        "total_api_calls":    int(row[2] or 0),
        "avg_processing_ms":  round(float(row[3] or 0), 1),
        "last_activity":      row[4],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUTURA PARA PAGAMENTOS (PREPARADA — integração futura)
# ══════════════════════════════════════════════════════════════════════════════

def create_subscription(user_id: int, plan_type: str,
                        payment_provider: str = "manual",
                        payment_provider_id: Optional[str] = None,
                        renewal_date: Optional[str] = None) -> Tuple[bool, str]:
    """
    Registra assinatura. Chamar após confirmação de pagamento pelo provedor.
    Compatível com Stripe, Mercado Pago ou ativação manual.
    """
    ok = _db_execute(
        """
        INSERT INTO maat_subscriptions
            (user_id, plan_type, status, renewal_date,
             payment_provider, payment_provider_id)
        VALUES (%s, %s, 'active', %s, %s, %s)
        """,
        (user_id, plan_type, renewal_date, payment_provider, payment_provider_id),
    )
    if ok:
        upgrade_plan(user_id, plan_type)
        return True, "Assinatura registrada."
    return False, "Erro ao registrar assinatura."


def cancel_subscription(user_id: int) -> Tuple[bool, str]:
    """Cancela assinatura ativa do usuário."""
    ok = _db_execute(
        """
        UPDATE maat_subscriptions SET status = 'cancelled', updated_at = NOW()
        WHERE user_id = %s AND status = 'active'
        """,
        (user_id,),
    )
    if ok:
        _db_execute(
            "UPDATE maat_users SET plan_type = 'analyst', simulation_limit = 5 "
            "WHERE id = %s",
            (user_id,),
        )
        return True, "Assinatura cancelada. Plano revertido para Analyst."
    return False, "Nenhuma assinatura ativa encontrada."


# ══════════════════════════════════════════════════════════════════════════════
# FAIR USE — PRIORIZAÇÃO INSTITUCIONAL
# ══════════════════════════════════════════════════════════════════════════════

def get_simulation_priority(user: User) -> int:
    """
    Retorna prioridade de processamento (menor = mais prioritário).
    Institutional: 1, Professional: 2, Analyst: 3
    """
    return {"institutional": 1, "professional": 2, "analyst": 3}.get(user.plan_type, 3)


def estimate_compute_cost(n_simulations: int,
                          model_complexity: str = "standard") -> float:
    """
    Estima custo computacional relativo de uma simulação.
    Usado para fair-use e monitoramento de infraestrutura.
    Retorna score entre 0 e 1 (1 = máximo custo).
    """
    base = {"standard": 0.3, "extended": 0.6, "full": 1.0}.get(model_complexity, 0.3)
    sim_factor = min(n_simulations / 10_000, 1.0)
    return round(base * sim_factor, 3)
