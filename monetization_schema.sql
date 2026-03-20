-- ============================================================
-- monetization_schema.sql
-- Rodar uma única vez no Neon via psql ou painel web
-- Tabelas pequenas — storage mínimo (dados operacionais)
-- ============================================================

-- ── 1. Usuários ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS maat_users (
    id                              SERIAL PRIMARY KEY,
    email                           TEXT UNIQUE NOT NULL,
    password_hash                   TEXT NOT NULL,
    plan_type                       TEXT NOT NULL DEFAULT 'analyst'
                                        CHECK (plan_type IN ('free','analyst','professional','institutional')),
    simulations_used_current_month  INT  NOT NULL DEFAULT 0,
    simulation_limit                INT  NOT NULL DEFAULT 5,
    credits_balance                 INT  NOT NULL DEFAULT 0,
    is_active                       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_reset_at                   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── 2. Assinaturas ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS maat_subscriptions (
    id                      SERIAL PRIMARY KEY,
    user_id                 INT NOT NULL REFERENCES maat_users(id) ON DELETE CASCADE,
    plan_type               TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active','cancelled','past_due','trialing')),
    renewal_date            DATE,
    payment_provider        TEXT,          -- 'stripe' | 'mercadopago' | 'manual'
    payment_provider_id     TEXT,          -- ID externo do provedor
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── 3. Log de uso ─────────────────────────────────────────────
-- Mantido enxuto: apenas últimos 90 dias (purge automático)
CREATE TABLE IF NOT EXISTS maat_usage_log (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             INT NOT NULL REFERENCES maat_users(id) ON DELETE CASCADE,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action_type         TEXT NOT NULL,   -- 'simulation' | 'ccee_fetch' | 'login'
    used_credit         BOOLEAN NOT NULL DEFAULT FALSE,
    api_calls_made      INT NOT NULL DEFAULT 0,
    processing_ms       INT,
    metadata            TEXT            -- JSON string opcional (sem objetos grandes)
);

-- Índice para queries de uso por usuário/período
CREATE INDEX IF NOT EXISTS idx_maat_usage_user_ts
    ON maat_usage_log (user_id, ts DESC);

-- Purge automático: manter apenas últimos 90 dias
-- (chamar periodicamente ou via pg_cron se disponível)
-- DELETE FROM maat_usage_log WHERE ts < NOW() - INTERVAL '90 days';

-- ── Limites padrão por plano ──────────────────────────────────
-- analyst:       5 simulações/mês, sem CCEE
-- professional: 30 simulações/mês, com CCEE
-- institutional: 500 simulações/mês, acesso total

-- ── Pacotes de crédito ────────────────────────────────────────
-- Definidos em código (monetization.py) — não precisa de tabela
-- 10 créditos  → R$  50
-- 50 créditos  → R$ 200
-- 100 créditos → R$ 350
