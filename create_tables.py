"""
create_tables.py — Cria e verifica tabelas de autenticação/monetização
Execute: python create_tables.py
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

url = os.getenv("DATABASE_URL_AUTH", "")
if not url:
    print("❌ DATABASE_URL_AUTH não definida no .env")
    exit(1)

conn = psycopg2.connect(url)
conn.autocommit = True
cur = conn.cursor()

TABLES = [
    ("maat_users", """
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
        )
    """),
    ("maat_subscriptions", """
        CREATE TABLE IF NOT EXISTS maat_subscriptions (
            id                  SERIAL PRIMARY KEY,
            user_id             INT NOT NULL REFERENCES maat_users(id) ON DELETE CASCADE,
            plan_type           TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'active'
                                    CHECK (status IN ('active','cancelled','past_due','trialing')),
            renewal_date        DATE,
            payment_provider    TEXT,
            payment_provider_id TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """),
    ("maat_usage_log", """
        CREATE TABLE IF NOT EXISTS maat_usage_log (
            id              BIGSERIAL PRIMARY KEY,
            user_id         INT NOT NULL REFERENCES maat_users(id) ON DELETE CASCADE,
            ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            action_type     TEXT NOT NULL,
            used_credit     BOOLEAN NOT NULL DEFAULT FALSE,
            api_calls_made  INT NOT NULL DEFAULT 0,
            processing_ms   INT,
            metadata        TEXT
        )
    """),
    ("idx_maat_usage_user_ts", """
        CREATE INDEX IF NOT EXISTS idx_maat_usage_user_ts
            ON maat_usage_log (user_id, ts DESC)
    """),
]

print(f"Conectado ao banco AUTH\n")
print("=== Criando tabelas ===")
for name, sql in TABLES:
    try:
        cur.execute(sql)
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print("\n=== Verificação completa ===")
for table in ["maat_users", "maat_subscriptions", "maat_usage_log"]:
    # Existe?
    cur.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = %s AND table_schema = 'public'", (table,))
    exists = cur.fetchone()[0] == 1

    # Quantas linhas?
    rows = "N/A"
    if exists:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        rows = f"{cur.fetchone()[0]:,} linhas"

    # Quais colunas?
    cols = []
    if exists:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = 'public' ORDER BY ordinal_position",
            (table,))
        cols = [(r[0], r[1]) for r in cur.fetchall()]

    status = "✅ existe" if exists else "❌ NÃO ENCONTRADA"
    print(f"\n  {status} — {table} ({rows})")
    for col, dtype in cols:
        print(f"    · {col:<40} {dtype}")

print("\n=== Usuários cadastrados ===")
try:
    cur.execute(
        "SELECT id, email, plan_type, simulations_used_current_month, "
        "simulation_limit, credits_balance, is_active, created_at "
        "FROM maat_users ORDER BY id")
    rows = cur.fetchall()
    if rows:
        for r in rows:
            active = "✅ ativo" if r[6] else "❌ inativo"
            print(f"  ID {r[0]:>3} | {r[1]:<35} | {r[2]:<15} | "
                  f"sims: {r[3]}/{r[4]} | créditos: {r[5]} | {active}")
    else:
        print("  (nenhum usuário cadastrado)")
except Exception as e:
    print(f"  ❌ {e}")

print("\n=== Logs de uso (últimos 10) ===")
try:
    cur.execute(
        "SELECT u.email, l.ts, l.action_type, l.used_credit, l.api_calls_made "
        "FROM maat_usage_log l JOIN maat_users u ON u.id = l.user_id "
        "ORDER BY l.ts DESC LIMIT 10")
    rows = cur.fetchall()
    if rows:
        for r in rows:
            credit = "crédito" if r[3] else "plano"
            print(f"  {str(r[1])[:19]} | {r[0]:<30} | {r[2]:<25} | via {credit} | {r[4]} api calls")
    else:
        print("  (nenhum log registrado)")
except Exception as e:
    print(f"  ❌ {e}")

cur.close()
conn.close()
print("\n✅ Verificação concluída.")
