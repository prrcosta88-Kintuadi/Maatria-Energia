"""
fix_constraint.py — Atualiza o CHECK constraint da tabela maat_users
para incluir o plano 'free'.
Execute: python fix_constraint.py
"""
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

url = os.getenv("DATABASE_URL_AUTH", "")
if not url:
    print("❌ DATABASE_URL_AUTH não definida.")
    exit(1)

conn = psycopg2.connect(url)
conn.autocommit = True
cur = conn.cursor()

print("Atualizando constraint de plan_type...\n")

steps = [
    # 1. Remover constraint antiga
    ("Removendo constraint antiga",
     "ALTER TABLE maat_users DROP CONSTRAINT IF EXISTS maat_users_plan_type_check"),

    # 2. Adicionar nova constraint com 'free'
    ("Adicionando nova constraint com 'free'",
     "ALTER TABLE maat_users ADD CONSTRAINT maat_users_plan_type_check "
     "CHECK (plan_type IN ('free','analyst','professional','institutional'))"),
]

for label, sql in steps:
    try:
        cur.execute(sql)
        print(f"  ✅ {label}")
    except Exception as e:
        print(f"  ❌ {label}: {e}")

# Verificar
cur.execute("""
    SELECT con.conname, pg_get_constraintdef(con.oid)
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    WHERE rel.relname = 'maat_users'
    AND con.contype = 'c'
""")
rows = cur.fetchall()
print("\nConstraints atuais em maat_users:")
for name, definition in rows:
    print(f"  {name}: {definition}")

cur.close()
conn.close()
print("\n✅ Pronto. Tente criar o usuário novamente.")
