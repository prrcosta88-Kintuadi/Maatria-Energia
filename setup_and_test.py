"""
setup_and_test.py — Setup e testes locais da MAÁTria Energia
═══════════════════════════════════════════════════════════════
Execute passo a passo conforme orientação.

Uso:
    python setup_and_test.py --step 1   # testar conexões
    python setup_and_test.py --step 2   # criar tabelas no banco AUTH
    python setup_and_test.py --step 3   # criar usuário master
    python setup_and_test.py --step 4   # testar autenticação
    python setup_and_test.py --step 5   # testar controle de uso
    python setup_and_test.py --step 6   # testar API CCEE
    python setup_and_test.py --all      # executar todos os steps
"""
import argparse
import os
import sys
from pathlib import Path

# Carregar .env local se existir
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env carregado")
except ImportError:
    print("⚠️  python-dotenv não instalado — variáveis devem estar no ambiente")

# ──────────────────────────────────────────────────────────────────────────────

def step1_test_connections():
    """Testa conexão com os dois bancos Neon."""
    print("\n" + "="*55)
    print("STEP 1 — Testar conexões Neon")
    print("="*55)

    import psycopg2

    # Banco operacional (ONS)
    url1 = os.getenv("DATABASE_URL", "")
    print(f"\n[1a] DATABASE_URL (banco ONS): {'definida' if url1 else '❌ NÃO DEFINIDA'}")
    if url1:
        try:
            conn = psycopg2.connect(url1)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM pld_historical")
            n = cur.fetchone()[0]
            conn.close()
            print(f"     ✅ Conectado — pld_historical: {n:,} linhas")
        except Exception as e:
            print(f"     ❌ Erro: {e}")

    # Banco AUTH (maatriaenergia-ccee)
    url2 = os.getenv("DATABASE_URL_AUTH", "")
    print(f"\n[1b] DATABASE_URL_AUTH (banco AUTH): {'definida' if url2 else '❌ NÃO DEFINIDA'}")
    if url2:
        try:
            conn = psycopg2.connect(url2)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            v = cur.fetchone()[0]
            conn.close()
            print(f"     ✅ Conectado — PostgreSQL: {v[:40]}...")
        except Exception as e:
            print(f"     ❌ Erro: {e}")
    else:
        print("     ⚠️  Adicione DATABASE_URL_AUTH ao .env")


def step2_create_tables():
    """Cria as tabelas de monetização no banco AUTH."""
    print("\n" + "="*55)
    print("STEP 2 — Criar tabelas no banco AUTH")
    print("="*55)

    sql_file = Path("monetization_schema.sql")
    if not sql_file.exists():
        print("❌ monetization_schema.sql não encontrado na raiz do projeto.")
        return

    import psycopg2
    url = os.getenv("DATABASE_URL_AUTH", "")
    if not url:
        print("❌ DATABASE_URL_AUTH não definida.")
        return

    ddl = sql_file.read_text(encoding="utf-8")
    # Filtrar apenas statements CREATE TABLE e CREATE INDEX
    statements = [s.strip() for s in ddl.split(";")
                  if s.strip() and not s.strip().startswith("--")
                  and ("CREATE" in s.upper() or "INSERT" in s.upper())]

    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur = conn.cursor()
    for stmt in statements:
        if not stmt.strip():
            continue
        try:
            cur.execute(stmt)
            first_line = stmt.strip().split("\n")[0][:60]
            print(f"  ✅ {first_line}")
        except Exception as e:
            print(f"  ⚠️  {str(e)[:80]}")
    cur.close()
    conn.close()
    print("\n✅ Tabelas criadas/verificadas no banco AUTH.")


def step3_create_master_user():
    """Cria usuário master institucional para testes."""
    print("\n" + "="*55)
    print("STEP 3 — Criar usuário master")
    print("="*55)

    sys.path.insert(0, str(Path.cwd()))
    from monetization import create_user, PLANS

    print("\nPlanos disponíveis:")
    for k, v in PLANS.items():
        print(f"  {k}: R${v['price_brl']}/mês — {v['description']}")

    email    = input("\nEmail do usuário master [admin@maatriaenergia.com.br]: ").strip()
    if not email:
        email = "admin@maatriaenergia.com.br"

    password = input("Senha (mín. 8 chars): ").strip()
    if len(password) < 8:
        print("❌ Senha muito curta.")
        return

    plan = input("Plano [institutional]: ").strip() or "institutional"

    ok, msg = create_user(email, password, plan)
    print(f"\n{'✅' if ok else '❌'} {msg}")

    if ok:
        print(f"\n   Email:  {email}")
        print(f"   Plano:  {plan}")
        print(f"   Limite: {PLANS[plan]['simulation_limit']} simulações/mês")


def step4_test_auth():
    """Testa autenticação com usuário existente."""
    print("\n" + "="*55)
    print("STEP 4 — Testar autenticação")
    print("="*55)

    sys.path.insert(0, str(Path.cwd()))
    from monetization import authenticate, PLANS

    email    = input("\nEmail: ").strip()
    password = input("Senha: ").strip()

    user = authenticate(email, password)
    if user:
        plan = PLANS.get(user.plan_type, {})
        print(f"\n✅ Autenticado com sucesso!")
        print(f"   ID:         {user.id}")
        print(f"   Email:      {user.email}")
        print(f"   Plano:      {user.plan_type} ({plan.get('label','')})")
        print(f"   Simulações: {user.simulations_used_current_month}/{user.simulation_limit}")
        print(f"   Créditos:   {user.credits_balance}")
        print(f"   CCEE:       {'✅ Sim' if user.has_ccee_access else '❌ Não'}")
    else:
        print("\n❌ Credenciais inválidas ou usuário inativo.")


def step5_test_usage_control():
    """Testa controle de uso (simulações e créditos)."""
    print("\n" + "="*55)
    print("STEP 5 — Testar controle de uso")
    print("="*55)

    sys.path.insert(0, str(Path.cwd()))
    from monetization import authenticate, can_run_simulation, consume_simulation, add_credits

    email    = input("\nEmail: ").strip()
    password = input("Senha: ").strip()

    user = authenticate(email, password)
    if not user:
        print("❌ Autenticação falhou.")
        return

    print(f"\n📊 Estado atual:")
    print(f"   Simulações usadas: {user.simulations_used_current_month}/{user.simulation_limit}")
    print(f"   Créditos:          {user.credits_balance}")

    # Testar can_run
    can, reason = can_run_simulation(user)
    print(f"\n🔍 can_run_simulation: {'✅ Sim' if can else '❌ Não'} — {reason}")

    if can:
        resp = input("\nExecutar simulação de teste? (s/n): ").strip().lower()
        if resp == "s":
            ok, tipo = consume_simulation(user.id, "test_simulation")
            print(f"   {'✅' if ok else '❌'} Consumido via: {tipo}")

    # Testar créditos
    resp = input("\nAdicionar 5 créditos de teste? (s/n): ").strip().lower()
    if resp == "s":
        ok, msg = add_credits(user.id, 5, "test")
        print(f"   {'✅' if ok else '❌'} {msg}")


def step6_test_ccee_api():
    """Testa acesso à API CCEE com cookie."""
    print("\n" + "="*55)
    print("STEP 6 — Testar API CCEE")
    print("="*55)

    cookie = os.getenv("CCEE_COOKIE", "")
    print(f"\nCCEE_COOKIE: {'definido ✅' if cookie else '❌ NÃO DEFINIDO'}")
    if not cookie:
        print("Defina CCEE_COOKIE no .env para testar.")
        print("(Copie via F12 → Network → Copy as cURL no browser)")
        return

    import requests
    import json

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/145.0.0.0",
        "Accept": "application/json",
        "Cookie": cookie,
    }

    # Dataset de montantes (2026)
    rid = "7143897d-d1b7-445e-ba53-5864e5a99688"
    url = f"https://dadosabertos.ccee.org.br/api/3/action/datastore_search?resource_id={rid}&limit=2"

    print(f"\n[6a] Testando dataset de montantes (2026)...")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(f"     HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("result", {}).get("total", "?")
            records = data.get("result", {}).get("records", [])
            print(f"     ✅ Total: {total} registros")
            if records:
                print(f"     Campos: {list(records[0].keys())}")
        else:
            print(f"     ❌ Erro: {resp.text[:200]}")
    except Exception as e:
        print(f"     ❌ {e}")

    # Dataset ESS (2026)
    rid_ess = "bf1fb5ee-bff1-4ff7-a791-83dd892438f5"
    url_ess = f"https://dadosabertos.ccee.org.br/api/3/action/datastore_search?resource_id={rid_ess}&limit=2"

    print(f"\n[6b] Testando dataset ESS horário (2026)...")
    try:
        resp = requests.get(url_ess, headers=headers, timeout=10)
        print(f"     HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("result", {}).get("total", "?")
            records = data.get("result", {}).get("records", [])
            print(f"     ✅ Total: {total} registros")
            if records:
                print(f"     Campos: {list(records[0].keys())}")
                print(f"     Linha 1: {records[0]}")
        else:
            print(f"     ❌ {resp.text[:200]}")
    except Exception as e:
        print(f"     ❌ {e}")


def run_all():
    step1_test_connections()
    step2_create_tables()
    step3_create_master_user()
    step4_test_auth()
    step5_test_usage_control()
    step6_test_ccee_api()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1,2,3,4,5,6])
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    steps = {
        1: step1_test_connections,
        2: step2_create_tables,
        3: step3_create_master_user,
        4: step4_test_auth,
        5: step5_test_usage_control,
        6: step6_test_ccee_api,
    }

    if args.all:
        run_all()
    elif args.step:
        steps[args.step]()
    else:
        print(__doc__)
