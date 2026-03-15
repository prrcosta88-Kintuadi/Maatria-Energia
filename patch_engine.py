"""
patch_engine.py — adiciona _CCEE_SUB_API ao premium_engine.py
Execute: python patch_engine.py
"""
import ast, sys
from pathlib import Path

TARGET = Path("premium_engine.py")
if not TARGET.exists():
    print("Erro: premium_engine.py nao encontrado.")
    sys.exit(1)

c = TARGET.read_text(encoding="utf-8")

# Verificar se a DEFINICAO existe (nao apenas o uso)
if "_CCEE_SUB_API = {" in c:
    print("OK: _CCEE_SUB_API ja definido -- nada a fazer.")
    sys.exit(0)

ANCHOR = "def fetch_ccee_market_data("
if ANCHOR not in c:
    print("Erro: ancora nao encontrada.")
    sys.exit(1)

INSERT = '_CCEE_SUB_API = {\n    "SE": "SUDESTE", "NE": "NORDESTE", "S": "SUL", "N": "NORTE",\n}\n\n'
c = c.replace(ANCHOR, INSERT + ANCHOR, 1)

try:
    ast.parse(c)
except SyntaxError as e:
    print(f"Erro de sintaxe: {e}")
    sys.exit(1)

TARGET.write_text(c, encoding="utf-8")
print("OK: _CCEE_SUB_API definido com sucesso.")
c2 = TARGET.read_text(encoding="utf-8")
print("Definicao : OK" if "_CCEE_SUB_API = {" in c2 else "Definicao : FALHOU")
print("Uso       : OK" if "_CCEE_SUB_API.get(" in c2 else "Uso       : FALHOU")
