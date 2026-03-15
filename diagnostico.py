"""
diagnostico.py — encontra _CCEE_SUB_API e mostra o contexto exato
"""
from pathlib import Path

c = Path("premium_engine.py").read_text(encoding="utf-8")
lines = c.splitlines()

print("=== Linhas com _CCEE_SUB_API ===")
for i, line in enumerate(lines, 1):
    if "_CCEE_SUB_API" in line:
        # mostrar contexto de 2 linhas antes e depois
        start = max(0, i-3)
        end   = min(len(lines), i+3)
        for j in range(start, end):
            marker = ">>>" if j == i-1 else "   "
            print(f"{marker} {j+1:4d}: {lines[j]}")
        print()

print("=== Total de ocorrências ===")
print(f"  {c.count('_CCEE_SUB_API')} ocorrência(s)")
