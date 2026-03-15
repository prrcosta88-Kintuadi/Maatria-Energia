import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM geracao_tipo_hora WHERE tipo_geracao = 'other'")
print(f"Registros 'other' antes: {cur.fetchone()[0]:,}")

cur.execute("DELETE FROM geracao_tipo_hora WHERE tipo_geracao = 'other'")
print(f"Removidos: {cur.rowcount:,}")
conn.commit()

cur.execute("SELECT tipo_geracao, COUNT(*) FROM geracao_tipo_hora GROUP BY 1 ORDER BY 1")
print("\nTipos restantes:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

conn.close()
print("\nConcluído.")
