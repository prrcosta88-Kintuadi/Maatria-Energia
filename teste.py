import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

print("=== valores distintos de tipo_geracao ===")
cur.execute("SELECT DISTINCT tipo_geracao, COUNT(*) FROM geracao_tipo_hora GROUP BY 1 ORDER BY 1")
for row in cur.fetchall():
    print(row)

conn.close()