import duckdb, pandas as pd

con = duckdb.connect("data/kintuadi.duckdb", read_only=True)

print("=== pld_historical sample ===")
df = con.execute("SELECT * FROM pld_historical WHERE pld > 0 LIMIT 5").df()
print(df.to_string())

print(f"\nSubmercados: {con.execute('SELECT DISTINCT submercado FROM pld_historical').df()['submercado'].tolist()}")
print(f"Período: {con.execute('SELECT MIN(data), MAX(data) FROM pld_historical WHERE pld>0').fetchone()}")
print(f"Linhas: {con.execute('SELECT COUNT(*) FROM pld_historical WHERE pld>0').fetchone()[0]:,}")

# Query do engine
try:
    r = con.execute("""
        SELECT date_trunc('week', data) + INTERVAL '1 day' AS semana,
               UPPER(TRIM(submercado)) AS sub, AVG(pld) AS pld_medio
        FROM pld_historical
        WHERE UPPER(TRIM(submercado)) IN ('SUDESTE','SE') AND pld>0 AND YEAR(data)>=2021
        GROUP BY 1,2 ORDER BY 1 DESC LIMIT 5
    """).df()
    print(f"\nQuery semanal:\n{r.to_string()}")
except Exception as e:
    print(f"\nErro query: {e}")

# Ver training_dataset gerado
import pandas as pd
try:
    ds = pd.read_csv("data/models/training_dataset.csv", index_col=0, parse_dates=True)
    print(f"\nDataset: {ds.shape}")
    pld_cols = [c for c in ds.columns if 'pld_real' in c]
    print(f"Colunas PLD real: {pld_cols}")
    if pld_cols:
        print(ds[pld_cols].dropna().tail(5).to_string())
    else:
        print("SEM colunas pld_real — modelo treinado SEM targets reais")
except Exception as e:
    print(f"Dataset erro: {e}")

con.close()