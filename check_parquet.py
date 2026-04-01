import duckdb, json

path = "data/core_section_economic.parquet"
con = duckdb.connect()

# Ver esquema e primeiros dados
print("=== SCHEMA ===")
print(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchdf().to_string())

print("\n=== COLUNAS DISPONÍVEIS ===")
row = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 1").fetchone()
desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchdf()
for i, col in enumerate(desc['column_name']):
    v = row[i]
    if isinstance(v, str):
        print(f"  {col}: string[{len(v)} chars] preview={v[:120]!r}")
    else:
        print(f"  {col}: {type(v).__name__} = {str(v)[:80]}")

con.close()