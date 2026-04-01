import duckdb

con = duckdb.connect("data/kintuadi.duckdb", read_only=True)

print("=== TABELAS ===")
tables = con.execute("SHOW TABLES").fetchdf()
print(tables.to_string())

print("\n=== SCHEMA DE CADA TABELA ===")
for t in tables['name']:
    try:
        schema = con.execute(f"DESCRIBE {t}").fetchdf()
        count  = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        minmax = con.execute(f"""
            SELECT MIN(din_instante)::text, MAX(din_instante)::text
            FROM {t}
            WHERE din_instante IS NOT NULL
        """).fetchone() if 'din_instante' in schema['column_name'].values else (None, None)
        print(f"\n[{t}] — {count:,} linhas | período: {minmax[0]} → {minmax[1]}")
        print(schema[['column_name','column_type']].to_string(index=False))
    except Exception as e:
        print(f"\n[{t}] — erro: {e}")

con.close()