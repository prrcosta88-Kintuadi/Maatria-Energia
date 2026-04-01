import duckdb, json

path = "data/core_section_economic.parquet"
con = duckdb.connect()
row = con.execute(f"SELECT section_json FROM read_parquet('{path}') LIMIT 1").fetchone()
con.close()

data = json.loads(row[0])

def show_structure(d, prefix="", depth=0):
    if depth > 2: return
    if isinstance(d, dict):
        for k, v in list(d.items())[:30]:
            if isinstance(v, dict):
                keys = list(v.keys())
                sample = str(keys[:3])[1:-1] if keys else "(vazio)"
                print(f"{prefix}{k}: dict[{len(v)}] keys=[{sample}]")
                if depth < 1:
                    show_structure(v, prefix + "  ", depth + 1)
            elif isinstance(v, list):
                print(f"{prefix}{k}: list[{len(v)}]")
            elif isinstance(v, (int, float)):
                print(f"{prefix}{k}: {v}")
            elif isinstance(v, str):
                print(f"{prefix}{k}: {v[:80]!r}")
            else:
                print(f"{prefix}{k}: {type(v).__name__}")

show_structure(data)