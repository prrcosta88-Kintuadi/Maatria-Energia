import duckdb, pandas as pd, numpy as np

con = duckdb.connect("data/kintuadi.duckdb", read_only=True)

cmo_df = con.execute("""
    SELECT date_trunc('week', din_instante) + INTERVAL '1 day' AS semana,
           AVG(val_cmo) AS cmo_real
    FROM cmo
    WHERE UPPER(TRIM(id_subsistema)) IN ('SUDESTE','SE','SE/CO')
      AND val_cmo > 0 AND YEAR(din_instante) >= 2022
    GROUP BY 1 ORDER BY 1
""").df()

pld_df = con.execute("""
    SELECT date_trunc('week', data) + INTERVAL '1 day' AS semana,
           AVG(pld) AS pld_real,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pld) AS pld_p50
    FROM pld_historical
    WHERE UPPER(TRIM(submercado)) IN ('SUDESTE','SE') AND pld > 0
      AND YEAR(data) >= 2022
    GROUP BY 1 ORDER BY 1
""").df()

ena_df = con.execute("""
    SELECT date_trunc('week', ena_data) + INTERVAL '1 day' AS semana,
           AVG(ena_bruta_regiao_percentualmlt) AS ena_mlt
    FROM ena_diario_subsistema
    WHERE UPPER(TRIM(id_subsistema)) IN ('SUDESTE','SE','SE/CO')
      AND YEAR(ena_data) >= 2022
    GROUP BY 1 ORDER BY 1
""").df()
con.close()

for d in [cmo_df, pld_df, ena_df]:
    d["semana"] = pd.to_datetime(d["semana"])
    d.set_index("semana", inplace=True)

df = cmo_df.join(pld_df).join(ena_df).dropna(subset=["cmo_real","pld_real"])
df["cmo_prev"] = df["cmo_real"].shift(1)
df["erro"]     = df["cmo_prev"] - df["pld_real"]
df["erro_pct"] = df["erro"] / df["pld_real"] * 100
df = df.dropna(subset=["cmo_prev"])

print(f"Obs: {len(df)}  {df.index[0].date()} → {df.index[-1].date()}")
e = df["erro"]
print(f"Bias={e.mean():.1f}  MAE={e.abs().mean():.1f}  "
      f"Std={e.std():.1f}  P10={e.quantile(.1):.1f}  P90={e.quantile(.9):.1f}")
print(f"R²={df['cmo_prev'].corr(df['pld_real'])**2:.4f}")

meses = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
         7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
df["mes"] = df.index.month
for mes, g in df.groupby("mes"):
    sub = g["erro"]
    print(f"  {meses[mes]}: bias={sub.mean():>8.1f}  MAE={sub.abs().mean():>6.1f}  "
          f"[{sub.quantile(.1):>7.1f}, {sub.quantile(.9):>7.1f}]  n={len(sub)}")

q33, q66 = df["ena_mlt"].quantile([.33,.66])
for lbl, mask in [
    (f"Seco <{q33:.0f}%",  df["ena_mlt"] < q33),
    (f"Normal",            (df["ena_mlt"]>=q33)&(df["ena_mlt"]<q66)),
    (f"Úmido >{q66:.0f}%", df["ena_mlt"] >= q66),
]:
    sub = df[mask]["erro"].dropna()
    print(f"  {lbl}: bias={sub.mean():>8.1f}  MAE={sub.abs().mean():>6.1f}  "
          f"P10={sub.quantile(.1):>7.1f}  P90={sub.quantile(.9):>7.1f}  n={len(sub)}")

for lag in [1,2,4,8]:
    print(f"  Autocorr lag{lag}: {df['erro'].autocorr(lag=lag):.3f}")

q = df["cmo_prev"].quantile([.25,.50,.75])
print("\n=== CALIBRAÇÃO BANDAS ===")
for lbl, lo, hi in [("<R$20",0,20),("R$20-60",20,60),("R$60-250",60,250),(">R$250",250,9999)]:
    sub = df[(df["cmo_prev"]>=lo)&(df["cmo_prev"]<hi)]
    if len(sub)<3: continue
    cmo_m = sub["cmo_prev"].mean()
    e = sub["erro"]
    print(f"  {lbl:<10} n={len(sub):>3} "
          f"bias={e.mean()/cmo_m:>+.3f}x  "
          f"p10={abs(e.quantile(.1))/cmo_m:.3f}x  "
          f"p90={abs(e.quantile(.9))/cmo_m:.3f}x")

df[["cmo_prev","pld_real","erro","erro_pct","ena_mlt"]].to_csv("data/models/erro_ons_historico.csv")
print("\nSalvo em data/models/erro_ons_historico.csv")