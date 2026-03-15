import requests, json

BASE = "https://dadosabertos.ccee.org.br/api/3/action"
RESOURCES = {
    2023: "67390467-e175-402f-8bf1-491a80d01a01",
    2024: "05c25b5e-aeed-4494-a203-93d68d070b2a",
    2025: "f5e2e2ce-9388-458f-86d2-c038dc18d997",
    2026: "7143897d-d1b7-445e-ba53-5864e5a99688",
}

# Cole aqui o cookie completo (mesmo usado no teste anterior)
COOKIE = "b3f442d57b21fff3f2786cf5f7ea936b=c88d591495ce7cc418d94b47879889e0; rxVisitor=1773508142031D5LFL1C594IHQUEQE2P1OA4VT6MT5AE2; _ga=GA1.1.207492672.1773508143; dtCookie=v_4_srv_3_sn_EA830F47641D1BF3A515DBE0551A0BC9_app-3A6b70a4dc8616698f_1_app-3A49cf763be474f9b7_1_ol_0_perc_100000_mul_1_rcs-3Acss_0; dtSa=-; _ga_KK4ZJ4NW5X=GS2.1.s1773521062$o2$g1$t1773521097$j25$l0$h0; rxvt=1773522897880|1773519363027; dtPC=3$521097087_374h-vCKTLCVKJCRJQDFURLTEHKFKPHUUTMFJP-0e0; ckan=e725983942c01c788c594daab1b92d0b5bceaa2911e16892ffd745dfb31686e0e4e37ebd"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    "sec-fetch-dest": "document", "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none", "upgrade-insecure-requests": "1",
    "Cookie": COOKIE,
}

def fetch(rid, filters=None, limit=5):
    params = {"resource_id": rid, "limit": limit}
    if filters:
        params["filters"] = json.dumps(filters)
    resp = requests.get(f"{BASE}/datastore_search", params=params, headers=HEADERS, timeout=15)
    return resp.status_code, resp.json() if resp.status_code == 200 else {}

print("="*60)
print("Testando todos os anos — sem filtro e com filtro SUDESTE")
print("="*60)

for ano, rid in RESOURCES.items():
    # Sem filtro
    status, data = fetch(rid, limit=3)
    result = data.get("result", {})
    total  = result.get("total", "?")
    subs   = list({r.get("SUBMERCADO") for r in result.get("records", [])})
    meses  = list({r.get("MES_REFERENCIA") for r in result.get("records", [])})

    # Com filtro SUDESTE
    status2, data2 = fetch(rid, filters={"SUBMERCADO": "SUDESTE"}, limit=2)
    total_se = data2.get("result", {}).get("total", "?")

    print(f"\nAno {ano} — HTTP {status}")
    print(f"  Total registros : {total}")
    print(f"  Submercados     : {subs}")
    print(f"  Meses amostra   : {sorted(meses)}")
    print(f"  Total SE (filtro SUDESTE): {total_se}")
