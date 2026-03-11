"""
generate_tab_parquets.py
========================
Converte os core_section_*.parquet (formato JSON-string) nos parquets tabulares
core_tab_*.parquet que o app.py lê via DuckDB SQL sem json.loads().

Uso:
    python generate_tab_parquets.py          # lê de ./data, salva em ./data
    python generate_tab_parquets.py --dir data

Commit os 5 arquivos gerados sem Git LFS:
    git lfs untrack "data/core_tab_*.parquet"
    git add data/core_tab_*.parquet
    git commit -m "add tabular section parquets"
    git push
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_lfs(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            h = f.read(512)
        return b"git-lfs.github.com/spec/v1" in h or b"oid sha256:" in h
    except Exception:
        return False


def _read_section_json(parquet_path: Path) -> dict:
    """Lê a coluna section_json do parquet de seção e retorna o dict."""
    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT section_json FROM read_parquet(?) LIMIT 1",
            [str(parquet_path)]
        ).fetchone()
    finally:
        con.close()
    if not row or not row[0]:
        raise ValueError(f"section_json vazio em {parquet_path}")
    val = row[0]
    return json.loads(val) if isinstance(val, str) else val


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        print(f"  AVISO: DataFrame vazio, {path.name} não salvo.")
        return
    tmp = path.with_suffix(".tmp.parquet")
    con = duckdb.connect()
    try:
        con.register("_df", df)
        con.execute(f"COPY _df TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(tmp, path)
    finally:
        con.close()
    print(f"  ✓ {path.name}  ({path.stat().st_size:,} bytes, {len(df):,} linhas)")


# ── conversores por seção ─────────────────────────────────────────────────────

def convert_economic(data: dict, out_dir: Path) -> None:
    """Exporta séries horárias de economic como (instante, serie, valor)."""
    rows = []
    for key in [
        "sin_cost_hourly", "T_prudencia_hourly", "T_hidro_hourly",
        "T_eletric_hourly", "T_sistemica_hourly", "CVaR_implicit_hourly",
        "Risk_Aversion_Gap_hourly", "curtailment_loss_hourly",
        "hydro_gap_hourly", "required_hydro_hourly",
        "mandatory_generation_hourly", "thermal_prudential_dispatch_hourly",
        "infra_marginal_rent_hourly",
    ]:
        for ts, val in (data.get(key) or {}).items():
            try:
                rows.append({"instante": ts, "serie": key, "valor": float(val)})
            except Exception:
                pass
    # system_state (string)
    for ts, val in (data.get("system_state_hourly") or {}).items():
        rows.append({"instante": ts, "serie": "system_state_hourly", "valor": str(val)})
    # normalization
    for sub_key, d in (data.get("normalization_hourly") or {}).items():
        if isinstance(d, dict):
            for ts, val in d.items():
                try:
                    rows.append({"instante": ts, "serie": f"norm_{sub_key}", "valor": float(val)})
                except Exception:
                    pass
    if rows:
        df = pd.DataFrame(rows)
        df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
        _save_parquet(df, out_dir / "core_tab_economic.parquet")


def convert_advanced(data: dict, out_dir: Path) -> None:
    """Exporta painel_horario_renovavel + CMO como (instante, serie, valor)."""
    rows = []
    for rec in (data.get("painel_horario_renovavel") or []):
        ts = rec.get("instante")
        if not ts:
            continue
        for col in ["gfom_pct", "ipr", "isr", "ear", "ena"]:
            if col in rec:
                try:
                    rows.append({"instante": ts, "serie": col, "valor": float(rec[col])})
                except Exception:
                    pass
    cmo = (data.get("aderencia_fisico_economica") or {}).get("cmo_horario_por_submercado") or {}
    for sm, d in cmo.items():
        if isinstance(d, dict):
            for ts, val in d.items():
                try:
                    rows.append({"instante": ts, "serie": f"cmo_{sm}", "valor": float(val)})
                except Exception:
                    pass
    if rows:
        df = pd.DataFrame(rows)
        df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
        _save_parquet(df, out_dir / "core_tab_advanced.parquet")


def convert_operacao(data: dict, out_dir: Path) -> None:
    """Exporta geração e carga como (instante, fonte, tipo, valor)."""
    rows = []
    for fonte, payload in (data.get("generation") or {}).items():
        for rec in (payload or {}).get("serie", []):
            try:
                rows.append({
                    "instante": rec["instante"],
                    "fonte": fonte,
                    "tipo": "geracao",
                    "valor": float(rec["geracao"]),
                })
            except Exception:
                pass
    for sm, payload in (data.get("load") or {}).items():
        for rec in (payload or {}).get("serie", []):
            try:
                rows.append({
                    "instante": rec["instante"],
                    "fonte": sm,
                    "tipo": "carga",
                    "valor": float(rec["carga"]),
                })
            except Exception:
                pass
    if rows:
        df = pd.DataFrame(rows)
        df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
        _save_parquet(df, out_dir / "core_tab_operacao.parquet")


def convert_ccee(data: dict, out_dir: Path) -> None:
    """Exporta registros CCEE como DataFrame tabular."""
    records = (data.get("data") or [])
    if records:
        df = pd.DataFrame(records)
        _save_parquet(df, out_dir / "core_tab_ccee.parquet")


def convert_renewables(data: dict, out_dir: Path) -> None:
    """Exporta curtailment como (instante, fonte, valor)."""
    rows = []
    for fonte in ["solar", "eolica"]:
        for rec in ((data.get("curtailment") or {}).get(fonte) or {}).get("serie", []):
            try:
                rows.append({
                    "instante": rec["instante"],
                    "fonte": fonte,
                    "valor": float(rec["valor"]),
                })
            except Exception:
                pass
    if rows:
        df = pd.DataFrame(rows)
        df["instante"] = pd.to_datetime(df["instante"], errors="coerce")
        _save_parquet(df, out_dir / "core_tab_renewables.parquet")


# ── main ──────────────────────────────────────────────────────────────────────

SECTIONS = {
    "advanced_metrics": convert_advanced,
    "economic":         convert_economic,
    "operacao":         convert_operacao,
    "ccee":             convert_ccee,
    "renewables":       convert_renewables,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data", help="Diretório com os core_section_*.parquet")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Diretório '{data_dir}' não encontrado.")
        sys.exit(1)

    print(f"\n📂 Lendo de: {data_dir.resolve()}\n")

    for section, converter in SECTIONS.items():
        src = data_dir / f"core_section_{section}.parquet"
        print(f"[{section}]")
        if not src.exists():
            print(f"  ✗ {src.name} não encontrado — pulando.")
            continue
        if _is_lfs(src):
            print(f"  ✗ {src.name} é ponteiro LFS — baixe o objeto real primeiro.")
            continue
        try:
            print(f"  Lendo {src.name}...")
            data = _read_section_json(src)
            converter(data, data_dir)
            del data
        except Exception as e:
            print(f"  ERRO: {e}")

    print("\n✅ Concluído. Commit os arquivos core_tab_*.parquet sem LFS:\n")
    print("  git lfs untrack 'data/core_tab_*.parquet'")
    print("  git add data/core_tab_*.parquet")
    print('  git commit -m "add tabular section parquets"')
    print("  git push\n")


if __name__ == "__main__":
    main()
