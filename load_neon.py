# -*- coding: utf-8 -*-
"""
load_neon.py — Carga inicial do Neon PostgreSQL (schema agregado)
=================================================================
Lê arquivos CSV/XLSX de data/ons/ e data/ccee_pld_*.csv e insere
nas tabelas do Neon com pré-agregação:

  Geracao_Usina_Horaria  → geracao_tipo_hora    (500 usinas → 6 tipos)
  Restricao_*            → restricao_renovavel  (200 usinas → 4 subsistemas)

Filtro: dados >= 2021 (quando PLD horário começou).

Uso:
    python load_neon.py              # carrega tudo
    python load_neon.py --dry-run    # mostra sem inserir
    python load_neon.py --only pld
    python load_neon.py --only geracao|carga|cmo|ear|ena|restricao|intercambio
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import db_neon
except ImportError:
    print("ERRO: db_neon.py nao encontrado.")
    sys.exit(1)

try:
    import psycopg2.extras
except ImportError:
    print("ERRO: psycopg2 nao instalado. Execute: pip install psycopg2-binary")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ANO_MIN = 2021       # Filtro geral — dados a partir de 2021

# ─── cache de datas máximas no Neon (consultado uma vez por execução) ─────────

_MAX_DATE_CACHE: Dict[str, Optional[datetime]] = {}

def _neon_max_date(table: str, date_col: str = "din_instante") -> Optional[datetime]:
    """Retorna o MAX(date_col) já carregado na tabela do Neon.
    Resultado em cache para evitar múltiplas consultas por execução.
    """
    key = f"{table}.{date_col}"
    if key not in _MAX_DATE_CACHE:
        try:
            row = db_neon.fetchone(f"SELECT MAX({date_col}) FROM {table}")
            _MAX_DATE_CACHE[key] = row[0] if row and row[0] else None
        except Exception:
            _MAX_DATE_CACHE[key] = None
    return _MAX_DATE_CACHE[key]


def _should_skip_file(fpath: Path, table: str,
                      date_col: str = "din_instante") -> bool:
    """Retorna True se o arquivo pode ser pulado com segurança.

    Regra:
    - Arquivos do ANO CORRENTE: NUNCA pulados.
      O mesmo arquivo xlsx/csv é substituído diariamente com dados acrescidos,
      então sempre precisa ser reprocessado para capturar o dia mais recente.
    - Arquivos de ANOS ANTERIORES: pulados se o Neon já tem dados daquele
      mês completo (MAX da tabela ultrapassa o último dia do mês do arquivo).
      Anos anteriores são histórico imutável — não mudam após o fechamento.
    """
    yr = _year(fpath.name)
    if not yr:
        return False   # não consegue extrair ano → processar por segurança

    # Ano corrente: sempre reprocessar (arquivo cresce a cada dia)
    if yr >= datetime.now().year:
        return False

    # Anos anteriores: consultar MAX no Neon
    max_dt = _neon_max_date(table, date_col)
    if max_dt is None:
        return False   # tabela vazia → processar tudo

    mo = _month(fpath.name)

    # Arquivo sem mês no nome (granularidade anual): pula se o Neon já tem
    # dados do ano seguinte
    if mo is None:
        return max_dt.year > yr

    # Arquivo mensal: pula se o Neon já tem dados além do último dia do mês
    file_last_day = datetime(yr, mo, 28)   # 28 é conservador (todo mês tem)
    return max_dt >= file_last_day

ANO_MIN_RESTRICAO = 2023  # Restrições eólica/solar: apenas a partir de 2023

# ── Mapeamento nom_tipousina → 6 tipos do dashboard ─────────────────────────
_TIPO_MAP: Dict[str, str] = {
    # ── nomes exatos conforme arquivo ONS Geracao_Usina_Horaria_{ano}-{mes} ──
    "HIDROELÉTRICA": "hydro",  "TÉRMICA": "thermal",
    "FOTOVOLTAICA":  "solar",  "EOLIELÉTRICA": "wind",  "NUCLEAR": "nuclear",
    # ── variantes sem acento / abreviações / nomes alternativos ──────────────
    "SOLAR": "solar",          "FOTOVOLT": "solar",
    "EOLICA": "wind",          "EÓLICA": "wind",        "EOLICO": "wind",
    "EOLIELETRICA": "wind",    "EOLIELETRICO": "wind",  "EOLIELÉTRICO": "wind",
    "EOL": "wind",
    "UHE": "hydro",           "PCH": "hydro",          "CGH": "hydro",
    "HIDROELETRICA": "hydro", "HIDRÁULICA": "hydro",   "HIDRAULICA": "hydro",
    "HIDRO": "hydro",
    "TERMICA": "thermal",     "TÉRMICA": "thermal",    "UTE": "thermal",
    "BIOMASSA": "thermal",    "GAS": "thermal",         "GÁS": "thermal",
    "OLEO": "thermal",        "ÓLEO": "thermal",        "CARVAO": "thermal",
    "CARVÃO": "thermal",      "DERIVADOS": "thermal",
    "NUCLEAR": "nuclear",     "UTN": "nuclear",
}

_SUB_MAP: Dict[str, str] = {
    "N": "N", "NORTE": "N",
    "NE": "NE", "NORDESTE": "NE",
    "SE": "SE", "SUDESTE": "SE", "SUDESTE/CO": "SE",
    "S": "S", "SUL": "S",
}


def _tipo(raw) -> str:
    c = str(raw).upper().strip()
    if c in _TIPO_MAP:
        return _TIPO_MAP[c]
    for k, v in _TIPO_MAP.items():
        if k in c:
            return v
    return "other"


def _sub(raw) -> Optional[str]:
    return _SUB_MAP.get(str(raw).upper().strip()) if raw else None


def _read(path) -> pd.DataFrame:
    try:
        p = str(path).lower()
        if p.endswith(".xlsx"):
            # Verificar quantas abas existem
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet_names = wb.sheetnames
            wb.close()

            if len(sheet_names) == 1:
                return pd.read_excel(path, engine="openpyxl", sheet_name=0)

            # Múltiplas abas: concatenar todas (ex: Geracao_Usina_Horaria_2021.xlsx)
            logger.info(f"  {len(sheet_names)} abas encontradas: {sheet_names}")
            frames = []
            for sheet in sheet_names:
                try:
                    df_sheet = pd.read_excel(path, engine="openpyxl", sheet_name=sheet)
                    if not df_sheet.empty:
                        frames.append(df_sheet)
                        logger.info(f"    Aba '{sheet}': {len(df_sheet):,} linhas")
                except Exception as e:
                    logger.warning(f"    Erro na aba '{sheet}': {e}")
            if not frames:
                return pd.DataFrame()
            result = pd.concat(frames, ignore_index=True)
            logger.info(f"  Total concatenado: {len(result):,} linhas")
            return result

        for enc in ("utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, sep=None, engine="python",
                                   on_bad_lines="skip", encoding=enc)
            except UnicodeDecodeError:
                continue
    except Exception as e:
        logger.warning(f"Erro ao ler {path}: {e}")
    return pd.DataFrame()


def _year(filename) -> Optional[int]:
    m = re.search(r"(\d{4})(?:-\d{2})?$", Path(filename).stem)
    return int(m.group(1)) if m else None


def _month(filename) -> Optional[int]:
    """Extrai mês do nome do arquivo, ex: Geracao_Usina_Horaria_2023-05 → 5."""
    m = re.search(r"\d{4}-(\d{2})$", Path(filename).stem)
    return int(m.group(1)) if m else None


def _ts(series: pd.Series) -> pd.Series:
    """Converte para datetime com suporte a formato brasileiro dd-mm-yy."""
    r = pd.to_datetime(series, errors="coerce", dayfirst=True)
    mask = r.isna()
    if mask.any():
        try:
            r[mask] = pd.to_datetime(series[mask], format="%d-%m-%y %H:%M:%S", errors="coerce")
        except Exception:
            pass
    return r


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def _insert(conn, sql: str, data: list):
    if data:
        psycopg2.extras.execute_batch(conn.cursor(), sql, data, page_size=500)


def _null(v):
    return None if (isinstance(v, float) and np.isnan(v)) else v


def _find(base: Path, patterns: list) -> list:
    """Busca arquivos por múltiplos padrões, deduplicando por path real.
    Necessário porque em Windows rglob é case-insensitive e padrões
    maiúsculos/minúsculos retornam os mesmos arquivos."""
    seen = {}
    for pat in patterns:
        for p in sorted(base.rglob(pat)):
            key = p.resolve()
            if key not in seen:
                seen[key] = p
    return sorted(seen.values())


# ════════════════════════════════════════════════════════════════════════════
# LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_geracao(ons_dir: Path, dry_run: bool) -> int:
    """Geracao_Usina_Horaria → geracao_tipo_hora (agregado por tipo, >= 2021)"""
    files = _find(ons_dir, ["Geracao_Usina_Horaria_*.xlsx",
                             "geracao_usina_horaria_*.xlsx",
                             "Geracao_Usina_Horaria_*.csv"])
    if not files:
        logger.warning("Nenhum arquivo Geracao_Usina_Horaria encontrado.")
        return 0

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            logger.info(f"Skip (ano={yr}): {fpath.name}")
            continue

        logger.info(f"Lendo geracao: {fpath.name}")
        if _should_skip_file(fpath, "geracao_tipo_hora", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        # Fallback: nom_subsistema -> id_subsistema
        if "id_subsistema" not in df.columns and "nom_subsistema" in df.columns:
            df["id_subsistema"] = df["nom_subsistema"]

        for col in ("din_instante", "id_subsistema", "nom_tipousina", "val_geracao"):
            if col not in df.columns:
                logger.warning(f"Coluna faltando: {col} em {fpath.name}")
                break
        else:
            df["din_instante"] = _ts(df["din_instante"])
            df = df.dropna(subset=["din_instante"])
            df = df[df["din_instante"].dt.year >= ANO_MIN]
            df["id_subsistema"] = df["id_subsistema"].apply(_sub)
            df["tipo_geracao"]  = df["nom_tipousina"].apply(_tipo)
            df["val_geracao"]   = _num(df["val_geracao"]).fillna(0.0)
            df = df.dropna(subset=["id_subsistema"])

            agg = df.groupby(["din_instante","id_subsistema","tipo_geracao"],
                             as_index=False)["val_geracao"].sum()
            n = len(agg)
            logger.info(f"  {len(df):,} linhas → {n:,} agregadas")

            if not dry_run:
                sql = (
                    "INSERT INTO geracao_tipo_hora "
                    "(din_instante,id_subsistema,tipo_geracao,val_geracao_mw) "
                    "VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT (din_instante,id_subsistema,tipo_geracao) "
                    "DO UPDATE SET val_geracao_mw=EXCLUDED.val_geracao_mw"
                )
                data = [(r.din_instante.to_pydatetime(), r.id_subsistema,
                         r.tipo_geracao, float(r.val_geracao))
                        for r in agg.itertuples(index=False)]
                try:
                    with db_neon.get_conn() as conn:
                        _insert(conn, sql, data)
                    logger.info(f"  OK: geracao_tipo_hora <- {n:,}")
                except Exception as e:
                    logger.error(f"  Erro: {e}")
            total += n
    return total


def load_carga(ons_dir: Path, dry_run: bool) -> int:
    """Curva_Carga → curva_carga por subsistema (>= 2021)"""
    files = _find(ons_dir, ["Curva_Carga_*.xlsx",
                             "curva_carga_*.xlsx",
                             "Curva_Carga_*.csv"])
    if not files:
        logger.warning("Nenhum arquivo Curva_Carga encontrado.")
        return 0

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            continue

        logger.info(f"Lendo carga: {fpath.name}")
        if _should_skip_file(fpath, "curva_carga", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        if "val_cargaenergiahomwmed" not in df.columns or "din_instante" not in df.columns:
            logger.warning(f"Colunas faltando em {fpath.name}")
            continue

        sub_col = next((c for c in ("id_subsistema","nom_subsistema") if c in df.columns), None)
        if not sub_col:
            logger.warning(f"Sem coluna subsistema em {fpath.name}")
            continue

        df["din_instante"] = _ts(df["din_instante"])
        df = df.dropna(subset=["din_instante"])
        df = df[df["din_instante"].dt.year >= ANO_MIN]
        df["id_subsistema"] = df[sub_col].apply(_sub)
        df = df.dropna(subset=["id_subsistema"])
        df["val_cargaenergiahomwmed"] = _num(df["val_cargaenergiahomwmed"])

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            sql = (
                "INSERT INTO curva_carga (din_instante,id_subsistema,val_cargaenergiahomwmed) "
                "VALUES (%s,%s,%s) "
                "ON CONFLICT (din_instante,id_subsistema) "
                "DO UPDATE SET val_cargaenergiahomwmed=EXCLUDED.val_cargaenergiahomwmed"
            )
            data = [(r.din_instante.to_pydatetime(), r.id_subsistema,
                     _null(r.val_cargaenergiahomwmed))
                    for r in df[["din_instante","id_subsistema",
                                  "val_cargaenergiahomwmed"]].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: curva_carga <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_cmo(ons_dir: Path, dry_run: bool) -> int:
    """CMO_SemiHorario → cmo (>= 2021)"""
    files = _find(ons_dir, ["CMO_SemiHorario_*.xlsx",
                             "CMO_*.xlsx",
                             "cmo_*.xlsx"])
    if not files:
        logger.warning("Nenhum arquivo CMO encontrado.")
        return 0

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            continue

        logger.info(f"Lendo CMO: {fpath.name}")
        if _should_skip_file(fpath, "cmo", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        if "din_instante" not in df.columns or "val_cmo" not in df.columns:
            logger.warning(f"Colunas faltando em {fpath.name}")
            continue

        sub_col = next((c for c in ("id_subsistema","nom_subsistema") if c in df.columns), None)
        df["din_instante"] = _ts(df["din_instante"])
        df = df.dropna(subset=["din_instante"])
        df = df[df["din_instante"].dt.year >= ANO_MIN]

        if sub_col:
            df["id_subsistema"] = df[sub_col].apply(_sub)
            df = df.dropna(subset=["id_subsistema"])
        else:
            df["id_subsistema"] = "SE"

        df["val_cmo"] = _num(df["val_cmo"])

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            sql = (
                "INSERT INTO cmo (din_instante,id_subsistema,val_cmo) "
                "VALUES (%s,%s,%s) "
                "ON CONFLICT (din_instante,id_subsistema) "
                "DO UPDATE SET val_cmo=EXCLUDED.val_cmo"
            )
            data = [(r.din_instante.to_pydatetime(), r.id_subsistema, _null(r.val_cmo))
                    for r in df[["din_instante","id_subsistema","val_cmo"]].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: cmo <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_ear(ons_dir: Path, dry_run: bool) -> int:
    """EAR_Diario_Subsistema → ear_diario_subsistema (>= 2021)"""
    files = _find(ons_dir, ["EAR_Diario_Subsistema_*.xlsx",
                         "ear_diario_subsistema_*.xlsx"])
    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            continue
        logger.info(f"Lendo EAR: {fpath.name}")
        if _should_skip_file(fpath, "ear_diario_subsistema", "ear_data"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in ("ear_data","din_instante") if c in df.columns), None)
        sub_col  = next((c for c in ("id_subsistema","nom_subsistema") if c in df.columns), None)
        if not date_col or not sub_col:
            logger.warning(f"Colunas faltando em {fpath.name}")
            continue

        df["ear_data"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.date
        df = df.dropna(subset=["ear_data"])
        df["id_subsistema"] = df[sub_col].apply(_sub)
        df = df.dropna(subset=["id_subsistema"])

        cols = ["ear_data","id_subsistema"]
        for c in ("ear_verif_subsistema_mwmes","ear_max_subsistema","ear_pct_subsistema"):
            if c in df.columns:
                df[c] = _num(df[c])
                cols.append(c)

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            col_str = ", ".join(cols)
            ph_str  = ", ".join(["%s"]*len(cols))
            sql = (f"INSERT INTO ear_diario_subsistema ({col_str}) VALUES ({ph_str}) "
                   "ON CONFLICT (ear_data,id_subsistema) DO NOTHING")
            data = [tuple(_null(v) for v in row)
                    for row in df[cols].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: ear_diario_subsistema <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_ena(ons_dir: Path, dry_run: bool) -> int:
    """ENA_Diario_Subsistema → ena_diario_subsistema (>= 2021)"""
    files = _find(ons_dir, ["ENA_Diario_Subsistema_*.xlsx",
                         "ena_diario_subsistema_*.xlsx"])
    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            continue
        logger.info(f"Lendo ENA: {fpath.name}")
        if _should_skip_file(fpath, "ena_diario_subsistema", "ena_data"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in ("ena_data","din_instante") if c in df.columns), None)
        sub_col  = next((c for c in ("id_subsistema","nom_subsistema") if c in df.columns), None)
        if not date_col or not sub_col:
            logger.warning(f"Colunas faltando em {fpath.name}")
            continue

        df["ena_data"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dt.date
        df = df.dropna(subset=["ena_data"])
        df["id_subsistema"] = df[sub_col].apply(_sub)
        df = df.dropna(subset=["id_subsistema"])

        cols = ["ena_data","id_subsistema"]
        for c in ("ena_bruta_regiao_mwmed","ena_armazenavel_regiao_mwmed"):
            if c in df.columns:
                df[c] = _num(df[c])
                cols.append(c)

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            col_str = ", ".join(cols)
            ph_str  = ", ".join(["%s"]*len(cols))
            sql = (f"INSERT INTO ena_diario_subsistema ({col_str}) VALUES ({ph_str}) "
                   "ON CONFLICT (ena_data,id_subsistema) DO NOTHING")
            data = [tuple(_null(v) for v in row)
                    for row in df[cols].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: ena_diario_subsistema <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_restricao(ons_dir: Path, dry_run: bool) -> int:
    """
    Restricao_Eolica/Fotovoltaica → restricao_renovavel agregada por subsistema (>= 2021)

    Por hora + subsistema + fonte:
      - val_geracao, val_geracaolimitada, val_disponibilidade, val_geracaoreferencia → SOMA
      - cod_razaorestricao  → distinct concatenados com "|"  ex: "ENE|CNF"
      - dsc_restricao       → distinct concatenados com " | " (sem duplicatas)
    """
    sources = [
        ("wind",  _find(ons_dir, ["Restricao_Eolica_*.xlsx",
                                    "restricao_eolica_*.xlsx"])),
        ("solar", _find(ons_dir, ["Restricao_Fotovoltaica_*.xlsx",
                                    "restricao_fotovoltaica_*.xlsx"])),
    ]

    _NUM_COLS = ("val_geracao", "val_geracaolimitada",
                 "val_disponibilidade", "val_geracaoreferencia")

    def _distinct_join(series: pd.Series, sep: str) -> str:
        """Retorna valores únicos não-nulos concatenados por sep."""
        vals = series.dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        return sep.join(dict.fromkeys(vals))  # preserva ordem, remove dups

    total = 0
    for fonte, files in sources:
        for fpath in files:
            yr = _year(fpath.name)
            if not yr or yr < ANO_MIN_RESTRICAO:
                logger.info(f"Skip restricao (ano={yr} < {ANO_MIN_RESTRICAO}): {fpath.name}")
                continue
            logger.info(f"Lendo restricao {fonte}: {fpath.name}")
        if _should_skip_file(fpath, "restricao_renovavel", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
            df = _read(fpath)
            if df.empty:
                continue
            df.columns = [c.strip().lower() for c in df.columns]

            if "din_instante" not in df.columns:
                logger.warning(f"Sem din_instante em {fpath.name}")
                continue

            sub_col = next((c for c in ("id_subsistema", "nom_subsistema")
                            if c in df.columns), None)
            df["din_instante"] = _ts(df["din_instante"])
            df = df.dropna(subset=["din_instante"])
            df = df[df["din_instante"].dt.year >= ANO_MIN_RESTRICAO]
            df["id_subsistema"] = df[sub_col].apply(_sub) if sub_col else "SE"
            df = df.dropna(subset=["id_subsistema"])
            df["fonte"] = fonte

            # Colunas numéricas presentes no arquivo
            num_present = [c for c in _NUM_COLS if c in df.columns]
            for c in num_present:
                df[c] = _num(df[c]).fillna(0.0)

            grp_cols = ["din_instante", "id_subsistema", "fonte"]

            # ── Agregação numérica (soma) ────────────────────────────────────
            agg_num = df.groupby(grp_cols, as_index=False)[num_present].sum()

            # ── Agregação textual (distinct por grupo) ───────────────────────
            text_rows = []
            has_cod = "cod_razaorestricao" in df.columns
            has_dsc = "dsc_restricao" in df.columns

            if has_cod or has_dsc:
                for key, grp in df.groupby(grp_cols):
                    row = {"din_instante": key[0],
                           "id_subsistema": key[1],
                           "fonte": key[2]}
                    if has_cod:
                        row["cod_razoes"] = _distinct_join(
                            grp["cod_razaorestricao"], "|")
                    if has_dsc:
                        row["dsc_restricoes"] = _distinct_join(
                            grp["dsc_restricao"], " | ")
                    text_rows.append(row)
                agg_txt = pd.DataFrame(text_rows)
                agg = agg_num.merge(agg_txt, on=grp_cols, how="left")
            else:
                agg = agg_num

            n = len(agg)
            logger.info(f"  {len(df):,} linhas → {n:,} agregadas")

            if not dry_run:
                ins_cols = grp_cols + num_present
                if "cod_razoes" in agg.columns:
                    ins_cols.append("cod_razoes")
                if "dsc_restricoes" in agg.columns:
                    ins_cols.append("dsc_restricoes")

                col_str = ", ".join(ins_cols)
                ph_str  = ", ".join(["%s"] * len(ins_cols))
                upd_cols = [c for c in ins_cols if c not in grp_cols]
                upd = ", ".join(f"{c}=EXCLUDED.{c}" for c in upd_cols)
                sql = (
                    f"INSERT INTO restricao_renovavel ({col_str}) "
                    f"VALUES ({ph_str}) "
                    f"ON CONFLICT (din_instante,id_subsistema,fonte) "
                    f"DO UPDATE SET {upd}"
                )
                data = []
                for r in agg[ins_cols].itertuples(index=False):
                    row = []
                    for i, c in enumerate(ins_cols):
                        v = r[i]
                        if c == "din_instante":
                            row.append(v.to_pydatetime() if hasattr(v, "to_pydatetime") else v)
                        else:
                            row.append(_null(v) if not isinstance(v, str) else (v or None))
                    data.append(tuple(row))
                try:
                    with db_neon.get_conn() as conn:
                        _insert(conn, sql, data)
                    logger.info(f"  OK: restricao_renovavel ({fonte}) <- {n:,}")
                except Exception as e:
                    logger.error(f"  Erro: {e}")
            total += n
    return total


def load_intercambio(ons_dir: Path, dry_run: bool) -> int:
    """Intercambio_Nacional → intercambio (>= 2021)"""
    files = _find(ons_dir, ["Intercambio_Nacional_*.xlsx",
                             "intercambio_nacional_*.xlsx"])
    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            continue
        logger.info(f"Lendo intercambio: {fpath.name}")
        if _should_skip_file(fpath, "intercambio", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        needed = {"din_instante","id_subsistema_origem","id_subsistema_destino","val_intercambiomwmed"}
        if not needed.issubset(df.columns):
            logger.warning(f"Colunas faltando em {fpath.name}: {needed - set(df.columns)}")
            continue

        df["din_instante"] = _ts(df["din_instante"])
        df = df.dropna(subset=["din_instante"])
        df = df[df["din_instante"].dt.year >= ANO_MIN]
        df["id_subsistema_origem"]  = df["id_subsistema_origem"].apply(_sub)
        df["id_subsistema_destino"] = df["id_subsistema_destino"].apply(_sub)
        df["val_intercambiomwmed"]  = _num(df["val_intercambiomwmed"])
        df = df.dropna(subset=["id_subsistema_origem","id_subsistema_destino"])

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            sql = (
                "INSERT INTO intercambio "
                "(din_instante,id_subsistema_origem,id_subsistema_destino,val_intercambiomwmed) "
                "VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (din_instante,id_subsistema_origem,id_subsistema_destino) "
                "DO UPDATE SET val_intercambiomwmed=EXCLUDED.val_intercambiomwmed"
            )
            data = [(r.din_instante.to_pydatetime(), r.id_subsistema_origem,
                     r.id_subsistema_destino, _null(r.val_intercambiomwmed))
                    for r in df[["din_instante","id_subsistema_origem",
                                  "id_subsistema_destino","val_intercambiomwmed"]].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: intercambio <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_disponibilidade(ons_dir: Path, dry_run: bool) -> int:
    """
    Disponibilidade_Usina → disponibilidade_tipo_hora agregada por tipo (>= 2021)
    core_analysis usa: id_tipousina (UHE/UTE/UTN) + val_dispsincronizada
    Agrega por (din_instante, id_subsistema, tipo_geracao) → soma val_dispsincronizada
    """
    files = _find(ons_dir, ["Disponibilidade_Usina_*.xlsx",
                             "disponibilidade_usina_*.xlsx",
                             "Disponibilidade_Usina_*.csv"])
    if not files:
        logger.warning("Nenhum arquivo Disponibilidade_Usina encontrado.")
        return 0

    # Mapeamento id_tipousina → tipo_geracao
    _TIPO_DISP = {"UHE": "hydro", "PCH": "hydro", "CGH": "hydro",
                  "UTE": "thermal", "UTN": "nuclear",
                  "EOL": "wind",   "UFV": "solar"}

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            logger.info(f"Skip (ano={yr}): {fpath.name}")
            continue

        logger.info(f"Lendo disponibilidade: {fpath.name}")
        if _should_skip_file(fpath, "disponibilidade_tipo_hora", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        tipo_col = next((c for c in ("id_tipousina", "nom_tipousina") if c in df.columns), None)
        sub_col  = next((c for c in ("id_subsistema", "nom_subsistema") if c in df.columns), None)
        if "din_instante" not in df.columns or not tipo_col or "val_dispsincronizada" not in df.columns:
            logger.warning(f"Colunas faltando em {fpath.name}: tipo={tipo_col}")
            continue

        df["din_instante"] = _ts(df["din_instante"])
        df = df.dropna(subset=["din_instante"])
        df = df[df["din_instante"].dt.year >= ANO_MIN]
        df["id_subsistema"] = df[sub_col].apply(_sub) if sub_col else "SE"
        df["tipo_geracao"]  = df[tipo_col].str.upper().str.strip().map(_TIPO_DISP)
        df = df.dropna(subset=["id_subsistema", "tipo_geracao"])
        df["val_dispsincronizada"] = _num(df["val_dispsincronizada"]).fillna(0.0)

        agg = df.groupby(["din_instante", "id_subsistema", "tipo_geracao"],
                         as_index=False)["val_dispsincronizada"].sum()
        n = len(agg)
        logger.info(f"  {len(df):,} linhas → {n:,} agregadas")

        if not dry_run:
            sql = (
                "INSERT INTO disponibilidade_tipo_hora "
                "(din_instante,id_subsistema,tipo_geracao,val_disp_sincronizada) "
                "VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (din_instante,id_subsistema,tipo_geracao) "
                "DO UPDATE SET val_disp_sincronizada=EXCLUDED.val_disp_sincronizada"
            )
            data = [(r.din_instante.to_pydatetime(), r.id_subsistema,
                     r.tipo_geracao, float(r.val_dispsincronizada))
                    for r in agg.itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: disponibilidade_tipo_hora <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_gfom(ons_dir: Path, dry_run: bool) -> int:
    """
    Despacho_GFOM → despacho_gfom AGREGADO por hora (SIN total, >= 2021)
    core_analysis usa SUM por hora de: val_verifgeracao, val_verifordemmerito,
    val_verifinflexpura, val_verifinflexibilidade, val_verifinflexembutmerito,
    val_verifordemdemeritoacimadainflex, val_verifrazaoeletrica,
    val_verifconstrainedoff, val_verifgfom
    """
    files = _find(ons_dir, ["Despacho_GFOM_*.xlsx",
                             "despacho_gfom_*.xlsx",
                             "Despacho_GFOM_*.csv"])
    if not files:
        logger.warning("Nenhum arquivo Despacho_GFOM encontrado.")
        return 0

    _GFOM_COLS = [
        "val_verifgeracao", "val_verifordemmerito", "val_verifinflexpura",
        "val_verifinflexibilidade", "val_verifinflexembutmerito",
        "val_verifordemdemeritoacimadainflex", "val_verifrazaoeletrica",
        "val_verifconstrainedoff", "val_verifgfom",
    ]

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            logger.info(f"Skip (ano={yr}): {fpath.name}")
            continue

        logger.info(f"Lendo GFOM: {fpath.name}")
        if _should_skip_file(fpath, "despacho_gfom", "din_instante"):
            logger.info(f"  SKIP (já no Neon): {fpath.name}")
            continue
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        if "din_instante" not in df.columns:
            logger.warning(f"Sem din_instante em {fpath.name}")
            continue

        df["din_instante"] = _ts(df["din_instante"])
        df = df.dropna(subset=["din_instante"])
        df = df[df["din_instante"].dt.year >= ANO_MIN]

        # Colunas presentes no arquivo (nomes podem variar levemente)
        num_present = [c for c in _GFOM_COLS if c in df.columns]
        if not num_present:
            logger.warning(f"Nenhuma coluna GFOM encontrada em {fpath.name}. Colunas: {list(df.columns[:8])}")
            continue

        for c in num_present:
            df[c] = _num(df[c]).fillna(0.0)

        agg = df.groupby("din_instante", as_index=False)[num_present].sum()
        n = len(agg)
        logger.info(f"  {len(df):,} linhas → {n:,} agregadas")

        if not dry_run:
            ins_cols = ["din_instante"] + num_present
            col_str  = ", ".join(ins_cols)
            ph_str   = ", ".join(["%s"] * len(ins_cols))
            upd      = ", ".join(f"{c}=EXCLUDED.{c}" for c in num_present)
            sql = (
                f"INSERT INTO despacho_gfom ({col_str}) VALUES ({ph_str}) "
                f"ON CONFLICT (din_instante) DO UPDATE SET {upd}"
            )
            data = []
            for r in agg.itertuples(index=False):
                row = [r.din_instante.to_pydatetime()]
                for c in num_present:
                    row.append(_null(getattr(r, c)))
                data.append(tuple(row))
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: despacho_gfom <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_cvu(ons_dir: Path, dry_run: bool) -> int:
    """
    CVU_Usina_Termica → cvu_usina_termica
    core_analysis usa: dat_fimsemana + val_cvu (média semanal SIN)
    Filtro: ano >= ANO_MIN (2021+)
    """
    files = _find(ons_dir, ["CVU_Usina_Termica_*.xlsx",
                             "cvu_usina_termica_*.xlsx",
                             "CVU_Usina_Termica_*.csv"])
    if not files:
        logger.warning("Nenhum arquivo CVU_Usina_Termica encontrado.")
        return 0

    total = 0
    for fpath in files:
        yr = _year(fpath.name)
        if not yr or yr < ANO_MIN:
            logger.info(f"Skip (ano={yr}): {fpath.name}")
            continue
        logger.info(f"Lendo CVU: {fpath.name}")
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in ("dat_fimsemana", "dat_iniciosemana") if c in df.columns), None)
        if not date_col or "val_cvu" not in df.columns:
            logger.warning(f"Colunas faltando em {fpath.name}")
            continue

        df["dat_fimsemana"]   = pd.to_datetime(df.get("dat_fimsemana",   df[date_col]),
                                                errors="coerce", dayfirst=True).dt.date
        df["dat_iniciosemana"] = pd.to_datetime(df.get("dat_iniciosemana", df[date_col]),
                                                errors="coerce", dayfirst=True).dt.date
        df["val_cvu"] = _num(df["val_cvu"])
        df = df.dropna(subset=["dat_fimsemana", "val_cvu"])
        df = df[df["val_cvu"] > 0]

        # Manter nom_usina e id_subsistema se existirem
        keep = ["dat_iniciosemana", "dat_fimsemana", "val_cvu"]
        for opt in ("nom_usina", "id_subsistema"):
            if opt in df.columns:
                keep.append(opt)

        df = df[keep].drop_duplicates(subset=["dat_fimsemana"] +
                                      (["nom_usina"] if "nom_usina" in keep else []))
        n = len(df)
        logger.info(f"  {n:,} linhas")

        if not dry_run:
            col_str = ", ".join(keep)
            ph_str  = ", ".join(["%s"] * len(keep))
            sql = (
                f"INSERT INTO cvu_usina_termica ({col_str}) VALUES ({ph_str}) "
                f"ON CONFLICT (dat_iniciosemana, nom_usina) DO UPDATE SET val_cvu=EXCLUDED.val_cvu"
                if "nom_usina" in keep else
                f"INSERT INTO cvu_usina_termica ({col_str}) VALUES ({ph_str}) "
                f"ON CONFLICT DO NOTHING"
            )
            data = [tuple(_null(v) if not isinstance(v, str) else v
                          for v in row)
                    for row in df[keep].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: cvu_usina_termica <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


def load_pld(data_dir: Path, dry_run: bool) -> int:
    """
    ccee_pld_*.csv (em data/) → pld_historical
    MES_REFERENCIA=202603 → ano=2026, mes=3
    Filtro: ano >= ANO_MIN
    """
    csv_files = sorted(data_dir.glob("ccee_pld_*.csv"))
    if not csv_files:
        csv_files = sorted((data_dir / "ccee").glob("ccee_pld_*.csv"))
    if not csv_files:
        logger.warning(f"Nenhum ccee_pld_*.csv em {data_dir}")
        return 0

    # Usar arquivo mais recente por ano
    by_year: Dict[int, Path] = {}
    for f in csv_files:
        m = re.search(r"ccee_pld_(\d{4})_", f.name)
        if m:
            yr = int(m.group(1))
            if yr not in by_year or f.name > by_year[yr].name:
                by_year[yr] = f

    total = 0
    for year, fpath in sorted(by_year.items()):
        if year < ANO_MIN:
            logger.info(f"Skip PLD ano={year}: {fpath.name}")
            continue

        logger.info(f"Lendo PLD: {fpath.name} (ano={year})")
        df = _read(fpath)
        if df.empty:
            continue
        df.columns = [c.strip().upper() for c in df.columns]

        needed = {"MES_REFERENCIA","SUBMERCADO","DIA","HORA","PLD_HORA"}
        if not needed.issubset(df.columns):
            logger.warning(f"Colunas faltando: {needed - set(df.columns)}")
            continue

        df["MES_REFERENCIA"] = pd.to_numeric(df["MES_REFERENCIA"], errors="coerce")
        df = df.dropna(subset=["MES_REFERENCIA"])
        df["_ano"] = (df["MES_REFERENCIA"] // 100).astype(int)
        df = df[df["_ano"] >= ANO_MIN]
        if df.empty:
            continue

        df["PLD_HORA"]       = _num(df["PLD_HORA"])
        df["DIA"]            = pd.to_numeric(df["DIA"], errors="coerce").astype("Int64")
        df["HORA"]           = pd.to_numeric(df["HORA"], errors="coerce").astype("Int64")
        df["MES_REFERENCIA"] = df["MES_REFERENCIA"].astype(int)
        df["SUBMERCADO"]     = df["SUBMERCADO"].str.upper().str.strip()
        df = df.dropna(subset=["DIA","HORA","PLD_HORA"])

        n = len(df)
        logger.info(f"  {n:,} linhas")
        if not dry_run:
            sql = (
                "INSERT INTO pld_historical (mes_referencia,submercado,dia,hora,pld_hora) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (mes_referencia,submercado,dia,hora) "
                "DO UPDATE SET pld_hora=EXCLUDED.pld_hora"
            )
            data = [(int(r.MES_REFERENCIA), r.SUBMERCADO,
                     int(r.DIA), int(r.HORA), _null(r.PLD_HORA))
                    for r in df[["MES_REFERENCIA","SUBMERCADO","DIA","HORA","PLD_HORA"]].itertuples(index=False)]
            try:
                with db_neon.get_conn() as conn:
                    _insert(conn, sql, data)
                logger.info(f"  OK: pld_historical <- {n:,}")
            except Exception as e:
                logger.error(f"  Erro: {e}")
        total += n
    return total


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

LOADERS = {
    "geracao":         ("geracao_tipo_hora",        lambda d, dr: load_geracao(d / "ons", dr)),
    "carga":           ("curva_carga",              lambda d, dr: load_carga(d / "ons", dr)),
    "cmo":             ("cmo",                      lambda d, dr: load_cmo(d / "ons", dr)),
    "ear":             ("ear_diario_subsistema",    lambda d, dr: load_ear(d / "ons", dr)),
    "ena":             ("ena_diario_subsistema",    lambda d, dr: load_ena(d / "ons", dr)),
    "restricao":       ("restricao_renovavel",      lambda d, dr: load_restricao(d / "ons", dr)),
    "intercambio":     ("intercambio",              lambda d, dr: load_intercambio(d / "ons", dr)),
    "disponibilidade": ("disponibilidade_tipo_hora",lambda d, dr: load_disponibilidade(d / "ons", dr)),
    "gfom":            ("despacho_gfom",            lambda d, dr: load_gfom(d / "ons", dr)),
    "cvu":             ("cvu_usina_termica",        lambda d, dr: load_cvu(d / "ons", dr)),
    "pld":             ("pld_historical",           lambda d, dr: load_pld(d, dr)),
}


def main():
    parser = argparse.ArgumentParser(
        description=f"Carga Neon PostgreSQL — dados a partir de {ANO_MIN}"
    )
    parser.add_argument("--dir",     default="data")
    parser.add_argument("--only",    default="",
                        help=f"Loader: {', '.join(LOADERS.keys())}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not db_neon.is_configured():
        print("ERRO: DATABASE_URL nao configurada.")
        sys.exit(1)

    row = db_neon.fetchone("SELECT version()")
    if not row:
        print("ERRO: Sem conexao com Neon.")
        sys.exit(1)
    logger.info(f"Conexao OK: {row[0][:60]}")
    logger.info(f"Filtro: ano >= {ANO_MIN} | dir: {args.dir}")
    if args.dry_run:
        logger.info("[DRY-RUN] Nenhuma linha sera inserida.")

    data_dir = Path(args.dir)
    only = args.only.lower().strip()

    if only and only not in LOADERS:
        print(f"ERRO: --only '{only}' invalido. Opcoes: {', '.join(LOADERS.keys())}")
        sys.exit(1)

    to_run = {only: LOADERS[only]} if only else LOADERS

    total = 0
    results = {}
    for key, (table, fn) in to_run.items():
        logger.info("=" * 55)
        logger.info(f">>> {key} → {table}")
        logger.info("=" * 55)
        try:
            n = fn(data_dir, args.dry_run)
        except Exception as e:
            logger.error(f"Falha em {key}: {e}")
            n = 0
        results[table] = n
        total += n

    logger.info("=" * 55)
    logger.info("RESUMO:")
    for t, n in results.items():
        logger.info(f"  {t}: {n:,}")
    logger.info(f"  TOTAL: {total:,}")
    if args.dry_run:
        logger.info("[DRY-RUN] Nada inserido.")
    logger.info("=" * 55)

    if not args.dry_run:
        logger.info("Contagem final no Neon:")
        for table in [t for t, _ in LOADERS.values()]:
            try:
                r = db_neon.fetchone(f'SELECT COUNT(*) FROM "{table}"')
                c = r[0] if r else "erro"
                logger.info(f"  {table}: {c:,}" if isinstance(c, int) else f"  {table}: {c}")
            except Exception:
                logger.info(f"  {table}: (nao existe)")


if __name__ == "__main__":
    main()
