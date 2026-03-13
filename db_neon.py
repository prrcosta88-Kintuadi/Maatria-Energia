"""
db_neon.py — Módulo de conexão compartilhado com Neon PostgreSQL
================================================================
Usado por:
  - integrated_collector_v2.py  (INSERT/UPSERT dos dados brutos ONS + CCEE)
  - core_analysis.py            (SELECT para processar análise)
  - app.py                      (SELECT para o dashboard)

Configuração:
  Defina a variável de ambiente DATABASE_URL com a connection string do Neon:
  postgresql://usuario:senha@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require

  Localmente: crie um arquivo .env na raiz do projeto com:
    DATABASE_URL=postgresql://...

  No Render: adicione DATABASE_URL em Environment → Add Environment Variable
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Dependências opcionais ────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_OK = True
except ImportError:
    _PSYCOPG2_OK = False
    logger.warning("psycopg2 não instalado. Instale com: pip install psycopg2-binary")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv opcional — variáveis de ambiente do sistema bastam


# ── Conexão ───────────────────────────────────────────────────────────────────

def get_database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL")


def is_configured() -> bool:
    return bool(get_database_url()) and _PSYCOPG2_OK


@contextmanager
def get_conn() -> Generator:
    """
    Context manager que abre e fecha a conexão automaticamente.

    Uso:
        with db_neon.get_conn() as conn:
            df = pd.read_sql("SELECT ...", conn)
    """
    url = get_database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL não configurada. "
            "Defina a variável de ambiente ou crie um arquivo .env"
        )
    if not _PSYCOPG2_OK:
        raise RuntimeError("psycopg2 não instalado. Execute: pip install psycopg2-binary")

    conn = psycopg2.connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Helpers de leitura ────────────────────────────────────────────────────────

def fetchdf(sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
    """
    Executa uma query SELECT e retorna um DataFrame.
    Retorna DataFrame vazio em caso de erro ou banco não configurado.
    """
    if not is_configured():
        return pd.DataFrame()
    try:
        with get_conn() as conn:
            return pd.read_sql(sql, conn, params=params)
    except Exception as e:
        logger.error(f"fetchdf error: {e}\nSQL: {sql[:200]}")
        return pd.DataFrame()


def fetchone(sql: str, params: Optional[List[Any]] = None) -> Optional[tuple]:
    """Executa query e retorna a primeira linha ou None."""
    if not is_configured():
        return None
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            return cur.fetchone()
    except Exception as e:
        logger.error(f"fetchone error: {e}")
        return None


def execute(sql: str, params: Optional[List[Any]] = None) -> None:
    """Executa DDL ou DML (CREATE, INSERT, UPDATE, DELETE)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or [])


def table_exists(table_name: str) -> bool:
    """Verifica se uma tabela existe no schema public."""
    row = fetchone(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s",
        [table_name.lower()]
    )
    return row is not None


def upsert_df(
    df: pd.DataFrame,
    table: str,
    conflict_cols: List[str],
    schema: str = "public",
) -> int:
    """
    Insere DataFrame no PostgreSQL com ON CONFLICT DO NOTHING.
    Retorna número de linhas inseridas.

    Para UPSERT real (atualizar se já existir), use upsert_df_update().
    """
    if df.empty:
        return 0
    if not is_configured():
        return 0

    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join([f'"{c}"' for c in cols])
    conflict = ", ".join([f'"{c}"' for c in conflict_cols])

    sql = (
        f'INSERT INTO {schema}."{table}" ({col_names}) '
        f"VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO NOTHING"
    )

    rows_inserted = 0
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            data = [tuple(row) for row in df.itertuples(index=False, name=None)]
            psycopg2.extras.execute_batch(cur, sql, data, page_size=1000)
            rows_inserted = cur.rowcount
    except Exception as e:
        logger.error(f"upsert_df error em {table}: {e}")
        raise

    return rows_inserted


# ── DDL — criação das tabelas ─────────────────────────────────────────────────

SCHEMA_SQL = """
-- Schema otimizado para Neon free tier (~93 MB)
-- Geração e disponibilidade agregadas por tipo, restrições por subsistema
-- Dados a partir de 2021 apenas

-- Geração horária AGREGADA por tipo (6 tipos: solar, wind, hydro, thermal, nuclear, mmgd)
CREATE TABLE IF NOT EXISTS geracao_tipo_hora (
    din_instante    TIMESTAMP,
    id_subsistema   VARCHAR(10),
    tipo_geracao    VARCHAR(20),
    val_geracao_mw  DOUBLE PRECISION,
    ano             INTEGER,
    mes             INTEGER,
    PRIMARY KEY (din_instante, id_subsistema, tipo_geracao)
);
CREATE INDEX IF NOT EXISTS idx_gth_instante ON geracao_tipo_hora (din_instante);
CREATE INDEX IF NOT EXISTS idx_gth_tipo     ON geracao_tipo_hora (tipo_geracao);
CREATE INDEX IF NOT EXISTS idx_gth_sub      ON geracao_tipo_hora (id_subsistema);

-- Curva de carga por subsistema
CREATE TABLE IF NOT EXISTS curva_carga (
    din_instante                TIMESTAMP,
    id_subsistema               VARCHAR(10),
    nom_subsistema              VARCHAR(60),
    val_cargaenergiahomwmed     DOUBLE PRECISION,
    ano                         INTEGER,
    mes                         INTEGER,
    PRIMARY KEY (din_instante, id_subsistema)
);
CREATE INDEX IF NOT EXISTS idx_cc_instante ON curva_carga (din_instante);

-- CMO semi-horário
CREATE TABLE IF NOT EXISTS cmo (
    din_instante    TIMESTAMP,
    id_subsistema   VARCHAR(10),
    nom_subsistema  VARCHAR(60),
    val_cmo         DOUBLE PRECISION,
    ano             INTEGER,
    mes             INTEGER,
    PRIMARY KEY (din_instante, id_subsistema)
);
CREATE INDEX IF NOT EXISTS idx_cmo_instante ON cmo (din_instante);

-- EAR diário por subsistema
CREATE TABLE IF NOT EXISTS ear_diario_subsistema (
    id_subsistema                       VARCHAR(10),
    nom_subsistema                      VARCHAR(60),
    ear_data                            DATE,
    ear_max_subsistema                  DOUBLE PRECISION,
    ear_verif_subsistema_mwmes          DOUBLE PRECISION,
    ear_verif_subsistema_percentual     DOUBLE PRECISION,
    ano                                 INTEGER,
    mes                                 INTEGER,
    PRIMARY KEY (ear_data, id_subsistema)
);
CREATE INDEX IF NOT EXISTS idx_ear_data ON ear_diario_subsistema (ear_data);

-- ENA diário por subsistema
CREATE TABLE IF NOT EXISTS ena_diario_subsistema (
    id_subsistema                           VARCHAR(10),
    nom_subsistema                          VARCHAR(60),
    ena_data                                DATE,
    ena_bruta_regiao_mwmed                  DOUBLE PRECISION,
    ena_bruta_regiao_percentualmlt          DOUBLE PRECISION,
    ena_armazenavel_regiao_mwmed            DOUBLE PRECISION,
    ena_armazenavel_regiao_percentualmlt    DOUBLE PRECISION,
    ano                                     INTEGER,
    mes                                     INTEGER,
    PRIMARY KEY (ena_data, id_subsistema)
);
CREATE INDEX IF NOT EXISTS idx_ena_data ON ena_diario_subsistema (ena_data);

-- CVU de usinas térmicas (mantida por usina — pequena)
CREATE TABLE IF NOT EXISTS cvu_usina_termica (
    dat_iniciosemana        DATE,
    dat_fimsemana           DATE,
    ano_referencia          INTEGER,
    mes_referencia          INTEGER,
    num_revisao             INTEGER,
    nom_semanaoperativa     VARCHAR(60),
    cod_usinaplanejamento   INTEGER,
    id_subsistema           VARCHAR(10),
    nom_subsistema          VARCHAR(60),
    nom_usina               VARCHAR(120),
    val_cvu                 DOUBLE PRECISION,
    ano                     INTEGER,
    PRIMARY KEY (dat_iniciosemana, nom_usina)
);

-- Disponibilidade AGREGADA por tipo de usina
CREATE TABLE IF NOT EXISTS disponibilidade_tipo_hora (
    din_instante            TIMESTAMP,
    id_subsistema           VARCHAR(10),
    tipo_geracao            VARCHAR(20),
    val_potencia_instalada  DOUBLE PRECISION,
    val_disp_operacional    DOUBLE PRECISION,
    val_disp_sincronizada   DOUBLE PRECISION,
    ano                     INTEGER,
    mes                     INTEGER,
    PRIMARY KEY (din_instante, id_subsistema, tipo_geracao)
);
CREATE INDEX IF NOT EXISTS idx_dth_instante ON disponibilidade_tipo_hora (din_instante);

-- Restrição renovável AGREGADA por subsistema (eólica e solar separadas)
-- Valores numéricos: soma de todas as usinas do subsistema
-- cod_razoes / dsc_restricoes: distinct de códigos/descrições que ocorreram naquela hora
CREATE TABLE IF NOT EXISTS restricao_renovavel (
    din_instante            TIMESTAMP,
    id_subsistema           VARCHAR(10),
    fonte                   VARCHAR(10),       -- wind | solar
    val_geracao             DOUBLE PRECISION,  -- soma MW gerado
    val_geracaolimitada     DOUBLE PRECISION,  -- soma MW limitado
    val_disponibilidade     DOUBLE PRECISION,  -- soma MW disponível
    val_geracaoreferencia   DOUBLE PRECISION,  -- soma MW referência
    cod_razoes              TEXT,              -- códigos distinct, ex: "ENE|CNF"
    dsc_restricoes          TEXT,              -- descrições distinct separadas por " | "
    ano                     INTEGER,
    mes                     INTEGER,
    PRIMARY KEY (din_instante, id_subsistema, fonte)
);
CREATE INDEX IF NOT EXISTS idx_rr_instante ON restricao_renovavel (din_instante);

-- Despacho GFOM AGREGADO por hora (SIN total — sem usina, sem patamar)
-- Colunas usadas pelo core_analysis: val_verifgeracao, val_verifordemmerito,
-- val_verifinflexpura, val_verifinflexibilidade, val_verifinflexembutmerito,
-- val_verifordemdemeritoacimadainflex, val_verifrazaoeletrica,
-- val_verifconstrainedoff, val_verifgfom
CREATE TABLE IF NOT EXISTS despacho_gfom (
    din_instante                            TIMESTAMP,
    val_verifgeracao                        DOUBLE PRECISION,
    val_verifordemmerito                    DOUBLE PRECISION,
    val_verifinflexpura                     DOUBLE PRECISION,
    val_verifinflexibilidade                DOUBLE PRECISION,
    val_verifinflexembutmerito              DOUBLE PRECISION,
    val_verifordemdemeritoacimadainflex     DOUBLE PRECISION,
    val_verifrazaoeletrica                  DOUBLE PRECISION,
    val_verifconstrainedoff                 DOUBLE PRECISION,
    val_verifgfom                           DOUBLE PRECISION,
    ano                                     INTEGER,
    mes                                     INTEGER,
    PRIMARY KEY (din_instante)
);
CREATE INDEX IF NOT EXISTS idx_gfom_instante ON despacho_gfom (din_instante);

-- Intercâmbio nacional
CREATE TABLE IF NOT EXISTS intercambio (
    din_instante            TIMESTAMP,
    id_subsistema_origem    VARCHAR(10),
    nom_subsistema_origem   VARCHAR(60),
    id_subsistema_destino   VARCHAR(10),
    nom_subsistema_destino  VARCHAR(60),
    val_intercambiomwmed    DOUBLE PRECISION,
    ano                     INTEGER,
    mes                     INTEGER,
    PRIMARY KEY (din_instante, id_subsistema_origem, id_subsistema_destino)
);

-- PLD histórico CCEE
CREATE TABLE IF NOT EXISTS pld_historical (
    mes_referencia              INTEGER,
    submercado                  VARCHAR(20),
    periodo_comercializacao     INTEGER,
    dia                         INTEGER,
    hora                        INTEGER,
    pld_hora                    DOUBLE PRECISION,
    ano                         INTEGER,
    mes                         INTEGER,
    PRIMARY KEY (mes_referencia, submercado, dia, hora)
);
CREATE INDEX IF NOT EXISTS idx_pld_mes ON pld_historical (mes_referencia);
CREATE INDEX IF NOT EXISTS idx_pld_sub ON pld_historical (submercado);
"""


def create_tables() -> None:
    """Cria todas as tabelas no Neon. Execute uma vez na configuração inicial."""
    logger.info("Criando tabelas no Neon...")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
    logger.info("Tabelas criadas com sucesso.")


# ── CLI para setup ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Utilitários Neon PostgreSQL")
    parser.add_argument("--create-tables", action="store_true", help="Cria as tabelas no Neon")
    parser.add_argument("--test",          action="store_true", help="Testa a conexão")
    parser.add_argument("--list-tables",   action="store_true", help="Lista tabelas existentes")
    args = parser.parse_args()

    if args.test:
        if not get_database_url():
            print("❌ DATABASE_URL não definida.")
        else:
            row = fetchone("SELECT version()")
            if row:
                print(f"✅ Conexão OK — {row[0]}")
            else:
                print("❌ Falha na conexão.")

    elif args.create_tables:
        create_tables()
        print("✅ Tabelas criadas.")

    elif args.list_tables:
        df = fetchdf(
            "SELECT table_name, pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size "
            "FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
        )
        if df.empty:
            print("Nenhuma tabela encontrada.")
        else:
            print(df.to_string(index=False))

    else:
        parser.print_help()
