#!/usr/bin/env python3
"""
update_neon.py — Rotina diária de atualização Kintuadi Energy
─────────────────────────────────────────────────────────────
Fluxo:
  1. Coleta novos arquivos (ons_collector_v2 + ccee_collector_v2
     via integrated_collector_v2)
  2. Carrega dados novos no Neon PostgreSQL (load_neon.py --only ...)
  3. Faz commit vazio no GitHub → Render detecta e reinicia o app

Uso:
  python update_neon.py              # execução completa
  python update_neon.py --dry-run    # simula sem gravar no Neon nem fazer push
  python update_neon.py --skip-collect  # pula coleta, só atualiza Neon + push
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path("logs") / f"update_neon_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

# ─── utilitários git ─────────────────────────────────────────────────────────

def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def _git_push_empty_commit(message: str) -> bool:
    """Cria um commit vazio e faz push para triggar o Render."""
    try:
        # Verifica se há algo para commitar ou cria commit vazio
        _git("add", "-A")
        result = _git("diff", "--cached", "--quiet", check=False)
        if result.returncode == 0:
            # Nada staged — cria commit vazio para forçar rebuild no Render
            _git("commit", "--allow-empty", "-m", message)
            log.info("Commit vazio criado para triggar rebuild no Render.")
        else:
            _git("commit", "-m", message)
            log.info("Commit com alterações criado.")

        _git("push")
        log.info("Push para GitHub realizado com sucesso.")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Falha no git push: {e.stderr or e}")
        return False


# ─── limpeza do ano corrente ─────────────────────────────────────────────────

def _clear_current_year_ons(data_dir: str = "data") -> None:
    """Remove arquivos ONS do ano corrente para forçar re-download.

    Arquivos do ano em curso são sempre incompletos (dados do dia anterior
    ainda não consolidados). Anos anteriores já estão completos e não são
    tocados.
    """
    ano = datetime.now().year
    target = Path(data_dir) / "ons" / str(ano)
    if not target.exists():
        log.info(f"Pasta {target} não encontrada — nada a limpar.")
        return
    removed = 0
    for f in target.rglob("*"):
        if f.is_file():
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                log.warning(f"Não foi possível remover {f}: {e}")
    log.info(f"Limpeza do ano {ano}: {removed} arquivo(s) removido(s) em {target}")


# ─── etapa 1: coleta ─────────────────────────────────────────────────────────

def step_collect(dry_run: bool, data_dir: str = "data") -> bool:
    log.info("=" * 60)
    log.info("ETAPA 1 — Coleta de novos dados (ONS + CCEE)")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY-RUN] Pulando limpeza e coleta.")
        return True

    # Limpar arquivos do ano corrente antes de coletar para garantir
    # que dados incompletos do dia anterior sejam re-baixados
    _clear_current_year_ons(data_dir)

    try:
        from scripts.integrated_collector_v2 import KintuadiIntegratedCollectorV2
        collector = KintuadiIntegratedCollectorV2()
        result = collector.quick_collect(persist_mode="incremental")
        if result:
            log.info("Coleta concluída com sucesso.")
            return True
        else:
            log.error("Coleta retornou resultado vazio/falso.")
            return False
    except Exception as e:
        log.error(f"Falha na coleta: {e}", exc_info=True)
        return False


# ─── etapa 2: carga no Neon ──────────────────────────────────────────────────

def step_load_neon(dry_run: bool, data_dir: str = "data") -> bool:
    log.info("=" * 60)
    log.info("ETAPA 2 — Carga incremental no Neon PostgreSQL")
    log.info("=" * 60)

    cmd = [
        sys.executable, "load_neon.py",
        "--dir", data_dir,
    ]
    if dry_run:
        cmd.append("--dry-run")

    log.info(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        log.info("Carga no Neon concluída com sucesso.")
        return True
    else:
        log.error(f"load_neon.py terminou com código {result.returncode}.")
        return False


# ─── etapa 3: push para triggar Render ───────────────────────────────────────

def step_trigger_render(dry_run: bool) -> bool:
    log.info("=" * 60)
    log.info("ETAPA 3 — Commit + push para triggar rebuild no Render")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY-RUN] Pulando push.")
        return True

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = f"chore: atualização automática diária [{ts}]"
    return _git_push_empty_commit(message)


# ─── pipeline principal ───────────────────────────────────────────────────────

def run(dry_run: bool = False, skip_collect: bool = False, data_dir: str = "data") -> bool:
    Path("logs").mkdir(exist_ok=True)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   KINTUADI ENERGY — ATUALIZAÇÃO DIÁRIA NEON              ║")
    log.info(f"║   {datetime.now().strftime('%d/%m/%Y %H:%M:%S'):<54}║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # Etapa 1 — coleta
    if not skip_collect:
        ok = step_collect(dry_run, data_dir)
        if not ok:
            log.warning("Coleta falhou — continuando com dados existentes em data/.")
    else:
        log.info("Etapa 1 pulada (--skip-collect).")

    # Etapa 2 — carga Neon
    ok = step_load_neon(dry_run, data_dir)
    if not ok:
        log.error("Carga no Neon falhou. Abortando.")
        return False

    # Etapa 3 — push → Render rebuild
    ok = step_trigger_render(dry_run)
    if not ok:
        log.warning("Push falhou — Neon foi atualizado mas o Render não foi notificado.")
        log.warning("Faça um push manual para triggar o rebuild: git push")

    log.info("✅ Pipeline finalizado.")
    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Atualização diária Neon + Render")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sem gravar no Neon nem fazer push")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Pula a coleta ONS/CCEE, usa arquivos existentes em data/")
    parser.add_argument("--dir", default="data",
                        help="Diretório dos arquivos de dados (default: data)")
    args = parser.parse_args()

    ok = run(dry_run=args.dry_run, skip_collect=args.skip_collect, data_dir=args.dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
