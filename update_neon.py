#!/usr/bin/env python3
"""
update_neon.py - Rotina diaria de atualizacao Kintuadi Energy.

Fluxo:
  1. Coleta novos arquivos (ONS + CCEE)
  2. Carrega dados novos no Neon PostgreSQL
  3. Faz commit/push para trigger de rebuild no Render
  4. Gera relatorios periodicos
  5. Atualiza o snapshot diario do AdaptivePLDForwardEngine
  6. Executa o retreino semanal do stack de forecast
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


def _git(*args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def _git_push_empty_commit(message: str) -> bool:
    """Cria um commit vazio e faz push para trigger do Render."""
    try:
        _git("add", "-A")
        result = _git("diff", "--cached", "--quiet", check=False)
        if result.returncode == 0:
            _git("commit", "--allow-empty", "-m", message)
            log.info("Commit vazio criado para trigger do rebuild no Render.")
        else:
            _git("commit", "-m", message)
            log.info("Commit com alteracoes criado.")

        _git("push")
        log.info("Push para GitHub realizado com sucesso.")
        return True
    except subprocess.CalledProcessError as exc:
        log.error("Falha no git push: %s", exc.stderr or exc)
        return False


def _clear_current_year_ons(data_dir: str = "data") -> None:
    """Remove arquivos ONS do ano corrente para forcar re-download."""
    year = datetime.now().year
    target = Path(data_dir) / "ons" / str(year)
    if not target.exists():
        log.info("Pasta %s nao encontrada - nada a limpar.", target)
        return

    removed = 0
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except Exception as exc:
            log.warning("Nao foi possivel remover %s: %s", path, exc)
    log.info("Limpeza do ano %s: %s arquivo(s) removido(s) em %s", year, removed, target)


def step_collect(dry_run: bool, data_dir: str = "data") -> bool:
    log.info("=" * 60)
    log.info("ETAPA 1 - Coleta de novos dados (ONS + CCEE)")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY-RUN] Pulando limpeza e coleta.")
        return True

    _clear_current_year_ons(data_dir)

    try:
        from scripts.integrated_collector_v2 import KintuadiIntegratedCollectorV2

        collector = KintuadiIntegratedCollectorV2()
        result = collector.quick_collect(persist_mode="incremental")
        if result:
            log.info("Coleta concluida com sucesso.")
            return True
        log.error("Coleta retornou resultado vazio/falso.")
        return False
    except Exception as exc:
        log.error("Falha na coleta: %s", exc, exc_info=True)
        return False


def step_load_neon(dry_run: bool, data_dir: str = "data") -> bool:
    log.info("=" * 60)
    log.info("ETAPA 2 - Carga incremental no Neon PostgreSQL")
    log.info("=" * 60)

    cmd = [sys.executable, "load_neon.py", "--dir", data_dir]
    if dry_run:
        cmd.append("--dry-run")

    log.info("Executando: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        log.info("Carga no Neon concluida com sucesso.")
        return True

    log.error("load_neon.py terminou com codigo %s.", result.returncode)
    return False


def step_trigger_render(dry_run: bool) -> bool:
    log.info("=" * 60)
    log.info("ETAPA 3 - Commit + push para trigger do rebuild no Render")
    log.info("=" * 60)

    if dry_run:
        log.info("[DRY-RUN] Pulando push.")
        return True

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    message = f"chore: atualizacao automatica diaria [{ts}]"
    return _git_push_empty_commit(message)


def _run_forecast_retrain(
    dry_run: bool = False,
    pmo_xlsx: str = "data/ons/PMOs/validacao_pmo.xlsx",
    model_dir: str = "data/models",
    duckdb_path: str = "data/kintuadi.duckdb",
    force: bool = False,
    adaptive_only: bool = False,
    include_adaptive: bool = True,
) -> None:
    """
    Etapa 6 - Retreino semanal do stack de forecast.

    Por padrao, roda toda segunda-feira quando chamado pelo pipeline diario.
    Quando `force=True`, executa imediatamente.
    """
    now = datetime.now()
    if not force and now.weekday() != 0:
        log.info("Retreino de forecast: nao e segunda-feira, pulando.")
        return

    if adaptive_only:
        log.info("Retreino de forecast em modo adaptive-only: modelos legados de PLD serao pulados.")

    try:
        from pld_forecast_engine import build_and_train
    except ImportError:
        build_and_train = None
        log.warning("pld_forecast_engine.py nao encontrado - retreino legado pulado.")

    if build_and_train is not None and not adaptive_only:
        try:
            log.info("Retreinando modelos legados de PLD (semanal)...")
            if not dry_run:
                build_and_train(
                    xlsx_path=Path(pmo_xlsx),
                    model_dir=Path(model_dir),
                )
                log.info("Modelos legados de PLD retreinados com sucesso.")
            else:
                log.info("  [dry-run] retreino legado de PLD pulado.")
        except Exception as exc:
            log.warning("Retreino legado de PLD falhou: %s", exc)

    try:
        from adaptive_pld_forward_engine import AdaptivePLDForwardEngine
    except ImportError:
        AdaptivePLDForwardEngine = None
        log.warning("adaptive_pld_forward_engine.py nao encontrado - snapshot estrutural pulado.")

    if AdaptivePLDForwardEngine is not None and include_adaptive:
        try:
            log.info("Gerando snapshot estrutural do AdaptivePLDForwardEngine (6 meses)...")
            if not dry_run:
                engine = AdaptivePLDForwardEngine(
                    duckdb_path=duckdb_path,
                    horizon_hours=24 * 30 * 6,
                    n_paths=1000,
                )
                result = engine.run(persist=True)
                log.info(
                    "Snapshot estrutural persistido com sucesso. run_id=%s | linhas=%s",
                    result.run_id,
                    len(result.hourly_table),
                )
            else:
                log.info("  [dry-run] snapshot estrutural pulado.")
        except Exception as exc:
            log.warning("Snapshot estrutural adaptativo falhou: %s", exc)


def _run_adaptive_forward_refresh(
    dry_run: bool = False,
    duckdb_path: str = "data/kintuadi.duckdb",
    horizon_hours: int = 24 * 30 * 6,
    n_paths: int = 1000,
) -> None:
    """
    Refresh diario do forward adaptativo com auto-recalibracao.

    A rotina usa o PLD realizado mais recente ja carregado no banco para
    recalcular o snapshot estrutural sem esperar a janela semanal do PMO.
    """
    try:
        from adaptive_pld_forward_engine import AdaptivePLDForwardEngine
    except ImportError:
        log.warning("adaptive_pld_forward_engine.py nao encontrado - refresh adaptativo diario pulado.")
        return

    try:
        log.info("Atualizando snapshot adaptativo diario com feedback do PLD realizado...")
        if not dry_run:
            engine = AdaptivePLDForwardEngine(
                duckdb_path=duckdb_path,
                horizon_hours=horizon_hours,
                n_paths=n_paths,
            )
            result = engine.run(persist=True)
            log.info(
                "Snapshot adaptativo diario persistido com sucesso. run_id=%s | linhas=%s",
                result.run_id,
                len(result.hourly_table),
            )
        else:
            log.info("  [dry-run] refresh adaptativo diario pulado.")
    except Exception as exc:
        log.warning("Refresh adaptativo diario falhou: %s", exc)


def _run_reports(dry_run: bool = False) -> None:
    """
    Gera relatorios periodicos via report_engine.py.
    - Flash diario: toda execucao
    - Relatorio semanal: toda segunda-feira
    - Relatorio mensal: todo dia 1
    """
    try:
        from report_engine import run as _report_run
    except ImportError:
        log.warning("report_engine.py nao encontrado - relatorios pulados.")
        return

    now = datetime.now()

    try:
        log.info("Gerando flash diario...")
        if not dry_run:
            _report_run(period="daily")
        else:
            log.info("  [dry-run] flash diario pulado.")
    except Exception as exc:
        log.warning("Flash diario falhou: %s", exc)

    if now.weekday() == 0:
        try:
            log.info("Gerando relatorio semanal (segunda-feira)...")
            if not dry_run:
                _report_run(period="weekly")
            else:
                log.info("  [dry-run] semanal pulado.")
        except Exception as exc:
            log.warning("Relatorio semanal falhou: %s", exc)

    if now.day == 1:
        try:
            log.info("Gerando relatorio mensal (dia 1)...")
            if not dry_run:
                _report_run(period="monthly")
            else:
                log.info("  [dry-run] mensal pulado.")
        except Exception as exc:
            log.warning("Relatorio mensal falhou: %s", exc)


def run(
    dry_run: bool = False,
    skip_collect: bool = False,
    data_dir: str = "data",
    allow_neon_failure: bool = False,
) -> bool:
    Path("logs").mkdir(exist_ok=True)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   KINTUADI ENERGY — ATUALIZAÇÃO DIÁRIA NEON              ║")
    log.info(f"║   {datetime.now().strftime('%d/%m/%Y %H:%M:%S'):<54}║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    if not skip_collect:
        ok = step_collect(dry_run, data_dir)
        if not ok:
            log.warning("Coleta falhou - continuando com dados existentes em data/.")
    else:
        log.info("Etapa 1 pulada (--skip-collect).")

    neon_ok = step_load_neon(dry_run, data_dir)
    if not neon_ok:
        if dry_run or allow_neon_failure:
            log.warning("Carga no Neon falhou. Continuando em modo local com DuckDB.")
        else:
            log.error("Carga no Neon falhou. Abortando.")
            return False

    if neon_ok:
        ok = step_trigger_render(dry_run)
        if not ok:
            log.warning("Push falhou - Neon foi atualizado mas o Render nao foi notificado.")
            log.warning("Faca um push manual para trigger do rebuild: git push")
    else:
        log.warning("Push para Render pulado porque o Neon nao foi atualizado nesta execucao.")

    _run_reports(dry_run)

    _run_adaptive_forward_refresh(
        dry_run=dry_run,
        duckdb_path=str(Path(data_dir) / "kintuadi.duckdb"),
    )

    _run_forecast_retrain(
        dry_run=dry_run,
        pmo_xlsx=str(Path(data_dir) / "ons" / "PMOs" / "validacao_pmo.xlsx"),
        model_dir=str(Path(data_dir) / "models"),
        duckdb_path=str(Path(data_dir) / "kintuadi.duckdb"),
        include_adaptive=False,
    )

    log.info("Pipeline finalizado.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualizacao diaria Neon + Render")
    parser.add_argument("--dry-run", action="store_true", help="Simula sem gravar no Neon nem fazer push")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Pula a coleta ONS/CCEE, usa arquivos existentes em data/",
    )
    parser.add_argument(
        "--allow-neon-failure",
        action="store_true",
        help="Continua o pipeline em modo local mesmo se a carga no Neon falhar",
    )
    parser.add_argument(
        "--report-only",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="Gera apenas um relatorio especifico, sem atualizar o Neon",
    )
    parser.add_argument(
        "--retrain-forecast",
        action="store_true",
        help="Retreina o stack de forecast e recalcula o snapshot estrutural de 6 meses",
    )
    parser.add_argument(
        "--force-weekly-retrain",
        action="store_true",
        help="Executa o retreino/refresh de forecast mesmo fora da janela semanal padrao",
    )
    parser.add_argument(
        "--adaptive-forward-only",
        action="store_true",
        help="Ao usar --retrain-forecast, recalcula apenas o AdaptivePLDForwardEngine",
    )
    parser.add_argument("--dir", default="data", help="Diretorio dos arquivos de dados (default: data)")
    args = parser.parse_args()

    if args.report_only:
        from report_engine import run as _report_run

        _report_run(period=args.report_only)
        sys.exit(0)

    if args.retrain_forecast:
        _run_forecast_retrain(
            dry_run=args.dry_run,
            pmo_xlsx=str(Path(args.dir) / "ons" / "PMOs" / "validacao_pmo.xlsx"),
            model_dir=str(Path(args.dir) / "models"),
            duckdb_path=str(Path(args.dir) / "kintuadi.duckdb"),
            force=args.force_weekly_retrain,
            adaptive_only=args.adaptive_forward_only,
        )
        sys.exit(0)

    ok = run(
        dry_run=args.dry_run,
        skip_collect=args.skip_collect,
        data_dir=args.dir,
        allow_neon_failure=args.allow_neon_failure,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
