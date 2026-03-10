# run_collector.py atualizado
#!/usr/bin/env python3
"""
Kintuadi Energy - Coletor Principal v2.0
"""

import os
import argparse
import sys
import subprocess
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Configura o ambiente"""
    
    # Cria diretórios necessários
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Configura Streamlit
    config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    if not os.path.exists(config_file):
        config_content = """[browser]
gatherUsageStats = false

[server]
address = "localhost"
port = 8501

[theme]
base = "dark"
"""
        with open(config_file, 'w') as f:
            f.write(config_content)
        logger.info("Configuração do Streamlit criada")
    
    # Carrega variáveis de ambiente
    load_dotenv()

def check_dependencies():
    """Verifica dependências"""
    try:
        import streamlit
        import pandas
        import plotly
        import requests
        logger.info("✅ Dependências verificadas")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        return False

def print_banner():
    """Exibe banner do sistema"""
    banner = f"""
╔{'═'*60}╗
║{'KINTUADI ENERGY INTELLIGENCE v2.0':^60}║
╠{'═'*60}╣
║{'Plataforma de Análise do Mercado de Energia':^60}║
║{datetime.now().strftime('%d/%m/%Y %H:%M:%S'):^60}║
╚{'═'*60}╝
    """
    print(banner)

def run_collector_v2(persist_mode: str = "full"):
    """Executa o coletor v2.0"""
    try:
        from scripts.integrated_collector_v2 import KintuadiIntegratedCollectorV2
        collector = KintuadiIntegratedCollectorV2()
        return collector.quick_collect(persist_mode=persist_mode)
    except ImportError as e:
        logger.error(f"Erro ao importar coletor v2: {e}")
        return None

def run_dashboard():
    """Executa o dashboard"""
    print("\n🌐 Iniciando Kintuadi Dashboard...")
    print("   Acesse: http://localhost:8501")
    print("   Pressione Ctrl+C para encerrar\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard_integrado.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard encerrado pelo usuário")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar dashboard: {e}")
        print("\n💡 Tente manualmente: python -m streamlit run dashboard_integrado.py")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")



def load_latest_raw_data():
    latest_file = Path("data") / "kintuadi_latest.json"
    if not latest_file.exists():
        return None
    try:
        with latest_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Falha ao ler {latest_file}: {e}")
        return None


def run_core_analysis():
    try:
        from scripts.core_analysis import build_core_analysis
    except Exception as e:
        logger.error(f"Erro ao importar core_analysis: {e}")
        return None

    raw = load_latest_raw_data()
    if not raw:
        logger.error("kintuadi_latest.json indisponível para gerar core.")
        return None

    try:
        core = build_core_analysis(raw, output_dir="data")
        return core
    except Exception as e:
        logger.error(f"Falha ao gerar core_analysis_latest (json/parquet): {e}")
        return None


def _get_current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def _get_default_remote() -> str:
    remotes = subprocess.check_output(["git", "remote"], text=True).splitlines()
    remotes = [r.strip() for r in remotes if r.strip()]
    if not remotes:
        raise RuntimeError("Nenhum remote git configurado.")
    return "origin" if "origin" in remotes else remotes[0]


def _has_upstream() -> bool:
    probe = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _push_with_upstream_fallback() -> None:
    if _has_upstream():
        subprocess.run(["git", "push"], check=True)
        return

    branch = _get_current_branch()
    remote = _get_default_remote()
    logger.info(f"Branch '{branch}' sem upstream; configurando tracking em {remote}/{branch}.")
    subprocess.run(["git", "push", "--set-upstream", remote, branch], check=True)


def publish_core_to_github(push: bool = True):
    src = Path("data") / "core_analysis_latest.parquet"
    dst = Path("core_analysis_latest.parquet")
    if not src.exists():
        logger.error("core_analysis_latest.parquet não encontrado em data/.")
        return False

    shutil.copy2(src, dst)

    try:
        subprocess.run(["git", "add", "core_analysis_latest.parquet"], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            logger.info("Sem alterações no core_analysis_latest.parquet para commit.")
            return True

        msg = f"Atualiza core_analysis_latest.parquet [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        logger.info("Commit do parquet realizado com sucesso.")

        if push:
            try:
                _push_with_upstream_fallback()
                logger.info("Push para GitHub realizado com sucesso.")
            except Exception as push_e:
                logger.warning(f"Commit feito, mas push falhou: {push_e}")
        return True
    except Exception as e:
        logger.error(f"Falha ao commitar parquet no git: {e}")
        return False


def run_pipeline_and_publish(push: bool = True, persist_mode: str = "full"):
    logger.info("Iniciando pipeline completo: coleta → integração → análise → publicação do parquet")
    collected = run_collector_v2(persist_mode=persist_mode)
    if not collected:
        logger.error("Coleta/integração falhou.")
        return False

    core = run_core_analysis()
    if not core:
        logger.error("Análise (core) falhou.")
        return False

    return publish_core_to_github(push=push)

def main():
    """Função principal"""
    
    # Configura ambiente
    setup_environment()
    
    # Verifica dependências
    if not check_dependencies():
        print("\n⚠️ Instale as dependências:")
        print("pip install -r requirements.txt")
        return
    
    # Exibe banner
    print_banner()
    
    # Menu principal
    while True:
        print("\n" + "="*60)
        print("🎯 MENU PRINCIPAL")
        print("="*60)
        print("1. Coleta completa + Dashboard (persist completo)")
        print("2. Apenas coletar dados (persist completo)")
        print("3. Apenas abrir dashboard")
        print("4. Coleta incremental (somente ano/mês correntes)")
        print("5. Verificar sistema")
        print("6. Pipeline completo + commit/push do parquet (persist completo)")
        print("7. Abrir dashboard_espelho.py")
        print("8. Pipeline completo + commit/push do parquet (persist incremental)")
        print("9. Sair")
        print("="*60)
        
        choice = input("\nEscolha (1-9): ").strip()
        
        if choice == "1":
            # Coleta completa + Dashboard
            print("\n📊 Executando coleta completa...")
            if run_collector_v2(persist_mode="full"):
                print("\n✅ Coleta concluída! Iniciando dashboard...")
                run_dashboard()
        
        elif choice == "2":
            # Apenas coleta
            print("\n📥 Coletando dados...")
            run_collector_v2(persist_mode="full")
        
        elif choice == "3":
            # Apenas dashboard
            run_dashboard()
        
        elif choice == "4":
            # Coleta incremental
            print("\n⚡ Coleta incremental (aproveitando dados históricos)...")
            run_collector_v2(persist_mode="incremental")
        
        elif choice == "5":
            # Verificar sistema
            check_system()
        
        elif choice == "6":
            print("\n🚀 Pipeline completo + publicação do parquet (persist completo)...")
            run_pipeline_and_publish(push=True, persist_mode="full")

        elif choice == "7":
            print("\n🌐 Iniciando Dashboard Espelho...")
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_espelho.py"], check=True)
            except Exception as e:
                print(f"❌ Erro ao executar dashboard_espelho: {e}")

        elif choice == "8":
            print("\n🚀 Pipeline completo + publicação do parquet (persist incremental)...")
            run_pipeline_and_publish(push=True, persist_mode="incremental")

        elif choice == "9":
            print("\n👋 Até logo!")
            break
        
        else:
            print("❌ Opção inválida")

def check_system():
    """Verifica status do sistema"""
    print("\n🔍 VERIFICANDO SISTEMA")
    print("-"*40)
    
    # Verifica diretórios
    dirs = ["data", "logs", "scripts"]
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d}/")
        else:
            print(f"❌ {d}/ (não existe)")
    
    # Verifica arquivos principais
    files = ["dashboard_integrado.py", "dashboard_espelho.py", "requirements.txt", "run_collector.py", "core_analysis_latest.parquet"]
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f}")
        else:
            print(f"❌ {f} (não existe)")
    
    # Verifica dados recentes
    import glob
    recent_files = glob.glob("data/kintuadi_*.json")
    if recent_files:
        latest = max(recent_files, key=os.path.getmtime)
        print(f"✅ Dados mais recentes: {os.path.basename(latest)}")
    else:
        print("❌ Nenhum dado coletado encontrado")
    
    print("-"*40)

def parse_args():
    parser = argparse.ArgumentParser(description="Kintuadi collector and parquet publisher")
    parser.add_argument("--daily-parquet", action="store_true", help="Executa pipeline e publica apenas core_analysis_latest.parquet")
    parser.add_argument("--persist-mode", choices=["full", "incremental"], default="incremental", help="Modo de persistência da coleta")
    parser.add_argument("--no-push", action="store_true", help="Faz commit local sem push")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.daily_parquet:
            setup_environment()
            ok = run_pipeline_and_publish(push=not args.no_push, persist_mode=args.persist_mode)
            sys.exit(0 if ok else 1)
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Programa interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
