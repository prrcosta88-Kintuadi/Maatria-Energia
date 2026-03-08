# setup_env.py
import os
import sys

def setup_environment():
    """Configura o ambiente Kintuadi Energy"""
    
    print("🔧 CONFIGURAÇÃO DO AMBIENTE KINTUADI ENERGY")
    print("="*50)
    
    # Verifica se .env existe
    env_file = ".env"
    
    if os.path.exists(env_file):
        print(f"✅ Arquivo {env_file} já existe")
        response = input("Deseja sobrescrever? (s/N): ").strip().lower()
        if response != 's':
            print("Manutenção do arquivo atual.")
            return
    else:
        print(f"📄 Criando arquivo {env_file}...")
    
    # Coleta informações do usuário
    print("\n📝 CONFIGURAÇÃO DAS CREDENCIAIS ONS")
    print("(Deixe em branco se não tiver acesso à API histórica)")
    print("-"*40)
    
    ons_username = input("Email ONS: ").strip()
    ons_password = input("Senha ONS: ").strip()
    
    # Configurações padrão
    config = f"""# KINTUADI ENERGY - CONFIGURAÇÕES

# CREDENCIAIS ONS (API histórica)
ONS_USERNAME={ons_username if ons_username else 'SEU_EMAIL_AQUI'}
ONS_PASSWORD={ons_password if ons_password else 'SUA_SENHA_AQUI'}

# APIS PÚBLICAS
ONS_PUBLIC_API_URL=https://integra.ons.org.br/api
CCEE_API_URL=https://dadosabertos.ccee.org.br/api/3/action

# DIRETÓRIOS
DATA_DIR=data
LOG_DIR=logs
AUDIT_DIR=audit_logs

# CONFIGURAÇÕES DE CACHE (minutos)
CACHE_TTL_ONS=30
CACHE_TTL_CCEE=60

# LIMITES
MAX_RESERVATORIOS=100
MAX_PLD_RECORDS=500
MAX_HISTORICAL_DAYS=30

# LOGGING
LOG_LEVEL=INFO
"""
    
    # Salva o arquivo
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print(f"\n✅ Arquivo {env_file} criado/atualizado com sucesso!")
    
    # Cria diretórios necessários
    directories = ['data', 'logs', 'audit_logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Diretório criado: {directory}/")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Edite o arquivo .env com suas credenciais reais")
    print("2. Execute: python test_ons_volume_api.py")
    print("3. Depois: python run_collector.py")
    
    # Mostra conteúdo do arquivo
    print(f"\n📋 CONTEÚDO DO {env_file}:")
    print("-"*40)
    print(config)

if __name__ == "__main__":
    setup_environment()