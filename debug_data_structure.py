# debug_data_structure.py
import json
import os

def debug_data():
    """Debug da estrutura de dados"""
    
    print("🔍 DEBUG DA ESTRUTURA DE DADOS")
    print("=" * 60)
    
    # Encontra o arquivo mais recente
    data_dir = "data"
    if not os.path.exists(data_dir):
        print("❌ Pasta 'data' não existe")
        return
    
    # Procura arquivos dashboard
    dashboard_files = [f for f in os.listdir(data_dir) if f.startswith('kintuadi_dashboard_')]
    
    if not dashboard_files:
        print("❌ Nenhum arquivo dashboard encontrado")
        # Procura qualquer JSON
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if json_files:
            print(f"📁 JSONs encontrados: {json_files}")
            file_to_check = os.path.join(data_dir, json_files[0])
        else:
            print("❌ Nenhum JSON encontrado")
            return
    else:
        # Pega o mais recente
        latest = max(dashboard_files)
        file_to_check = os.path.join(data_dir, latest)
    
    print(f"\n📄 Analisando: {os.path.basename(file_to_check)}")
    
    try:
        with open(file_to_check, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 ESTRUTURA COMPLETA (recursiva):")
        print_structure(data)
        
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")

def print_structure(obj, indent=0, path=""):
    """Imprime estrutura recursivamente"""
    indent_str = "  " * indent
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, (dict, list)) and indent < 3:  # Limita profundidade
                print(f"{indent_str}{key}: {type(value).__name__}")
                print_structure(value, indent + 1, new_path)
            else:
                # Mostra valor se for importante
                if key in ['volume_medio', 'pld_medio', 'status', 'total_reservatorios', 'registros']:
                    print(f"{indent_str}{key}: {value}")

if __name__ == "__main__":
    debug_data()