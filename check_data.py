# check_data.py
import json
import os
from datetime import datetime

print("🔍 VERIFICANDO DADOS COLETADOS")
print("=" * 50)

# Lista arquivos na pasta data
data_dir = "data"
if os.path.exists(data_dir):
    print(f"Arquivos em '{data_dir}':")
    for file in sorted(os.listdir(data_dir)):
        size = os.path.getsize(os.path.join(data_dir, file))
        modified = datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, file)))
        print(f"  📄 {file} ({size} bytes, {modified.strftime('%H:%M')})")
        
        # Mostra conteúdo do latest
        if file == "kintuadi_latest.json":
            print(f"\n  Conteúdo de {file}:")
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  Estrutura: {list(data.keys())}")
                
                if 'ons' in data:
                    print(f"  ONS: {data['ons']}")
                if 'ccee' in data:
                    print(f"  CCEE: {data['ccee']}")
else:
    print(f"❌ Pasta '{data_dir}' não existe!")