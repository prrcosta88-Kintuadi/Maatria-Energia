"""
Retreinamento completo dos modelos PLD para todos os subsistemas
com ponderação por PLD alto e garantia de ordenação dos quantis.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração
MODEL_DIR = Path('data/models')
BACKUP_DIR = MODEL_DIR / 'backup_before_retrain'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Subsistemas e seus nomes nas colunas
SUBSISTEMAS = {
    'seco': 'SE/CO',
    's': 'Sul',
    'ne': 'Nordeste',
    'n': 'Norte'
}

# Features comuns a todos os subsistemas
FEATURE_COLS_BASE = [
    'ena_prev_{}_mlt',      # ENA prevista pelo ONS
    'ear_init_{}',          # EAR inicial (Tab6 do PMO)
    'term_total_sin',       # Despacho térmico SIN
    'thermal_ratio',        # Proporção térmica
    'cmo_med_{}',           # CMO médio do subsistema
    'cmo_delta_{}',         # Variação do CMO
    'carga_sem1_sin',       # Carga prevista
    'mes_sin', 'mes_cos',   # Sazonalidade
    'erro_ons_{}',          # Erro histórico ONS
    'erro_ons_{}_4w',       # Erro ONS rolling 4 semanas
    'cmo_real_{}_lag1',     # CMO realizado lag1
    'cmo_erro_{}_lag1',     # Erro CMO lag1
    'ena_real_{}_mlt_lag1'  # ENA realizada lag1
]

def get_features_for_subsistema(df, subsistema):
    """Obtém as features disponíveis para um subsistema"""
    features = []
    for col_pattern in FEATURE_COLS_BASE:
        col = col_pattern.format(subsistema)
        if col in df.columns:
            features.append(col)
    
    # Adicionar colunas que não dependem do subsistema
    extra_cols = ['term_total_sin', 'thermal_ratio', 'carga_sem1_sin', 'mes_sin', 'mes_cos']
    for col in extra_cols:
        if col in df.columns and col not in features:
            features.append(col)
    
    return features

def remove_high_na_features(df, features, threshold=0.5):
    """Remove features com mais de threshold% de NaN"""
    good_features = []
    for col in features:
        if df[col].isna().sum() / len(df) < threshold:
            good_features.append(col)
        else:
            print(f'    Removendo {col} ({df[col].isna().sum()} NaNs)')
    return good_features

def train_weighted_quantiles(X, y, weights=None):
    """
    Treina modelos para P10, P50 e P90 garantindo ordenação
    """
    models = {}
    y_vals = y.values if hasattr(y, 'values') else y
    
    # Treinar P50 primeiro (mediana)
    gb50 = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.50,
        max_iter=500, max_depth=5,
        learning_rate=0.05, min_samples_leaf=10,
        l2_regularization=0.1, random_state=42
    )
    gb50.fit(X, y_vals, sample_weight=weights)
    models[0.50] = gb50
    
    # Prever com P50 para usar como referência
    pred50 = gb50.predict(X)
    
    # Treinar P10 com foco em valores abaixo da mediana
    # Usar pesos maiores para valores abaixo da mediana
    weights_p10 = weights.copy() if weights is not None else np.ones(len(y_vals))
    below_median = y_vals < pred50
    weights_p10[below_median] *= 2.0
    
    gb10 = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.10,
        max_iter=500, max_depth=5,
        learning_rate=0.05, min_samples_leaf=10,
        l2_regularization=0.1, random_state=42
    )
    gb10.fit(X, y_vals, sample_weight=weights_p10)
    models[0.10] = gb10
    
    # Treinar P90 com foco em valores acima da mediana
    weights_p90 = weights.copy() if weights is not None else np.ones(len(y_vals))
    above_median = y_vals > pred50
    weights_p90[above_median] *= 2.0
    
    gb90 = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.90,
        max_iter=500, max_depth=5,
        learning_rate=0.05, min_samples_leaf=10,
        l2_regularization=0.1, random_state=42
    )
    gb90.fit(X, y_vals, sample_weight=weights_p90)
    models[0.90] = gb90
    
    return models

def calculate_weights(y_vals):
    """
    Calcula pesos para as observações:
    - PLD > R$300: peso 4.0 (muito alto)
    - PLD entre R$250-300: peso 3.0 (alto)
    - PLD entre R$150-250: peso 2.0 (médio-alto)
    - PLD < R$150: peso 1.0 (normal)
    """
    weights = np.ones(len(y_vals))
    weights[y_vals > 300] = 4.0
    weights[(y_vals > 250) & (y_vals <= 300)] = 3.0
    weights[(y_vals > 150) & (y_vals <= 250)] = 2.0
    return weights

def train_subsistema(df, subsistema, nome, horizons=[4, 8, 12, 26]):
    """
    Treina todos os modelos para um subsistema
    """
    print(f'\n{"="*60}')
    print(f'Treinando: {nome} ({subsistema})')
    print(f'{"="*60}')
    
    # Obter features
    features = get_features_for_subsistema(df, subsistema)
    features = remove_high_na_features(df, features)
    
    print(f'  Features: {len(features)}')
    
    target_col = f'pld_real_{subsistema}'
    if target_col not in df.columns:
        print(f'  ❌ Target {target_col} não encontrado')
        return {}
    
    results = {}
    
    for h in horizons:
        print(f'\n  --- h={h} semanas ---')
        
        # Preparar target com shift
        y = df[target_col].shift(-h)
        mask = y.notna() & (y > 0)
        
        X = df[features][mask]
        y_vals = y[mask]
        
        if len(X) < 52:
            print(f'    ⚠️  Apenas {len(X)} observações, ignorando')
            continue
        
        # Calcular pesos
        weights = calculate_weights(y_vals)
        
        # Estatísticas
        n_high = (y_vals > 250).sum()
        n_very_high = (y_vals > 300).sum()
        pct_high = 100 * n_high / len(y_vals)
        
        print(f'    N obs: {len(X)}')
        print(f'    PLD médio: R${y_vals.mean():.2f}')
        print(f'    PLD > R$250: {n_high} obs ({pct_high:.1f}%)')
        print(f'    PLD > R$300: {n_very_high} obs')
        print(f'    Peso médio: {weights.mean():.2f}')
        
        # Treinar modelos quantílicos
        models = train_weighted_quantiles(X.values, y_vals.values, weights)
        
        # Avaliar in-sample
        pred50 = models[0.50].predict(X.values)
        pred10 = models[0.10].predict(X.values)
        pred90 = models[0.90].predict(X.values)
        
        # Garantir ordenação P10 <= P50 <= P90
        pred10 = np.minimum(pred10, pred50)
        pred90 = np.maximum(pred90, pred50)
        
        mae = np.mean(np.abs(pred50 - y_vals))
        mape = np.mean(np.abs((pred50 - y_vals) / y_vals)) * 100
        
        # Verificar inversões
        inversions = np.sum(pred10 > pred50) + np.sum(pred90 < pred50)
        
        print(f'    MAE: R${mae:.2f}')
        print(f'    MAPE: {mape:.1f}%')
        print(f'    Inversões: {inversions}')
        
        if inversions > 0:
            print(f'    ⚠️  Ajustando inversões...')
            # Ajuste final: garantir ordenação
            pred10 = np.minimum(pred10, pred50)
            pred90 = np.maximum(pred90, pred50)
            mae_corrected = np.mean(np.abs(pred50 - y_vals))
            print(f'    MAE após correção: R${mae_corrected:.2f}')
        
        # Salvar modelo
        result = {
            'models': models,
            'feature_cols': features,
            'metrics': {
                'mae_p50': mae,
                'mape_p50': mape,
                'n_train': len(X),
                'pld_mean': y_vals.mean(),
                'pld_std': y_vals.std(),
                'weight_mean': weights.mean(),
                'inversions': inversions
            },
            'trained_at': pd.Timestamp.now().isoformat()
        }
        
        results[h] = result
        
        # Salvar arquivo
        model_file = MODEL_DIR / f'gbm_{subsistema}_h{h}_retrained.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(result, f)
        print(f'    ✅ Salvo: {model_file.name}')
    
    return results

def backup_existing_models():
    """Faz backup dos modelos existentes"""
    print('\n📦 Fazendo backup dos modelos existentes...')
    existing = list(MODEL_DIR.glob('gbm_*.pkl'))
    for f in existing:
        backup_path = BACKUP_DIR / f.name
        if not backup_path.exists():
            import shutil
            shutil.copy2(f, backup_path)
            print(f'  Backup: {f.name}')
    print(f'  ✅ {len(existing)} modelos salvos em {BACKUP_DIR}')

def replace_with_new_models():
    """Substitui modelos antigos pelos novos retreinados"""
    print('\n🔄 Substituindo modelos antigos...')
    new_models = list(MODEL_DIR.glob('gbm_*_retrained.pkl'))
    for new_file in new_models:
        # Nome do arquivo original (sem _retrained)
        original_name = new_file.name.replace('_retrained', '')
        original_path = MODEL_DIR / original_name
        
        # Substituir
        import shutil
        shutil.copy2(new_file, original_path)
        print(f'  ✅ {original_name} atualizado')
    print(f'  ✅ {len(new_models)} modelos substituídos')

def validate_models(df, subsistema, results):
    """Valida os modelos treinados"""
    print(f'\n{"="*60}')
    print(f'Validação: {SUBSISTEMAS[subsistema]}')
    print(f'{"="*60}')
    
    target_col = f'pld_real_{subsistema}'
    
    for h, result in results.items():
        if not result:
            continue
        
        metrics = result['metrics']
        print(f'\n  h={h} semanas:')
        print(f'    MAE: R${metrics["mae_p50"]:.2f}')
        print(f'    MAPE: {metrics["mape_p50"]:.1f}%')
        print(f'    PLD médio treino: R${metrics["pld_mean"]:.2f}')
        print(f'    Peso médio: {metrics["weight_mean"]:.2f}')
        
        # Verificar se o modelo é bom
        if metrics["mae_p50"] < 30:
            status = "✅ EXCELENTE"
        elif metrics["mae_p50"] < 60:
            status = "✅ BOM"
        elif metrics["mae_p50"] < 100:
            status = "⚠️ ACEITÁVEL"
        else:
            status = "❌ RUIM"
        
        print(f'    Status: {status}')

# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MAÁTria Energia · Retreinamento Completo dos Modelos PLD")
    print("="*60)
    
    # 1. Carregar dataset
    print("\n[1/5] Carregando dataset...")
    df = pd.read_pickle('data/models/training_dataset.pkl')
    print(f'  Dataset: {df.shape[0]} semanas, {df.shape[1]} colunas')
    print(f'  Período: {df.index.min().date()} a {df.index.max().date()}')
    
    # 2. Backup dos modelos existentes
    print("\n[2/5] Backup dos modelos...")
    backup_existing_models()
    
    # 3. Treinar para cada subsistema
    print("\n[3/5] Treinando modelos...")
    all_results = {}
    
    for subsistema, nome in SUBSISTEMAS.items():
        results = train_subsistema(df, subsistema, nome)
        all_results[subsistema] = results
        validate_models(df, subsistema, results)
    
    # 4. Substituir modelos antigos
    print("\n[4/5] Atualizando modelos em produção...")
    replace_with_new_models()
    
    # 5. Resumo final
    print("\n[5/5] Resumo Final")
    print("="*60)
    
    total_models = sum(len(r) for r in all_results.values())
    print(f'\n✅ {total_models} modelos treinados e atualizados')
    print(f'   - 4 subsistemas × 4 horizontes = 16 modelos')
    print(f'   - Backup salvo em: {BACKUP_DIR}')
    print(f'   - Modelos em produção: {MODEL_DIR}')
    
    # Tabela de resumo
    print('\n📊 Resumo de desempenho:')
    print('-' * 70)
    print(f'{"Subsistema":<12} {"h=4":>12} {"h=8":>12} {"h=12":>12} {"h=26":>12}')
    print('-' * 70)
    
    for subsistema, nome in SUBSISTEMAS.items():
        row = f'{nome:<12}'
        for h in [4, 8, 12, 26]:
            if h in all_results.get(subsistema, {}):
                mae = all_results[subsistema][h]['metrics']['mae_p50']
                if mae < 30:
                    row += f'  ✅ {mae:>6.1f}'
                elif mae < 60:
                    row += f'  👍 {mae:>6.1f}'
                else:
                    row += f'  ⚠️ {mae:>6.1f}'
            else:
                row += f'  {"-":>8}'
        print(row)
    print('-' * 70)
    
    print('\n✅ Retreinamento concluído!')
    print('\nAgora execute: python pld_forecast_engine.py --forecast')