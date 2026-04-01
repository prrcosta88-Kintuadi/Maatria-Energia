import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

# Carregar dataset
df = pd.read_pickle('data/models/training_dataset.pkl')
print('Dataset carregado:', df.shape)

# Features que o modelo usa
FEATURE_COLS = [
    'ena_prev_seco_mlt', 'ear_init_seco', 'term_total_sin', 'thermal_ratio',
    'cmo_med_seco', 'cmo_delta_seco', 'carga_sem1_sin', 'mes_sin', 'mes_cos',
    'erro_ons_seco', 'erro_ons_seco_4w', 'spdi', 'structural_drift',
    'cmo_real_seco_lag1', 'cmo_erro_seco_lag1', 'ena_real_seco_mlt_lag1'
]

available = [c for c in FEATURE_COLS if c in df.columns]
print(f'Features disponíveis: {len(available)}')

# Remover colunas com muitos NaN
for col in available[:]:  # usar cópia para iterar
    if df[col].isna().sum() > len(df) * 0.5:
        print(f'  Removendo {col} ({df[col].isna().sum()} NaNs)')
        available.remove(col)

print(f'Features finais: {len(available)}')
print(f'Features: {available}')

# Função para treinar com ponderação
def train_weighted(h, df, available):
    y = df['pld_real_seco'].shift(-h)
    mask = y.notna() & (y > 0)
    X = df[available][mask]
    y_vals = y[mask]
    
    if len(X) < 52:
        print(f'  h={h}: apenas {len(X)} observações, ignorando')
        return None
    
    # Pesos: 3x para PLD > 250, 2x para PLD entre 150-250
    weights = np.ones(len(y_vals))
    weights[y_vals > 250] = 3.0
    weights[(y_vals > 150) & (y_vals <= 250)] = 2.0
    
    print(f'\n=== h={h} semanas ===')
    print(f'  N obs: {len(X)}')
    print(f'  PLD médio: R${y_vals.mean():.2f}')
    print(f'  PLD > R$250: {(y_vals > 250).sum()} obs ({100*(y_vals>250).sum()/len(y_vals):.1f}%)')
    print(f'  PLD > R$300: {(y_vals > 300).sum()} obs')
    print(f'  Peso médio: {weights.mean():.2f}')
    
    # Treinar para os 3 quantis
    models = {}
    for q in [0.10, 0.50, 0.90]:
        gb = HistGradientBoostingRegressor(
            loss='quantile', quantile=q,
            max_iter=500, max_depth=5,
            learning_rate=0.05, min_samples_leaf=10,
            l2_regularization=0.1, random_state=42
        )
        gb.fit(X.values, y_vals.values, sample_weight=weights)
        models[q] = gb
    
    # Avaliação in-sample
    y_pred = models[0.50].predict(X.values)
    mae = np.mean(np.abs(y_pred - y_vals))
    print(f'  MAE in-sample: R${mae:.2f}')
    
    return {
        'models': models,
        'feature_cols': available,
        'metrics': {'mae_p50': mae, 'n_train': len(X)},
        'trained_at': pd.Timestamp.now().isoformat()
    }

# Treinar e salvar cada horizonte
model_dir = Path('data/models')
for h in [4, 8, 12, 26]:
    result = train_weighted(h, df, available)
    if result:
        with open(model_dir / f'gbm_seco_h{h}_weighted.pkl', 'wb') as f:
            pickle.dump(result, f)
        print(f'  ✅ Modelo salvo: gbm_seco_h{h}_weighted.pkl')