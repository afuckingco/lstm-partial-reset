import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import download_stock_data, add_synthetic_drift, create_sequences
from src.model import SimpleLSTM, train_model, predict
from src.utils import compute_metrics, paired_ttest

# ========== PARAMETERS ==========
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
DRIFT_POINT = 0.7        # drift occurs at 70% of the series
DRIFT_FACTOR = 1.5       # price increases 50% after drift
WINDOW = 5
FEATURES = ['price_lag1', 'return', 'volatility']
TARGET = 'price'
EPOCHS = 20
BATCH = 32
LR = 0.001
RESET_RATIO = 0.03
RESET_FREQ = 15
N_ITER = 30              # number of bootstrap iterations

# ========== LOAD DATA ==========
print("Downloading stock data...")
df_raw = download_stock_data(TICKER, START_DATE, END_DATE)

all_base_mse = []
all_prop_mse = []

for seed in tqdm(range(N_ITER), desc="Experiment iterations"):
    # Add synthetic drift (randomized drift point slightly)
    np.random.seed(seed)
    df = add_synthetic_drift(df_raw.copy(), drift_point=DRIFT_POINT, drift_factor=DRIFT_FACTOR)
    
    # Create sequences
    X, y = create_sequences(df, FEATURES, TARGET, window=WINDOW)
    if len(X) < 50:
        continue
    
    # Temporal split (70% train, 30% test)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, len(FEATURES))).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, len(FEATURES))).reshape(X_test.shape)
    
    # Baseline model (no reset)
    model_base = SimpleLSTM(input_size=len(FEATURES))
    train_model(model_base, X_train_scaled, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR)
    pred_base = predict(model_base, X_test_scaled)
    mse_base = compute_metrics(y_test, pred_base)['MSE']
    
    # Proposed model (with periodic reset)
    model_prop = SimpleLSTM(input_size=len(FEATURES))
    train_model(model_prop, X_train_scaled, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                reset_every=RESET_FREQ, reset_ratio=RESET_RATIO)
    pred_prop = predict(model_prop, X_test_scaled)
    mse_prop = compute_metrics(y_test, pred_prop)['MSE']
    
    all_base_mse.append(mse_base)
    all_prop_mse.append(mse_prop)

# ========== RESULTS ==========
base_mean, base_std = np.mean(all_base_mse), np.std(all_base_mse)
prop_mean, prop_std = np.mean(all_prop_mse), np.std(all_prop_mse)
t_stat, p_val = ttest_rel(all_base_mse, all_prop_mse)
improvement = (base_mean - prop_mean) / base_mean * 100

print("\n" + "="*60)
print("STOCK PRICE PREDICTION WITH PERIODIC RESET")
print(f"Iterations: {len(all_base_mse)}")
print(f"Baseline MSE: {base_mean:.6f} ± {base_std:.6f}")
print(f"Proposed MSE: {prop_mean:.6f} ± {prop_std:.6f}")
print(f"Improvement: {improvement:.2f}%")
print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4e}")

# Save results
os.makedirs('../results', exist_ok=True)
df_res = pd.DataFrame({'mse_base': all_base_mse, 'mse_prop': all_prop_mse})
df_res.to_csv('../results/stock_experiment.csv', index=False)

# Boxplot
plt.figure(figsize=(6,5))
sns.boxplot(data=df_res[['mse_base', 'mse_prop']])
plt.xticks([0,1], ['Baseline', 'Proposed'])
plt.ylabel('MSE')
plt.title(f'Stock Prediction MSE Comparison ({TICKER})')
plt.savefig('../results/boxplot_stock.png', dpi=300, bbox_inches='tight')
plt.show()