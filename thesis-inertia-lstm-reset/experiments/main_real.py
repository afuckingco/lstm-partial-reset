import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import SimpleLSTM, train_model, predict
from src.preprocessing import load_and_prepare_real_data

def main():
    DATA_PATH = '../data/student_lifestyle_dataset.csv'  # ganti sesuai nama file
    N_ITER = 10
    WINDOW = 5
    EPOCHS = 20
    BATCH = 32
    LR = 0.001
    RESET_RATIO = 0.03
    RESET_FREQ = 15

    all_base, all_prop = [], []
    for i in tqdm(range(N_ITER)):
        try:
            X, y, _ = load_and_prepare_real_data(DATA_PATH, window=WINDOW)
        except Exception as e:
            print(f"Error load data: {e}")
            continue
        if len(X) < 30: continue
        split = int(0.7*len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1,4)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1,4)).reshape(X_test.shape)
        # baseline
        model_base = SimpleLSTM()
        train_model(model_base, X_train, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR)
        pred_base = predict(model_base, X_test)
        mse_base = np.mean((pred_base - y_test)**2)
        # proposed
        model_prop = SimpleLSTM()
        train_model(model_prop, X_train, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                    reset_every=RESET_FREQ, reset_ratio=RESET_RATIO)
        pred_prop = predict(model_prop, X_test)
        mse_prop = np.mean((pred_prop - y_test)**2)
        all_base.append(mse_base)
        all_prop.append(mse_prop)

    if not all_base: return
    base_mean, base_std = np.mean(all_base), np.std(all_base)
    prop_mean, prop_std = np.mean(all_prop), np.std(all_prop)
    t_stat, p_val = ttest_rel(all_base, all_prop)
    improvement = (base_mean - prop_mean)/base_mean*100
    print("="*50)
    print("HASIL EKSPERIMEN DATA RIIL")
    print(f"MSE Baseline: {base_mean:.4f} ± {base_std:.4f}")
    print(f"MSE Proposed: {prop_mean:.4f} ± {prop_std:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"t-test: t={t_stat:.3f}, p={p_val:.4e}")
    os.makedirs('../results', exist_ok=True)
    df_res = pd.DataFrame({'mse_base': all_base, 'mse_prop': all_prop})
    df_res.to_csv('../results/real_data_results.csv', index=False)
    plt.figure()
    sns.boxplot(data=df_res)
    plt.savefig('../results/boxplot_real_data.png')
    plt.show()

if __name__ == '__main__':
    main()
