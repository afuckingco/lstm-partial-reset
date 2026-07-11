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
from src.data_generator import generate_data_sudden
from src.model import SimpleLSTM, train_model, predict

def main():
    N_ITER = 42
    N_SUBJECTS = 17
    N_DAYS = 120
    DRIFT_DAY = 80
    WINDOW = 5
    EPOCHS = 20
    BATCH = 32
    LR = 0.001
    RESET_RATIO = 0.03
    RESET_FREQ = 15

    all_base, all_prop = [], []
    for seed in tqdm(range(N_ITER)):
        df = generate_data_sudden(seed, n_subjects=N_SUBJECTS, n_days=N_DAYS, drift_day=DRIFT_DAY)
        for subj in df['subject'].unique():
            subj_data = df[df['subject']==subj].sort_values('day')
            subj_data['stress_lag1'] = subj_data['stress'].shift(1)
            subj_data = subj_data.dropna()
            X = subj_data[['stress_lag1','mood','sleep','social']].values
            y = subj_data['stress'].values
            if len(X) < WINDOW+1: continue
            X_seq, y_seq = [], []
            for i in range(len(X)-WINDOW):
                X_seq.append(X[i:i+WINDOW])
                y_seq.append(y[i+WINDOW])
            X_seq, y_seq = np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
            if len(X_seq) < 30: continue
            split = int(0.7*len(X_seq))
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]
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

    base_mean, base_std = np.mean(all_base), np.std(all_base)
    prop_mean, prop_std = np.mean(all_prop), np.std(all_prop)
    t_stat, p_val = ttest_rel(all_base, all_prop)
    improvement = (base_mean - prop_mean)/base_mean*100
    print("="*50)
    print("HASIL EKSPERIMEN UTAMA (Tesis)")
    print(f"Total sampel: {len(all_base)}")
    print(f"MSE Baseline: {base_mean:.4f} ± {base_std:.4f}")
    print(f"MSE Proposed: {prop_mean:.4f} ± {prop_std:.4f}")
    print(f"Penurunan: {improvement:.2f}%")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4e}")
    os.makedirs('../results', exist_ok=True)
    df_res = pd.DataFrame({'mse_base': all_base, 'mse_prop': all_prop})
    df_res.to_csv('../results/tesis_results.csv', index=False)
    plt.figure(figsize=(6,5))
    sns.boxplot(data=df_res[['mse_base','mse_prop']])
    plt.xticks([0,1], ['Baseline', 'Proposed'])
    plt.ylabel('MSE')
    plt.title('Boxplot MSE')
    plt.savefig('../results/boxplot_tesis.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
