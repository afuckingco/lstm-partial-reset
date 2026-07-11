# experiments/main_real_student.py (versi auto-detect kolom)
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

def load_student_lifestyle(file_path, window=5):
    df = pd.read_csv(file_path)
    print("Kolom yang tersedia:", df.columns.tolist())
    
    # Mapping stress level
    stress_map = {'Low': 2.5, 'Moderate': 5.0, 'High': 7.5}
    
    # Cari kolom stress (biasanya 'Stress_Level')
    stress_col = None
    for col in df.columns:
        if 'stress' in col.lower():
            stress_col = col
            break
    if stress_col is None:
        raise KeyError("Kolom stress tidak ditemukan.")
    df['stress'] = df[stress_col].map(stress_map)
    
    # Cari kolom study (Study_Hours atau Study_Hours_Per_Day)
    study_col = None
    for col in df.columns:
        if 'study' in col.lower() and 'hour' in col.lower():
            study_col = col
            break
    if study_col is None:
        # alternatif cari kolom dengan kata 'study'
        for col in df.columns:
            if 'study' in col.lower():
                study_col = col
                break
    if study_col is None:
        raise KeyError("Kolom study tidak ditemukan.")
    
    # Cari kolom activity (Physical_Activity atau Physical_Activity_Per_Day)
    activity_col = None
    for col in df.columns:
        if 'physical' in col.lower() or 'activity' in col.lower():
            activity_col = col
            break
    if activity_col is None:
        raise KeyError("Kolom physical activity tidak ditemukan.")
    
    # Cari kolom sleep
    sleep_col = None
    for col in df.columns:
        if 'sleep' in col.lower():
            sleep_col = col
            break
    if sleep_col is None:
        raise KeyError("Kolom sleep tidak ditemukan.")
    
    # Cari kolom social
    social_col = None
    for col in df.columns:
        if 'social' in col.lower():
            social_col = col
            break
    if social_col is None:
        raise KeyError("Kolom social tidak ditemukan.")
    
    # Hitung mood sebagai rata-rata activity dan study, skala 0-10
    df['mood'] = (df[activity_col] + df[study_col]) / 2
    max_mood = df['mood'].max()
    if max_mood > 0:
        df['mood'] = df['mood'] / max_mood * 10
    else:
        df['mood'] = 5  # fallback
    
    df['sleep'] = df[sleep_col]
    df['social'] = df[social_col]
    
    df_sel = df[['stress', 'mood', 'sleep', 'social']].copy()
    df_sel = df_sel.sort_index()
    df_sel['stress_lag1'] = df_sel['stress'].shift(1)
    df_sel = df_sel.dropna().reset_index(drop=True)
    
    features = ['stress_lag1', 'mood', 'sleep', 'social']
    X = df_sel[features].values
    y = df_sel['stress'].values
    
    X_seq, y_seq = [], []
    for i in range(len(X)-window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), df_sel

def main():
    DATA_PATH = '../data/student_lifestyle_dataset.csv'  # pastikan file ada
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
            X, y, _ = load_student_lifestyle(DATA_PATH, window=WINDOW)
        except Exception as e:
            print(f"Error load data: {e}")
            continue
        if len(X) < 30:
            print("Data terlalu pendek.")
            continue
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1,4)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1,4)).reshape(X_test.shape)
        
        model_base = SimpleLSTM()
        train_model(model_base, X_train, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR)
        pred_base = predict(model_base, X_test)
        mse_base = np.mean((pred_base - y_test)**2)
        
        model_prop = SimpleLSTM()
        train_model(model_prop, X_train, y_train, epochs=EPOCHS, batch_size=BATCH, lr=LR,
                    reset_every=RESET_FREQ, reset_ratio=RESET_RATIO)
        pred_prop = predict(model_prop, X_test)
        mse_prop = np.mean((pred_prop - y_test)**2)
        
        all_base.append(mse_base)
        all_prop.append(mse_prop)

    if not all_base:
        print("Tidak ada data yang diproses.")
        return

    base_mean, base_std = np.mean(all_base), np.std(all_base)
    prop_mean, prop_std = np.mean(all_prop), np.std(all_prop)
    t_stat, p_val = ttest_rel(all_base, all_prop)
    improvement = (base_mean - prop_mean) / base_mean * 100

    print("="*50)
    print("HASIL EKSPERIMEN DENGAN STUDENT LIFESTYLE DATASET")
    print(f"MSE Baseline: {base_mean:.4f} ± {base_std:.4f}")
    print(f"MSE Proposed: {prop_mean:.4f} ± {prop_std:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4e}")

    os.makedirs('../results', exist_ok=True)
    df_res = pd.DataFrame({'mse_base': all_base, 'mse_prop': all_prop})
    df_res.to_csv('../results/real_data_results.csv', index=False)
    plt.figure(figsize=(6,5))
    sns.boxplot(data=df_res[['mse_base','mse_prop']])
    plt.xticks([0,1], ['Baseline', 'Proposed'])
    plt.ylabel('MSE')
    plt.title('Perbandingan MSE pada Student Lifestyle Dataset')
    plt.savefig('../results/boxplot_real_data.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()