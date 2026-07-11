import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_prepare_real_data(file_path, window=5):
    df = pd.read_csv(file_path)
    # Sesuaikan mapping kolom di sini
    # contoh mapping untuk dataset General Mental Health
    mapping = {'stress':'Stress Level', 'mood':'Mood', 'sleep':'Sleep Hours', 'social':'Social Interaction'}
    # cek kolom yang tersedia
    cols = {}
    for k, v in mapping.items():
        if v in df.columns:
            cols[k] = v
        else:
            # coba alternatif
            alt = k.capitalize()
            if alt in df.columns:
                cols[k] = alt
            else:
                raise KeyError(f"Kolom {v} atau {alt} tidak ditemukan dalam dataset")
    df_sel = df[[cols['stress'], cols['mood'], cols['sleep'], cols['social']]].copy()
    df_sel.columns = ['stress','mood','sleep','social']
    df_sel = df_sel.sort_index()
    df_sel['stress_lag1'] = df_sel['stress'].shift(1)
    df_sel = df_sel.dropna().reset_index(drop=True)
    features = ['stress_lag1','mood','sleep','social']
    X = df_sel[features].values
    y = df_sel['stress'].values
    X_seq, y_seq = [], []
    for i in range(len(X)-window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), df_sel
