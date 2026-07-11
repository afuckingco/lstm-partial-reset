import os

# Base directory
base_dir = os.path.join(os.getcwd(), "thesis_inertia")

# Struktur folder
folders = [
    "data",
    "results",
    "src",
    "experiments",
    "dashboard",
    "thesis"
]

# File kosong atau boilerplate (isi dengan kode final)
files_content = {
    # src/__init__.py
    os.path.join("src", "__init__.py"): "",
    
    # src/data_generator.py
    os.path.join("src", "data_generator.py"): '''\
import numpy as np
import pandas as pd

def generate_data_sudden(seed, n_subjects=17, n_days=120, drift_day=80):
    np.random.seed(seed)
    records = []
    for subj in range(n_subjects):
        inertia = np.random.uniform(0.2, 0.9)
        stress = np.random.uniform(3, 7)
        for day in range(n_days):
            drift = 2.0 if day >= drift_day else 0.0
            change = (drift + np.random.normal(0,0.1) - stress) * (1 - inertia)
            stress += change
            stress = np.clip(stress, 1, 10)
            mood = 10 - stress + np.random.normal(0,0.3)
            if day >= drift_day:
                mood += 0.5
            mood = np.clip(mood, 1, 10)
            sleep = 7 - 0.1*(stress-5) + np.random.normal(0,0.3)
            sleep = np.clip(sleep, 4, 10)
            social = 6 - 0.2*(stress-5) + np.random.normal(0,0.4)
            social = np.clip(social, 1, 10)
            records.append([subj, day, stress, mood, sleep, social, inertia])
    return pd.DataFrame(records, columns=['subject','day','stress','mood','sleep','social','inertia'])
''',
    
    # src/model.py
    os.path.join("src", "model.py"): '''\
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(model, X, y, epochs=20, batch_size=32, lr=0.001,
                reset_every=None, reset_ratio=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).view(-1,1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        if reset_every and (epoch+1) % reset_every == 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        n = p.numel()
                        n_reset = max(1, int(n * reset_ratio))
                        idx = torch.randperm(n)[:n_reset]
                        p.data.view(-1)[idx] = torch.randn(n_reset, device=p.device) * 0.01
    return model

def predict(model, X):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(Xt).cpu().numpy()
    return pred.squeeze()
''',
    
    # src/preprocessing.py (untuk data riil)
    os.path.join("src", "preprocessing.py"): '''\
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
''',
    
    # experiments/main_tesis.py
    os.path.join("experiments", "main_tesis.py"): '''\
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
''',
    
    # experiments/main_real.py (template)
    os.path.join("experiments", "main_real.py"): '''\
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
''',
    
    # dashboard/app.py (sederhana)
    os.path.join("dashboard", "app.py"): '''\
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleLSTM, train_model, predict
from src.data_generator import generate_data_sudden

st.set_page_config(layout="wide")
st.title("Periodic Partial Reset LSTM Dashboard")

data_source = st.sidebar.radio("Data source", ["Generate synthetic", "Upload CSV"])
window = st.sidebar.slider("Window", 3, 10, 5)
reset_ratio = st.sidebar.slider("Reset ratio (%)", 1, 10, 3) / 100
reset_freq = st.sidebar.slider("Reset frequency (epochs)", 5, 30, 15)
epochs = st.sidebar.slider("Epochs", 10, 50, 20)

if data_source == "Generate synthetic":
    df = generate_data_sudden(seed=42, n_subjects=1, n_days=200, drift_day=120)
    df = df[df['subject']==0].copy()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)

# Preprocess
df['stress_lag1'] = df['stress'].shift(1)
df = df.dropna().reset_index(drop=True)
features = ['stress_lag1','mood','sleep','social']
X = df[features].values
y = df['stress'].values
X_seq, y_seq = [], []
for i in range(len(X)-window):
    X_seq.append(X[i:i+window])
    y_seq.append(y[i+window])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)
split = int(0.7*len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1,4)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1,4)).reshape(X_test.shape)

model_base = SimpleLSTM()
train_model(model_base, X_train, y_train, epochs=epochs)
pred_base = predict(model_base, X_test)
mse_base = np.mean((pred_base - y_test)**2)

model_prop = SimpleLSTM()
train_model(model_prop, X_train, y_train, epochs=epochs, reset_every=reset_freq, reset_ratio=reset_ratio)
pred_prop = predict(model_prop, X_test)
mse_prop = np.mean((pred_prop - y_test)**2)

col1, col2, col3 = st.columns(3)
col1.metric("MSE Baseline", f"{mse_base:.4f}")
col2.metric("MSE Proposed", f"{mse_prop:.4f}")
improve = (mse_base-mse_prop)/mse_base*100
col3.metric("Improvement", f"{improve:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test, name='Actual', line=dict(color='black')))
fig.add_trace(go.Scatter(y=pred_base, name='Baseline', line=dict(dash='dash')))
fig.add_trace(go.Scatter(y=pred_prop, name='Proposed', line=dict(dash='dot')))
st.plotly_chart(fig, use_container_width=True)
''',
    
    # requirements.txt di root
    "requirements.txt": '''\
numpy
pandas
torch
scikit-learn
scipy
matplotlib
seaborn
tqdm
river
streamlit
plotly
''',
    
    # README.md
    "README.md": '''\
# Periodic Partial Reset LSTM untuk Prediksi Stres

## Cara menjalankan:
1. Buat virtual environment: `python -m venv venv` lalu aktifkan.
2. Install requirements: `pip install -r requirements.txt`
3. Jalankan eksperimen utama: `cd experiments && python main_tesis.py`
4. Jalankan dashboard: `streamlit run dashboard/app.py`
''',
    
    # thesis (opsional) – buat folder kosong dan file placeholder
    os.path.join("thesis", ".gitkeep"): "",
}

# Buat folder
os.makedirs(base_dir, exist_ok=True)
for folder in folders:
    full_path = os.path.join(base_dir, folder)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created folder: {full_path}")

# Buat file-file
for rel_path, content in files_content.items():
    full_path = os.path.join(base_dir, rel_path)
    # Pastikan folder parent-nya sudah ada
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created file: {full_path}")

print("\n✅ Semua folder dan file berhasil dibuat di:", base_dir)
print("Sekarang Anda dapat masuk ke folder thesis_inertia dan mulai bekerja.")