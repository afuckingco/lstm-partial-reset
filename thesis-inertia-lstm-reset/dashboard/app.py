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
