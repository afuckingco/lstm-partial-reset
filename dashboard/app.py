import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import download_energy_data, add_synthetic_drift, create_sequences
from src.model import SimpleLSTM, train_model, predict

st.set_page_config(layout="wide")
st.title("🔋 Energy Consumption Prediction with Periodic LSTM Reset")

drift_factor = st.sidebar.slider("Synthetic Drift Factor", 1.0, 2.0, 1.3, 0.05)
reset_ratio = st.sidebar.slider("Reset Ratio (%)", 1, 10, 3) / 100
reset_freq = st.sidebar.slider("Reset Frequency (epochs)", 5, 30, 15)
epochs = st.sidebar.slider("Epochs", 10, 50, 20)

if st.sidebar.button("Run Experiment"):
    with st.spinner("Loading data and training..."):
        df_raw = download_energy_data()
        df = add_synthetic_drift(df_raw, drift_point=0.7, drift_factor=drift_factor)
        X, y = create_sequences(df, window=24)
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
        model_base = SimpleLSTM(input_size=1)
        train_model(model_base, X_train_scaled, y_train, epochs=epochs)
        pred_base = predict(model_base, X_test_scaled)
        mse_base = np.mean((pred_base - y_test)**2)
        model_prop = SimpleLSTM(input_size=1)
        train_model(model_prop, X_train_scaled, y_train, epochs=epochs,
                    reset_every=reset_freq, reset_ratio=reset_ratio)
        pred_prop = predict(model_prop, X_test_scaled)
        mse_prop = np.mean((pred_prop - y_test)**2)
        improvement = (mse_base - mse_prop) / mse_base * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline MSE", f"{mse_base:.6f}")
        col2.metric("Proposed MSE", f"{mse_prop:.6f}")
        col3.metric("Improvement", f"{improvement:.2f}%", delta=f"{improvement:.1f}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name='Actual', line=dict(color='black')))
        fig.add_trace(go.Scatter(y=pred_base, name='Baseline', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(y=pred_prop, name='Proposed', line=dict(dash='dot')))
        fig.update_layout(title="Energy Consumption Prediction", xaxis_title="Time step", yaxis_title="Power (kW)")
        st.plotly_chart(fig, use_container_width=True)