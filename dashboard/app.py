import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import download_stock_data, add_synthetic_drift, create_sequences
from src.model import SimpleLSTM, train_model, predict

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction with Periodic LSTM Reset")

# Sidebar controls
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
drift_factor = st.sidebar.slider("Synthetic Drift Factor", 1.0, 2.5, 1.5, 0.1)
reset_ratio = st.sidebar.slider("Reset Ratio (%)", 1, 10, 3) / 100
reset_freq = st.sidebar.slider("Reset Frequency (epochs)", 5, 30, 15)
epochs = st.sidebar.slider("Epochs", 10, 50, 20)

if st.sidebar.button("Run Experiment"):
    with st.spinner("Downloading data and training models..."):
        df_raw = download_stock_data(ticker, str(start_date), str(end_date))
        if df_raw.empty:
            st.error("No data found. Check ticker or date range.")
            st.stop()
        
        # Add synthetic drift
        df = add_synthetic_drift(df_raw, drift_point=0.7, drift_factor=drift_factor)
        
        features = ['price_lag1', 'return', 'volatility']
        target = 'price'
        window = 5
        
        X, y = create_sequences(df, features, target, window=window)
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)
        
        # Baseline
        model_base = SimpleLSTM(input_size=len(features))
        train_model(model_base, X_train_scaled, y_train, epochs=epochs)
        pred_base = predict(model_base, X_test_scaled)
        mse_base = np.mean((pred_base - y_test)**2)
        
        # Proposed with reset
        model_prop = SimpleLSTM(input_size=len(features))
        train_model(model_prop, X_train_scaled, y_train, epochs=epochs,
                    reset_every=reset_freq, reset_ratio=reset_ratio)
        pred_prop = predict(model_prop, X_test_scaled)
        mse_prop = np.mean((pred_prop - y_test)**2)
        
        improvement = (mse_base - mse_prop) / mse_base * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline MSE", f"{mse_base:.6f}")
        col2.metric("Proposed MSE", f"{mse_prop:.6f}")
        col3.metric("Improvement", f"{improvement:.2f}%", delta=f"{improvement:.1f}%" if improvement>0 else None)
        
        # Plot predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name="Actual", line=dict(color='black')))
        fig.add_trace(go.Scatter(y=pred_base, name="Baseline (no reset)", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(y=pred_prop, name="Proposed (periodic reset)", line=dict(dash='dot')))
        fig.update_layout(title=f"{ticker} Price Prediction Comparison", xaxis_title="Time Step", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: show raw data preview
        with st.expander("Show raw data preview"):
            st.dataframe(df.tail(20))