import yfinance as yf
import pandas as pd
import numpy as np

def download_stock_data(ticker="AAPL", start="2020-01-01", end="2024-12-31"):
    """
    Download stock data from Yahoo Finance.
    Returns DataFrame with columns: price, price_lag1, return, volatility
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Close']].copy()
    df.columns = ['price']
    
    # Lag feature (past price)
    df['price_lag1'] = df['price'].shift(1)
    
    # Returns and rolling volatility
    df['return'] = df['price'].pct_change()
    df['volatility'] = df['return'].rolling(5).std()
    
    # Drop NaN rows from shifting and rolling
    df = df.dropna()
    return df

def add_synthetic_drift(df, drift_point=0.5, drift_factor=1.5):
    """
    Inject artificial sudden drift into the price series.
    drift_point: fraction of length where drift occurs (0 to 1)
    drift_factor: multiplier for price after drift
    """
    df = df.copy()
    idx = int(len(df) * drift_point)
    df.loc[df.index[idx]:, 'price'] = df.loc[df.index[idx]:, 'price'] * drift_factor
    # Recalculate lag feature after drift injection
    df['price_lag1'] = df['price'].shift(1)
    df = df.dropna()
    return df

def create_sequences(df, features, target, window=5):
    """
    Convert time series data into sequences for LSTM.
    """
    X = df[features].values
    y = df[target].values
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)