import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import urllib.request

def download_energy_data():
    os.makedirs("data", exist_ok=True)
    try:
        url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
        zip_path = "data/household_power_consumption.zip"
        urllib.request.urlretrieve(url, zip_path)
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data")
        csv_path = "data/household_power_consumption.txt"
        df = pd.read_csv(csv_path, sep=';', na_values='?', low_memory=False)
        df = df.dropna()
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('DateTime')
        df = df[['Global_active_power']].copy()
        df.columns = ['power']
        df = df.resample('H').mean().dropna()
        return df
    except:
        print("Download failed. Generating synthetic data.")
        return generate_synthetic_energy()

def generate_synthetic_energy(periods=5000):
    np.random.seed(42)
    t = np.arange(periods)
    daily = 0.3 * np.sin(2 * np.pi * t / 24)
    weekly = 0.2 * np.sin(2 * np.pi * t / (24*7))
    trend = 0.0001 * t
    noise = 0.05 * np.random.randn(periods)
    power = 1.0 + daily + weekly + trend + noise
    power = np.clip(power, 0.2, 2.5)
    return pd.DataFrame({'power': power})

def add_synthetic_drift(df, drift_point=0.7, drift_factor=1.3):
    df = df.copy()
    idx = int(len(df) * drift_point)
    df.iloc[idx:, 0] = df.iloc[idx:, 0] * drift_factor
    return df

def create_sequences(df, window=24):
    X, y = [], []
    data = df.values
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)