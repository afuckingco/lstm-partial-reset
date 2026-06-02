```markdown
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

# Stock Price Prediction with Periodic Partial Reset LSTM

> **Portfolio Project** – Adapting to sudden market changes using periodic weight reset.

This repository implements **periodic partial reset** on LSTM to handle **sudden concept drift** in stock price prediction. Inspired by Newton’s law of inertia and psychological inertia, the method randomly resets **3% of weights every 15 epochs**, helping the model escape local minima during market regime shifts.

## 📊 Key Results (AAPL stock with synthetic drift)

| Model | MSE | Improvement |
|-------|-----|-------------|
| Baseline (no reset) | 75528.23 | – |
| Proposed (periodic reset) | 76112.83 | **-0.77%** |

> ⚠️ On this synthetic drift scenario, the periodic reset did **not** improve performance. This negative result is still valuable – it highlights the sensitivity of the method to parameter choices and dataset characteristics. With proper tuning (e.g., reset ratio 1%, frequency 5 epochs), the method might yield positive gains.

## 🗂️ Repository Structure

```
stock-reset-lstm/
├── data/                     # (empty) place for CSV datasets
├── experiments/              # main experiment script (main_stock.py)
├── src/                      # core modules
│   ├── data_loader.py        # download & preprocess stock data
│   ├── model.py              # LSTM + periodic reset logic
│   └── utils.py              # metrics, visualization
├── dashboard/                # Streamlit interactive dashboard
│   └── app.py
├── results/                  # output CSV and plots (auto-generated)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/afiqandico13/stock-reset-lstm.git
cd stock-reset-lstm
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 3. Run the main experiment
```bash
cd experiments
python main_stock.py
```

This will download AAPL stock data (2020–2024), inject a synthetic drift at 70% of the series, and compare baseline vs periodic reset LSTM over 30 bootstrap iterations.

### 4. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard lets you choose any ticker, adjust reset parameters, and see predictions in real time.

## 📈 Visual Results

![Boxplot](results/boxplot_stock.png)

*Boxplot of MSE over 30 iterations – baseline vs proposed.*

## 🧠 Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Reset ratio | 3% | Small enough to avoid catastrophic forgetting, adjustable |
| Reset frequency | every 15 epochs | One reset in a 20‑epoch training |
| LSTM hidden size | 32 | Lightweight, runs on CPU |
| Window size | 5 days | Uses one week of past prices |

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Inspired by Newton’s law of inertia and Kuppens et al. (2010) on emotional inertia.
- Stock data from Yahoo Finance (yfinance library).
- Built with PyTorch, Streamlit, and scikit-learn.

---

**📧 Contact**  
For questions or collaboration, please open an issue on GitHub.
```
