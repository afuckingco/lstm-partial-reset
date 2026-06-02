[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

# Stock Price Prediction with Periodic Partial Reset LSTM

> **Portfolio Project** – Adapting to sudden market changes using periodic weight reset.

This repository implements **periodic partial reset** on LSTM to handle **sudden concept drift** in stock price prediction. The method randomly resets **3% of weights every 15 epochs**, helping the model adapt to regime changes.

## 📊 Key Results (AAPL stock with synthetic drift)

| Model | MSE | Improvement |
|-------|-----|-------------|
| Baseline (no reset) | 75528.23 | – |
| Proposed (periodic reset) | 76112.83 | **-0.77%** (slight degradation) |

> ⚠️ On this synthetic drift scenario, the periodic reset did not improve performance, highlighting the need for per‑dataset parameter tuning.

## 🗂️ Repository Structure
