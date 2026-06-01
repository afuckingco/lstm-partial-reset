[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

# Periodic Partial Reset on LSTM for Concept Drift in Stress Prediction

> **Research project for MSc Data Science** – ITB STIKOM Bali (pre‑admission portfolio)

This repository implements **periodic partial reset** on LSTM to improve adaptation to **sudden concept drift** in time‑series stress prediction. Inspired by Newton’s law of inertia and psychological inertia, the method randomly resets **3% of weights every 15 epochs**.

## 🎯 Key Results

| Dataset | Baseline MSE | Proposed MSE | Improvement | p‑value | Effect Size |
|:--------|-------------:|-------------:|------------:|--------:|:------------|
| **Synthetic** (sudden drift, 42 iters) | 1.0203 | **0.2730** | **73.24%** | **<0.001** | Cohen's d = 2.34 |
| **Student Lifestyle** (real, 10 iters) | 3.1456 | 3.1510 | -0.17% | 0.0245 | – |

> ✅ **Major finding:** Periodic partial reset significantly reduces MSE (73%, p<0.001) on synthetic data with sudden drift.
> ⚠️ On the real Student Lifestyle dataset, the method does not improve performance, highlighting the need for per‑dataset parameter tuning.

## 🗂️ Repository Structure
