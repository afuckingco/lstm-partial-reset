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
stock-reset-lstm/
├── data/ # (empty, for dataset)
├── experiments/ # main experiment script
├── src/ # data_loader, model, utils
├── dashboard/ # Streamlit app
├── results/ # output CSV and plots
└── requirements.txt

text

## 🚀 Getting Started

```bash
git clone https://github.com/afiqandico13/stock-reset-lstm.git
cd stock-reset-lstm
pip install -r requirements.txt
cd experiments
python main_stock.py
🌐 Live Demo (Streamlit)
bash
streamlit run dashboard/app.py
📜 License
MIT

text

**Edit README langsung di GitHub** (klik file → pensil) atau di lokal lalu push lagi.

### 3. 🏷️ Buat GitHub Release (v1.0.0)

- Buka repositori di GitHub → klik **Releases** → **Create a new release**.
- Tag version: `v1.0.0`
- Title: `Initial release – periodic reset LSTM for stock prediction`
- Description: ringkasan hasil (sama seperti di README).
- Klik **Publish release**.

### 4. 🌐 (Opsional) Deploy Dashboard ke Streamlit Cloud

Jika ingin dashboard bisa diakses online:

- Buka [share.streamlit.io](https://share.streamlit.io)
- Login dengan GitHub.
- Klik **New app** → pilih repo `afiqandico13/stock-reset-lstm` → branch `main` → file `dashboard/app.py`.
- Deploy. Anda akan mendapat URL publik (contoh: `https://stock-reset-lstm.streamlit.app`).
- Tempelkan badge ke README.

### 5. 📈 Analisis Hasil (Jika Ingin Perbaiki Performa)

Hasil MSE Anda menunjukkan **proposed lebih buruk** (improvement -0.77%). Ini adalah temuan. Anda bisa:

- **Optimasi parameter reset** (coba rasio 1% atau frekuensi 5 epoch).
- **Hapus synthetic drift** dan amati pada data asli.
- **Bandingkan dengan model lain** (ARIMA, Prophet).

Tuliskan di README sebagai "lesson learned" – justru menarik untuk portofolio.

### 6. 🔗 Tambahkan ke Profil GitHub & LinkedIn

- **Pin repositori** di profil GitHub agar terlihat di halaman utama.
- **Buat postingan di LinkedIn** dengan ringkasan proyek, tautan GitHub, dan screenshot.
- **Masukkan link ke CV** (jika melamar kerja).

---

## ✅ Checklist Final Proyek Portofolio

- [x] Kode lengkap dan berjalan.
- [x] Repositori publik di GitHub.
- [x] README profesional (badge, struktur, cara pakai).
- [x] Hasil eksperimen (positif atau negatif) disajikan jujur.
- [x] Dashboard Streamlit (lokal atau cloud).
- [ ] (Opsional) GitHub Release v1.0.0.
- [ ] (Opsional) Deploy dashboard ke Streamlit Cloud.

---

## 🚀 Kesimpulan

Anda telah berhasil **menyelesaikan proyek portofolio kedua** dengan struktur dan kualitas yang sama baiknya dengan tesis pertama. Proyek ini menunjukkan kemampuan Anda dalam:

- Implementasi LSTM dengan mekanisme reset periodik.
- Eksperimen time series dengan concept drift.
- Pengembangan dashboard interaktif.
- Dokumentasi dan publikasi kode.

**Selamat! 🎉** Proyek ini bisa menjadi nilai tambah besar saat melamar kerja atau studi lanjut. Jika ingin membuat proyek ketiga (misal prediksi energi, kualitas udara), saya siap bantu. Ada yang ingin ditanyakan lagi?
