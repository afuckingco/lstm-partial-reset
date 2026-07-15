```markdown
```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ cat README.md
```

# 🧠 LSTM Partial Reset — Adaptive Time-Series Research Suite

> A comprehensive research suite exploring concept drift adaptation in Long Short-Term Memory (LSTM) networks through periodic partial weight resets. Applied and validated across three distinct, high-volatility domains: physiological stress prediction, energy consumption forecasting, and financial market modeling.

<div align="center">

[![Status](https://img.shields.io/badge/STATUS-CORE_RESEARCH-d1242f?style=for-the-badge&labelColor=1e1e2e)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-89b4fa?style=for-the-badge&labelColor=1e1e2e)](LICENSE)

</div>

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ ./train.py --pipeline --domain all
```

```text
[Pipeline] Time-Series Ingestion → Sliding Window Normalization → Partial-Reset LSTM Training → Drift Detection → Evaluation Metrics
Domains Active: 3 | Architecture: Adaptive LSTM | Forgetting Mitigation: ENABLED | Status: VALIDATED
```
> *Concept drift adaptation in LSTM models — periodic partial reset approach. All three domains live in one repository, each preserving its own commit history.*

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ htop --modules
```

## ⚙️ Core Research Modules

| Subproject | Domain | Priority | Description |
|------------|--------|----------|-------------|
| **[thesis-inertia-lstm-reset](./thesis-inertia-lstm-reset)** | Stress Prediction | ![](https://img.shields.io/badge/CORE-d1242f?style=flat) | Primary thesis research. Adapts to physiological concept drift using inertia-based partial resets to maintain baseline accuracy without catastrophic forgetting. |
| **[energy-reset-lstm](./energy-reset-lstm)** | Energy Forecasting | ![](https://img.shields.io/badge/IMPORTANT-ff6b00?style=flat) | Models seasonal and anomalous consumption patterns. Resets trigger on detected structural breaks in grid load data. |
| **[stock-reset-lstm](./stock-reset-lstm)** | Stock Prediction | ![](https://img.shields.io/badge/IMPORTANT-ff6b00?style=flat) | Financial time-series adaptation. Mitigates performance decay during market regime shifts via scheduled hidden-state regularization. |

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ htop --stack
```

## 🛠️ Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **Core Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) 3.9+ | Standard for deep learning research and data manipulation. |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Dynamic computation graph, ideal for custom weight-reset logic and gradient manipulation. |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) / ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Efficient handling of large-scale, multi-variate time-series arrays. |
| **Evaluation** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Robust metrics (RMSE, MAE, R²) and statistical drift detection (KS-test, ADWIN). |
| **Environment** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive experimentation, visualization, and reproducible research notebooks. |

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ ./setup.sh
```

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/afuckingco/lstm-partial-reset.git
cd lstm-partial-reset

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Navigate to a specific domain subproject
cd thesis-inertia-lstm-reset

# 5. Run the training pipeline with partial reset enabled
python train.py --config configs/partial_reset.yaml --epochs 100
```
> **⚠️ Note:** Datasets are not included in this repository due to size and privacy constraints. Refer to the `README.md` inside each subproject directory for data acquisition and preprocessing instructions.

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ tree -L 2 -I 'venv|__pycache__|.ipynb_checkpoints|data'
```

## 📂 Project Structure

```text
lstm-partial-reset/
├── thesis-inertia-lstm-reset/    # Core thesis: Stress prediction with inertia
├── energy-reset-lstm/            # Energy consumption forecasting
├── stock-reset-lstm/             # Financial market regime adaptation
├── src/                          # Shared utilities (custom LSTM layers, reset logic)
│   ├── models/
│   │   └── adaptive_lstm.py      # Core implementation of the partial reset mechanism
│   └── utils/
│       └── drift_detector.py     # Statistical tests for triggering resets
├── notebooks/                    # Cross-domain EDA and comparative analysis
├── requirements.txt              # Global Python dependencies
└── README.md                     # Master documentation
```

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ cat KNOWN_LIMITATIONS.md
```

## ⚠️ Research Limitations & Trade-offs

- **Reset Frequency Tuning**: The optimal interval for partial resets is highly domain-dependent. Too frequent resets cause instability; too infrequent leads to concept drift lag.
- **Computational Overhead**: Continuous drift monitoring and selective weight manipulation add ~15-20% training time overhead compared to standard LSTMs.
- **Data Stationarity Assumption**: While designed for drift, extreme, non-reverting structural breaks (e.g., black swan financial events) may still require full model retraining rather than partial resets.

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ echo $ROADMAP
```

## 📈 Future Improvements

- [ ] **Automated Hyperparameter Search**: Integrate Optuna to dynamically find the optimal reset threshold and inertia coefficient per dataset.
- [ ] **Transformer Baseline**: Implement a Temporal Fusion Transformer (TFT) with similar reset mechanics for comparative benchmarking.
- [ ] **Edge Deployment**: Quantize the adapted models for on-device, real-time inference (e.g., wearable stress monitoring).
- [ ] **Unified API**: Abstract the domain-specific data loaders into a single, cohesive `TimeSeriesAdapter` interface.

---

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ connect --author
```

## 👤 Author

**Afiq Andico Pangimpian** — Machine learning researcher, systems thinker, and builder of adaptive architectures.

<div align="center">
  <a href="https://github.com/afuckingco" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/in/pangimpian" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
  <a href="mailto:anotherwaltzcompany@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
</div>

> *Systems decay. Models drift. The solution is not to rebuild from scratch, but to know exactly what to let go, and what to keep.*

```console
┌──(test㉿afuckingco)-[~/projects/lstm-partial-reset]
└─$ exit
```
> *Connection closed. Build something useful.*
```
