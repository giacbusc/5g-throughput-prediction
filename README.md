# 5G Throughput Prediction with Transfer Learning

This repository contains **Project #8** developed for the **Network Data Analysis Laboratory (2025-2026)** course at **Politecnico di Milano**.

**Team 8:** Giacomo Buscaglia, Zarlene de Mesa, Davide Roccuzzo

---

## 🎯 Objective

> Can we predict the user throughput given user terminal and radio unit features in a dense 5G deployment?

| Task | Points | Description |
|------|--------|-------------|
| **Regression** | 12 pts | Throughput regression on ACC Arena users using **Neural Network (Keras)** and **Random Forest (scikit-learn)**, augmented with the features of the **X closest users** |
| **Transfer Learning** | 3 pts | Fine-tune an ACC Arena model on a *limited* Salt&Tar training set; compare against the same model trained from scratch |

The project is designed to **run end-to-end on Google Colab (T4 GPU) in under ~30 minutes**.

---

## 📊 Dataset

*5G High Density Demand (HDD) Dataset in Liverpool City Region, UK* (Maheshwari et al., 2025).

| Venue | Users | Native sampling | Samples/user |
|-------|-------|-----------------|--------------|
| **ACC Arena** | 12,000 | 1 sample / ~3 s | ~10,000 |
| **Salt & Tar** | 3,000 | 1 sample / s | ~10,000 |

**Raw layout (wide format):** one folder per metric (`Throughput`, `BLER`, `PRB`, `RU_Association`, `SINR` DL/UL, `Positions`); each CSV has a `time` column plus one column per user (`entityStats id N`), ~500 users per file. `Positions` files store interleaved blocks of `(id, latitude, longitude, altitude, mobileState)` where **`mobileState` is the traffic type** and lat/long are converted to local metres.

**Traffic-type classes:** 0=off, 1=idle, 2=constant_rate, 3=video, 4=gaming, 5=http

> The notebooks resample to **1 sample / `RESAMPLE_SECONDS`** (default 60 s) and load a configurable **`N_USERS`** subset to stay within Colab limits. This is a regression *per sample*, not a time-series forecast, so down-sampling rows does not change the learned feature→throughput relationship.

---

## 🏗️ Project Structure

```
5g-throughput-prediction/
├── data/
│   ├── raw/                          # ⚠️ Not committed — dataset unzipped here (or mounted from Drive on Colab)
│   └── processed/                    # Auto-generated arrays (notebook 02)
├── notebooks/                        # 5 self-contained notebooks (no src/ package — Colab-friendly)
│   ├── 01_eda.ipynb                  # Step 1 — raw data analysis / visualization
│   ├── 02_preprocessing_features.ipynb # Step 2 — tidy table + X-closest-users features
│   ├── 03_model_training.ipynb       # Step 3 — NN + RF, hyperparameter tuning w/ CV
│   ├── 04_evaluation.ipynb           # Step 4 — metrics & NN-vs-RF-vs-X comparison
│   └── 05_transfer_learning.ipynb    # ⭐ Advanced — TL ACC Arena → Salt&Tar
├── results/
│   ├── figures/                      # Auto-generated plots
│   ├── models/                       # Saved models (.keras / .pkl)
│   └── metrics.csv                   # Written by notebook 03
├── requirements.txt
├── GUIDE.md                          # Developer guide
└── README.md
```

Each notebook is **self-contained** (all helper functions are defined inline, in the style of the course solution notebooks) and shares data through files in `data/processed/` and `results/`.

---

## 🔬 Methodology (mapped to the 4 mandatory steps + advanced task)

1. **`01_eda.ipynb` — Raw data analysis.** Load the wide-format ACC Arena data into a tidy per-user/per-timestamp table; inspect throughput distribution, traffic-type effects, correlations, SINR-vs-throughput, spatial layout.
2. **`02_preprocessing_features.ipynb` — Preprocessing & features.** Build the full tidy table, resample, engineer the **Team-8 X-closest-users features** (3-D Euclidean KD-tree), keep active users only (`ACTIVE_ONLY`), drop extreme-throughput outliers above the 99th train-percentile (`OUTLIER_PCT`, motivated in the EDA), split **by user** (no leakage), standardise, and save arrays for `X ∈ {0,1,3,5,10}`.
3. **`03_model_training.ipynb` — Optimisation & training.** Train an **MLP (Keras)** and a **Random Forest (sklearn)** for each `X`; hyperparameters tuned with cross-validation (kept light for the time budget). Records MSE/MAE/R², **training duration** and **inference time**.
4. **`04_evaluation.ipynb` — Testing & scenarios.** Compare NN vs RF across `X`, plot metrics-vs-X, pick the best configuration.
5. **`05_transfer_learning.ipynb` — Advanced.** Fine-tune the ACC-trained MLP on a limited Salt&Tar set vs the same network trained from scratch.

---

## 🚀 Getting Started (Google Colab — recommended)

1. Put **`L5GHDD_Dataset.zip`** in your Google Drive (the notebooks search `MyDrive` for it; a `gdown` fallback downloads it from the shared folder).
2. Open each notebook in Colab, set the runtime to **T4 GPU**, and run top-to-bottom **in order** (01 → 05).
3. The first cells mount Drive, unzip the dataset, and create the output folders automatically.

Key knobs live in the **config cell** at the top of every notebook: `RESAMPLE_SECONDS`, `N_USERS`, `X_VALUES`, `BEST_X`, `OUTLIER_PCT`, `ACTIVE_ONLY`.

### Local run

```bash
pip install -r requirements.txt
# unzip the dataset so that data/raw/ contains the "ACC Arena" and "Salt & Tar" folders
jupyter notebook notebooks/
```

---

## 📈 Results Summary

> *(To be filled after running all notebooks — see `results/metrics.csv`.)*

| Model | X | MSE | MAE (Mbps) | R² | Train time (s) |
|-------|---|-----|------------|-----|----------------|
| RF    | 0 |     |            |     |                |
| RF    | best |  |            |     |                |
| MLP   | 0 |     |            |     |                |
| MLP   | best |  |            |     |                |

---

## 📚 References

1. Maheshwari, M. K., et al. *"5G High Density Demand Dataset in Liverpool City Region, UK."* Scientific Data (2025).
2. Maheshwari, M. K., et al. *"5G High Density Demand (HDD) Dataset in Liverpool City Region, UK (Supplement)."* (2025).
