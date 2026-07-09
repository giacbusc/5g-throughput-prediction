# 5G Throughput Prediction — Developer Guide

## Project Overview
**Team 8 — Project #8**: Throughput Prediction in a Dense 5G deployment with Transfer Learning
**Course**: Network Data Analysis Laboratory (2025-2026), Politecnico di Milano
**Authors**: Giacomo Buscaglia, Zarlene de Mesa, Davide Roccuzzo

## Task Summary
- **Primary (12 pts)**: Throughput regression on ACC Arena users comparing **NN (Keras)** and **Random Forest (sklearn)**.
  Team-8-specific feature: the **features of the X closest users** (3-D Euclidean distance on x,y,z). Experiment X ∈ {3,5,10}
  (X=0/1 dropped by team decision: heavy user co-location makes a single arbitrary neighbour uninformative).
- **Advanced (3 pts)**: Transfer Learning ACC Arena → Salt&Tar. Fine-tuned model vs the same model trained from scratch on a limited Salt&Tar set.

## Design constraints
- **Must run end-to-end on Google Colab (T4 GPU) in < ~30 min.**
- Notebooks are **self-contained**: helper functions are defined **inline** (course-solution style), not imported from a package. There is **no `src/`**.
- Heavy choices are exposed as constants in the first config cell: `RESAMPLE_SECONDS` (default 60), `N_USERS` (default 1500, sampled **at random** from the full ~12k ACC Arena population, seeded), `X_VALUES`, `BEST_X`, `OUTLIER_PCT`, `ACTIVE_ONLY`.

## Raw data format (important)
Wide format, one folder per metric under each venue (`ACC Arena/`, `Salt & Tar/`):
- **Standard metrics** (`Throughput`, `BLER`, `PRB`, `RU_Association`, `SINR` DL/UL): first column `time`, remaining columns `entityStats id <N>` (one per user, ~500 users/file, ~10k rows).
- **Positions**: first column `timeStamp`, then repeated 5-column blocks `(entityStats id, latitude, longitude, altitude, mobileState)`.
  - `mobileState` → **traffic_type** (0–5); `latitude/longitude` → converted to local **metres** (x,y); `altitude` → z.
- Time grids across metrics are slightly offset/jittered → aligned with `reindex(method="nearest")` onto a common grid.

## Notebooks (run in order)
```
notebooks/
├── 01_eda.ipynb                    # Step 1 — raw data analysis/visualization
├── 02_preprocessing_features.ipynb # Step 2 — tidy table + X-closest-users features, saves arrays
├── 03_model_training.ipynb         # Step 3 — NN + RF, hyperparameter tuning w/ CV, saves models + metrics.csv
├── 04_evaluation.ipynb             # Step 4 — NN vs RF vs X comparison, picks best
└── 05_transfer_learning.ipynb      # Advanced — TL ACC → Salt&Tar
```
Inter-notebook handoff: `02` writes `data/processed/acc_X{x}.npz` + scaler + column list; `03` writes `results/models/*` and `results/metrics.csv`; `04`/`05` read those.

## Pipeline conventions
- **Target**: raw throughput in Mbps (no transform — matches the course solution notebooks' style). The heavy
  tail is handled in preprocessing: samples above the **99th train-percentile** are dropped (`OUTLIER_PCT`);
  EDA shows the top ~1% samples carry ~2/3 of total variance and would dominate MSE/R² otherwise.
- **Split**: by `user_id` (70/15/15) to avoid leaking a user's samples across splits. Outlier threshold computed on train only, after the split.
- **Feature schema** (fixed, so TL weights transfer): standardised numeric `[bler, prb, sinr_dl, sinr_ul, x, y, z]` + one-hot `traffic_type` (6 classes) + per-neighbour `[dist, sinr_dl, sinr_ul, prb, bler, throughput]` × X. `ru_id` is venue-specific and excluded.
- **Metrics**: MSE, MAE, R², training duration.
- `RANDOM_SEED = 42` everywhere.

## Data access on Colab
First cell mounts Drive and unzips `L5GHDD_Dataset.zip` (searched under `MyDrive`, with a `gdown` fallback on the shared folder id). Locally, just unzip so `data/raw/` contains the `ACC Arena` and `Salt & Tar` folders.

## Stack
Python 3.10+, numpy/pandas/scipy, scikit-learn, TensorFlow/Keras, matplotlib. See `requirements.txt`.
