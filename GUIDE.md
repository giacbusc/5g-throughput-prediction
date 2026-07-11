# 5G Throughput Prediction — Developer Guide

## Project Overview
**Team 8 — Project #8**: Throughput Prediction in a Dense 5G deployment with Transfer Learning
**Course**: Network Data Analysis Laboratory (2025-2026), Politecnico di Milano
**Authors**: Giacomo Buscaglia, Zarlene de Mesa, Davide Roccuzzo

## Task Summary
- **Primary (12 pts)**: Throughput regression on ACC Arena users comparing **NN (Keras)** and **Random Forest (sklearn)**.
  Team-8-specific feature: the **features of the X closest users** (3-D Euclidean distance on x,y,z). Experiment X ∈ {3,5,10}
  × two **neighbour encodings** — `pos` (ordered per-neighbour columns `nb0_*, nb1_*, ...`) and `agg` (order-invariant
  aggregates: `nb_prb_sum` ≈ cell load, `nb_sinr_dl/ul_mean`, `nb_bler_mean`, `nb_active_count`) —
  plus a **X=0 no-neighbour baseline** that quantifies the neighbours' net contribution. Rationale: ~99% user co-location
  makes the distance ordering arbitrary (ties at ~0 m), so positional columns are permutation noise; aggregates encode the
  contention mechanism (shared PRB budget) directly. X=1 dropped by team decision (a single arbitrary neighbour is uninformative).
- **Advanced (3 pts)**: Transfer Learning ACC Arena → Salt&Tar. Fine-tuned model vs the same model trained from scratch on a limited Salt&Tar set.

## Design constraints
- **Must run end-to-end on Google Colab (T4 GPU) in < ~30 min.**
- Notebooks are **self-contained**: helper functions are defined **inline** (course-solution style), not imported from a package. There is **no `src/`**.
- Heavy choices are exposed as constants in the first config cell: `RESAMPLE_SECONDS` (default 120 — **the dataset-size knob**: coarsen the time grid rather than subsample users, per the professor's guidance), `N_USERS` (default `None` = **ALL** ~12k ACC users, so the X-closest neighbourhoods are the real ones; an int subsample biases them and is for debug runs only), `X_VALUES`, `ENCODINGS`, `BEST_X`, `BEST_ENC`, `OUTLIER_PCT`, `ACTIVE_ONLY`.

## Raw data format (important)
Wide format, one folder per metric under each venue (`ACC Arena/`, `Salt & Tar/`):
- **Standard metrics** (`Throughput`, `BLER`, `PRB`, `RU_Association`, `SINR` DL/UL): first column `time`, remaining columns `entityStats id <N>` (one per user, ~500 users/file, ~10k rows).
- **Positions**: first column `timeStamp`, then repeated 5-column blocks `(entityStats id, latitude, longitude, altitude, mobileState)`.
  - `mobileState` → **traffic_type** (0–5); `latitude/longitude` → converted to local **metres** (x,y); `altitude` → z.
- **Timestamps are jittered** (measured on the raw files): ACC Arena has a nominal 3s cadence but ~1/3 of the steps are 4s, with duplicate stamps (diff 0s) and holes up to 7s; Salt&Tar runs at ~1s. Metrics are offset ~3s from each other (throughput starts at t=1215, the rest at 1212). Alignment: duplicates dropped, then `reindex(method="nearest", tolerance=max(RESAMPLE_SECONDS/2, 4s))` onto a common grid — the 4s floor keeps every grid point matched even at fine grids (0 NaN verified at 120s on both venues).
- Some Salt&Tar position files have lowercase names (`positions_salt_tar_UE_Id_500_999.csv`): globs must stay case-insensitive-friendly (`*.csv`, not `Positions_*.csv`).

## Notebooks (run in order)
```
notebooks/
├── 01_eda.ipynb                    # Step 1 — raw data analysis/visualization
├── 02_preprocessing_features.ipynb # Step 2 — tidy table + X-closest-users features, saves arrays
├── 03_model_training.ipynb         # Step 3 — NN + RF, hyperparameter tuning w/ CV, saves models + metrics.csv
├── 04_evaluation.ipynb             # Step 4 — NN vs RF vs X comparison, picks best
├── 05_transfer_learning.ipynb      # Advanced — TL ACC → Salt&Tar
└── 06_same_traffic_neighbors.ipynb # Experiment — neighbours drawn from the SAME traffic class
```
Notebook 06 is a self-contained experiment (run after 03 so `metrics.csv` exists for the comparison):
same pipeline/split/outlier rule as 02-03, but neighbours are searched among users with the **same
`traffic_type` at the same instant** (grouping by `(time, traffic_type)`), aggregated encoding only.
Writes `results/metrics_sametype.csv` (`enc="agg_same"`) + `models/*_X{x}_same.*` and a comparison chart
against the X=0 baseline and generic-pool `agg`.
Inter-notebook handoff: `02` writes `data/processed/acc_X0.npz` (baseline), `acc_X{x}.npz` (pos) and `acc_X{x}_agg.npz` (agg) + scaler + column list each; `03` trains the 7 scenarios × 2 models, writes `results/models/*` (`nn/rf_X{x}[_agg]`) and `results/metrics.csv` (with an `enc` column); `04`/`05` read those (`05` follows `BEST_X`/`BEST_ENC`).

## Pipeline conventions
- **Target**: raw throughput in Mbps (no transform — matches the course solution notebooks' style). The heavy
  tail is handled in preprocessing: samples above the **99th train-percentile** are dropped (`OUTLIER_PCT`);
  EDA shows the top ~1% samples carry ~2/3 of total variance and would dominate MSE/R² otherwise.
- **Split**: by `user_id` (70/15/15) to avoid leaking a user's samples across splits. Outlier threshold computed on train only, after the split.
- **Feature schema** (fixed per scenario, so TL weights transfer): standardised numeric `[bler, prb, sinr_dl, sinr_ul, x, y, z]` + one-hot `traffic_type` (6 classes) + neighbour features per the encoding — `pos`: per-neighbour `[dist, sinr_dl, sinr_ul, prb, bler]` × X; `agg`: `[nb_prb_sum, nb_sinr_dl_mean, nb_sinr_ul_mean, nb_bler_mean, nb_active_count]` (constant size in X); X=0: none. `ru_id` is venue-specific and excluded. **Neighbour `throughput` is excluded by design** (2026-07-11): it is the target variable measured on other users — not a UT/RU feature available at inference — and under co-location it would open a label-sharing channel between test users.
- **Metrics**: MSE, MAE, R², training duration.
- `RANDOM_SEED = 42` everywhere.

## Data access on Colab
First cell mounts Drive and unzips `L5GHDD_Dataset.zip` (searched under `MyDrive`, with a `gdown` fallback on the shared folder id). Locally, just unzip so `data/raw/` contains the `ACC Arena` and `Salt & Tar` folders.

## Stack
Python 3.10+, numpy/pandas/scipy, scikit-learn, TensorFlow/Keras, matplotlib. See `requirements.txt`.
