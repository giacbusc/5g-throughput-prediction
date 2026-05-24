# 5G Throughput Prediction — CLAUDE.md

## Project Overview
**Team 8 — Project #8**: Throughput Prediction in a Dense 5G deployment with Transfer Learning  
**Course**: Network Data Analysis Laboratory (2025-2026), Politecnico di Milano  
**Authors**: Giacomo Buscaglia, Zarlene de Mesa, Davide Roccuzzo

## Task Summary
- **Primary (12 pts)**: Throughput regression on ACC Arena users using NN + Random Forest.  
  Team-8-specific feature: features of the **X closest users** (3D Euclidean distance). Experiment X ∈ {1,3,5,10}.
- **Advanced (3 pts)**: Transfer Learning from ACC Arena → Salt&Tar. Compare fine-tuned model vs model trained from scratch on limited Salt&Tar data.

## Dataset
- `data/raw/` — original CSVs from the HDD Liverpool dataset  
  - ACC Arena: 12k users, sampled every 3 s, ~10k samples/user  
  - Salt&Tar: 3k users, sampled every 1 s, ~10k samples/user  
- Columns: `timestamp, user_id, x, y, z, traffic_type, ru_id, sinr_dl, sinr_ul, throughput, prb, bler`
- `traffic_type` classes: 0=off, 1=idle, 2=constant_rate, 3=video, 4=gaming, 5=http

## Project Structure
```
5g-throughput-prediction/
├── data/
│   ├── raw/                  # Original dataset files (not committed)
│   └── processed/            # Preprocessed arrays / feature matrices
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb   # X-closest-users features
│   ├── 04_model_training.ipynb        # NN + RF + hyperparameter tuning
│   ├── 05_evaluation.ipynb            # Metrics, comparison plots
│   └── 06_transfer_learning.ipynb     # Advanced: TL ACC Arena → Salt&Tar
├── src/
│   ├── data/
│   │   ├── loader.py         # Load raw CSVs, split by venue
│   │   └── preprocessor.py   # Scaling, encoding, train/val/test split
│   ├── features/
│   │   └── closest_users.py  # Build X-nearest-neighbor feature matrix
│   ├── models/
│   │   ├── neural_network.py # MLP regressor (PyTorch)
│   │   ├── random_forest.py  # RF regressor (scikit-learn)
│   │   └── transfer_learning.py  # Fine-tuning utilities
│   └── utils/
│       ├── metrics.py        # MSE, MAE, R², training duration
│       └── visualization.py  # Reusable plotting helpers
├── results/
│   ├── figures/              # Saved plots (PNG/PDF)
│   └── models/               # Saved model weights (.pt, .pkl)
├── reports/                  # Final report (PDF/LaTeX)
├── requirements.txt
└── README.md
```

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/

# Run preprocessing from CLI
python -m src.data.preprocessor --venue acc_arena

# Run training
python -m src.models.neural_network --config configs/nn_acc_arena.yaml
```

## Coding Conventions
- Python 3.10+
- All notebooks are numbered and self-contained; they import from `src/`
- Models save checkpoints to `results/models/<model_name>_<timestamp>.pt/pkl`
- Figures save to `results/figures/<notebook_step>_<description>.png`
- Use `tqdm` for long loops; use `joblib` for parallel CV
- Random seeds: fix `RANDOM_SEED = 42` everywhere for reproducibility

## Evaluation Metrics
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (coefficient of determination)
- Training duration (seconds)
- Inference time (ms/sample)

## Notes
- The X-closest-users feature (Team 8) uses **3D Euclidean distance** on (x, y, z) position columns
- For Transfer Learning: freeze early layers of the NN trained on ACC Arena, fine-tune on Salt&Tar
- All hyperparameter searches use Optuna (TPE sampler) with 5-fold CV
