"""
src/utils/metrics.py
=====================
Regression evaluation metrics for throughput prediction.

All functions accept 1-D numpy arrays and return Python scalars.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time_s: float = 0.0,
    inference_time_ms: float = 0.0,
) -> dict:
    """Compute all regression metrics and return a summary dict.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predictions (Mbps).
    training_time_s : float
        Wall-clock training time in seconds (optional).
    inference_time_ms : float
        Inference time per sample in ms (optional).

    Returns
    -------
    dict with keys: mse, rmse, mae, r2, mape,
                    training_time_s, inference_time_ms_per_sample
    """
    return {
        "mse":                  mse(y_true, y_pred),
        "rmse":                 rmse(y_true, y_pred),
        "mae":                  mae(y_true, y_pred),
        "r2":                   r2(y_true, y_pred),
        "mape_%":               mape(y_true, y_pred),
        "training_time_s":      training_time_s,
        "inference_time_ms":    inference_time_ms,
    }


def measure_inference_time(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_runs: int = 5,
) -> float:
    """Return the average inference time in ms per sample over *n_runs*.

    Parameters
    ----------
    predict_fn : callable
        A function that accepts X and returns predictions.
    X : np.ndarray
        Input array for timing.
    n_runs : int
        Number of repetitions for averaging.

    Returns
    -------
    float
        Average inference time in **ms per sample**.
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(X)
        times.append(time.perf_counter() - t0)
    avg_per_sample_ms = (np.mean(times) / len(X)) * 1000
    return float(avg_per_sample_ms)


def print_metrics(metrics: dict, title: str = "") -> None:
    """Pretty-print a metrics dictionary."""
    header = f"─── {title} ───" if title else "─── Metrics ───"
    print(header)
    print(f"  MSE              : {metrics['mse']:.4f} Mbps²")
    print(f"  RMSE             : {metrics['rmse']:.4f} Mbps")
    print(f"  MAE              : {metrics['mae']:.4f} Mbps")
    print(f"  R²               : {metrics['r2']:.4f}")
    print(f"  MAPE             : {metrics['mape_%']:.2f} %")
    if metrics.get("training_time_s"):
        print(f"  Training time    : {metrics['training_time_s']:.1f} s")
    if metrics.get("inference_time_ms"):
        print(f"  Inference time   : {metrics['inference_time_ms']:.4f} ms/sample")
    print()
