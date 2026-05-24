"""
src/utils/visualization.py
===========================
Reusable plotting helpers used across all notebooks.

All functions return a :class:`matplotlib.figure.Figure` so callers can
save them to ``results/figures/`` with ``fig.savefig(...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Global aesthetic settings
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
FIGSIZE_DEFAULT = (10, 6)
RESULTS_FIGURES = Path("results/figures")


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def plot_throughput_distribution(
    df: pd.DataFrame,
    venue: str = "",
    bins: int = 60,
    save: bool = False,
) -> plt.Figure:
    """Histogram + KDE of throughput values."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    sns.histplot(df["throughput"], bins=bins, kde=True, ax=ax, color="steelblue")
    ax.set_xlabel("Throughput (Mbps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Throughput Distribution{' — ' + venue if venue else ''}")
    _maybe_save(fig, f"01_throughput_dist_{venue}.png", save)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    venue: str = "",
    save: bool = False,
) -> plt.Figure:
    """Pearson correlation heatmap."""
    if columns is None:
        columns = ["sinr_dl", "sinr_ul", "prb", "bler", "throughput"]
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, square=True, ax=ax,
    )
    ax.set_title(f"Correlation Heatmap{' — ' + venue if venue else ''}")
    fig.tight_layout()
    _maybe_save(fig, f"01_corr_heatmap_{venue}.png", save)
    return fig


def plot_spatial_throughput(
    df: pd.DataFrame,
    venue: str = "",
    sample_frac: float = 0.05,
    save: bool = False,
) -> plt.Figure:
    """2D scatter of user positions coloured by mean throughput."""
    sample = df.sample(frac=sample_frac, random_state=42) if len(df) > 10_000 else df
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        sample["x"], sample["y"],
        c=sample["throughput"], cmap="plasma",
        s=5, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Throughput (Mbps)")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(f"Spatial Throughput Distribution{' — ' + venue if venue else ''}")
    _maybe_save(fig, f"01_spatial_throughput_{venue}.png", save)
    return fig


def plot_traffic_type_distribution(
    df: pd.DataFrame,
    venue: str = "",
    save: bool = False,
) -> plt.Figure:
    """Bar chart of traffic type counts with labels."""
    labels = {0: "off", 1: "idle", 2: "const_rate", 3: "video", 4: "gaming", 5: "http"}
    counts = df["traffic_type"].value_counts().sort_index()
    counts.index = [labels.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("tab10"))
    ax.set_xlabel("Traffic Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Traffic Type Distribution{' — ' + venue if venue else ''}")
    ax.tick_params(axis="x", rotation=30)
    _maybe_save(fig, f"01_traffic_type_{venue}.png", save)
    return fig


def plot_boxplot_by_traffic(
    df: pd.DataFrame,
    feature: str = "throughput",
    venue: str = "",
    save: bool = False,
) -> plt.Figure:
    """Boxplot of *feature* grouped by traffic type."""
    labels = {0: "off", 1: "idle", 2: "const_rate", 3: "video", 4: "gaming", 5: "http"}
    df2 = df.copy()
    df2["traffic_label"] = df2["traffic_type"].map(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df2, x="traffic_label", y=feature, ax=ax, palette="tab10")
    ax.set_xlabel("Traffic Type")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by Traffic Type{' — ' + venue if venue else ''}")
    _maybe_save(fig, f"01_box_{feature}_{venue}.png", save)
    return fig


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict,
    model_name: str = "",
    save: bool = False,
) -> plt.Figure:
    """Loss curves for a training run (output of train_mlp)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"],   label="Val MSE")
    ax.axvline(history["best_epoch"] - 1, color="red", linestyle="--",
               alpha=0.6, label=f"Best epoch ({history['best_epoch']})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training History{' — ' + model_name if model_name else ''}")
    ax.legend()
    _maybe_save(fig, f"04_training_history_{model_name}.png", save)
    return fig


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    max_points: int = 2000,
    save: bool = False,
) -> plt.Figure:
    """Scatter of predictions vs. ground truth."""
    idx = np.random.choice(len(y_true), min(max_points, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true[idx], y_pred[idx], s=10, alpha=0.4, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("True Throughput (Mbps)")
    ax.set_ylabel("Predicted Throughput (Mbps)")
    ax.set_title(f"Predicted vs True{' — ' + model_name if model_name else ''}")
    ax.legend()
    _maybe_save(fig, f"05_pred_vs_true_{model_name}.png", save)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save: bool = False,
) -> plt.Figure:
    """Residual distribution plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, s=8, alpha=0.3, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicted Throughput (Mbps)")
    axes[0].set_ylabel("Residual (Mbps)")
    axes[0].set_title("Residuals vs Predicted")

    sns.histplot(residuals, bins=60, kde=True, ax=axes[1], color="coral")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (Mbps)")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(f"Residual Analysis{' — ' + model_name if model_name else ''}")
    fig.tight_layout()
    _maybe_save(fig, f"05_residuals_{model_name}.png", save)
    return fig


def plot_metrics_comparison(
    results: dict[str, dict],
    metrics: Sequence[str] = ("rmse", "mae", "r2"),
    save: bool = False,
) -> plt.Figure:
    """Bar chart comparing multiple models across metrics.

    Parameters
    ----------
    results : dict
        ``{model_name: metrics_dict}`` where each ``metrics_dict`` is the
        output of :func:`~src.utils.metrics.evaluate`.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        names = list(results.keys())
        vals  = [results[m].get(metric, 0) for m in names]
        colors = sns.color_palette("tab10", len(names))
        bars = ax.bar(names, vals, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Model Comparison")
    fig.tight_layout()
    _maybe_save(fig, "05_model_comparison.png", save)
    return fig


def plot_x_sensitivity(
    x_values: list[int],
    metrics_per_x: list[dict],
    metric: str = "rmse",
    model_name: str = "",
    save: bool = False,
) -> plt.Figure:
    """Line plot of a metric vs. number of closest neighbours X."""
    vals = [m[metric] for m in metrics_per_x]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, vals, marker="o", linewidth=2, color="steelblue")
    ax.set_xlabel("X (number of closest users)")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"{metric.upper()} vs. Number of Closest Users (X)"
        f"{' — ' + model_name if model_name else ''}"
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _maybe_save(fig, f"05_x_sensitivity_{metric}_{model_name}.png", save)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "",
    save: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of feature importances (Random Forest)."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Top-{top_n} Feature Importances"
        f"{' — ' + model_name if model_name else ''}"
    )
    fig.tight_layout()
    _maybe_save(fig, f"05_feature_importance_{model_name}.png", save)
    return fig


# ---------------------------------------------------------------------------
# Transfer learning helpers
# ---------------------------------------------------------------------------

def plot_transfer_learning_comparison(
    results_by_strategy: dict[str, dict],
    metrics: Sequence[str] = ("rmse", "mae", "r2"),
    train_sizes: Optional[list[int]] = None,
    save: bool = False,
) -> plt.Figure:
    """Compare fine-tuned vs. scratch-trained model metrics.

    Parameters
    ----------
    results_by_strategy : dict
        ``{'fine_tune': metrics_dict, 'scratch': metrics_dict}``
    train_sizes : list of int, optional
        If provided, plot metric vs. Salt&Tar training set size.
    """
    if train_sizes is not None:
        # Plot each metric as a function of training size
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            for strategy, vals_list in results_by_strategy.items():
                ax.plot(train_sizes, [v[metric] for v in vals_list],
                        marker="o", label=strategy)
            ax.set_xlabel("Salt&Tar training set size")
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())
            ax.legend()
        fig.suptitle("Transfer Learning: Metric vs. Training Set Size")
    else:
        fig = plot_metrics_comparison(results_by_strategy, metrics=metrics)
        fig.suptitle("Transfer Learning: Fine-tune vs. Scratch")

    fig.tight_layout()
    _maybe_save(fig, "06_tl_comparison.png", save)
    return fig


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _maybe_save(fig: plt.Figure, fname: str, save: bool) -> None:
    if save:
        RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
        path = RESULTS_FIGURES / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
