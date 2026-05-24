"""
src/models/neural_network.py
=============================
MLP (Multi-Layer Perceptron) regressor built with PyTorch.

Supports:
  - Configurable depth / width
  - Dropout regularisation
  - Early stopping
  - Checkpoint saving
  - Fine-tuning (freeze / unfreeze layers) for Transfer Learning

Usage
-----
>>> from src.models.neural_network import MLPRegressor, train_mlp
>>> model = MLPRegressor(input_dim=20)
>>> history = train_mlp(model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """Fully-connected feedforward network for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Number of neurons in each hidden layer.
    dropout : float
        Dropout probability (applied after each hidden layer). 0 = no dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # single output: throughput

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Transfer Learning helpers
    # ------------------------------------------------------------------

    def freeze_feature_extractor(self, n_layers_to_keep_trainable: int = 1) -> None:
        """Freeze all layers except the last *n_layers_to_keep_trainable* blocks.

        Used for Transfer Learning: call this after loading a model trained on
        ACC Arena, then fine-tune only the top layers on Salt&Tar data.

        Parameters
        ----------
        n_layers_to_keep_trainable : int
            Number of hidden blocks (from the end) to keep trainable, plus the
            final output layer (always trainable).
        """
        all_params = list(self.net.parameters())
        # Freeze all first
        for p in all_params:
            p.requires_grad = False

        # Always unfreeze the output linear layer
        list(self.net.children())[-1].weight.requires_grad = True
        list(self.net.children())[-1].bias.requires_grad = True

        # Unfreeze the last n blocks (each block = Linear + BN + ReLU + Dropout)
        block_size = 4  # Linear, BN, ReLU, Dropout
        for block_idx in range(n_layers_to_keep_trainable):
            start = -(block_idx + 1) * block_size - 1  # -1 for output layer
            for layer in list(self.net.children())[start: start + block_size]:
                for p in layer.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Trainable parameters: %d / %d (%.1f%%)",
            trainable, total, 100 * trainable / total,
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (full fine-tuning)."""
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mlp(
    model: MLPRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 10,
    checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> dict:
    """Train *model* and return a history dictionary.

    Parameters
    ----------
    model : MLPRegressor
    X_train, y_train : np.ndarray
    X_val, y_val : np.ndarray
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    batch_size : int
    max_epochs : int
    patience : int
        Early-stopping patience (epochs without val loss improvement).
    checkpoint_path : path-like, optional
        If given, save the best model weights here.
    verbose : bool
        Print epoch summaries.

    Returns
    -------
    dict with keys: ``train_loss``, ``val_loss``, ``best_epoch``,
    ``training_time_s``.
    """
    model = model.to(DEVICE)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5, verbose=False
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
        train_loss = running_loss / len(y_train)

        # --- Validation ---
        val_loss = _evaluate_loss(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            logger.info(
                "Epoch %3d/%d  train_MSE=%.4f  val_MSE=%.4f  best=%d",
                epoch, max_epochs, train_loss, val_loss, best_epoch,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    elapsed = time.time() - start_time
    logger.info(
        "Training done in %.1f s | best_epoch=%d | best_val_MSE=%.4f",
        elapsed, best_epoch, best_val_loss,
    )

    # Reload best weights
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "training_time_s": elapsed,
    }


def predict_mlp(
    model: MLPRegressor,
    X: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run inference and return predictions as a 1-D numpy array."""
    model.eval()
    model.to(DEVICE)
    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def _evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            total_loss += criterion(pred, y_batch).item() * len(y_batch)
            total_n += len(y_batch)
    return total_loss / total_n
