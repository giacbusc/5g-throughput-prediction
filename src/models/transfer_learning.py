"""
src/models/transfer_learning.py
================================
Transfer Learning utilities for the advanced task (Team 8):
  Train on ACC Arena → fine-tune on (limited) Salt&Tar data.

Two strategies are compared:
  1. **Fine-tuning**: load ACC Arena weights, freeze early layers, retrain top layers.
  2. **Scratch baseline**: train the same architecture from scratch on the limited
     Salt&Tar training set.

Usage
-----
>>> from src.models.transfer_learning import TransferLearner
>>> tl = TransferLearner(base_model_path="results/models/mlp_acc_arena.pt",
...                      input_dim=20)
>>> tl.fine_tune(X_salt_train, y_salt_train, X_salt_val, y_salt_val,
...              n_layers_to_keep_trainable=1)
>>> preds = tl.predict(X_salt_test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.neural_network import MLPRegressor, train_mlp, predict_mlp, DEVICE

logger = logging.getLogger(__name__)


class TransferLearner:
    """Wrapper that handles loading a pre-trained model and fine-tuning it.

    Parameters
    ----------
    base_model_path : path-like
        Path to ``*.pt`` file saved by :func:`~src.models.neural_network.train_mlp`.
    input_dim : int
        Number of input features (must match the pre-trained model).
    hidden_dims : list of int
        Architecture of the pre-trained model (must match).
    dropout : float
        Dropout used in the pre-trained model (must match).
    """

    def __init__(
        self,
        base_model_path: str | Path,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.2,
    ):
        self.base_model_path = Path(base_model_path)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.model: Optional[MLPRegressor] = None
        self.history: Optional[dict] = None

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        n_layers_to_keep_trainable: int = 1,
        lr: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 256,
        checkpoint_path: Optional[str | Path] = None,
    ) -> dict:
        """Load ACC Arena weights and fine-tune on Salt&Tar data.

        Parameters
        ----------
        n_layers_to_keep_trainable : int
            How many hidden blocks (from the top) + the output layer to keep
            trainable. Passed to :meth:`~MLPRegressor.freeze_feature_extractor`.
        lr : float
            A smaller learning rate than for the original training is typical.
        Returns
        -------
        dict
            Training history from :func:`~src.models.neural_network.train_mlp`.
        """
        self.model = self._load_base_model()
        self.model.freeze_feature_extractor(n_layers_to_keep_trainable)

        logger.info(
            "Fine-tuning: keeping %d top block(s) trainable on %d samples.",
            n_layers_to_keep_trainable, len(y_train),
        )

        self.history = train_mlp(
            self.model,
            X_train, y_train, X_val, y_val,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )
        return self.history

    # ------------------------------------------------------------------
    # Scratch baseline
    # ------------------------------------------------------------------

    @staticmethod
    def train_from_scratch(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        checkpoint_path: Optional[str | Path] = None,
    ) -> tuple[MLPRegressor, dict]:
        """Train the same MLP architecture from random init on limited data.

        Returns
        -------
        (model, history)
        """
        model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims,
                             dropout=dropout)
        history = train_mlp(
            model,
            X_train, y_train, X_val, y_val,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )
        return model, history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the fine-tuned model."""
        if self.model is None:
            raise RuntimeError("Call fine_tune() before predict().")
        return predict_mlp(self.model, X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_base_model(self) -> MLPRegressor:
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Base model checkpoint not found: {self.base_model_path}\n"
                "Train the ACC Arena model first (notebook 04_model_training.ipynb)."
            )
        model = MLPRegressor(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        state_dict = torch.load(self.base_model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("Loaded base model from %s", self.base_model_path)
        return model
