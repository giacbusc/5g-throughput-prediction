"""
src/data/preprocessor.py
========================
Data cleaning, feature scaling, one-hot encoding, and train/val/test splitting.

Typical usage
-------------
>>> from src.data.loader import load_venue
>>> from src.data.preprocessor import Preprocessor
>>> df = load_venue("acc_arena")
>>> prep = Preprocessor()
>>> X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

# Numeric features to standardise (zero-mean, unit-variance)
NUMERIC_FEATURES = ["sinr_dl", "sinr_ul", "prb", "bler", "x", "y", "z"]

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = ["traffic_type"]

# Target column
TARGET = "throughput"

# Columns that are identifiers / not used as model features
ID_COLUMNS = ["timestamp", "user_id", "ru_id", "venue"]


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor:
    """Full preprocessing pipeline for a single venue DataFrame.

    Steps
    -----
    1. Drop rows with NaN in any required column.
    2. Remove outliers (throughput ≤ 0, extreme SINR values).
    3. One-hot encode ``traffic_type``.
    4. StandardScale all numeric features.
    5. Split into train / val / test.

    The fitted ``ColumnTransformer`` (scaler + encoder) is stored in
    ``self.column_transformer`` and can be saved/loaded with joblib.

    Parameters
    ----------
    val_size : float
        Fraction of data for validation (default 0.10).
    test_size : float
        Fraction of data for test (default 0.15).
    random_state : int
        Random seed for reproducibility.
    remove_zero_throughput : bool
        If True, rows where throughput == 0 are dropped before splitting.
    """

    def __init__(
        self,
        val_size: float = 0.10,
        test_size: float = 0.15,
        random_state: int = 42,
        remove_zero_throughput: bool = True,
    ):
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.remove_zero_throughput = remove_zero_throughput

        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        extra_feature_cols: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """Fit the pipeline on *df* and return split arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Full venue DataFrame (output of :func:`~src.data.loader.load_venue`).
        extra_feature_cols : list of str, optional
            Additional columns to include as numeric features
            (e.g., closest-user feature columns added by
            :mod:`src.features.closest_users`).

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        """
        df = self._clean(df)
        X_raw, y = self._extract_xy(df, extra_feature_cols)
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = (
            self._split(X_raw, y)
        )

        self.column_transformer = self._build_transformer(
            X_train_raw, extra_feature_cols
        )
        X_train = self.column_transformer.fit_transform(X_train_raw)
        X_val   = self.column_transformer.transform(X_val_raw)
        X_test  = self.column_transformer.transform(X_test_raw)

        self.feature_names_out_ = self._get_feature_names(extra_feature_cols)

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d | features: %d",
            len(y_train), len(y_val), len(y_test), X_train.shape[1],
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame,
                  extra_feature_cols: Optional[list[str]] = None) -> np.ndarray:
        """Transform a new DataFrame using the already-fitted pipeline.

        Call :meth:`fit_transform` first.
        """
        if self.column_transformer is None:
            raise RuntimeError("Call fit_transform() before transform().")
        df = self._clean(df)
        X_raw, _ = self._extract_xy(df, extra_feature_cols)
        return self.column_transformer.transform(X_raw)

    def save(self, path: str | Path) -> None:
        """Persist the fitted transformer to *path* (joblib format)."""
        joblib.dump(self.column_transformer, path)
        logger.info("Saved preprocessor → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a previously saved preprocessor."""
        obj = cls()
        obj.column_transformer = joblib.load(path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n_start = len(df)
        df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])

        # Remove physically impossible values
        df = df[df["bler"].between(0.0, 1.0)]
        df = df[df["prb"] >= 0]
        df = df[df["throughput"] >= 0]

        if self.remove_zero_throughput:
            df = df[df["throughput"] > 0]

        n_dropped = n_start - len(df)
        if n_dropped:
            logger.info("Dropped %d rows during cleaning (%.1f%%)",
                        n_dropped, 100 * n_dropped / n_start)
        return df.reset_index(drop=True)

    def _extract_xy(
        self,
        df: pd.DataFrame,
        extra_feature_cols: Optional[list[str]],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if extra_feature_cols:
            cols = cols + [c for c in extra_feature_cols if c not in cols]
        X_raw = df[cols].copy()
        y = df[TARGET].values.astype(np.float32)
        return X_raw, y

    def _split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ):
        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Then split val from remaining
        val_ratio = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_transformer(
        self,
        X_sample: pd.DataFrame,
        extra_feature_cols: Optional[list[str]],
    ) -> ColumnTransformer:
        numeric_cols = NUMERIC_FEATURES.copy()
        if extra_feature_cols:
            numeric_cols += [c for c in extra_feature_cols
                             if c not in numeric_cols and
                             c not in CATEGORICAL_FEATURES]

        transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(
                    categories=[list(range(6))],  # traffic_type: 0-5
                    sparse_output=False,
                    handle_unknown="ignore",
                ), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return transformer

    def _get_feature_names(
        self, extra_feature_cols: Optional[list[str]]
    ) -> list[str]:
        numeric_cols = NUMERIC_FEATURES.copy()
        if extra_feature_cols:
            numeric_cols += [c for c in extra_feature_cols
                             if c not in numeric_cols and
                             c not in CATEGORICAL_FEATURES]
        cat_names = [f"traffic_type_{i}" for i in range(6)]
        return numeric_cols + cat_names
