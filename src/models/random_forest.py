"""
src/models/random_forest.py
============================
Random Forest regressor wrapper with hyperparameter tuning via Optuna.

Usage
-----
>>> from src.models.random_forest import RFRegressor
>>> rf = RFRegressor()
>>> rf.tune(X_train, y_train, n_trials=50)
>>> rf.fit(X_train, y_train)
>>> preds = rf.predict(X_test)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class RFRegressor:
    """Thin wrapper around :class:`sklearn.ensemble.RandomForestRegressor`.

    Adds:
    - Optuna-based hyperparameter search (:meth:`tune`)
    - Consistent save/load interface
    - Feature importance accessor

    Parameters
    ----------
    n_estimators : int
        Number of trees (default 200; overridden by :meth:`tune`).
    random_state : int
        Random seed.
    n_jobs : int
        Parallelism for both fitting and tuning (-1 = all cores).
    """

    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.params = {**self.DEFAULT_PARAMS, "random_state": random_state,
                       "n_jobs": n_jobs, **kwargs}
        self.model: Optional[RandomForestRegressor] = None
        self.training_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFRegressor":
        """Fit the Random Forest on training data."""
        logger.info("Fitting Random Forest with params: %s", self.params)
        self.model = RandomForestRegressor(**self.params)
        t0 = time.time()
        self.model.fit(X, y)
        self.training_time_s = time.time() - t0
        logger.info("Training done in %.1f s.", self.training_time_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return throughput predictions."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
    ) -> dict:
        """Search hyperparameters with Optuna (TPE sampler, *cv*-fold CV).

        After calling this method, :attr:`params` is updated with the best
        found configuration.

        Parameters
        ----------
        X, y : np.ndarray
            Training data.
        n_trials : int
            Number of Optuna trials.
        cv : int
            Number of cross-validation folds.
        scoring : str
            Scikit-learn scoring string.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        logger.info("Starting RF hyperparameter search (%d trials, %d-fold CV)…",
                    n_trials, cv)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_categorical(
                    "max_depth", [None, 5, 10, 20, 30, 50]
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
                ),
                "random_state": self.params["random_state"],
                "n_jobs": self.params["n_jobs"],
            }
            rf = RandomForestRegressor(**params)
            scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))  # negative MSE → maximise

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        best["random_state"] = self.params["random_state"]
        best["n_jobs"] = self.params["n_jobs"]
        self.params = best

        logger.info("Best RF params: %s (score=%.4f)", best, study.best_value)
        return best

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        joblib.dump(self.model, path)
        logger.info("Saved RF model → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RFRegressor":
        obj = cls()
        obj.model = joblib.load(path)
        return obj

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        return self.model.feature_importances_
