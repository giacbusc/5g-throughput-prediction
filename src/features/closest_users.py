"""
src/features/closest_users.py
==============================
Team-8 specific feature engineering: for each user sample, find the X closest
users (by 3D Euclidean distance on position columns x, y, z) at the same
timestamp and append their features as additional input columns.

This implements the "features from X closest users" requirement from the
project assignment (Project #8).

Usage
-----
>>> from src.data.loader import load_venue
>>> from src.features.closest_users import add_closest_user_features
>>> df = load_venue("acc_arena")
>>> df_enriched = add_closest_user_features(df, X=5)
>>> # df_enriched contains extra columns: neighbor_1_sinr_dl, ..., neighbor_5_bler
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Features we copy from each neighbour
NEIGHBOR_FEATURES = ["sinr_dl", "sinr_ul", "prb", "bler", "traffic_type", "throughput"]


def add_closest_user_features(
    df: pd.DataFrame,
    X: int,
    position_cols: List[str] = ("x", "y", "z"),
    neighbor_features: List[str] = NEIGHBOR_FEATURES,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Augment *df* with features of the X spatially closest users.

    For each row (user at timestamp t), the function:
    1. Groups all rows by timestamp.
    2. Within each timestamp group, builds a KD-tree of (x, y, z) positions.
    3. Queries the K+1 nearest neighbours (the +1 excludes the row itself).
    4. Appends the neighbour features as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Venue DataFrame with columns ``x, y, z, user_id, timestamp`` and
        the columns listed in *neighbor_features*.
    X : int
        Number of closest neighbours to include.
    position_cols : tuple of str
        Column names for the 3D spatial coordinates.
    neighbor_features : list of str
        Feature columns to copy from each neighbour.
    fill_value : float
        Value used when fewer than *X* other users exist at a timestamp.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ``X * len(neighbor_features)`` extra columns.
        New column naming: ``neighbor_{rank}_{feature}`` (rank = 1 … X).
    """
    if X <= 0:
        raise ValueError(f"X must be a positive integer, got {X}.")

    pos_cols = list(position_cols)
    missing = set(pos_cols + neighbor_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    # Pre-allocate result arrays
    new_col_names = [
        f"neighbor_{rank}_{feat}"
        for rank in range(1, X + 1)
        for feat in neighbor_features
    ]
    result = np.full((len(df), len(new_col_names)), fill_value, dtype=np.float32)

    df = df.reset_index(drop=True)

    timestamps = df["timestamp"].unique()
    logger.info(
        "Building closest-user features (X=%d) over %d timestamps …",
        X, len(timestamps),
    )

    for ts in timestamps:
        mask = df["timestamp"] == ts
        idx = df.index[mask].tolist()
        if len(idx) < 2:
            continue  # only one user at this timestamp — keep fill_value

        positions = df.loc[idx, pos_cols].values.astype(float)
        feat_matrix = df.loc[idx, neighbor_features].values.astype(float)

        # Build KD-tree on positions within this timestamp
        tree = cKDTree(positions)

        # Query X+1 neighbours (first result is the point itself)
        k = min(X + 1, len(idx))
        _, nn_indices = tree.query(positions, k=k)

        for local_i, (global_i, neighbours) in enumerate(zip(idx, nn_indices)):
            # Exclude self (index 0 in nn_indices is always the point itself)
            neighbour_local_ids = [n for n in neighbours if n != local_i][:X]

            for rank, nb_local in enumerate(neighbour_local_ids):
                col_start = rank * len(neighbor_features)
                result[global_i, col_start: col_start + len(neighbor_features)] = (
                    feat_matrix[nb_local]
                )
            # If fewer than X neighbours exist, remaining columns keep fill_value

    # Attach new columns to a copy of df
    df_out = df.copy()
    for col_idx, col_name in enumerate(new_col_names):
        df_out[col_name] = result[:, col_idx]

    logger.info("Added %d neighbour columns to DataFrame.", len(new_col_names))
    return df_out


def get_neighbor_column_names(X: int, neighbor_features: List[str] = NEIGHBOR_FEATURES) -> List[str]:
    """Return the list of column names that :func:`add_closest_user_features` adds."""
    return [
        f"neighbor_{rank}_{feat}"
        for rank in range(1, X + 1)
        for feat in neighbor_features
    ]
