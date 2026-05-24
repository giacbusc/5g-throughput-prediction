"""
src/data/loader.py
==================
Utilities to load the 5G HDD Liverpool dataset from raw CSV files.

Expected raw file structure (place files in data/raw/):
    acc_arena.csv   — ACC Arena venue (12k users, sampled every 3 s)
    salt_tar.csv    — Salt & Tar venue (3k users, sampled every 1 s)

Expected columns (adapt COLUMN_MAP if the actual headers differ):
    timestamp, user_id, x, y, z,
    traffic_type, ru_id,
    sinr_dl, sinr_ul,
    throughput, prb, bler
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalisation map  (raw header → standard name)
# Adjust if the CSV headers differ from the standard names below.
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, str] = {
    # Possible raw names → standard name
    "Timestamp": "timestamp",
    "Time": "timestamp",
    "UserID": "user_id",
    "User_ID": "user_id",
    "user id": "user_id",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "TrafficType": "traffic_type",
    "Traffic_Type": "traffic_type",
    "RU": "ru_id",
    "RU_ID": "ru_id",
    "SINR_DL": "sinr_dl",
    "DL_SINR": "sinr_dl",
    "SINR_UL": "sinr_ul",
    "UL_SINR": "sinr_ul",
    "Throughput": "throughput",
    "PRB": "prb",
    "BLER": "bler",
}

# The 12 standard columns we expect after normalisation
REQUIRED_COLUMNS = [
    "timestamp", "user_id",
    "x", "y", "z",
    "traffic_type", "ru_id",
    "sinr_dl", "sinr_ul",
    "throughput", "prb", "bler",
]

TRAFFIC_LABELS = {
    0: "off",
    1: "idle",
    2: "constant_rate",
    3: "video",
    4: "gaming",
    5: "http",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_venue(
    venue: str,
    data_dir: str | Path = "data/raw",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a single venue CSV and return a normalised DataFrame.

    Parameters
    ----------
    venue : str
        Either ``'acc_arena'`` or ``'salt_tar'``.
    data_dir : path-like
        Directory that contains the raw CSV files.
    nrows : int, optional
        If given, only the first *nrows* rows are loaded (useful for quick
        prototyping).

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with columns in :data:`REQUIRED_COLUMNS`.
    """
    data_dir = Path(data_dir)
    fname = data_dir / f"{venue}.csv"

    if not fname.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {fname}\n"
            "Please place the raw CSV in data/raw/ and rename it to "
            f"'{venue}.csv'."
        )

    logger.info("Loading %s from %s …", venue, fname)
    df = pd.read_csv(fname, nrows=nrows)

    # --- Normalise column names -------------------------------------------
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"After column normalisation, the following required columns are "
            f"still missing: {missing}.\n"
            f"Current columns: {list(df.columns)}\n"
            "Update COLUMN_MAP in src/data/loader.py to match your CSV headers."
        )

    df = df[REQUIRED_COLUMNS].copy()

    # --- Basic type coercion ----------------------------------------------
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["user_id"] = df["user_id"].astype(int)
    df["traffic_type"] = df["traffic_type"].astype(int)
    df["ru_id"] = df["ru_id"].astype(int)
    for col in ["x", "y", "z", "sinr_dl", "sinr_ul", "throughput", "prb", "bler"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Add venue label --------------------------------------------------
    df["venue"] = venue

    logger.info(
        "Loaded %s: %d rows, %d unique users, %d unique RUs",
        venue,
        len(df),
        df["user_id"].nunique(),
        df["ru_id"].nunique(),
    )
    return df


def load_all(
    data_dir: str | Path = "data/raw",
    nrows: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Load both venues and return a dictionary ``{venue_name: DataFrame}``.

    Parameters
    ----------
    data_dir : path-like
        Directory containing ``acc_arena.csv`` and ``salt_tar.csv``.
    nrows : int, optional
        Forward-passed to :func:`load_venue`.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    return {
        venue: load_venue(venue, data_dir=data_dir, nrows=nrows)
        for venue in ("acc_arena", "salt_tar")
    }


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a quick summary DataFrame (count, dtype, nulls, basic stats)."""
    info = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notnull().sum(),
        "null": df.isnull().sum(),
        "null_%": (df.isnull().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    numeric_stats = df.describe().T[["mean", "std", "min", "50%", "max"]]
    numeric_stats.columns = ["mean", "std", "min", "median", "max"]
    return info.join(numeric_stats, how="left")
