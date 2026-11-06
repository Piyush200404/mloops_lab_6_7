"""Data stage: load, validate, clean and split.

Only data handling lives here. Feature transformations that must also run at
inference time (imputing, scaling, one-hot encoding) are NOT done here; they
live inside the sklearn Pipeline built in train.py so training and inference
use exactly the same preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import project_path


def load_data(path) -> pd.DataFrame:
    """Load the raw CSV and check that the expected columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python -m src.generate_data` first."
        )
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame, cfg: dict) -> None:
    """Raise ValueError if required columns are missing."""
    required = (
        cfg["data"]["numeric_features"]
        + cfg["data"]["categorical_features"]
        + [cfg["data"]["target"]]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_neighborhood(series: pd.Series) -> pd.Series:
    """Normalise text categories: strip whitespace, lower-case."""
    return series.astype("object").str.strip().str.lower()


def preprocess_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Row-level cleaning: duplicates, formats, invalid values.

    Missing *feature* values are kept (NaN) and imputed later inside the model
    pipeline, so the imputer is fitted on training data only (no leakage).
    Rows with a missing/invalid target are dropped.
    """
    check_columns(df, cfg)
    d = cfg["data"]
    df = df.drop_duplicates().copy()

    for col in d["categorical_features"]:
        df[col] = clean_neighborhood(df[col])
    allowed = d["allowed_neighborhoods"]
    df.loc[~df["neighborhood"].isin(allowed), "neighborhood"] = np.nan

    df = df.dropna(subset=[d["target"]])

    # Turn out-of-range values into NaN (they get imputed later); drop bad targets.
    for col, (lo, hi) in d["valid_ranges"].items():
        if col == d["target"]:
            df = df[df[col].between(lo, hi)]
        else:
            df.loc[~df[col].between(lo, hi) & df[col].notna(), col] = np.nan

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, cfg: dict):
    """Split into train/test. The test set is never used to fit anything."""
    d = cfg["data"]
    features = d["numeric_features"] + d["categorical_features"]
    X = df[features]
    y = df[d["target"]]
    return train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
    )


def save_processed(df: pd.DataFrame, cfg: dict) -> Path:
    out = project_path(cfg["paths"]["processed_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
