"""Prediction stage: load the saved pipeline and score new records."""
import os
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, project_path

_cfg = load_config()
FEATURES = _cfg["data"]["numeric_features"] + _cfg["data"]["categorical_features"]


def model_path() -> Path:
    """MODEL_PATH env var wins (used in Docker); otherwise the config path."""
    env = os.getenv("MODEL_PATH")
    return Path(env) if env else project_path(_cfg["paths"]["model"])


def load_model(path=None):
    path = Path(path) if path else model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict(model, records) -> list:
    """records: list of dicts (or a DataFrame) with the feature columns.

    Column names/order are enforced here so callers cannot reorder features.
    """
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return [float(p) for p in model.predict(df[FEATURES])]


if __name__ == "__main__":
    # Used for "load the saved model in a separate Python process" (Lab A4).
    m = load_model()
    samples = [
        {"area_sqft": 1200, "bedrooms": 2, "bathrooms": 2, "age_years": 10, "location_score": 6.5, "neighborhood": "suburban"},
        {"area_sqft": 2500, "bedrooms": 4, "bathrooms": 3, "age_years": 3, "location_score": 8.5, "neighborhood": "urban"},
        {"area_sqft": 900, "bedrooms": 1, "bathrooms": 1, "age_years": 40, "location_score": 3.0, "neighborhood": "rural"},
        {"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2, "age_years": 15, "location_score": 5.0, "neighborhood": "suburban"},
        {"area_sqft": 3200, "bedrooms": 5, "bathrooms": 4, "age_years": 1, "location_score": 9.0, "neighborhood": "urban"},
    ]
    for rec, p in zip(samples, predict(m, samples)):
        print(f"{rec} -> predicted price: {p:,.2f}")
