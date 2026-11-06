"""Training stage: build the full pipeline, fit, evaluate, save.

Run:  python -m src.train [--config configs/config.yaml]
"""
import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config, project_path
from src.data import load_data, preprocess_data, save_processed, split_data
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def build_pipeline(cfg: dict) -> Pipeline:
    """Preprocessing + estimator in ONE object (saved together)."""
    d, m = cfg["data"], cfg["model"]

    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, d["numeric_features"]),
            ("cat", categorical, d["categorical_features"]),
        ]
    )

    if m["type"] == "random_forest":
        est = RandomForestRegressor(
            n_estimators=m["n_estimators"],
            max_depth=m["max_depth"],
            random_state=cfg["training"]["random_state"],
            n_jobs=-1,
        )
    elif m["type"] == "linear_regression":
        est = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {m['type']}")
    return Pipeline([("preprocess", pre), ("model", est)])


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict) -> Pipeline:
    """Fit the pipeline on TRAINING data only."""
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    return pipe


def _log_to_mlflow(cfg, metrics, model_path, report_path):
    """Optional MLflow tracking. Never breaks training if MLflow is missing."""
    ml = cfg.get("mlflow", {})
    if not ml.get("enabled"):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not installed; skipping experiment tracking.")
        return
    try:
        uri = ml.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(ml["experiment_name"])
        with mlflow.start_run():
            mlflow.log_params(
                {
                    **{f"model.{k}": v for k, v in cfg["model"].items()},
                    **{f"training.{k}": v for k, v in cfg["training"].items()},
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(report_path))
        log.info("Run logged to MLflow (%s)", uri or "local ./mlruns")
    except Exception as exc:  # tracking server down etc.
        log.warning("MLflow logging failed (%s). Training result is still saved.", exc)


def main(config_path=None) -> None:
    cfg = load_config(config_path)

    log.info("Loading data")
    raw = load_data(project_path(cfg["paths"]["raw_data"]))
    log.info("Cleaning data (%d raw rows)", len(raw))
    clean = preprocess_data(raw, cfg)
    save_processed(clean, cfg)
    log.info("%d rows after cleaning", len(clean))

    X_train, X_test, y_train, y_test = split_data(clean, cfg)
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    log.info("Training model (%s)", cfg["model"]["type"])
    pipe = train_model(X_train, y_train, cfg)

    model_path = project_path(cfg["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    log.info("Saved full inference pipeline -> %s", model_path)

    metrics = evaluate_model(pipe, X_test, y_test, cfg)
    log.info("Held-out metrics: %s", metrics)

    _log_to_mlflow(
        cfg, metrics, model_path, project_path(cfg["paths"]["evaluation_report"])
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the house-price model")
    ap.add_argument("--config", default=None, help="path to config YAML")
    main(ap.parse_args().config)
