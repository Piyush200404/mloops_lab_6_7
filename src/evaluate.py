"""Evaluation stage: metrics on held-out data + saved reports.

Can be used from train.py or on its own:
    python -m src.evaluate
(re-creates the same test split using the config seed).
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import load_config, project_path

log = logging.getLogger("evaluate")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, cfg: dict) -> dict:
    """Compute metrics, write reports/metrics.json and an .md report + sample predictions."""
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_test_rows": int(len(y_test)),
    }

    metrics_path = project_path(cfg["paths"]["metrics_report"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sample = X_test.head(10).copy()
    sample["actual_price"] = y_test.head(10).to_numpy()
    sample["predicted_price"] = preds[:10].round(2)
    sample.to_csv(project_path(cfg["paths"]["predictions_sample"]), index=False)

    report = f"""# Evaluation Report

| Metric | Value |
|---|---|
| MAE  | {metrics['mae']:,.2f} |
| RMSE | {metrics['rmse']:,.2f} |
| R²   | {metrics['r2']:.4f} |
| Held-out rows | {metrics['n_test_rows']} |

Model type: `{cfg['model']['type']}`  |  random_state: `{cfg['training']['random_state']}`
|  test_size: `{cfg['training']['test_size']}`

Metrics were computed on a held-out test split that was never used to fit the
preprocessing steps or the estimator.
"""
    project_path(cfg["paths"]["evaluation_report"]).write_text(report, encoding="utf-8")
    return metrics


def main() -> None:
    from src.data import load_data, preprocess_data, split_data

    cfg = load_config()
    model = joblib.load(project_path(cfg["paths"]["model"]))
    clean = preprocess_data(load_data(project_path(cfg["paths"]["raw_data"])), cfg)
    _, X_test, _, y_test = split_data(clean, cfg)
    print(json.dumps(evaluate_model(model, X_test, y_test, cfg), indent=2))


if __name__ == "__main__":
    main()
