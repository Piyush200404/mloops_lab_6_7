"""Shared fixtures. Trains a tiny model once if models/model.joblib is missing."""
import pytest

from src.config import load_config, project_path


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    cfg = load_config()
    if not project_path(cfg["paths"]["model"]).exists():
        from src import generate_data, train

        if not project_path(cfg["paths"]["raw_data"]).exists():
            generate_data.main()
        train.main()
