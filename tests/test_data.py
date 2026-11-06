import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import check_columns, preprocess_data, split_data
from src.generate_data import generate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_missing_column_raises(cfg):
    df = generate(50).drop(columns=["bedrooms"])
    with pytest.raises(ValueError):
        check_columns(df, cfg)


def test_cleaning_removes_duplicates_and_invalid(cfg):
    raw = generate(300)
    clean = preprocess_data(raw, cfg)
    assert clean.duplicated().sum() == 0
    assert clean["age_years"].dropna().ge(0).all()           # -5 became NaN
    assert set(clean["neighborhood"].dropna()) <= {"urban", "suburban", "rural"}
    assert clean["price"].notna().all()


def test_split_has_no_overlap_and_right_size(cfg):
    clean = preprocess_data(generate(400), cfg)
    X_tr, X_te, y_tr, y_te = split_data(clean, cfg)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_te) == pytest.approx(len(clean) * cfg["training"]["test_size"], abs=1)
