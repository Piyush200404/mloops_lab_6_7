import pytest

from src.predict import FEATURES, load_model, predict

SAMPLE = {
    "area_sqft": 1800, "bedrooms": 3, "bathrooms": 2,
    "age_years": 12, "location_score": 7.5, "neighborhood": "suburban",
}


def test_model_loads_and_predicts_one_value_per_row():
    model = load_model()
    out = predict(model, [SAMPLE, {**SAMPLE, "area_sqft": 2400}])
    assert len(out) == 2
    assert all(isinstance(x, float) and x > 0 for x in out)


def test_predict_rejects_missing_feature():
    model = load_model()
    bad = {k: v for k, v in SAMPLE.items() if k != "bedrooms"}
    with pytest.raises(ValueError):
        predict(model, [bad])


def test_larger_house_costs_more():
    model = load_model()
    small, big = predict(model, [{**SAMPLE, "area_sqft": 900}, {**SAMPLE, "area_sqft": 3500}])
    assert big > small


def test_pipeline_handles_missing_values_and_unknown_category():
    model = load_model()
    rec = {**SAMPLE, "area_sqft": float("nan"), "neighborhood": "coastal"}
    assert len(predict(model, [rec])) == 1  # imputer + handle_unknown="ignore"
