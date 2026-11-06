from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID = {
    "area_sqft": 1800, "bedrooms": 3, "bathrooms": 2,
    "age_years": 12, "location_score": 7.5, "neighborhood": "suburban",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float) and body["prediction"] > 0


def test_predict_missing_field_returns_422():
    bad = {k: v for k, v in VALID.items() if k != "area_sqft"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_type_returns_422():
    assert client.post("/predict", json={**VALID, "bedrooms": "three"}).status_code == 422


def test_predict_invalid_business_value_returns_422():
    assert client.post("/predict", json={**VALID, "area_sqft": -50}).status_code == 422
    assert client.post("/predict", json={**VALID, "neighborhood": "mars"}).status_code == 422


def test_predict_rejects_unknown_field():
    assert client.post("/predict", json={**VALID, "colour": "red"}).status_code == 422


def test_neighborhood_is_case_insensitive():
    r = client.post("/predict", json={**VALID, "neighborhood": "  URBAN "})
    assert r.status_code == 200
