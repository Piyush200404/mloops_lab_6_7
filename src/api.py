"""FastAPI inference service.

Run locally:  uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
Docs:         http://127.0.0.1:8000/docs
"""

import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from typing import Literal
# import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.predict import FEATURES, load_model, model_path, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

_model = None  # cached pipeline


def get_model():
    """Load the pipeline once and reuse it (also loads lazily on first call)."""
    global _model
    if _model is None:
        _model = load_model()
        log.info("Model loaded from %s", model_path())
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except FileNotFoundError:
        log.error("Model file not found at startup: %s", model_path())
    yield


app = FastAPI(title="House Price Prediction API", version="1.0.0", lifespan=lifespan)


class PredictionInput(BaseModel):
    """Request schema - must match the columns the pipeline was trained on."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "area_sqft": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 12,
                "location_score": 7.5,
                "neighborhood": "suburban",
            }
        },
    )

    area_sqft: float = Field(..., gt=0, le=10000, description="Built-up area in square feet")
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=10)
    age_years: float = Field(..., ge=0, le=100, description="Age of the house in years")
    location_score: float = Field(..., ge=0, le=10, description="Location quality, 0-10")
    neighborhood: Literal["urban", "suburban", "rural"]

    @field_validator("neighborhood", mode="before")
    @classmethod
    def _normalise(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class PredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: float = Field(..., description="Predicted house price")
    model_file: str


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """Log details server-side; never leak internals to the client."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health():
    try:
        get_model()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model is not available")
    row = {name: getattr(payload, name) for name in FEATURES}  # exact names/order
    result = predict(model, [row])[0]
    log.info("prediction served: %.2f", result)
    return PredictionOutput(prediction=round(result, 2), model_file=model_path().name)


@app.get("/mlflow-check")
def mlflow_check():
    """Used in the Docker Compose lab to prove API -> MLflow connectivity."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return {"configured": False, "reachable": False}
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3) as r:
            return {"configured": True, "uri": uri, "reachable": r.status == 200}
    except Exception:
        return {"configured": True, "uri": uri, "reachable": False}
