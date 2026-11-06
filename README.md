# MLOps Lab 7-9: House-Price ML Pipeline -> FastAPI -> Docker

A reproducible ML service. Synthetic house-price data -> cleaning -> sklearn Pipeline
(preprocessing + RandomForest) -> evaluation -> `models/model.joblib` -> FastAPI `POST /predict` -> Docker.
Only **synthetic** data is used (no personal/confidential data).

```
Data -> Cleaning -> Feature Eng. -> Training -> Evaluation -> Saved Pipeline
                                                                  |
                                                       FastAPI POST /predict
                                                                  |
                                                          Docker container
```

## Project structure
```
mlops/
├── configs/config.yaml        # paths, seed, test size, hyperparameters, MLflow
├── data/raw|processed/        # CSV data (DVC tracks raw)
├── models/model.joblib        # full inference pipeline (preprocessing + model)
├── reports/                   # metrics.json, evaluation_report.md, sample_predictions.csv
├── src/
│   ├── data.py       load_data, preprocess_data, split_data
│   ├── train.py      build_pipeline, train_model, main (+ optional MLflow)
│   ├── evaluate.py   evaluate_model
│   ├── predict.py    load_model, predict
│   ├── api.py        FastAPI app (/health, /predict, /mlflow-check)
│   ├── config.py     YAML loader
│   └── generate_data.py   creates the synthetic dataset
├── tests/            # 14 tests (data, predict, API)
├── scripts/          # sample_requests.sh, compare_predictions.py
├── screenshots/      # put your screenshots here (see SCREENSHOT_GUIDE.md)
├── docs/REPORT_TEMPLATE.md
├── Dockerfile  .dockerignore  docker-compose.yml
├── requirements.txt (runtime)   requirements-dev.txt (train/test/mlflow/dvc)
└── README.md  .gitignore
```

## Quick start (local)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.generate_data          # 1) create synthetic data
python -m src.train                  # 2) clean, train, evaluate, save pipeline
python -m src.evaluate               #    re-run evaluation on held-out data
python -m src.predict                # 3) load model in a separate process, predict 5 records
python -m pytest -v                  # 4) tests
```
Outputs: `models/model.joblib`, `reports/metrics.json`, `reports/evaluation_report.md`.

Configuration lives in `configs/config.yaml` (seed, test size, `n_estimators`, `max_depth`, model type).
Use another file with `python -m src.train --config path/to/other.yaml`.

### Design decisions
- **Same preprocessing in training and inference**: imputer, scaler and one-hot encoder are inside one
  sklearn `Pipeline`, saved as a single file.
- **No leakage**: split happens first; imputers/scalers are fitted on the training split only.
- **Importing a module never trains** (`if __name__ == "__main__"` guards).
- Unknown categories are ignored by the encoder; missing values are imputed.

## API
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Swagger UI: http://127.0.0.1:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness + whether the model is loaded |
| POST | /predict | predict one house price |
| GET | /mlflow-check | is `MLFLOW_TRACKING_URI` reachable (Compose lab) |

**Request**
```json
{"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2, "age_years": 12,
 "location_score": 7.5, "neighborhood": "suburban"}
```
**Response (200)**
```json
{"prediction": 395254.85, "model_file": "model.joblib"}
```
Constraints: `area_sqft` in (0, 10000], bedrooms/bathrooms 1-10, `age_years` 0-100,
`location_score` 0-10, `neighborhood` in urban|suburban|rural, no extra fields.

**Invalid input -> 422** (service keeps running; internal errors return a generic 500 and are only logged server-side)
```json
{"detail":[{"type":"missing","loc":["body","area_sqft"],"msg":"Field required", "...": "..."}]}
```
curl:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"area_sqft":1800,"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'
```
Run all valid/invalid examples: `bash scripts/sample_requests.sh`

| Test | Expected | Observed |
|---|---|---|
| Valid input | prediction returned | 200 |
| Missing field | validation error | 422 |
| Wrong type | validation error | 422 |
| Invalid business value (area < 0, unknown neighborhood) | validation error | 422 |
| Health check | healthy | 200 `ok` |

### Online vs batch inference (B6 design)
Today the API is **online inference** (one record, immediate answer, latency matters).
To score a CSV with many records I would:
- **small files (up to a few thousand rows)**: add a synchronous `POST /predict/batch` that accepts a JSON list
  or an uploaded CSV, validates every row with the same Pydantic schema, runs one vectorised `model.predict`
  on a DataFrame and returns predictions plus per-row errors;
- **large files**: return `202 Accepted` + job id, process in a background worker/queue (Celery/RQ or a
  scheduled job), store results in object storage and expose `GET /jobs/{id}`;
- **periodic scoring (nightly)**: a separate batch process (cron/Airflow) that loads the same
  `model.joblib`, so throughput and scheduling matter more than latency.

## Docker
```bash
python -m src.train                                    # make sure models/model.joblib exists
docker build -t ml-inference-api:1.0 .
docker image ls
docker run --name ml-api -p 8000:8000 ml-inference-api:1.0     # port busy? use -p 8001:8000
# new terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"area_sqft":1800,"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'
docker ps
docker logs ml-api
docker stop ml-api && docker rm ml-api                 # clean up
```
Use another model without rebuilding (volume + env var):
```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models:ro" -e MODEL_PATH=/models/model.joblib ml-inference-api:1.0
```
Notes: the runtime image uses only `requirements.txt` (no pytest/mlflow). `.dockerignore` keeps `.git`, `.venv`,
data, secrets and notebooks out. **Never bake credentials into the image.**
`requirements.txt` pins scikit-learn 1.5.2: the pickled pipeline must be loaded with the same scikit-learn version
that trained it (re-run `python -m src.train` if you change versions).

### Docker Compose (FastAPI + MLflow)
```bash
docker compose config
docker compose up --build -d
docker compose ps
curl http://localhost:8000/mlflow-check        # {"configured":true,...,"reachable":true}
# log a training run into the Compose MLflow server:
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train     # then open http://localhost:5000
docker compose logs
docker compose down            # keeps the volume (experiments survive)
docker compose down -v         # also deletes the volume
```
The API reaches MLflow at `http://mlflow:5000` (service name on the Compose network). MLflow metadata is in
SQLite and artifacts under `/mlflow/artifacts`, both on the `mlflow-data` volume.
**Status:** Compose file is provided as a tested-on-paper design scaffold; verify the MLflow image tag/flags with
the current MLflow docs, and keep the server version equal to the `mlflow` client in `requirements-dev.txt`.

## Final integration check (local vs container)
```bash
docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0
python scripts/compare_predictions.py --url http://localhost:8000     # writes reports/local_vs_container.md
```
Tolerance: 0.01 (the API rounds to two decimals); same library versions give identical predictions.

## DVC (dataset versioning)
```bash
dvc init
dvc add data/raw/house_prices.csv
git add data/raw/house_prices.csv.dvc data/raw/.gitignore .dvc
git commit -m "Track dataset with DVC"
# optional remote: dvc remote add -d storage <path-or-url> && dvc push
```

## Why Docker? (concept notes)
Code works on one laptop but fails on another because of different OS, Python/library versions, missing system
packages, paths and environment variables. A container packages the code, Python runtime and exact dependencies so
they behave the same everywhere. Containers do **not** guarantee identical hardware/CPU architecture or performance,
that the model/data are correct, that secrets and external services (databases, MLflow) are available, or that the
app is secure.

## Reproduce from scratch
`git clone <repo> && cd mlops` -> Quick start -> Docker section.

## Challenges and resolutions
Fill in from your own experience (port conflicts, model path in the container, scikit-learn version mismatch, ...).
