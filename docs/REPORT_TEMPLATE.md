# Report.pdf template (fill in, then export to PDF)

Export: paste into Word/Google Docs, insert screenshots from /screenshots, File -> Download as PDF.
Or: `pandoc docs/REPORT_TEMPLATE.md -o Report.pdf`

---
# MLOps Lab 7-9: ML Pipeline, FastAPI and Docker
Name: ______  Roll no: ______  GitHub: ______

## 1. Objectives
(3-4 lines: modularise the pipeline, serve it with FastAPI, containerise with Docker.)

## 2. Architecture
Data -> Cleaning -> Feature engineering (inside sklearn Pipeline) -> Training -> Evaluation -> models/model.joblib
-> FastAPI POST /predict -> Docker container (+ MLflow in Compose)

### 2.1 Pipeline stages (Part A1)
| Stage | Input | Output | Purpose | One possible failure |
|---|---|---|---|---|
| Data loading | CSV path | DataFrame | read data | file missing / wrong columns |
| Cleaning | raw DataFrame | clean DataFrame | remove duplicates, invalid values | dropping too many rows |
| Feature engineering | clean features | numeric matrix | impute, scale, one-hot | data leakage if fit on test data |
| Training | X_train, y_train | fitted pipeline | learn price function | overfitting |
| Evaluation | model, X_test, y_test | metrics + report | measure generalisation | wrong metric / leakage |
| Prediction | saved pipeline + records | prices | serve new data | feature mismatch |

## 3. Implementation
- Modules (data.py, train.py, evaluate.py, predict.py, api.py) and what each does.
- Config file and CLI.   - Why a full Pipeline is saved.   - MLflow.   - Dockerfile line-by-line.   - Compose.
![Folder structure](screenshots/01_folder_structure.png)
![Training](screenshots/03_train_cli.png)

## 4. Results
### 4.1 Model metrics (screenshots 04)
### 4.2 API tests (screenshots 09-19)
| Test | Expected | Observed |
|---|---|---|
| Valid input | prediction | 200 |
| Missing field | 422 | 422 |
| Wrong type | 422 | 422 |
| Invalid business value | 422 | 422 |
| Health | ok | ok |
### 4.3 Docker evidence (screenshots 20-31)
### 4.4 Compose evidence (screenshots 32-38)
### 4.5 Local vs container (screenshot 39) - tolerance 0.01

## 5. Concepts
- Online vs batch inference (see README section "Batch design").
- Why code works on one laptop but not another; what containers do NOT guarantee
  (same CPU architecture/kernel, data/secrets, correct app config, security, performance).

## 6. Challenges and resolutions
| Challenge | Resolution |
|---|---|
| (e.g. port 8000 busy) | mapped -p 8001:8000 |
| (e.g. model not found in container) | COPY models/ and MODEL_PATH env var |

## 7. Reflection
(What you learned, what you would improve: CI, model registry, monitoring, auth, batch endpoint.)

## 8. Reproduce (for a teammate)
See README "Quick start".
