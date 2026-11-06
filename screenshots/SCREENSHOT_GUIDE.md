# Screenshot Guide - what to run, what to capture, what to name it

Save every image in THIS folder using the exact filename shown (PNG).
Tip: make the terminal font large, and keep the command AND its output in the same shot.
Windows: Win+Shift+S  |  macOS: Cmd+Shift+4  |  Linux: PrintScreen

Run all commands from the project root (the folder that contains `src/`).
Activate your venv first:
  Windows:   .venv\Scripts\activate
  macOS/Linux: source .venv/bin/activate

## PART A - Pipeline (Session 7)

| # | Filename | Command to run | What must be visible |
|---|----------|----------------|----------------------|
| 01 | 01_folder_structure.png | VS Code Explorer expanded (or `tree -L 2 -I ".venv|__pycache__"`) | src/, tests/, configs/, models/, Dockerfile, etc. |
| 02 | 02_generate_data.png | `python -m src.generate_data` | "Wrote 1510 rows ..." |
| 03 | 03_train_cli.png | `python -m src.train` | log lines: loading, cleaning, train/test sizes, saved pipeline, held-out metrics |
| 04 | 04_evaluation_report.png | open `reports/evaluation_report.md` (or `cat reports/metrics.json`) | MAE, RMSE, R2 |
| 05 | 05_predict_separate_process.png | `python -m src.predict` | 5 predictions from a fresh Python process |
| 06 | 06_pytest_pass.png | `python -m pytest -v` | all tests PASSED (green) |
| 07 | 07_import_no_training.png | `python -c "import src.train; print('import ok, no training started')"` | message printed, no training logs |
| 08 | 08_mlflow_ui.png (optional) | `mlflow ui` then open http://127.0.0.1:5000 | experiment "house-price-service" with a run + metrics |

## PART B - FastAPI (Session 8)

Terminal 1: `uvicorn src.api:app --reload --host 127.0.0.1 --port 8000`

| # | Filename | Action | What must be visible |
|---|----------|--------|----------------------|
| 09 | 09_uvicorn_running.png | Terminal 1 | "Uvicorn running on http://127.0.0.1:8000" and "Model loaded" |
| 10 | 10_swagger_home.png | browser -> http://127.0.0.1:8000/docs | list of endpoints: /health, /predict, /mlflow-check |
| 11 | 11_swagger_health.png | /docs -> GET /health -> Try it out -> Execute | 200 + `{"status":"ok","model_loaded":true}` |
| 12 | 12_swagger_predict_valid.png | POST /predict -> Try it out -> Execute (keep the example body) | request body + 200 response with prediction |
| 13 | 13_curl_valid.png | Terminal 2: `bash scripts/sample_requests.sh` (or run each curl by hand) | valid request + prediction |
| 14 | 14_curl_missing_field.png | same script / curl without `area_sqft` | 422 "Field required" |
| 15 | 15_curl_wrong_type.png | same script / `"area_sqft":"big"` | 422 parsing error |
| 16 | 16_curl_invalid_business.png | `"area_sqft": -100` | 422 "greater than 0" |
| 17 | 17_postman_valid.png | Postman: POST http://127.0.0.1:8000/predict, Body -> raw -> JSON | request + response + status 200 |
| 18 | 18_postman_invalid.png | Postman with a missing field | status 422 |
| 19 | 19_api_tests_pass.png | `python -m pytest tests/test_api.py -v` | API tests passed |

Windows PowerShell hint for curl: use `curl.exe` and put JSON in single quotes with escaped double quotes,
or simply use the Swagger UI / Postman / `scripts/sample_requests.sh` from Git Bash.

## PART C - Docker (Session 9)

| # | Filename | Command | What must be visible |
|---|----------|---------|----------------------|
| 20 | 20_docker_version.png | `docker --version` and `docker info` (top part) | Docker installed and engine running |
| 21 | 21_docker_build.png | `docker build -t ml-inference-api:1.0 .` | final lines "Successfully built"/"naming to ... ml-inference-api:1.0" |
| 22 | 22_docker_image_ls.png | `docker image ls` | ml-inference-api 1.0 with size |
| 23 | 23_docker_run.png | `docker run --name ml-api -p 8000:8000 ml-inference-api:1.0` | uvicorn startup + "Model loaded" |
| 24 | 24_docker_ps.png | (new terminal) `docker ps` | container ml-api, port 0.0.0.0:8000->8000/tcp |
| 25 | 25_container_health.png | `curl http://localhost:8000/health` | `{"status":"ok","model_loaded":true}` |
| 26 | 26_container_swagger.png | browser -> http://localhost:8000/docs | Swagger UI served by the container |
| 27 | 27_container_predict.png | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"area_sqft":1800,"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'` | prediction from the host |
| 28 | 28_docker_logs.png | `docker logs ml-api` | startup + "prediction served" lines |
| 29 | 29_docker_inspect.png | `docker image inspect ml-inference-api:1.0 --format "{{.Size}} {{.Config.ExposedPorts}}"` and `docker history ml-inference-api:1.0` | size + layers |
| 30 | 30_recreate_container.png | `docker stop ml-api` -> `docker rm ml-api` -> run again -> `curl .../health` | still works after recreation |
| 31 | 31_env_var_override.png | `docker run --rm -p 8001:8000 -e MODEL_PATH=/app/models/model.joblib ml-inference-api:1.0` then `curl localhost:8001/health` | different host port + env var working |

## Docker Compose (if you implement it)

| # | Filename | Command | What must be visible |
|---|----------|---------|----------------------|
| 32 | 32_compose_config.png | `docker compose config` | valid merged config, no errors |
| 33 | 33_compose_up.png | `docker compose up --build` | both services starting |
| 34 | 34_compose_ps.png | `docker compose ps` | api + mlflow both "running" |
| 35 | 35_mlflow_ui_compose.png | browser -> http://localhost:5000 | MLflow UI from the container |
| 36 | 36_api_to_mlflow.png | `curl http://localhost:8000/mlflow-check` | `"reachable": true` |
| 37 | 37_mlflow_run_via_compose.png | Terminal on host: `MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.train` (PowerShell: `$env:MLFLOW_TRACKING_URI="http://localhost:5000"; python -m src.train`) then refresh the MLflow UI | new run visible in the UI |
| 38 | 38_compose_persistence.png | `docker compose down` -> `docker compose up -d` -> refresh MLflow UI | the earlier run is STILL there (volume works) |

## PART D - Final integration

| # | Filename | Command | What must be visible |
|---|----------|---------|----------------------|
| 39 | 39_local_vs_container.png | local API stopped, container running: `python scripts/compare_predictions.py --url http://localhost:8000` | table with local vs container values, "All consistent: True" |
| 40 | 40_github_repo.png | your GitHub repository page | file tree + README |
| 41 | 41_dvc_status.png | `dvc status` (or `dvc push` output) | dataset tracked by DVC |

## Clean-up evidence (optional but nice)
| 42 | 42_cleanup.png | `docker compose down` / `docker stop ml-api && docker rm ml-api` / `docker ps -a` | everything removed |

## Before you submit - checklist
- No passwords, tokens or personal info visible in any screenshot.
- Each screenshot shows both the command and its output.
- Filenames match the table (the report refers to them).
