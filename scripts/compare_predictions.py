"""Part D: compare LOCAL model vs CONTAINER API predictions.

Usage:
    python scripts/compare_predictions.py --url http://localhost:8000
Writes reports/local_vs_container.md (use it as evidence / screenshot).
"""
import argparse
import json
import urllib.request
from pathlib import Path

from src.predict import load_model, predict

TOL = 1e-6  # documented numerical tolerance (same library versions -> identical)

SAMPLES = [
    {"area_sqft": 1200, "bedrooms": 2, "bathrooms": 2, "age_years": 10, "location_score": 6.5, "neighborhood": "suburban"},
    {"area_sqft": 2500, "bedrooms": 4, "bathrooms": 3, "age_years": 3, "location_score": 8.5, "neighborhood": "urban"},
    {"area_sqft": 900, "bedrooms": 1, "bathrooms": 1, "age_years": 40, "location_score": 3.0, "neighborhood": "rural"},
    {"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2, "age_years": 15, "location_score": 5.0, "neighborhood": "suburban"},
    {"area_sqft": 3200, "bedrooms": 5, "bathrooms": 4, "age_years": 1, "location_score": 9.0, "neighborhood": "urban"},
]


def call_api(url, rec):
    req = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(rec).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["prediction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    url = ap.parse_args().url

    model = load_model()
    local = predict(model, SAMPLES)
    lines = ["| # | local | container | abs diff | OK |", "|---|---|---|---|---|"]
    ok_all = True
    for i, (rec, lp) in enumerate(zip(SAMPLES, local), 1):
        cp = call_api(url, rec)
        # API rounds to 2 decimals, so compare with 0.01 tolerance
        diff = abs(lp - cp)
        ok = diff <= 0.01 + TOL
        ok_all &= ok
        lines.append(f"| {i} | {lp:,.2f} | {cp:,.2f} | {diff:.4f} | {'yes' if ok else 'NO'} |")
    text = "\n".join(lines) + f"\n\nTolerance: 0.01 (API rounds to 2 decimals). All consistent: **{ok_all}**\n"
    print(text)
    out = Path("reports/local_vs_container.md")
    out.write_text("# Local vs Container Predictions\n\n" + text, encoding="utf-8")


if __name__ == "__main__":
    main()
