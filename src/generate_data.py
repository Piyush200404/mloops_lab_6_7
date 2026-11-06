"""Create a SYNTHETIC house-price dataset (no personal or confidential data).

Run:  python -m src.generate_data
The file intentionally contains a few missing values, duplicates and invalid
rows so the cleaning stage has something real to do.
"""
import numpy as np
import pandas as pd

from src.config import load_config, project_path


def generate(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(1800, 600, n_rows).clip(300, 6000).round(0)
    bedrooms = rng.integers(1, 7, n_rows)
    bathrooms = np.clip(bedrooms - rng.integers(0, 3, n_rows), 1, 6)
    age = rng.integers(0, 60, n_rows)
    location = rng.uniform(1, 10, n_rows).round(2)
    neighborhood = rng.choice(["urban", "suburban", "rural"], n_rows, p=[0.4, 0.4, 0.2])
    mult = pd.Series(neighborhood).map({"urban": 1.35, "suburban": 1.0, "rural": 0.7}).to_numpy()

    price = (
        50_000
        + area * 110 * mult
        + bedrooms * 8_000
        + bathrooms * 12_000
        - age * 900
        + location * 15_000
        + rng.normal(0, 20_000, n_rows)
    ).round(0)

    df = pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "location_score": location,
            "neighborhood": neighborhood,
            "price": price,
        }
    )

    # --- inject a little "dirt" for the cleaning stage ---
    idx = rng.choice(n_rows, 25, replace=False)
    df.loc[idx[:10], "area_sqft"] = np.nan          # missing numeric
    df.loc[idx[10:18], "neighborhood"] = np.nan     # missing categorical
    df.loc[idx[18:21], "age_years"] = -5            # invalid value
    df.loc[idx[21:23], "neighborhood"] = "  URBAN "  # inconsistent format
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)  # duplicates
    return df


def main() -> None:
    cfg = load_config()
    out = project_path(cfg["paths"]["raw_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(cfg["data"]["synthetic_rows"], cfg["training"]["random_state"])
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
