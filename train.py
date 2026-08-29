"""Train, compare, evaluate, and serialize the laptop price model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import build_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "laptop_price.csv"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/TMaiza/ECD_proyecto_g15/460355698121dae98e596dc516cab111ca546bf2/laptop_price.csv"


def make_pipeline(estimator) -> Pipeline:
    categorical = ["company", "type", "cpu_family", "gpu_brand", "os"]
    numerical = [
        "inches", "ram_gb", "weight_kg", "screen_width", "screen_height",
        "touchscreen", "ips", "cpu_ghz", "ssd_gb", "hdd_gb", "flash_gb",
    ]
    preprocessor = ColumnTransformer([
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
        ("number", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numerical),
    ])
    regressor = TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def main() -> None:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL, encoding="utf-8-sig").to_csv(DATA, index=False)
    raw = pd.read_csv(DATA, encoding="utf-8-sig")
    raw = raw.drop_duplicates().dropna(subset=["Price_euros"])
    X = build_features(raw)
    y = pd.to_numeric(raw["Price_euros"], errors="coerce")
    valid = y.notna() & (y > 0)
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.85,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.9,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    leaderboard = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = np.maximum(pipeline.predict(X_test), 0)
        metrics = {
            "model": name,
            "mae_eur": round(float(mean_absolute_error(y_test, predictions)), 2),
            "rmse_eur": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }
        leaderboard.append(metrics)
        fitted[name] = (pipeline, predictions)

    leaderboard.sort(key=lambda item: item["mae_eur"])
    winner = leaderboard[0]["model"]
    best_pipeline, best_predictions = fitted[winner]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, ARTIFACTS / "laptop_price_pipeline.joblib")
    pd.DataFrame({"actual_eur": y_test, "predicted_eur": best_predictions}).to_csv(
        ARTIFACTS / "test_predictions.csv", index=False
    )
    metadata = {
        "winner": winner,
        "records": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "leaderboard": leaderboard,
        "price_p10_eur": round(float(y.quantile(0.10)), 2),
        "price_p90_eur": round(float(y.quantile(0.90)), 2),
        "companies": sorted(X["company"].unique().tolist()),
        "types": sorted(X["type"].unique().tolist()),
        "cpu_families": sorted(X["cpu_family"].unique().tolist()),
        "gpu_brands": sorted(X["gpu_brand"].unique().tolist()),
        "operating_systems": sorted(X["os"].unique().tolist()),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
