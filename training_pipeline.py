"""
Training pipeline - multi-city, multi-horizon version.

Steps:
  1. Fetch all accumulated rows (all cities) from Hopsworks
  2. Engineer features (city-aware, gap-aware - see feature_engineering.py)
  3. Build 3 forecast targets per city: PM2.5 at t+24h, t+48h, t+72h
  4. Time-respecting train/test split (no shuffling - the last N% of each
     city's timeline is held out, since shuffling a time series leaks
     future information into training)
  5. Train Ridge, Random Forest, and Gradient Boosting (XGBoost if available,
     else sklearn's GradientBoostingRegressor as a dependency-free fallback)
     for each horizon
  6. Evaluate with RMSE / MAE / R^2
  7. Log every run's metrics to reports/metrics_log.csv
  8. Register the best model per horizon in the Hopsworks Model Registry,
     but only PROMOTE it to production if it beats the current production
     model's logged metric - a bad training day never silently overwrites
     a working deployed model.

Target choice: PM2.5, not OpenWeather's 1-5 'aqi' field. PM2.5 is on a
continuous concentration scale (ug/m3) present in BOTH the live OpenWeather
feed and the historical Open-Meteo backfill, so it's the one variable with a
consistent, comparable scale across your entire combined dataset. See
backfill_openmeteo.py's docstring for why 'aqi' can't be mixed directly.
"""

import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import hopsworks
from feature_engineering import engineer_features
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

HORIZONS = {"24h": 24, "48h": 48, "72h": 72}
TEST_FRACTION = 0.15  # last 15% of each city's timeline held out for testing
METRICS_LOG_PATH = "reports/metrics_log.csv"


# ---------------------------------------------------------------------------
# Data loading + target construction
# ---------------------------------------------------------------------------

def fetch_all_data():
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    return df, project


def add_future_targets(df, horizons=HORIZONS, tolerance_hours=2):
    """
    For each city, attach PM2.5 value at t+N hours as a target column,
    using merge_asof (forward-looking, nearest actual future row within a
    tolerance window) instead of a naive row-count shift - handles the
    same irregular-spacing problem as the lag features do, just looking
    forward instead of backward.
    """
    pieces = []
    for city, city_df in df.groupby("city", sort=False):
        city_df = city_df.sort_values("timestamp").reset_index(drop=True)
        result = city_df.copy()

        for label, hours in horizons.items():
            lookup = city_df[["timestamp", "pm2_5"]].copy()
            lookup["timestamp"] = lookup["timestamp"] - pd.Timedelta(hours=hours)
            lookup = lookup.rename(columns={"pm2_5": f"target_pm2_5_{label}"})

            result = pd.merge_asof(
                result.sort_values("timestamp"),
                lookup.sort_values("timestamp"),
                on="timestamp",
                direction="forward",
                tolerance=pd.Timedelta(hours=tolerance_hours),
            )
        pieces.append(result)

    return pd.concat(pieces, ignore_index=True).sort_values(["city", "timestamp"]).reset_index(drop=True)


def time_based_split(df, test_fraction=TEST_FRACTION):
    """Split each city's data along the time axis (no shuffling)."""
    train_pieces, test_pieces = [], []
    for city, city_df in df.groupby("city", sort=False):
        city_df = city_df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(city_df) * (1 - test_fraction))
        train_pieces.append(city_df.iloc[:split_idx])
        test_pieces.append(city_df.iloc[split_idx:])
    train = pd.concat(train_pieces, ignore_index=True)
    test = pd.concat(test_pieces, ignore_index=True)
    return train, test


def get_feature_columns(df):
    """Everything except identifiers, raw timestamp, and target columns."""
    exclude = {"city", "timestamp"}
    exclude |= {c for c in df.columns if c.startswith("target_pm2_5_")}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Model training + evaluation
# ---------------------------------------------------------------------------

def get_candidate_models():
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1
        )
    else:
        models["gradient_boosting"] = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
    return models


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_all(train_df, test_df, feature_cols):
    """Returns nested dict: results[horizon][model_name] = {metrics, model}"""
    results = {}

    for label in HORIZONS:
        target_col = f"target_pm2_5_{label}"

        train_rows = train_df.dropna(subset=feature_cols + [target_col])
        test_rows = test_df.dropna(subset=feature_cols + [target_col])

        if len(train_rows) < 50 or len(test_rows) < 10:
            print(f"  [{label}] not enough non-null rows yet (train={len(train_rows)}, "
                  f"test={len(test_rows)}) - skipping this horizon for now")
            continue

        X_train, y_train = train_rows[feature_cols], train_rows[target_col]
        X_test, y_test = test_rows[feature_cols], test_rows[target_col]

        results[label] = {}
        for name, model in get_candidate_models().items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            results[label][name] = {"model": model, "metrics": metrics}
            print(f"  [{label}] {name}: RMSE={metrics['rmse']:.2f} "
                  f"MAE={metrics['mae']:.2f} R2={metrics['r2']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Metrics logging
# ---------------------------------------------------------------------------

def log_metrics(results, run_timestamp):
    os.makedirs("reports", exist_ok=True)
    rows = []
    for horizon, model_results in results.items():
        for model_name, r in model_results.items():
            rows.append({
                "run_timestamp": run_timestamp,
                "horizon": horizon,
                "model": model_name,
                **r["metrics"],
            })

    new_log = pd.DataFrame(rows)
    if os.path.exists(METRICS_LOG_PATH):
        existing = pd.read_csv(METRICS_LOG_PATH)
        combined = pd.concat([existing, new_log], ignore_index=True)
    else:
        combined = new_log
    combined.to_csv(METRICS_LOG_PATH, index=False)
    print(f"Logged {len(new_log)} metric rows to {METRICS_LOG_PATH}")


# ---------------------------------------------------------------------------
# Model promotion (only replace production model if genuinely better)
# ---------------------------------------------------------------------------

def get_current_production_rmse(project, horizon):
    """Reads the currently-registered production model's stored RMSE, if any."""
    try:
        mr = project.get_model_registry()
        model = mr.get_model(f"aqi_pm25_{horizon}", version=None)  # latest version
        return model.training_metrics.get("rmse")
    except Exception:
        return None  # no production model registered yet


def promote_best_model(project, results, feature_cols):
    import joblib

    mr = project.get_model_registry()

    for horizon, model_results in results.items():
        best_name = min(model_results, key=lambda n: model_results[n]["metrics"]["rmse"])
        best = model_results[best_name]
        new_rmse = best["metrics"]["rmse"]

        current_rmse = get_current_production_rmse(project, horizon)

        if current_rmse is not None and new_rmse >= current_rmse:
            print(f"  [{horizon}] new best ({best_name}, RMSE={new_rmse:.2f}) did NOT beat "
                  f"current production (RMSE={current_rmse:.2f}) - not promoting")
            continue

        model_dir = f"model_artifacts/aqi_pm25_{horizon}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(best["model"], os.path.join(model_dir, "model.pkl"))
        with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        hw_model = mr.python.create_model(
            name=f"aqi_pm25_{horizon}",
            metrics=best["metrics"],
            description=f"Best model for {horizon} PM2.5 forecast: {best_name}",
        )
        hw_model.save(model_dir)
        print(f"  [{horizon}] promoted {best_name} (RMSE={new_rmse:.2f}"
              + (f", beat previous {current_rmse:.2f}" if current_rmse is not None else ", first model registered")
              + ")")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching data from Hopsworks...")
    raw_df, project = fetch_all_data()
    print(f"  {len(raw_df)} raw rows across {raw_df['city'].nunique()} cities")

    print("Engineering features...")
    df = engineer_features(raw_df)

    print("Building forecast targets (24h / 48h / 72h ahead)...")
    df = add_future_targets(df)

    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} feature columns")

    print("Splitting train/test (time-based, per city)...")
    train_df, test_df = time_based_split(df)
    print(f"  train: {len(train_df)} rows, test: {len(test_df)} rows")

    print("Training and evaluating models...")
    results = train_and_evaluate_all(train_df, test_df, feature_cols)

    if not results:
        print("No horizon had enough data to train on yet. "
              "This is expected early on - accumulate more history and re-run.")
        return

    run_timestamp = datetime.now(timezone.utc).isoformat()
    log_metrics(results, run_timestamp)

    print("Checking promotion (only replaces production model if better)...")
    promote_best_model(project, results, feature_cols)

    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
