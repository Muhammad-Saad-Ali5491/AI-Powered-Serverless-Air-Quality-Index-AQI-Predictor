"""
Training pipeline for the Pearls AQI Predictor.

Builds a 3-day-ahead (24h / 48h / 72h) multi-output AQI forecast using
three candidate model families:
  * Ridge Regression        (fast linear baseline)
  * Random Forest Regressor (non-linear, handles feature interactions)
  * TensorFlow dense network (deep-learning candidate)

All three are trained on the same time-based train/test split and
evaluated with RMSE / MAE / R2 (src.training.evaluate). The best model
(lowest overall RMSE) is saved to models/ along with a registry.json that
tracks every training run's metrics, so the CI training pipeline can pick
up the current champion automatically.

Run:  python -m src.training.train_model
"""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src import config
from src.features.feature_store import get_feature_store
from src.training.evaluate import evaluate_predictions, is_better
from src.utils.logging_utils import get_logger
from src.utils.paths import MODELS_DIR

logger = get_logger(__name__)

REGISTRY_PATH = MODELS_DIR / "registry.json"
HORIZONS_HOURS = [24, 48, 72]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def build_training_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Create the (X, y) supervised-learning matrix: for every row, y is the
    AQI `h` hours later, for h in HORIZONS_HOURS, computed per-city on a
    time-sorted series. Rows without a full 72h future window are dropped.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    feature_cols = [c for c in config.ALL_FEATURES if c in df.columns]

    frames = []
    for city, g in df.groupby("city"):
        g = g.reset_index(drop=True)
        targets = {}
        for h in HORIZONS_HOURS:
            targets[f"target_{h}h"] = g["aqi"].shift(-h)
        target_df = pd.DataFrame(targets)
        combined = pd.concat([g, target_df], axis=1)
        frames.append(combined)

    full = pd.concat(frames, ignore_index=True)
    target_cols = [f"target_{h}h" for h in HORIZONS_HOURS]
    full = full.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    X = full[feature_cols]
    y = full[target_cols]
    return X, y, feature_cols


def time_based_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
    n = len(X)
    split_idx = int(n * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def train_ridge(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))
    model.fit(X_scaled, y_train)
    return {"model": model, "scaler": scaler, "type": "ridge"}


def train_random_forest(X_train, y_train):
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
    )
    model.fit(X_train, y_train)
    return {"model": model, "scaler": None, "type": "random_forest"}


def train_tensorflow(X_train, y_train, X_val=None, y_val=None, epochs: int = 30):
    import tensorflow as tf

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(HORIZONS_HOURS)),  # 3 outputs: 24h/48h/72h
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    val_data = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_data = (scaler.transform(X_val), y_val)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss" if val_data is None else "val_loss",
                                                    patience=5, restore_best_weights=True)]

    model.fit(
        X_scaled, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=callbacks,
    )
    return {"model": model, "scaler": scaler, "type": "tensorflow"}


def predict_with(bundle: dict, X):
    model, scaler = bundle["model"], bundle["scaler"]
    X_in = scaler.transform(X) if scaler is not None else X
    preds = model.predict(X_in)
    return np.asarray(preds)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"runs": [], "champion": None}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str))


def _save_model_artifact(bundle: dict, name: str) -> str:
    if bundle["type"] == "tensorflow":
        path = MODELS_DIR / f"{name}.keras"
        bundle["model"].save(path)
        if bundle["scaler"] is not None:
            joblib.dump(bundle["scaler"], MODELS_DIR / f"{name}_scaler.joblib")
    else:
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(bundle, path)
    return str(path.name)


def _prune_stale_artifacts(champion_artifact: str) -> None:
    """
    Keep the models/ directory (and therefore the git repo, since CI commits
    it) bounded: delete every model artifact file that isn't the current
    champion. Without this, a daily training run would leave a new ~50-100MB
    file behind forever, and the repo would grow unbounded over months of
    unattended GitHub Actions runs.

    The registry.json history of past runs' metrics is kept (it's tiny —
    a few hundred bytes per run) so you can still see training history even
    though old model binaries are removed.
    """
    keep_names = {champion_artifact}
    if champion_artifact.endswith(".keras"):
        keep_names.add(champion_artifact.replace(".keras", "_scaler.joblib"))

    for path in MODELS_DIR.glob("*"):
        if path.name in ("registry.json", ".gitkeep"):
            continue
        if path.suffix not in (".joblib", ".keras"):
            continue
        if path.name not in keep_names:
            path.unlink()
            logger.info("Pruned stale model artifact: %s", path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_training(df: pd.DataFrame | None = None, epochs: int = 30, prune_stale_artifacts: bool = True) -> dict:
    store = get_feature_store()
    if df is None:
        df = store.read_features()

    if df.empty or len(df) < 100:
        raise ValueError(
            f"Not enough feature rows to train on ({len(df)}). "
            "Run the feature pipeline / backfill first."
        )

    X, y, feature_cols = build_training_matrix(df)
    if len(X) < 50:
        raise ValueError(f"Only {len(X)} usable training rows after building targets — need more history.")

    X_train, X_test, y_train, y_test = time_based_split(X, y)
    logger.info("Training rows: %d | Test rows: %d | Features: %d", len(X_train), len(X_test), len(feature_cols))

    candidates = {
        "ridge": train_ridge(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "tensorflow": train_tensorflow(X_train, y_train, X_test, y_test, epochs=epochs),
    }

    results = {}
    best_name, best_metrics, best_bundle = None, None, None
    for name, bundle in candidates.items():
        preds = predict_with(bundle, X_test)
        metrics = evaluate_predictions(y_test.values, preds)
        results[name] = metrics
        logger.info("[%s] overall RMSE=%.3f MAE=%.3f R2=%.3f",
                    name, metrics["overall"]["rmse"], metrics["overall"]["mae"], metrics["overall"]["r2"])
        if is_better(metrics, best_metrics):
            best_name, best_metrics, best_bundle = name, metrics, bundle

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{best_name}_{timestamp}"
    artifact_name = _save_model_artifact(best_bundle, run_id)

    registry = _load_registry()
    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_type": best_name,
        "artifact": artifact_name,
        "feature_columns": feature_cols,
        "horizons_hours": HORIZONS_HOURS,
        "metrics": {name: m for name, m in results.items()},
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "python_platform": platform.platform(),
    }
    registry["runs"].append(run_record)

    current_champion = registry.get("champion")
    if current_champion is None or is_better(best_metrics, current_champion.get("metrics", {}).get(current_champion.get("model_type"))):
        registry["champion"] = run_record
        logger.info("New champion model: %s (RMSE=%.3f)", best_name, best_metrics["overall"]["rmse"])

    _save_registry(registry)

    if prune_stale_artifacts:
        _prune_stale_artifacts(registry["champion"]["artifact"])

    logger.info("Training complete. Best model: %s -> %s", best_name, artifact_name)
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQI forecasting models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--keep-all-artifacts",
        action="store_true",
        help="Keep every trained model file instead of pruning to just the champion (uses more disk/repo space).",
    )
    args = parser.parse_args()
    run_training(epochs=args.epochs, prune_stale_artifacts=not args.keep_all_artifacts)
