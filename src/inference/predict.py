"""
Real-time inference: load the current champion model + latest feature row
for a city, and produce a 24h/48h/72h AQI forecast.

Used by both the Streamlit dashboard and the Flask API.
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features.feature_store import get_feature_store
from src.training.train_model import REGISTRY_PATH, HORIZONS_HOURS
from src.utils.aqi_calc import aqi_category
from src.utils.logging_utils import get_logger
from src.utils.paths import MODELS_DIR

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


def load_champion():
    if not REGISTRY_PATH.exists():
        raise ModelNotTrainedError("No model registry found. Run the training pipeline first.")
    registry = json.loads(REGISTRY_PATH.read_text())
    champion = registry.get("champion")
    if champion is None:
        raise ModelNotTrainedError("Registry exists but no champion model is recorded yet.")

    model_type = champion["model_type"]
    artifact = champion["artifact"]

    if model_type == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(MODELS_DIR / artifact)
        scaler_path = MODELS_DIR / artifact.replace(".keras", "_scaler.joblib")
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        bundle = {"model": model, "scaler": scaler, "type": "tensorflow"}
    else:
        bundle = joblib.load(MODELS_DIR / artifact)

    return bundle, champion


def _predict_raw(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    scaler = bundle.get("scaler")
    X_in = scaler.transform(X) if scaler is not None else X
    return np.asarray(bundle["model"].predict(X_in))


def forecast_city(city_name: str) -> dict:
    """
    Produce a 3-day AQI forecast for one city using the latest available
    feature row as the model input (autoregressive: current conditions ->
    24h/48h/72h ahead, matching how the model was trained).
    """
    city = config.get_city(city_name)  # validates the city name
    store = get_feature_store()
    latest = store.get_latest(city.name)
    if latest is None:
        raise ValueError(
            f"No feature data available yet for {city.name}. "
            "Run the feature pipeline (or backfill) first."
        )

    bundle, champion = load_champion()
    feature_cols = champion["feature_columns"]
    missing = [c for c in feature_cols if c not in latest.index]
    if missing:
        raise ValueError(f"Latest feature row for {city.name} is missing columns: {missing}")

    X = pd.DataFrame([latest[feature_cols]])
    preds = _predict_raw(bundle, X)[0]
    preds = np.clip(preds, 0, 500)

    base_time = pd.to_datetime(latest["timestamp"], utc=True)
    forecast = []
    for hours, value in zip(HORIZONS_HOURS, preds):
        aqi_value = int(round(value))
        forecast.append(
            {
                "target_time": (base_time + timedelta(hours=hours)).isoformat(),
                "horizon_hours": hours,
                "predicted_aqi": aqi_value,
                "category": aqi_category(aqi_value),
                "is_hazardous": aqi_value >= config.HAZARDOUS_AQI_THRESHOLD,
            }
        )

    return {
        "city": city.name,
        "model_type": champion["model_type"],
        "model_run_id": champion["run_id"],
        "as_of": base_time.isoformat(),
        "current_aqi": int(latest["aqi"]) if not pd.isna(latest["aqi"]) else None,
        "current_category": aqi_category(latest["aqi"]) if not pd.isna(latest["aqi"]) else "Unknown",
        "forecast": forecast,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def forecast_all_cities() -> list[dict]:
    results = []
    for city in config.CITIES:
        try:
            results.append(forecast_city(city.name))
        except Exception as exc:
            logger.warning("Could not forecast for %s: %s", city.name, exc)
            results.append({"city": city.name, "error": str(exc)})
    return results
