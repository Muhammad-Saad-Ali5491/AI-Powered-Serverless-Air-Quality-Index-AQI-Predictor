"""
Flask REST API for the Pearls AQI Predictor.

Endpoints:
  GET /health
  GET /cities
  GET /forecast/<city>
  GET /forecast            (all cities)
  GET /explain/<city>      (SHAP feature importance)
  GET /alerts               (cities currently forecast to be hazardous)

Run locally:  python app/flask_api.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify
from flask_cors import CORS

from src import config
from src.inference.predict import forecast_city, forecast_all_cities, ModelNotTrainedError
from src.explainability.shap_explain import explain_model
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/cities")
def cities():
    return jsonify({"cities": config.CITY_NAMES})


@app.get("/forecast/<city_name>")
def forecast_single(city_name: str):
    try:
        return jsonify(forecast_city(city_name))
    except ModelNotTrainedError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/forecast")
def forecast_all():
    return jsonify({"results": forecast_all_cities()})


@app.get("/explain/<city_name>")
def explain(city_name: str):
    try:
        return jsonify(explain_model(city=city_name))
    except (ModelNotTrainedError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/alerts")
def alerts():
    results = forecast_all_cities()
    hazardous = []
    for r in results:
        if "error" in r:
            continue
        for f in r.get("forecast", []):
            if f["is_hazardous"]:
                hazardous.append({"city": r["city"], **f})
    return jsonify({"hazardous_forecasts": hazardous, "threshold": config.HAZARDOUS_AQI_THRESHOLD})


if __name__ == "__main__":
    # host 0.0.0.0 so it also works inside containers; debug off by default for safety
    app.run(host="0.0.0.0", port=5000, debug=False)
