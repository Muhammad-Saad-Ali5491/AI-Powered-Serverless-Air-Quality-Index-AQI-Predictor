"""
SHAP-based feature importance / explainability for the champion AQI model.

Works with the tree-based (Random Forest) and linear (Ridge) models via
shap.Explainer's auto-detection; falls back to KernelExplainer for the
TensorFlow model (slower, so it's sampled).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap

from src.inference.predict import load_champion
from src.features.feature_store import get_feature_store
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_background(X: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    if len(X) <= n:
        return X
    return X.sample(n, random_state=42)


def _normalise_shap_values(shap_values) -> np.ndarray:
    """Return a 2-D (rows, features) array across SHAP API versions."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    return values


def explain_model(city: str | None = None, sample_size: int = 200) -> dict:
    """
    Returns mean absolute SHAP value per feature (global importance) for
    the first forecast horizon (24h), which is the primary target most
    users care about on the dashboard.
    """
    bundle, champion = load_champion()
    feature_cols = champion["feature_columns"]

    store = get_feature_store()
    df = store.read_features(city=city)
    if df.empty:
        raise ValueError("No feature data available to explain.")

    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    latest = X.iloc[[-1]]
    if len(X) > sample_size:
        X = pd.concat([X.iloc[:-1].sample(sample_size - 1, random_state=42), latest])

    scaler = bundle.get("scaler")
    X_model_input = scaler.transform(X) if scaler is not None else X.values

    model_type = bundle["type"]

    if model_type in {"random_forest", "extra_trees", "hist_gradient_boosting", "xgboost"}:
        # MultiOutputRegressor wraps one estimator per horizon; explain 24h.
        estimator = bundle["model"].estimators_[0]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_model_input, check_additivity=False)
    elif model_type == "ridge":
        estimator = bundle["model"].estimators_[0]
        explainer = shap.LinearExplainer(estimator, X_model_input)
        shap_values = explainer.shap_values(X_model_input)
    else:  # tensorflow — use a fast, sampled KernelExplainer
        model = bundle["model"]
        background = _get_background(pd.DataFrame(X_model_input, columns=feature_cols), n=30).values

        def predict_fn(data):
            return np.asarray(model.predict(data, verbose=0))[:, 0]

        explainer = shap.KernelExplainer(predict_fn, background)
        sample = pd.DataFrame(X_model_input, columns=feature_cols).sample(
            min(50, len(X_model_input)), random_state=42
        )
        shap_values = explainer.shap_values(sample.values, nsamples=100)

    shap_values = _normalise_shap_values(shap_values)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    latest_row = shap_values[-1]
    importance = sorted(
        zip(feature_cols, mean_abs_shap.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )
    contributions = sorted(
        zip(feature_cols, latest_row.tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )

    return {
        "model_type": model_type,
        "horizon": "24h",
        "feature_importance": [{"feature": f, "mean_abs_shap": v} for f, v in importance],
        "contributions": [{"feature": f, "shap_value": v} for f, v in contributions],
    }
