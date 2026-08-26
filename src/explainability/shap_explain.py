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

    X = df[feature_cols].dropna()
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)

    scaler = bundle.get("scaler")
    X_model_input = scaler.transform(X) if scaler is not None else X.values

    model_type = bundle["type"]

    if model_type == "random_forest":
        # MultiOutputRegressor wraps one RF per horizon; explain horizon 0 (24h)
        estimator = bundle["model"].estimators_[0]
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_model_input)
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

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_cols, mean_abs_shap.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )

    return {
        "model_type": model_type,
        "horizon": "24h",
        "feature_importance": [{"feature": f, "mean_abs_shap": v} for f, v in importance],
    }
