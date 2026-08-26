"""Evaluation metrics shared by all model types."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HORIZON_LABELS = ["24h", "48h", "72h"]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R2 both overall and per forecast horizon.
    y_true / y_pred shape: (n_samples, n_horizons)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "overall": {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        },
        "per_horizon": {},
    }

    n_horizons = y_true.shape[1] if y_true.ndim > 1 else 1
    labels = HORIZON_LABELS[:n_horizons] if n_horizons <= len(HORIZON_LABELS) else [f"h{i}" for i in range(n_horizons)]

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    for i, label in enumerate(labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics["per_horizon"][label] = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if len(set(yt)) > 1 else float("nan"),
        }
    return metrics


def is_better(candidate_metrics: dict, current_best_metrics: dict | None) -> bool:
    """Lower overall RMSE wins."""
    if current_best_metrics is None:
        return True
    return candidate_metrics["overall"]["rmse"] < current_best_metrics["overall"]["rmse"]
