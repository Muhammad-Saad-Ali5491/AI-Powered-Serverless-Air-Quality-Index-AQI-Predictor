"""Native XGBoost multi-output artifact support."""
from __future__ import annotations

import numpy as np


class NativeXGBoostMultiOutput:
    """Small inference adapter around one native Booster per forecast horizon."""

    def __init__(self, estimators):
        self.estimators_ = estimators

    def predict(self, X):
        import xgboost as xgb

        matrix = xgb.DMatrix(X)
        return np.column_stack([estimator.predict(matrix) for estimator in self.estimators_])
