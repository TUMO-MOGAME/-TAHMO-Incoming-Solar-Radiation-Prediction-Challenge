"""XGBoost wrapper. Skeleton — fill in after LightGBM baseline is validated."""
from __future__ import annotations

import numpy as np
import pandas as pd


class XGBoostRegressor:
    def __init__(self, params: dict, training: dict) -> None:
        self.params = params
        self.training = training
        self.booster_ = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "XGBoostRegressor":
        raise NotImplementedError("Implement once LightGBM baseline is validated.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implement once LightGBM baseline is validated.")
