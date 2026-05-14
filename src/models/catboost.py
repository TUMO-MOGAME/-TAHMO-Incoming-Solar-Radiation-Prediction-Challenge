"""CatBoost wrapper. Skeleton — fill in after LightGBM baseline is validated."""
from __future__ import annotations

import numpy as np
import pandas as pd


class CatBoostRegressorWrapper:
    def __init__(self, params: dict, training: dict) -> None:
        self.params = params
        self.training = training
        self.model_ = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "CatBoostRegressorWrapper":
        raise NotImplementedError("Implement once LightGBM baseline is validated.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implement once LightGBM baseline is validated.")
