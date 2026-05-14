"""LightGBM wrapper. Skeleton — fill in once we run our first CV."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LightGBMRegressor:
    """Thin wrapper around lightgbm.train with early stopping."""

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
    ) -> "LightGBMRegressor":
        # TODO: import lightgbm, build Dataset, train with early stopping.
        raise NotImplementedError("Implement when running first CV.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # TODO: self.booster_.predict(X, num_iteration=self.best_iteration_)
        raise NotImplementedError("Implement when running first CV.")
