"""Shared model interface.

All concrete model wrappers (lgbm, xgb, catboost) implement this minimal
protocol so they can be swapped via config.yaml without changing the CV loop.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """Minimal contract for a fittable + predictable regressor."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "Model":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...
