"""XGBoost regressor wrapper with early stopping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostRegressor:
    """Thin wrapper around `xgboost.train` with early stopping."""

    def __init__(self, params: dict, training: dict) -> None:
        self.params = dict(params)
        self.training = dict(training)
        self.booster_: xgb.Booster | None = None
        self.best_iteration_: int | None = None
        self.feature_names_: list[str] | None = None
        self.categorical_features_: list[str] | None = None

    def _to_dmatrix(self, X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
        enable_cat = bool(self.categorical_features_)
        return xgb.DMatrix(X, label=y, enable_categorical=enable_cat)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "XGBoostRegressor":
        self.feature_names_ = list(X_train.columns)
        self.categorical_features_ = list(categorical_features) if categorical_features else []

        dtrain = self._to_dmatrix(X_train, y_train)
        evals = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            dvalid = self._to_dmatrix(X_valid, y_valid)
            evals.append((dvalid, "valid"))

        params = dict(self.params)
        # Categorical handling lives on the DMatrix; tree_method=hist is required for it.
        if self.categorical_features_:
            params.setdefault("tree_method", "hist")

        early_stopping = (
            self.training.get("early_stopping_rounds")
            if X_valid is not None
            else None
        )

        self.booster_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.training.get("num_boost_round", 1000),
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=self.training.get("verbose_eval", 200),
        )
        self.best_iteration_ = (
            getattr(self.booster_, "best_iteration", None)
            or self.training.get("num_boost_round")
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Model has not been fitted")
        dmat = self._to_dmatrix(X)
        return self.booster_.predict(
            dmat,
            iteration_range=(0, (self.best_iteration_ or 0) + 1),
        )

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.booster_ is None:
            raise RuntimeError("Model has not been fitted")
        score_map = self.booster_.get_score(importance_type=importance_type)
        # XGBoost returns dict keyed by feature name; zero-importance features may be missing
        return (
            pd.DataFrame({
                "feature": self.feature_names_,
                "importance": [score_map.get(f, 0.0) for f in self.feature_names_],
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )