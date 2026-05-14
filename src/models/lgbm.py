"""LightGBM regressor wrapper with early stopping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


class LightGBMRegressor:
    """Thin wrapper around `lightgbm.train` with early stopping callbacks."""

    def __init__(self, params: dict, training: dict) -> None:
        self.params = dict(params)
        self.training = dict(training)
        self.booster_: lgb.Booster | None = None
        self.best_iteration_: int | None = None
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "LightGBMRegressor":
        self.feature_names_ = list(X_train.columns)
        cat = categorical_features if categorical_features else "auto"

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat, free_raw_data=False)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(
                X_valid, label=y_valid,
                categorical_feature=cat, reference=dtrain, free_raw_data=False,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")

        callbacks: list = []
        es = self.training.get("early_stopping_rounds")
        if es and X_valid is not None:
            callbacks.append(lgb.early_stopping(es, verbose=False))
        log_every = self.training.get("log_evaluation", 0)
        if log_every:
            callbacks.append(lgb.log_evaluation(period=log_every))

        self.booster_ = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=self.training.get("num_boost_round", 1000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster_.best_iteration or self.training.get("num_boost_round")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Model has not been fitted")
        return self.booster_.predict(X, num_iteration=self.best_iteration_)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self.booster_ is None:
            raise RuntimeError("Model has not been fitted")
        imp = self.booster_.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
