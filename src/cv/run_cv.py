"""Full CV loop: features -> per-fold fit -> evaluate -> save artifacts.

Usage:
    python -m src.cv.run_cv --config config/config.yaml \
        --features config/features.yaml --model config/model.yaml \
        --experiment exp_001_lgbm_baseline
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from src.cv.evaluate import evaluate
from src.cv.splitters import make_splitter
from src.features.build_features import build_feature_matrix
from src.models.lgbm import LightGBMRegressor
from src.utils.io import load_config, read_parquet, resolve_path, write_parquet
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def build_model(model_cfg: dict):
    active = model_cfg["active"]
    if active == "lightgbm":
        return LightGBMRegressor(model_cfg["lightgbm"]["params"], model_cfg["lightgbm"]["training"])
    raise NotImplementedError(f"Model '{active}' not yet wired into run_cv.py")


def main(config_path: str, features_path: str, model_path: str, experiment: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    fcfg = load_config(features_path)
    mcfg = load_config(model_path)
    set_seed(cfg["project"]["seed"])

    exp_dir = resolve_path(cfg["paths"]["experiments_dir"]) / experiment
    (exp_dir / "oof").mkdir(parents=True, exist_ok=True)

    for src_path, dest_name in [
        (config_path, "config.yaml"),
        (features_path, "features.yaml"),
        (model_path, "model.yaml"),
    ]:
        shutil.copy(resolve_path(src_path), exp_dir / dest_name)
    log.info("Experiment dir: %s", exp_dir)

    log.info("Loading interim train parquet...")
    train = read_parquet(cfg["paths"]["train_interim"])
    log.info("Train shape: %s", train.shape)

    target_col = cfg["columns"]["target"]
    id_col = cfg["columns"]["id"]
    station_col = cfg["columns"]["station"]

    log.info("Building feature matrix...")
    X, cat_features, _ = build_feature_matrix(train, cfg, fcfg)
    y = train[target_col].to_numpy()
    ids = train[id_col].to_numpy()
    stations = train[station_col].to_numpy()
    log.info("X shape: %s, n_features: %d, categorical: %s", X.shape, X.shape[1], cat_features)

    splitter = make_splitter(cfg)
    log.info("Splitter: %s", type(splitter).__name__)

    clip_min = cfg["evaluation"]["clip_min"]
    clip_max = cfg["evaluation"]["clip_max"]
    mbe_w = cfg["evaluation"]["mbe_weight"]
    rmse_w = cfg["evaluation"]["rmse_weight"]

    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_scores: list[dict] = []

    for fold in splitter.split(train):
        log.info("==== Fold %d (%s): train=%d valid=%d ====",
                 fold.fold_id, fold.description, len(fold.train_idx), len(fold.valid_idx))

        X_tr = X.iloc[fold.train_idx]
        y_tr = y[fold.train_idx]
        X_va = X.iloc[fold.valid_idx]
        y_va = y[fold.valid_idx]

        model = build_model(mcfg)
        model.fit(X_tr, y_tr, X_va, y_va, categorical_features=cat_features)

        preds = np.clip(model.predict(X_va), clip_min, clip_max)
        oof_pred[fold.valid_idx] = preds

        report = evaluate(y_va, preds, stations=stations[fold.valid_idx], mbe_weight=mbe_w, rmse_weight=rmse_w)
        log.info("Fold %d: n=%d MBE=%.3f RMSE=%.3f combined=%.3f best_iter=%s",
                 fold.fold_id, report.n, report.mbe, report.rmse, report.combined,
                 getattr(model, "best_iteration_", None))
        fold_scores.append({
            "fold": fold.fold_id,
            "description": fold.description,
            "n_train": int(len(fold.train_idx)),
            "n_valid": int(len(fold.valid_idx)),
            "mbe": report.mbe,
            "rmse": report.rmse,
            "combined": report.combined,
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
        })

        if hasattr(model, "feature_importance"):
            fi = model.feature_importance("gain")
            fi.to_csv(exp_dir / f"feature_importance_fold_{fold.fold_id}.csv", index=False)

    valid_mask = ~np.isnan(oof_pred)
    overall = evaluate(y[valid_mask], oof_pred[valid_mask],
                       stations=stations[valid_mask], mbe_weight=mbe_w, rmse_weight=rmse_w)

    log.info("==== OOF overall: n=%d MBE=%.4f RMSE=%.4f combined=%.4f ====",
             overall.n, overall.mbe, overall.rmse, overall.combined)

    pd.DataFrame(fold_scores).to_csv(exp_dir / "fold_scores.csv", index=False)
    overall_summary = {
        "n": int(overall.n),
        "mbe": float(overall.mbe),
        "abs_mbe": float(overall.abs_mbe),
        "rmse": float(overall.rmse),
        "combined": float(overall.combined),
        "splitter": type(splitter).__name__,
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "categorical_features": cat_features,
    }
    with open(exp_dir / "overall_score.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    oof_df = pd.DataFrame({
        id_col: ids[valid_mask],
        "y_true": y[valid_mask],
        "y_pred": oof_pred[valid_mask],
        station_col: stations[valid_mask],
    })
    write_parquet(oof_df, str(exp_dir / "oof" / "oof_predictions.parquet"))

    if overall.per_station is not None:
        overall.per_station.to_csv(exp_dir / "per_station_score.csv", index=False)

    log.info("Wrote artifacts to %s", exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    parser.add_argument("--model", default="config/model.yaml")
    parser.add_argument("--experiment", default="exp_001_lgbm_baseline")
    args = parser.parse_args()
    main(args.config, args.features, args.model, args.experiment)
