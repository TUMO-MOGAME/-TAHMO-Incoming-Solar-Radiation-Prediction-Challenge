"""Full CV loop. Skeleton — wire up model training once src/models/* is implemented.

Usage:
    python -m src.cv.run_cv --config config/config.yaml
"""
from __future__ import annotations

import argparse

from src.cv.splitters import make_splitter
from src.utils.io import load_config, read_parquet
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    log.info("Loading interim train parquet...")
    train = read_parquet(cfg["paths"]["train_interim"])
    log.info("Train shape: %s", train.shape)

    splitter = make_splitter(cfg)
    log.info("Using splitter: %s", type(splitter).__name__)

    for fold in splitter.split(train):
        log.info(
            "Fold %d (%s): train=%d valid=%d",
            fold.fold_id, fold.description,
            len(fold.train_idx), len(fold.valid_idx),
        )
        # TODO: wire up model training + evaluation once src/models/* is implemented.
        # 1. Build features for train + valid
        # 2. Fit model on train
        # 3. Predict on valid
        # 4. Compute evaluate(y_true, y_pred, stations)
        # 5. Save OOF predictions to experiments/exp_NNN_*/oof/fold_X.parquet

    log.info("CV scaffolding ran. Implement model wiring to produce scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
