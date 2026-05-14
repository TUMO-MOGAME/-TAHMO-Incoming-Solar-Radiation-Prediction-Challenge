"""Train the final model on all training data and persist it.

Usage:
    python -m src.train --config config/config.yaml
"""
from __future__ import annotations

import argparse

from src.utils.io import load_config, read_parquet
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main(config_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    log.info("Loading processed train data...")
    # TODO: load processed train, build feature matrix, fit model, persist to models/
    # train = read_parquet(cfg["paths"]["processed_dir"] + "/train.parquet")
    log.info("train.py is a skeleton. Implement after CV is producing scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)