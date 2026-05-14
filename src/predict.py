"""Generate predictions on the processed test set.

Usage:
    python -m src.predict --config config/config.yaml
"""
from __future__ import annotations

import argparse

from src.utils.io import load_config
from src.utils.logging import setup_logger


def main(config_path: str) -> None:
    log = setup_logger()
    cfg = load_config(config_path)
    log.info("predict.py is a skeleton. Implement after train.py is producing models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
