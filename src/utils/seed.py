"""Deterministic seeding for numpy, random, and (optionally) PYTHONHASHSEED."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and PYTHONHASHSEED for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
