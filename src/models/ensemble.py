"""Blending and stacking across multiple base models.

Built last, after we have 2-3 strong single models with OOF predictions
in experiments/exp_*/oof/.
"""
from __future__ import annotations

import numpy as np


def weighted_blend(predictions: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted average of model predictions. Weights need not sum to 1."""
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must be same length")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    stacked = np.vstack(predictions)
    return (w[:, None] * stacked).sum(axis=0)
