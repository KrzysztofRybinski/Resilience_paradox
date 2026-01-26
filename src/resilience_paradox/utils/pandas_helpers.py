"""Pandas helper utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gini(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if array.size == 0:
        return 0.0
    array = np.sort(array)
    index = np.arange(1, array.size + 1)
    return (np.sum((2 * index - array.size - 1) * array)) / (array.size * np.sum(array))


def hhi(values: pd.Series) -> float:
    total = values.sum()
    if total == 0:
        return 0.0
    shares = values / total
    return float((shares**2).sum())
