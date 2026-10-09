"""Feature construction for the machine-learning tier.

At each time stamp ``i`` the features use readings up to and including ``i``
only, so there is no look-ahead bias. Rows that do not yet have three hours
of history are filled with ``nan``.
"""

from __future__ import annotations

import numpy as np

DIFF_LAGS = (1, 3, 6, 12, 36)  # 5, 15, 30, 60, 180 minutes
ROLL_WINDOWS = (12, 36)  # 1 hour, 3 hours

FEATURE_NAMES: tuple[str, ...] = (
    "value",
    *(f"d{lag}" for lag in DIFF_LAGS),
    *(f"{stat}{window}" for window in ROLL_WINDOWS for stat in ("max", "min", "std")),
    "mean12",
)


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Windows ending at each index; shape (n - window + 1, window)."""
    return np.lib.stride_tricks.sliding_window_view(values, window)


def build_features(temperature) -> np.ndarray:
    """Return an array of shape ``(n, len(FEATURE_NAMES))``.

    Columns follow :data:`FEATURE_NAMES`: the reading itself, first
    differences at lags of 5 to 180 minutes, rolling maximum, minimum and
    standard deviation over 1 and 3 hours, and the 1-hour rolling mean.
    """
    values = np.asarray(temperature, dtype=float)
    if values.ndim != 1:
        raise ValueError("temperature must be a 1-D array")
    n = values.size
    columns: dict[str, np.ndarray] = {"value": values.copy()}
    for lag in DIFF_LAGS:
        column = np.full(n, np.nan)
        column[lag:] = values[lag:] - values[:-lag]
        columns[f"d{lag}"] = column
    for window in ROLL_WINDOWS:
        stats = {name: np.full(n, np.nan) for name in ("max", "min", "std", "mean")}
        if n >= window:
            view = _rolling(values, window)
            stats["max"][window - 1 :] = view.max(axis=1)
            stats["min"][window - 1 :] = view.min(axis=1)
            stats["std"][window - 1 :] = view.std(axis=1)
            stats["mean"][window - 1 :] = view.mean(axis=1)
        for name in ("max", "min", "std"):
            columns[f"{name}{window}"] = stats[name]
        if window == 12:
            columns["mean12"] = stats["mean"]
    features = np.column_stack([columns[name] for name in FEATURE_NAMES])
    warmup = max(max(DIFF_LAGS), max(ROLL_WINDOWS) - 1)
    features[: min(warmup, n)] = np.nan
    return features
