"""Point forecasters for a temperature series.

Every function receives the history observed so far (``history[-1]`` is the
latest reading) and a horizon in 5-minute steps, and uses nothing else, so a
forecast can never look into the future.

The ARIMA(5,1,0) model follows the thesis paper: an autoregression of order
5 on the first differences, re-estimated at every forecast origin on a
sliding 24-hour window by ridge-regularised least squares, with a stability
correction of the fitted coefficients.
"""

from __future__ import annotations

import numpy as np

from coldchain_monitor.data import STEPS_PER_HOUR

AR_ORDER = 5
AR_WINDOW = 24 * STEPS_PER_HOUR  # 288 steps
AR_RIDGE = 1e-3
STABILITY_LIMIT = 0.98


def _as_history(history) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty 1-D array")
    return values


def _check_horizon(steps: int) -> None:
    if steps < 1:
        raise ValueError("the horizon must be at least one step")


def persistence(history, steps: int) -> float:
    """Forecast = the latest reading."""
    _check_horizon(steps)
    return float(_as_history(history)[-1])


def moving_average(history, steps: int, window: int = STEPS_PER_HOUR) -> float:
    """Forecast = mean of the last ``window`` readings (1 hour by default)."""
    _check_horizon(steps)
    if window < 1:
        raise ValueError("window must be at least 1")
    return float(_as_history(history)[-window:].mean())


def linear_trend(history, steps: int, slope_window: int = 6) -> float:
    """Extrapolate the slope of the last ``slope_window`` steps (30 minutes)."""
    _check_horizon(steps)
    values = _as_history(history)
    if values.size <= slope_window:
        return float(values[-1])
    slope = (values[-1] - values[-1 - slope_window]) / slope_window
    return float(values[-1] + slope * steps)


def fit_ar_on_differences(
    history, order: int = AR_ORDER, ridge: float = AR_RIDGE
) -> np.ndarray:
    """Ridge least-squares AR(``order``) coefficients of the differenced series.

    Returns ``phi`` with ``phi[0]`` multiplying the most recent difference.
    A series too short to estimate gives all-zero coefficients, which makes
    the ARIMA forecast fall back to persistence.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    diffs = np.diff(_as_history(history))
    if diffs.size < order + 2:
        return np.zeros(order)
    lagged = np.column_stack(
        [diffs[order - k - 1 : diffs.size - k - 1] for k in range(order)]
    )
    target = diffs[order:]
    gram = lagged.T @ lagged + ridge * np.eye(order)
    return np.linalg.solve(gram, lagged.T @ target)


def companion_eigenvalues(phi) -> np.ndarray:
    """Eigenvalues of the companion matrix of AR coefficients ``phi``.

    The recursion ``d[t] = phi[0] d[t-1] + ... + phi[p-1] d[t-p]`` produces
    bounded forecasts exactly when every eigenvalue has modulus below one.
    """
    phi = np.asarray(phi, dtype=float)
    order = phi.size
    companion = np.zeros((order, order))
    companion[0, :] = phi
    if order > 1:
        companion[1:, :-1] = np.eye(order - 1)
    return np.linalg.eigvals(companion)


def stabilise(
    phi, limit: float = STABILITY_LIMIT, shrink: float = 0.95, max_iter: int = 500
) -> np.ndarray:
    """Shrink ``phi`` uniformly until all companion eigenvalues are below ``limit``.

    Without this correction an unstable fit makes multi-step forecasts
    diverge, which inflates the error by orders of magnitude.
    """
    if not 0 < limit <= 1 or not 0 < shrink < 1:
        raise ValueError("limit must be in (0, 1] and shrink in (0, 1)")
    phi = np.array(phi, dtype=float)
    for _ in range(max_iter):
        if np.max(np.abs(companion_eigenvalues(phi))) < limit:
            return phi
        phi *= shrink
    return np.zeros_like(phi)


def arima_forecast_path(
    history,
    steps: int,
    order: int = AR_ORDER,
    window: int = AR_WINDOW,
    ridge: float = AR_RIDGE,
) -> np.ndarray:
    """ARIMA(``order``,1,0) forecasts for 1..``steps`` steps ahead.

    The model is fitted to the last ``window`` differences of ``history``
    and stabilised before the multi-step recursion.
    """
    _check_horizon(steps)
    values = _as_history(history)[-(window + 1) :]
    phi = stabilise(fit_ar_on_differences(values, order, ridge))
    recent = np.zeros(order)
    diffs = np.diff(values)[-order:][::-1]  # most recent first
    recent[: diffs.size] = diffs
    path = np.empty(steps)
    level = values[-1]
    for step in range(steps):
        delta = float(phi @ recent)
        level += delta
        path[step] = level
        recent = np.concatenate(([delta], recent[:-1]))
    return path
