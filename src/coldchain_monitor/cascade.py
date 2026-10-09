"""Trigger of the two-tier monitoring cascade.

Tier 1 (persistence / ARIMA) runs on every reading. Tier 2, a heavier
machine-learning classifier, is switched on only while the signal is
volatile:

    Tier 2 active  <=>  std12 > tau0  or  std36 > tau1

where ``std12`` and ``std36`` are the rolling standard deviations over one
and three hours. The default thresholds are placeholders: tuning them on
real dairy sensor data is future work.
"""

from __future__ import annotations

import numpy as np

from coldchain_monitor.features import FEATURE_NAMES, build_features

DEFAULT_TAU0 = 0.30
DEFAULT_TAU1 = 0.45


def tier2_active(temperature, tau0: float = DEFAULT_TAU0, tau1: float = DEFAULT_TAU1):
    """Boolean array: ``True`` where the volatility trigger switches Tier 2 on.

    During the first three hours, before the rolling windows are full, the
    trigger is ``False``.
    """
    if tau0 < 0 or tau1 < 0:
        raise ValueError("thresholds must be non-negative")
    features = build_features(temperature)
    std12 = features[:, FEATURE_NAMES.index("std12")]
    std36 = features[:, FEATURE_NAMES.index("std36")]
    with np.errstate(invalid="ignore"):
        return (std12 > tau0) | (std36 > tau1)
