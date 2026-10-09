"""coldchain-monitor: forecasting and early warning for milk cold-storage temperature.

The package implements the always-on tier ("Tier 1") of the two-tier monitoring
cascade proposed for a dairy cold-storage digital twin: simple forecasters
(persistence, moving average, linear trend), a stabilised ARIMA(5,1,0) model,
excursion early-warning scores and the metrics used to compare them.
"""

from coldchain_monitor.cascade import tier2_active
from coldchain_monitor.data import (
    STEP_MINUTES,
    TemperatureLog,
    read_csv,
    synthetic_log,
    write_csv,
)
from coldchain_monitor.evaluation import EvaluationRow, evaluate, format_table
from coldchain_monitor.features import FEATURE_NAMES, build_features
from coldchain_monitor.forecasting import (
    arima_forecast_path,
    companion_eigenvalues,
    fit_ar_on_differences,
    linear_trend,
    moving_average,
    persistence,
    stabilise,
)
from coldchain_monitor.warning import (
    average_precision,
    event_metrics,
    excursion_labels,
    roc_auc,
    threshold_at_false_alarm_rate,
)

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "STEP_MINUTES",
    "EvaluationRow",
    "TemperatureLog",
    "arima_forecast_path",
    "average_precision",
    "build_features",
    "companion_eigenvalues",
    "evaluate",
    "event_metrics",
    "excursion_labels",
    "fit_ar_on_differences",
    "format_table",
    "linear_trend",
    "moving_average",
    "persistence",
    "read_csv",
    "roc_auc",
    "stabilise",
    "synthetic_log",
    "threshold_at_false_alarm_rate",
    "tier2_active",
    "write_csv",
    "__version__",
]
