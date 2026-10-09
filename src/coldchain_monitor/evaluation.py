"""Comparison of the Tier-1 methods on a set of temperature logs.

Alarm thresholds are calibrated on *calibration* logs and every reported
number is computed on separate *held-out* logs, mirroring the
leave-series-out idea of the thesis paper: a method is never judged on a
log that was used to tune it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from coldchain_monitor.data import STEPS_PER_HOUR, TemperatureLog
from coldchain_monitor.forecasting import arima_forecast_path, linear_trend
from coldchain_monitor.warning import (
    DEFAULT_LIMIT,
    EventMetrics,
    average_precision,
    event_metrics,
    excursion_labels,
    roc_auc,
    threshold_at_false_alarm_rate,
)

#: readings needed before the first forecast (3 hours of history)
WARMUP_STEPS = 36
#: a forecast is "hard" when the true change over the horizon exceeds this, degC
HARD_CHANGE = 0.5
#: methods that also give an alarm score
ALARM_METHODS = ("Persistence", "Linear trend", "ARIMA(5,1,0)")
POINT_METHODS = ("Persistence", "Moving average", "Linear trend", "ARIMA(5,1,0)")


@dataclass(frozen=True)
class EvaluationRow:
    """Result of one method at one horizon on the held-out logs."""

    horizon_hours: int
    method: str
    mae_all: float
    mae_hard: float
    roc_auc: float
    average_precision: float
    detection_rate: float
    median_lead_minutes: float
    false_alarms_per_day: float


def predictions(log: TemperatureLog, horizons_steps: Sequence[int], limit: float = DEFAULT_LIMIT):
    """Point forecasts and alarm scores of every method at every origin.

    Returns ``(points, scores)``: dictionaries keyed by ``(method, steps)``
    with arrays as long as the log (``nan`` where no forecast is made).
    The alarm score is the forecast margin above the limit; for ARIMA the
    highest value on the forecast path within the horizon is used.
    """
    values = log.temperature
    n = values.size
    longest = max(horizons_steps)
    points = {(m, h): np.full(n, np.nan) for m in POINT_METHODS for h in horizons_steps}
    scores = {(m, h): np.full(n, np.nan) for m in ALARM_METHODS for h in horizons_steps}
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    for i in range(WARMUP_STEPS, n):
        history = values[: i + 1]
        path = arima_forecast_path(history, longest)
        mean_hour = (cumulative[i + 1] - cumulative[i + 1 - STEPS_PER_HOUR]) / STEPS_PER_HOUR
        for h in horizons_steps:
            trend = linear_trend(history, h)
            points["Persistence", h][i] = values[i]
            points["Moving average", h][i] = mean_hour
            points["Linear trend", h][i] = trend
            points["ARIMA(5,1,0)", h][i] = path[h - 1]
            scores["Persistence", h][i] = values[i] - limit
            scores["Linear trend", h][i] = trend - limit
            scores["ARIMA(5,1,0)", h][i] = path[:h].max() - limit
    return points, scores


def evaluate(
    calibration_logs: Sequence[TemperatureLog],
    test_logs: Sequence[TemperatureLog],
    horizons_hours: Sequence[int] = (1, 3, 6),
    limit: float = DEFAULT_LIMIT,
    false_alarm_rate: float = 0.05,
) -> list[EvaluationRow]:
    """Compare the methods; thresholds from calibration logs, metrics from test logs."""
    if not calibration_logs or not test_logs:
        raise ValueError("both calibration and test logs are required")
    if any(h < 1 for h in horizons_hours):
        raise ValueError("horizons must be whole hours, at least 1")
    steps = [int(h) * STEPS_PER_HOUR for h in horizons_hours]

    def collect(logs):
        collected = []
        for log in logs:
            points, scores = predictions(log, steps, limit)
            collected.append((log.temperature, points, scores))
        return collected

    calibration, test = collect(calibration_logs), collect(test_logs)
    rows = []
    for hours, h in zip(horizons_hours, steps, strict=True):
        for method in POINT_METHODS:
            errors, hard = [], []
            for values, points, _ in test:
                forecast = points[method, h][:-h]
                actual = values[h:]
                ok = np.isfinite(forecast)
                errors.append(np.abs(forecast - actual)[ok])
                hard.append((np.abs(actual - values[:-h]) > HARD_CHANGE)[ok])
            errors, hard = np.concatenate(errors), np.concatenate(hard)
            mae_all = float(errors.mean()) if errors.size else float("nan")
            mae_hard = float(errors[hard].mean()) if hard.any() else float("nan")

            auc = ap = detection = lead = false_alarms = float("nan")
            if method in ALARM_METHODS:

                def pooled(dataset, method=method, h=h):
                    all_scores, all_labels = [], []
                    for values, _, scores in dataset:
                        labels, valid = excursion_labels(values, h, limit)
                        keep = valid & np.isfinite(scores[method, h])
                        all_scores.append(scores[method, h][keep])
                        all_labels.append(labels[keep])
                    return np.concatenate(all_scores), np.concatenate(all_labels)

                test_scores, test_labels = pooled(test)
                auc = roc_auc(test_scores, test_labels)
                ap = average_precision(test_scores, test_labels)
                cal_scores, cal_labels = pooled(calibration)
                threshold = threshold_at_false_alarm_rate(cal_scores, cal_labels, false_alarm_rate)
                total = EventMetrics(0, 0, (), 0, 0.0)
                for values, _, scores in test:
                    with np.errstate(invalid="ignore"):
                        alarms = scores[method, h] > threshold
                    total = total + event_metrics(values, alarms, h, limit)
                detection = total.detection_rate
                lead = total.median_lead_minutes
                false_alarms = total.false_alarms_per_day
            rows.append(
                EvaluationRow(
                    int(hours), method, mae_all, mae_hard, auc, ap, detection, lead, false_alarms
                )
            )
    return rows


def format_table(rows: Sequence[EvaluationRow]) -> str:
    """Plain-text table of evaluation rows."""

    def cell(value: float, spec: str) -> str:
        return "-" if math.isnan(value) else format(value, spec)

    header = (
        f"{'H':>3}  {'Method':<15} {'MAE':>6} {'MAE hard':>9} {'AUC':>6} {'AP':>6} "
        f"{'Det.':>5} {'Lead,min':>9} {'FA/day':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.horizon_hours:>2}h  {row.method:<15} {cell(row.mae_all, '.3f'):>6} "
            f"{cell(row.mae_hard, '.3f'):>9} {cell(row.roc_auc, '.3f'):>6} "
            f"{cell(row.average_precision, '.3f'):>6} {cell(row.detection_rate, '.2f'):>5} "
            f"{cell(row.median_lead_minutes, '.0f'):>9} {cell(row.false_alarms_per_day, '.2f'):>7}"
        )
    return "\n".join(lines)
