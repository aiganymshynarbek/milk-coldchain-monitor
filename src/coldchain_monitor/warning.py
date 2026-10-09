"""Excursion early warning: labels, ranking metrics and event-level metrics.

An *excursion* is a reading above the limit (4 degC by default). The warning
task at time ``i`` with horizon ``H`` asks: will the temperature exceed the
limit at any step in ``(i, i + H]``, given that it is within the limit now?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coldchain_monitor.data import STEP_MINUTES

DEFAULT_LIMIT = 4.0


def excursion_labels(temperature, steps: int, limit: float = DEFAULT_LIMIT):
    """Return ``(labels, valid)`` for the warning task.

    ``labels[i]`` is 1 when any of the next ``steps`` readings is above
    ``limit``. ``valid[i]`` is ``True`` where the task is defined: the
    current reading is within the limit and the full horizon is observed.
    """
    if steps < 1:
        raise ValueError("the horizon must be at least one step")
    values = np.asarray(temperature, dtype=float)
    n = values.size
    labels = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    if n <= steps:
        return labels, valid
    above = values > limit
    future = np.lib.stride_tricks.sliding_window_view(above[1:], steps).any(axis=1)
    labels[: future.size] = future.astype(int)
    valid[: future.size] = ~above[: future.size]
    return labels, valid


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(1, values.size + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve (ties get half credit); ``nan`` if one class is absent."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(scores, labels) -> float:
    """Area under the precision-recall curve; ``nan`` without positives."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    total = int(labels.sum())
    if total == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    hits = np.cumsum(sorted_labels)
    counts = np.arange(1, labels.size + 1)
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    precision = hits[last_of_tie] / counts[last_of_tie]
    recall = hits[last_of_tie] / total
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def threshold_at_false_alarm_rate(scores, labels, rate: float = 0.05) -> float:
    """Score threshold exceeded by at most ``rate`` of the negative examples."""
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")
    scores = np.asarray(scores, dtype=float)
    negatives = scores[~np.asarray(labels).astype(bool)]
    if negatives.size == 0:
        raise ValueError("no negative examples to calibrate on")
    return float(np.quantile(negatives, 1.0 - rate, method="higher"))


@dataclass(frozen=True)
class EventMetrics:
    """Event-level result for one log (or pooled over logs)."""

    onsets: int  #: excursion onsets with a full horizon of history
    detected: int  #: onsets preceded by an alarm within the horizon
    lead_minutes: tuple[float, ...]  #: lead time of each detected onset
    false_alarm_episodes: int  #: alarm episodes not followed by an excursion
    monitored_days: float

    @property
    def detection_rate(self) -> float:
        return self.detected / self.onsets if self.onsets else float("nan")

    @property
    def median_lead_minutes(self) -> float:
        return float(np.median(self.lead_minutes)) if self.lead_minutes else float("nan")

    @property
    def false_alarms_per_day(self) -> float:
        if self.monitored_days <= 0:
            return float("nan")
        return self.false_alarm_episodes / self.monitored_days

    def __add__(self, other: EventMetrics) -> EventMetrics:
        return EventMetrics(
            self.onsets + other.onsets,
            self.detected + other.detected,
            self.lead_minutes + other.lead_minutes,
            self.false_alarm_episodes + other.false_alarm_episodes,
            self.monitored_days + other.monitored_days,
        )


def event_metrics(temperature, alarms, steps: int, limit: float = DEFAULT_LIMIT) -> EventMetrics:
    """Score a boolean alarm sequence against the excursion onsets of a log.

    An onset is a reading above ``limit`` that follows a reading within it.
    It counts as detected when an alarm was raised in the ``steps`` readings
    before it; the lead time is measured from the earliest such alarm. An
    alarm episode (a run of consecutive alarms) is false when no excursion
    follows within the horizon of any of its readings.
    """
    values = np.asarray(temperature, dtype=float)
    alarms = np.asarray(alarms, dtype=bool)
    if alarms.shape != values.shape:
        raise ValueError("alarms and temperature must have the same length")
    labels, valid = excursion_labels(values, steps, limit)
    alarms = alarms & valid
    above = values > limit
    onsets = np.nonzero(above[1:] & ~above[:-1])[0] + 1
    onsets = onsets[onsets >= steps]
    leads = []
    for onset in onsets:
        fired = np.nonzero(alarms[onset - steps : onset])[0]
        if fired.size:
            leads.append(float((steps - fired[0]) * STEP_MINUTES))
    padded = np.r_[False, alarms, False]
    starts = np.nonzero(padded[1:] & ~padded[:-1])[0]
    ends = np.nonzero(~padded[1:] & padded[:-1])[0]
    false_episodes = sum(
        1 for start, end in zip(starts, ends, strict=True) if not labels[start:end].any()
    )
    days = int(valid.sum()) * STEP_MINUTES / (60.0 * 24.0)
    return EventMetrics(int(onsets.size), len(leads), tuple(leads), false_episodes, days)
