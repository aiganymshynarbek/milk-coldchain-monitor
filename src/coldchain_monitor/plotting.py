"""Matplotlib figures (Agg backend, so they also render on a CI server)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coldchain_monitor.data import TemperatureLog  # noqa: E402
from coldchain_monitor.evaluation import EvaluationRow  # noqa: E402
from coldchain_monitor.warning import DEFAULT_LIMIT  # noqa: E402

COLORS = {
    "Persistence": "#1f5fa8",
    "Moving average": "#7f8c9b",
    "Linear trend": "#d9730d",
    "ARIMA(5,1,0)": "#2a8a4a",
}
RED = "#b3261e"

STYLE = {
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig, path: str | Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_trace(
    log: TemperatureLog,
    alarms=None,
    tier2=None,
    limit: float = DEFAULT_LIMIT,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Temperature trace with the limit, excursions, alarms and Tier-2 periods."""
    hours = log.time_min / 60.0
    values = log.temperature
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        if tier2 is not None:
            ax.fill_between(
                hours, 0, 1, where=np.asarray(tier2, dtype=bool), step="post",
                transform=ax.get_xaxis_transform(), color="#f0a35a", alpha=0.25, lw=0,
                label="Tier 2 active (volatile)",
            )
        ax.step(hours, values, where="post", color=COLORS["Persistence"], lw=1.8,
                label="temperature")
        ax.axhline(limit, color=RED, ls="--", lw=1.6, label=f"limit {limit:g} °C")
        above = values > limit
        ax.fill_between(hours, limit, values, where=above, step="post", color=RED, alpha=0.3,
                        lw=0)
        if alarms is not None:
            fired = np.asarray(alarms, dtype=bool)
            ax.plot(hours[fired], np.full(int(fired.sum()), limit + 0.25), "v",
                    color=COLORS["ARIMA(5,1,0)"], ms=7, label="early-warning alarm")
        ax.set(xlabel="Time, h", ylabel="Temperature, °C", title=title or f"Log {log.name}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=4, frameon=False)
        fig.tight_layout()
        return _save(fig, path)


def _grouped_bars(ax, rows: Sequence[EvaluationRow], attribute: str, methods: Sequence[str]):
    horizons = sorted({row.horizon_hours for row in rows})
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        heights = [
            next(getattr(r, attribute) for r in rows
                 if r.method == method and r.horizon_hours == h)
            for h in horizons
        ]
        offsets = np.arange(len(horizons)) + (k - (len(methods) - 1) / 2) * width
        ax.bar(offsets, heights, width * 0.92, color=COLORS[method], label=method)
    ax.set_xticks(np.arange(len(horizons)), [f"{h} h" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.grid(axis="x", visible=False)


def plot_forecast_errors(rows: Sequence[EvaluationRow], path: str | Path | None = None):
    """Mean absolute error of the point forecasts: all origins and hard segments."""
    methods = [m for m in COLORS if any(r.method == m for r in rows)]
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        _grouped_bars(left, rows, "mae_all", methods)
        left.set(title="All forecasts", ylabel="MAE, °C")
        _grouped_bars(right, rows, "mae_hard", methods)
        right.set(title="Hard segments (change > 0.5 °C)")
        left.legend(loc="upper left")
        fig.suptitle("Point-forecast error on held-out logs", fontsize=18)
        fig.tight_layout()
        return _save(fig, path)


def plot_warning(rows: Sequence[EvaluationRow], path: str | Path | None = None):
    """Early-warning quality: ROC-AUC and detection rate at the calibrated threshold."""
    alarm_rows = [r for r in rows if not math.isnan(r.roc_auc)]
    methods = [m for m in COLORS if any(r.method == m for r in alarm_rows)]
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
        _grouped_bars(left, alarm_rows, "roc_auc", methods)
        left.set(title="Discrimination", ylabel="ROC-AUC", ylim=(0.5, 1.0))
        _grouped_bars(right, alarm_rows, "detection_rate", methods)
        right.set(title="Onsets detected at 5 % false alarms", ylabel="Detection rate",
                  ylim=(0, 1.05))
        left.legend(loc="upper right")
        fig.suptitle("Excursion early warning on held-out logs", fontsize=18)
        fig.tight_layout()
        return _save(fig, path)
