"""End-to-end demonstration on synthetic logs.

Run from the repository root::

    python examples/run_demo.py

Twenty synthetic milk-store logs are generated; ten calibrate the alarm
thresholds and ten are held out for the comparison. The figures go to
``docs/figures``, the table to ``docs/demo_results.txt`` and one example
log to ``data/example_log.csv``.

The logs are synthetic: the output demonstrates the software, not the
ranking of the methods on a real cold chain.
"""

from pathlib import Path

import numpy as np

from coldchain_monitor import (
    STEP_MINUTES,
    evaluate,
    excursion_labels,
    format_table,
    synthetic_log,
    threshold_at_false_alarm_rate,
    tier2_active,
    write_csv,
)
from coldchain_monitor.evaluation import predictions
from coldchain_monitor.plotting import plot_forecast_errors, plot_trace, plot_warning

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
N_LOGS = 20
STEPS_1H = 60 // STEP_MINUTES


def main() -> None:
    logs = [synthetic_log(hours=168, seed=seed) for seed in range(N_LOGS)]
    calibration, test = logs[: N_LOGS // 2], logs[N_LOGS // 2 :]
    write_csv(logs[0], ROOT / "data" / "example_log.csv")

    # 1. Compare the methods: thresholds from calibration logs, metrics from test logs.
    rows = evaluate(calibration, test, horizons_hours=(1, 3, 6))
    table = format_table(rows)
    print(table)
    (ROOT / "docs" / "demo_results.txt").write_text(table + "\n", encoding="utf-8")
    plot_forecast_errors(rows, FIGURES / "forecast_errors.png")
    plot_warning(rows, FIGURES / "early_warning.png")

    # 2. One held-out log with the 1-hour ARIMA alarm and the Tier-2 trigger.
    method = "ARIMA(5,1,0)"
    cal_scores, cal_labels = [], []
    for log in calibration:
        _, scores = predictions(log, [STEPS_1H])
        labels, valid = excursion_labels(log.temperature, STEPS_1H)
        keep = valid & np.isfinite(scores[method, STEPS_1H])
        cal_scores.append(scores[method, STEPS_1H][keep])
        cal_labels.append(labels[keep])
    threshold = threshold_at_false_alarm_rate(
        np.concatenate(cal_scores), np.concatenate(cal_labels), 0.05
    )
    example = test[2]
    _, scores = predictions(example, [STEPS_1H])
    with np.errstate(invalid="ignore"):
        alarms = (scores[method, STEPS_1H] > threshold) & (example.temperature <= 4.0)
    plot_trace(
        example,
        alarms=alarms,
        tier2=tier2_active(example.temperature),
        path=FIGURES / "trace.png",
        title="Held-out synthetic log with 1-hour ARIMA early warning",
    )

    share = np.mean([tier2_active(log.temperature).mean() for log in test])
    print(f"\nTier 2 would be active {share:.0%} of the time on the held-out logs")
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
