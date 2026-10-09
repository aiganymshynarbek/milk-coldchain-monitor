import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coldchain_monitor import __version__
from coldchain_monitor.cli import main
from coldchain_monitor.data import TemperatureLog, read_csv, synthetic_log, write_csv
from coldchain_monitor.evaluation import ALARM_METHODS, POINT_METHODS, evaluate, format_table
from coldchain_monitor.plotting import plot_forecast_errors, plot_trace, plot_warning

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name)


class SyntheticLogTest(unittest.TestCase):
    def test_same_seed_gives_same_log(self):
        np.testing.assert_array_equal(
            synthetic_log(hours=24, seed=3).temperature, synthetic_log(hours=24, seed=3).temperature
        )

    def test_length_and_duration(self):
        log = synthetic_log(hours=24, seed=0)
        self.assertEqual(len(log), 24 * 12 + 1)
        self.assertAlmostEqual(log.hours, 24.0)
        self.assertEqual(log.time_min[1], 5.0)

    def test_quantization_grid(self):
        values = synthetic_log(hours=48, seed=1, quantization=0.5).temperature
        np.testing.assert_allclose(values * 2, np.round(values * 2))

    def test_quiet_store_stays_in_the_thermostat_band(self):
        log = synthetic_log(
            hours=72, seed=2, doors_per_day=0, faults_per_week=0, quantization=0, noise=0
        )
        self.assertLess(log.temperature.max(), 2.5 + 0.6 + 0.1)
        self.assertGreater(log.temperature.min(), 2.5 - 0.6 - 0.1)

    def test_faults_cause_excursions(self):
        log = synthetic_log(hours=168, seed=0, faults_per_week=20)
        self.assertGreater(log.temperature.max(), 4.0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            synthetic_log(hours=0)
        with self.assertRaises(ValueError):
            TemperatureLog(np.array([3.0]))
        with self.assertRaises(ValueError):
            TemperatureLog(np.array([3.0, float("nan")]))


class CsvTest(TempDirTestCase):
    def test_round_trip(self):
        log = synthetic_log(hours=12, seed=4)
        loaded = read_csv(write_csv(log, self.folder / "sub" / "store_a.csv"))
        np.testing.assert_allclose(loaded.temperature, log.temperature, atol=5e-3)
        self.assertEqual(loaded.name, "store_a")

    def test_missing_column_is_reported(self):
        path = self.folder / "bad.csv"
        path.write_text("time_min,temp\n0,3.0\n5,3.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "temperature_c"):
            read_csv(path)

    def test_non_numeric_value_is_reported(self):
        path = self.folder / "bad.csv"
        path.write_text("time_min,temperature_c\n0,3.0\n5,n/a\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            read_csv(path)

    def test_too_short_file_is_reported(self):
        path = self.folder / "short.csv"
        path.write_text("time_min,temperature_c\n0,3.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            read_csv(path)


class EvaluationTest(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        logs = [synthetic_log(hours=96, seed=seed, faults_per_week=8) for seed in range(6)]
        cls.calibration, cls.test = logs[:3], logs[3:]
        cls.rows = evaluate(cls.calibration, cls.test, horizons_hours=(1, 3))

    def row(self, method, hours):
        return next(r for r in self.rows if r.method == method and r.horizon_hours == hours)

    def test_one_row_per_method_and_horizon(self):
        self.assertEqual(len(self.rows), 2 * len(POINT_METHODS))

    def test_metrics_are_in_their_ranges(self):
        for row in self.rows:
            with self.subTest(method=row.method, horizon=row.horizon_hours):
                self.assertGreaterEqual(row.mae_all, 0.0)
                self.assertGreater(row.mae_hard, 0.5)  # the change itself exceeds 0.5 °C
                if row.method in ALARM_METHODS:
                    self.assertTrue(0.0 <= row.roc_auc <= 1.0)
                    self.assertTrue(0.0 <= row.detection_rate <= 1.0)
                    self.assertGreaterEqual(row.false_alarms_per_day, 0.0)
                else:
                    self.assertTrue(math.isnan(row.roc_auc))

    def test_persistence_error_matches_direct_computation(self):
        errors = []
        for log in self.test:
            y = log.temperature
            errors.append(np.abs(y[36:-12] - y[48:]))
        self.assertAlmostEqual(
            self.row("Persistence", 1).mae_all, float(np.concatenate(errors).mean())
        )

    def test_error_grows_with_horizon(self):
        for method in POINT_METHODS:
            with self.subTest(method=method):
                self.assertLess(self.row(method, 1).mae_all, self.row(method, 3).mae_all)

    def test_warning_is_better_than_chance(self):
        self.assertGreater(self.row("Persistence", 1).roc_auc, 0.7)
        self.assertGreater(self.row("ARIMA(5,1,0)", 1).roc_auc, 0.7)

    def test_table_lists_every_method(self):
        table = format_table(self.rows)
        for method in POINT_METHODS:
            self.assertIn(method, table)
        self.assertEqual(len(table.splitlines()), 2 + len(self.rows))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], self.test)
        with self.assertRaises(ValueError):
            evaluate(self.calibration, self.test, horizons_hours=(0,))

    def test_figures_are_written_as_png(self):
        log = self.test[0]
        alarms = log.temperature > 3.4
        paths = [self.folder / name for name in ("trace.png", "errors.png", "warning.png")]
        plot_trace(log, alarms=alarms, tier2=alarms, path=paths[0])
        plot_forecast_errors(self.rows, paths[1])
        plot_warning(self.rows, paths[2])
        for path in paths:
            self.assertEqual(path.read_bytes()[:8], PNG_SIGNATURE)


class CliTest(TempDirTestCase):
    def make_logs(self, count):
        paths = []
        for seed in range(count):
            path = self.folder / f"log{seed}.csv"
            code, _, _ = run_cli("generate", str(path), "--hours", "60", "--seed", str(seed))
            self.assertEqual(code, 0)
            paths.append(str(path))
        return paths

    def test_generate_then_forecast(self):
        (path,) = self.make_logs(1)
        code, out, _ = run_cli("forecast", path, "--hours", "3")
        self.assertEqual(code, 0)
        self.assertIn("ARIMA(5,1,0)", out)
        self.assertIn("Status:", out)
        self.assertIn("Tier 2 trigger", out)

    def test_evaluate_prints_table(self):
        paths = self.make_logs(4)
        code, out, _ = run_cli(
            "evaluate", "--calibration", *paths[:2], "--test", *paths[2:], "--horizons", "1"
        )
        self.assertEqual(code, 0)
        self.assertIn("Persistence", out)
        self.assertIn("FA/day", out)

    def test_missing_file_gives_error_code(self):
        code, _, err = run_cli("forecast", str(self.folder / "nothing.csv"))
        self.assertEqual(code, 2)
        self.assertIn("error", err)

    def test_version_flag(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn(__version__, out.getvalue())


if __name__ == "__main__":
    unittest.main()
