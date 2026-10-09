"""Tests of the forecasters, in particular the ARIMA stability correction."""

import unittest

import numpy as np

from coldchain_monitor.forecasting import (
    arima_forecast_path,
    companion_eigenvalues,
    fit_ar_on_differences,
    linear_trend,
    moving_average,
    persistence,
    stabilise,
)


def simulate_ar_differences(phi, n, seed=0, sigma=0.05):
    """Series whose first differences follow an AR process with coefficients ``phi``."""
    rng = np.random.default_rng(seed)
    order = len(phi)
    diffs = np.zeros(n + order)
    for t in range(order, n + order):
        diffs[t] = float(np.dot(phi, diffs[t - order : t][::-1])) + rng.normal(0.0, sigma)
    return 3.0 + np.cumsum(diffs[order:])


class SimpleForecastersTest(unittest.TestCase):
    def test_persistence_returns_last_value(self):
        self.assertEqual(persistence([2.0, 2.5, 3.5], 12), 3.5)

    def test_moving_average_uses_last_window_only(self):
        history = np.r_[np.full(50, 9.0), np.full(12, 2.0)]
        self.assertAlmostEqual(moving_average(history, 6), 2.0)
        self.assertAlmostEqual(moving_average([1.0, 3.0], 6), 2.0)  # shorter than the window

    def test_linear_trend_is_exact_on_a_straight_line(self):
        history = 2.0 + 0.1 * np.arange(40)
        self.assertAlmostEqual(linear_trend(history, 12), history[-1] + 1.2)

    def test_linear_trend_falls_back_to_persistence_on_short_history(self):
        self.assertEqual(linear_trend([2.0, 3.0], 12), 3.0)

    def test_constant_series_is_forecast_as_constant_by_all_methods(self):
        history = np.full(300, 3.5)
        self.assertEqual(persistence(history, 36), 3.5)
        self.assertAlmostEqual(moving_average(history, 36), 3.5)
        self.assertAlmostEqual(linear_trend(history, 36), 3.5)
        np.testing.assert_allclose(arima_forecast_path(history, 36), 3.5)

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            persistence([], 1)
        with self.assertRaises(ValueError):
            persistence([1.0], 0)
        with self.assertRaises(ValueError):
            moving_average([1.0, 2.0], 1, window=0)
        with self.assertRaises(ValueError):
            fit_ar_on_differences([1.0, 2.0, 3.0], order=0)


class ArimaTest(unittest.TestCase):
    TRUE_PHI = np.array([0.5, -0.2, 0.1, 0.0, 0.0])

    def test_fit_recovers_known_coefficients(self):
        series = simulate_ar_differences(self.TRUE_PHI, 20000, seed=1)
        phi = fit_ar_on_differences(series, ridge=1e-6)
        np.testing.assert_allclose(phi, self.TRUE_PHI, atol=0.03)

    def test_short_series_gives_zero_coefficients(self):
        np.testing.assert_array_equal(fit_ar_on_differences([1.0, 2.0, 3.0]), np.zeros(5))

    def test_companion_eigenvalues_of_ar1(self):
        eigenvalues = companion_eigenvalues([0.7])
        np.testing.assert_allclose(eigenvalues, [0.7])

    def test_companion_eigenvalues_match_characteristic_roots(self):
        phi = np.array([0.5, 0.3])
        expected = np.roots([1.0, -0.5, -0.3])
        np.testing.assert_allclose(
            np.sort(companion_eigenvalues(phi).real), np.sort(expected.real), atol=1e-12
        )

    def test_stabilise_leaves_stable_coefficients_unchanged(self):
        np.testing.assert_array_equal(stabilise(self.TRUE_PHI), self.TRUE_PHI)

    def test_stabilise_brings_unstable_fit_inside_the_limit(self):
        unstable = np.array([1.4, 0.2, 0.0, 0.0, 0.0])
        self.assertGreater(np.max(np.abs(companion_eigenvalues(unstable))), 1.0)
        fixed = stabilise(unstable)
        self.assertLess(np.max(np.abs(companion_eigenvalues(fixed))), 0.98)
        # uniform rescaling keeps the direction of the coefficient vector
        self.assertAlmostEqual(fixed[1] / fixed[0], unstable[1] / unstable[0])

    def test_forecast_stays_bounded_on_an_accelerating_series(self):
        """Regression test: without the correction this forecast diverges."""
        history = 2.0 + 0.01 * 1.05 ** np.arange(150)
        raw = fit_ar_on_differences(history)
        self.assertGreater(np.max(np.abs(companion_eigenvalues(raw))), 0.98)
        path = arima_forecast_path(history, 72)
        self.assertTrue(np.all(np.isfinite(path)))
        last_step = history[-1] - history[-2]
        self.assertLess(abs(path[-1] - history[-1]), 60 * abs(last_step))

    def test_path_length_and_first_step(self):
        series = simulate_ar_differences(self.TRUE_PHI, 400, seed=2)
        path = arima_forecast_path(series, 36)
        self.assertEqual(path.shape, (36,))
        shorter = arima_forecast_path(series, 12)
        np.testing.assert_allclose(path[:12], shorter)

    def test_forecast_uses_only_the_sliding_window(self):
        series = simulate_ar_differences(self.TRUE_PHI, 600, seed=3)
        changed = series.copy()
        changed[:200] += 5.0  # older than the 288-step window
        np.testing.assert_allclose(
            arima_forecast_path(series, 12), arima_forecast_path(changed, 12)
        )

    def test_arima_beats_persistence_on_autocorrelated_differences(self):
        series = simulate_ar_differences(np.array([0.8, 0.0, 0.0, 0.0, 0.0]), 1500, seed=4)
        errors_arima, errors_persistence = [], []
        for i in range(400, 1490, 7):
            actual = series[i + 3]
            errors_arima.append(abs(arima_forecast_path(series[: i + 1], 3)[-1] - actual))
            errors_persistence.append(abs(series[i] - actual))
        self.assertLess(np.mean(errors_arima), 0.8 * np.mean(errors_persistence))


if __name__ == "__main__":
    unittest.main()
