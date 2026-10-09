import unittest

import numpy as np

from coldchain_monitor.cascade import tier2_active
from coldchain_monitor.data import synthetic_log
from coldchain_monitor.features import FEATURE_NAMES, build_features


class FeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = synthetic_log(hours=24, seed=5, quantization=0.0).temperature
        cls.features = build_features(cls.series)

    def column(self, name):
        return self.features[:, FEATURE_NAMES.index(name)]

    def test_shape_and_names(self):
        self.assertEqual(self.features.shape, (self.series.size, 13))
        self.assertEqual(len(set(FEATURE_NAMES)), 13)

    def test_warmup_rows_are_nan_and_the_rest_finite(self):
        self.assertTrue(np.all(np.isnan(self.features[:36])))
        self.assertTrue(np.all(np.isfinite(self.features[36:])))

    def test_values_against_direct_computation(self):
        i = 150
        y = self.series
        self.assertEqual(self.column("value")[i], y[i])
        self.assertAlmostEqual(self.column("d1")[i], y[i] - y[i - 1])
        self.assertAlmostEqual(self.column("d36")[i], y[i] - y[i - 36])
        self.assertAlmostEqual(self.column("max12")[i], y[i - 11 : i + 1].max())
        self.assertAlmostEqual(self.column("min36")[i], y[i - 35 : i + 1].min())
        self.assertAlmostEqual(self.column("std36")[i], y[i - 35 : i + 1].std())
        self.assertAlmostEqual(self.column("mean12")[i], y[i - 11 : i + 1].mean())

    def test_no_look_ahead(self):
        """Changing the future must not change the features of the past."""
        changed = self.series.copy()
        changed[200:] += 10.0
        np.testing.assert_array_equal(build_features(changed)[:200], self.features[:200])

    def test_short_series_gives_only_nan(self):
        self.assertTrue(np.all(np.isnan(build_features(np.arange(10.0)))))

    def test_rejects_two_dimensional_input(self):
        with self.assertRaises(ValueError):
            build_features(np.zeros((4, 4)))


class CascadeTest(unittest.TestCase):
    def test_flat_signal_never_triggers(self):
        self.assertFalse(tier2_active(np.full(200, 3.0)).any())

    def test_jump_triggers_then_trigger_expires(self):
        signal = np.full(300, 3.0)
        signal[100:] = 5.0
        active = tier2_active(signal)
        self.assertFalse(active[:100].any())
        self.assertTrue(active[101])
        self.assertFalse(active[100 + 36 + 1 :].any())

    def test_higher_thresholds_trigger_less_often(self):
        signal = synthetic_log(hours=72, seed=6).temperature
        low = tier2_active(signal, 0.1, 0.1).mean()
        high = tier2_active(signal, 0.6, 0.8).mean()
        self.assertGreater(low, high)

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            tier2_active(np.zeros(50), tau0=-1.0)


if __name__ == "__main__":
    unittest.main()
