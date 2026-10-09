import math
import unittest

import numpy as np

from coldchain_monitor.warning import (
    average_precision,
    event_metrics,
    excursion_labels,
    roc_auc,
    threshold_at_false_alarm_rate,
)


class LabelsTest(unittest.TestCase):
    def test_hand_worked_example(self):
        #            0    1    2    3    4    5    6
        series = [3.0, 3.0, 3.5, 4.5, 3.0, 3.0, 3.0]
        labels, valid = excursion_labels(series, steps=2)
        np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(valid, [True, True, True, False, True, False, False])

    def test_reading_equal_to_limit_is_not_an_excursion(self):
        labels, _ = excursion_labels([3.0, 4.0, 4.0, 3.0], steps=1)
        self.assertEqual(labels.sum(), 0)

    def test_series_shorter_than_horizon_has_no_valid_origin(self):
        _, valid = excursion_labels([3.0, 5.0], steps=5)
        self.assertFalse(valid.any())

    def test_rejects_zero_horizon(self):
        with self.assertRaises(ValueError):
            excursion_labels([3.0, 3.0], steps=0)


class RankingMetricsTest(unittest.TestCase):
    def test_perfect_and_inverted_ranking(self):
        labels = [0, 0, 1, 1]
        self.assertEqual(roc_auc([0.1, 0.2, 0.8, 0.9], labels), 1.0)
        self.assertEqual(roc_auc([0.9, 0.8, 0.2, 0.1], labels), 0.0)
        self.assertEqual(average_precision([0.1, 0.2, 0.8, 0.9], labels), 1.0)

    def test_textbook_value(self):
        # pairs (positive, negative): 0.8>0.1, 0.8>0.4, 0.35>0.1, 0.35<0.4 -> 3/4
        self.assertAlmostEqual(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
        # precision 1 at recall 1/2, precision 2/3 at recall 1
        self.assertAlmostEqual(average_precision([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 5 / 6)

    def test_constant_score_is_chance_level(self):
        labels = [0, 1, 0, 1, 0, 0]
        self.assertAlmostEqual(roc_auc(np.zeros(6), labels), 0.5)
        self.assertAlmostEqual(average_precision(np.zeros(6), labels), 2 / 6)

    def test_auc_is_invariant_to_monotone_transformations(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=500)
        labels = (scores + rng.normal(size=500) > 0.5).astype(int)
        self.assertAlmostEqual(roc_auc(scores, labels), roc_auc(np.exp(scores), labels))

    def test_single_class_gives_nan(self):
        self.assertTrue(math.isnan(roc_auc([0.1, 0.2], [0, 0])))
        self.assertTrue(math.isnan(average_precision([0.1, 0.2], [0, 0])))

    def test_threshold_respects_false_alarm_rate(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=4000)
        labels = np.zeros(4000, dtype=int)
        labels[:200] = 1
        threshold = threshold_at_false_alarm_rate(scores, labels, 0.05)
        rate = np.mean(scores[labels == 0] > threshold)
        self.assertLessEqual(rate, 0.05)
        self.assertGreater(rate, 0.04)

    def test_threshold_needs_negatives_and_a_valid_rate(self):
        with self.assertRaises(ValueError):
            threshold_at_false_alarm_rate([0.1, 0.2], [1, 1])
        with self.assertRaises(ValueError):
            threshold_at_false_alarm_rate([0.1, 0.2], [0, 1], rate=1.5)


class EventMetricsTest(unittest.TestCase):
    def setUp(self):
        self.series = np.full(60, 3.0)
        self.series[30:36] = 5.0  # one excursion, onset at index 30

    def test_alarm_before_onset_is_a_detection_with_lead_time(self):
        alarms = np.zeros(60, dtype=bool)
        alarms[24:30] = True  # first alarm 6 steps = 30 minutes before the onset
        result = event_metrics(self.series, alarms, steps=12)
        self.assertEqual((result.onsets, result.detected), (1, 1))
        self.assertEqual(result.lead_minutes, (30.0,))
        self.assertEqual(result.false_alarm_episodes, 0)
        self.assertEqual(result.detection_rate, 1.0)

    def test_no_alarm_means_missed_onset(self):
        result = event_metrics(self.series, np.zeros(60, dtype=bool), steps=12)
        self.assertEqual((result.onsets, result.detected), (1, 0))
        self.assertTrue(math.isnan(result.median_lead_minutes))

    def test_alarm_far_from_any_excursion_is_a_false_episode(self):
        alarms = np.zeros(60, dtype=bool)
        alarms[5:8] = True
        alarms[10] = True
        result = event_metrics(self.series, alarms, steps=12)
        self.assertEqual(result.detected, 0)
        self.assertEqual(result.false_alarm_episodes, 2)
        self.assertGreater(result.false_alarms_per_day, 0.0)

    def test_alarm_too_early_does_not_count_as_detection(self):
        alarms = np.zeros(60, dtype=bool)
        alarms[10] = True  # 20 steps before the onset, horizon is 12
        self.assertEqual(event_metrics(self.series, alarms, steps=12).detected, 0)

    def test_results_can_be_pooled(self):
        alarms = np.zeros(60, dtype=bool)
        alarms[28] = True
        hit = event_metrics(self.series, alarms, steps=12)
        miss = event_metrics(self.series, np.zeros(60, dtype=bool), steps=12)
        pooled = hit + miss
        self.assertEqual((pooled.onsets, pooled.detected), (2, 1))
        self.assertEqual(pooled.detection_rate, 0.5)
        self.assertAlmostEqual(pooled.monitored_days, 2 * hit.monitored_days)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            event_metrics(self.series, np.zeros(10, dtype=bool), steps=12)


if __name__ == "__main__":
    unittest.main()
