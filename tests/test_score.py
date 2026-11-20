"""Scoring contract: weights, bit order, pair estimates, and input validation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import (DISPLACEMENTS, estimate_observables, exact_observables,
                   score_observables, score_submission)


def submission():
    return {"temperature": 2.375,
            **{basis: [[0] * 16 for _ in range(128)] for basis in "XYZ"}}


class ScoreTests(unittest.TestCase):
    def test_exact_observables_have_zero_loss(self):
        for temperature in (0.01, 1.1875, 2.375, 4.0):
            means, pairs = exact_observables(temperature)
            self.assertEqual(score_observables(means, pairs, means, pairs)["score"], 0)

    def test_group_weights_are_equal_without_sqrt_or_half(self):
        zero_means, zero_pairs = np.zeros(3), np.zeros((3, 9))
        delta = 0.2
        mean_only = score_observables(np.full(3, delta), zero_pairs, zero_means, zero_pairs)
        pair_only = score_observables(zero_means, np.full((3, 9), delta), zero_means, zero_pairs)
        both = score_observables(np.full(3, delta), np.full((3, 9), delta), zero_means, zero_pairs)
        self.assertAlmostEqual(mean_only["score"], 27 * delta ** 2)
        self.assertAlmostEqual(pair_only["score"], 27 * delta ** 2)
        self.assertAlmostEqual(both["score"], 54 * delta ** 2)
        one_mean = score_observables([delta, 0, 0], zero_pairs, zero_means, zero_pairs)
        errors = zero_pairs.copy()
        errors[0, 0] = delta
        one_pair = score_observables(zero_means, errors, zero_means, zero_pairs)
        self.assertAlmostEqual(one_mean["score"] / one_pair["score"], 9)

    def test_averaging_precedes_squaring_and_keeps_directions(self):
        data = submission()
        # Vertical stripes: translation-averaged magnetization is zero, while
        # horizontal/vertical correlations have opposite signs. No C4 pooling.
        pattern = [q % 4 % 2 for q in range(16)]
        data["Z"] = [pattern.copy() for _ in range(128)]
        _, means, pairs = estimate_observables(data)
        np.testing.assert_array_equal(means, [1, 1, 0])
        np.testing.assert_array_equal(pairs[2], [(-1) ** dx for dx, dy in DISPLACEMENTS])
        self.assertEqual(pairs[2, DISPLACEMENTS.index((1, 0))], -1)
        self.assertEqual(pairs[2, DISPLACEMENTS.index((0, 1))], 1)

    def test_nine_displacements_and_translation_invariance(self):
        representatives = {min((dx, dy), ((-dx) % 4, (-dy) % 4))
                           for dx in range(4) for dy in range(4) if (dx, dy) != (0, 0)}
        self.assertEqual(set(DISPLACEMENTS), representatives)
        data = submission()
        rng = np.random.default_rng(17)
        for basis in "XYZ":
            data[basis] = rng.integers(0, 2, (128, 16)).tolist()
        _, means, pairs = estimate_observables(data)
        moved = {"temperature": data["temperature"]}
        permutation = [((q // 4 + 1) % 4) * 4 + (q % 4 + 2) % 4 for q in range(16)]
        for basis in "XYZ":
            moved[basis] = np.array(data[basis])[:, permutation].tolist()
        _, moved_means, moved_pairs = estimate_observables(moved)
        np.testing.assert_array_equal(means, moved_means)
        np.testing.assert_array_equal(pairs, moved_pairs)
        # Independent scalar implementation verifies wrapping and shot products.
        for axis, basis in enumerate("XYZ"):
            for index, (dx, dy) in enumerate(DISPLACEMENTS):
                total = 0
                for row in data[basis]:
                    for y in range(4):
                        for x in range(4):
                            i, j = x + 4*y, (x+dx) % 4 + 4*((y+dy) % 4)
                            total += (1-2*row[i]) * (1-2*row[j])
                self.assertEqual(pairs[axis, index], total / (128*16))

    def test_shot_products_not_product_of_means(self):
        data = submission()
        for basis in "XYZ":
            data[basis] = [[0] * 16 for _ in range(64)] + [[1] * 16 for _ in range(64)]
        _, means, pairs = estimate_observables(data)
        np.testing.assert_array_equal(means, 0)
        np.testing.assert_array_equal(pairs, 1)

    def test_score_matches_independent_all_plus_calculation(self):
        means, pairs = exact_observables(2.375)
        expected = 9 * sum((1 - float(v)) ** 2 for v in means.flat)
        expected += sum((1 - float(v)) ** 2 for v in pairs.flat)
        self.assertAlmostEqual(score_submission(submission())["score"], expected)
        self.assertAlmostEqual(means[0], 0.2714551814977822)
        np.testing.assert_array_equal(means[1:], 0)

    def test_invalid_submissions_are_rejected(self):
        for temperature in (True, 0, -1, float("nan"), "2.375", 3.0):
            data = submission()
            data["temperature"] = temperature
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                score_submission(data)
        for invalid in (2, -1, 0.5, "0", float("nan"), True):
            data = submission()
            data["Z"][0][0] = invalid
            with self.subTest(bit=invalid), self.assertRaises(ValueError):
                score_submission(data)
        for count in (0, 127, 129):
            data = submission()
            data["X"] = [[0] * 16 for _ in range(count)]
            with self.subTest(shots=count), self.assertRaises(ValueError):
                score_submission(data)
        data = submission()
        del data["Y"]
        with self.assertRaises(ValueError):
            score_submission(data)


if __name__ == "__main__":
    unittest.main()
