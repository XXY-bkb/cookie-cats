"""stats_utils 的单元测试（只用标准库 unittest，不装 pytest 也能跑）。

运行：python -m unittest discover -s tests -v

测试的意义：把"什么算对"写成可执行的断言。比如两比例检验，
我们手算一组数字，如果哪天有人改坏了公式，测试立刻失败。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_utils import (  # noqa: E402
    achieved_power,
    bonferroni,
    bootstrap_diff,
    cohens_h,
    diff_ci_newcombe,
    min_detectable_delta,
    required_sample_size,
    srm_check,
    two_proportion_test,
    wilson_interval,
)


class TestSRM(unittest.TestCase):
    def test_balanced_passes(self):
        r = srm_check([5000, 5000])
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.p_value, 1.0, places=6)

    def test_imbalanced_fails(self):
        # 44700 vs 45489：卡方 ≈ 6.9，p ≈ 0.0086
        r = srm_check([44700, 45489])
        self.assertLess(r.p_value, 0.05)
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.chi2, 6.9024, places=3)

    def test_custom_ratio(self):
        r = srm_check([100, 200], expected_ratio=[0.5, 0.5])
        self.assertLess(r.p_value, 0.001)


class TestProportionTest(unittest.TestCase):
    def setUp(self):
        # 构造一组"好算"的数字：40% vs 30%，各 1000 人
        self.r = two_proportion_test(400, 1000, 300, 1000)

    def test_rates_and_diff(self):
        self.assertAlmostEqual(self.r.p_control, 0.40)
        self.assertAlmostEqual(self.r.p_treatment, 0.30)
        self.assertAlmostEqual(self.r.diff, 0.10, places=12)

    def test_ci_contains_point_estimate(self):
        self.assertLess(self.r.ci_low, self.r.diff)
        self.assertGreater(self.r.ci_high, self.r.diff)

    def test_se_formula(self):
        expected = math.sqrt(0.4 * 0.6 / 1000 + 0.3 * 0.7 / 1000)
        self.assertAlmostEqual(self.r.se, expected, places=12)

    def test_ci_is_wider_than_2se(self):
        # 95% 区间 = 1.96 SE，这里验证实现没把 z 值写错成 1 或 3
        self.assertAlmostEqual(self.r.ci_high - self.r.diff, 1.959964 * self.r.se, places=3)

    def test_significant_when_far(self):
        self.assertTrue(self.r.significant)
        self.assertTrue(self.r.ci_excludes_zero)

    def test_not_significant_when_close(self):
        r = two_proportion_test(400, 1000, 395, 1000)
        self.assertFalse(r.significant)

    def test_relative_lift(self):
        self.assertAlmostEqual(self.r.rel_lift, 0.10 / 0.30, places=12)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            two_proportion_test(5, 0, 3, 10)
        with self.assertRaises(ValueError):
            two_proportion_test(11, 10, 3, 10)


class TestIntervals(unittest.TestCase):
    def test_wilson_inside_unit_interval(self):
        lo, hi = wilson_interval(0, 100)
        self.assertGreaterEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_wilson_centered_near_p(self):
        lo, hi = wilson_interval(500, 1000)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual((lo + hi) / 2, 0.5, places=3)

    def test_newcombe_matches_wald_in_mid_range(self):
        # 比例不极端时，两种方法结论应一致（数值接近但有差异，只验证同向）
        lo, hi = diff_ci_newcombe(400, 1000, 300, 1000)
        wald = two_proportion_test(400, 1000, 300, 1000)
        self.assertAlmostEqual(lo, wald.ci_low, delta=0.01)
        self.assertAlmostEqual(hi, wald.ci_high, delta=0.01)


class TestBootstrap(unittest.TestCase):
    def test_median_diff_of_shifted_samples(self):
        a = list(range(1000))  # 中位数 499.5
        b = list(range(100, 1100))  # 中位数 599.5
        r = bootstrap_diff(a, b, n_boot=500, seed=0)
        self.assertAlmostEqual(r.observed, -100.0, places=6)
        self.assertLess(r.ci_high, 0)  # 差值稳定为负

    def test_percentile_method_brackets_observed(self):
        import random

        random.seed(1)
        a = [random.gauss(10, 1) for _ in range(500)]
        b = [random.gauss(10, 1) for _ in range(500)]
        r = bootstrap_diff(a, b, n_boot=400, seed=7)
        self.assertLess(r.ci_low, r.observed)
        self.assertGreater(r.ci_high, r.observed)

    def test_rejects_small_n_boot(self):
        with self.assertRaises(ValueError):
            bootstrap_diff([1, 2, 3], [4, 5, 6], n_boot=10)


class TestPower(unittest.TestCase):
    def test_cohens_h_sign(self):
        self.assertGreater(cohens_h(0.45, 0.40), 0)
        self.assertAlmostEqual(cohens_h(0.4, 0.4), 0.0)

    def test_required_sample_grows_as_delta_shrinks(self):
        n_big = required_sample_size(0.18, 0.02)
        n_small = required_sample_size(0.18, 0.01)
        self.assertLess(n_big, n_small)

    def test_roundtrip_power(self):
        # 用 solve_power 算出的 n，反算功效应该刚好 80%
        n = required_sample_size(0.18, 0.01, power=0.8)
        self.assertAlmostEqual(achieved_power(0.18, 0.01, n), 0.8, places=2)

    def test_mde_inverts_power(self):
        mde = min_detectable_delta(0.18, 44700)
        self.assertAlmostEqual(achieved_power(0.18, mde, 44700), 0.8, places=2)

    def test_mde_for_real_data(self):
        # 真实实验规模下，80% 功效的 MDE 约 0.73pp
        self.assertAlmostEqual(min_detectable_delta(0.182, 44700) * 100, 0.73, places=1)


class TestBonferroni(unittest.TestCase):
    def test_threshold(self):
        # 0.05/3 ≈ 0.0167：0.01 和 0.015 过线，0.2 不过线
        r = bonferroni([0.01, 0.015, 0.2])
        self.assertAlmostEqual(r.alpha_adjusted, 0.05 / 3)
        self.assertEqual(r.reject, (True, True, False))

    def test_edge_case_exactly_at_threshold(self):
        r = bonferroni([0.025, 0.0251])
        self.assertEqual(r.reject, (False, False))  # 严格小于才拒绝

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bonferroni([])


if __name__ == "__main__":
    unittest.main()
