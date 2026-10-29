"""The statistics the paper's contrasts rest on: sign-flip p by convolution against full enumeration, Holm,
Wilson and Newcombe intervals, the seed-clustered interval."""
import numpy as np
import pytest

from analysis import matrix_stats as ms


def test_sign_flip_convolution_equals_enumeration():
    rng = np.random.default_rng(3)
    for _ in range(5):
        v = rng.integers(-6, 7, size=10) / 24.0
        assert ms.sign_flip_exact_conv(v, scale=24) == pytest.approx(ms.sign_flip_exact(v))


def test_sign_flip_all_zero_is_one():
    assert ms.sign_flip_exact(np.zeros(8)) == 1.0
    assert ms.sign_flip_exact_conv(np.zeros(8)) == 1.0


def test_sign_flip_dispatch_uses_exact_for_integer_multiples_and_monte_carlo_otherwise():
    assert ms.sign_flip_p([1 / 24, -2 / 24, 3 / 24, 1 / 24])["method"].startswith("exact_conv")
    mc = ms.sign_flip_p([0.1234567, -0.2345678, 0.3456789, 0.01234567])
    assert mc["method"].startswith("monte_carlo") and 0 < mc["p"] <= 1


def test_holm_adjustment():
    assert ms.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    # a missing p stays missing but still counts in the family size
    assert ms.holm([0.5, None, 0.01]) == [pytest.approx(1.0), None, pytest.approx(0.03)]


def test_wilson_interval():
    lo, hi = ms.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)


def test_newcombe_interval_contains_difference_and_is_antisymmetric():
    lo, hi = ms.newcombe10(8, 5, 2, 15)
    assert lo < (5 - 2) / 30 < hi
    lo2, hi2 = ms.newcombe10(8, 2, 5, 15)
    assert (lo2, hi2) == pytest.approx((-hi, -lo), abs=1e-9)


def test_cluster_interval_matches_t_interval():
    v = np.array([0.1, -0.05, 0.2, 0.0, 0.15, 0.05])
    r = ms.cluster_interval(v)
    half = 2.570581835636314 * v.std(ddof=1) / np.sqrt(len(v))
    assert r["estimate"] == pytest.approx(round(v.mean(), 4))
    assert r["ci95_t_seeds"] == pytest.approx([round(v.mean() - half, 4), round(v.mean() + half, 4)])


def test_no_between_seed_variation_is_not_estimable():
    r = ms.cluster_interval(np.full(5, 0.1))
    assert r["ci95_t_seeds"].startswith("not estimable")
