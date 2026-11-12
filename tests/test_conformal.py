import math

import numpy as np

from src.conformal import all_scores, calibrate, evaluate, prediction_sets, true_scores, wilson


def synth(rng, n, K=5):
    gold = rng.integers(0, K, n)
    logits = rng.normal(0, 2.0, (n, K))
    logits[np.arange(n), gold] += rng.normal(2.0, 2.0, n)
    P = np.exp(logits - logits.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True), gold


def test_quantile_index():
    s = np.arange(1, 501) / 500.0
    assert calibrate(s, 0.10) == s[450]  # ceil(501 * 0.9) = 451st smallest
    assert calibrate(np.arange(5.0), 0.10) == math.inf  # too few points: trivial guarantee


def test_coverage_is_about_90_percent_on_average():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(300):
        Pc, gc = synth(rng, 500); Pt, gt = synth(rng, 500)
        q = calibrate(true_scores(Pc, gc), 0.10)
        covs.append(evaluate(Pt, gt, q)["coverage"])
    assert 0.895 <= np.mean(covs) <= 0.915  # guarantee: >= 0.90, and tight at n=500


def test_saturated_probabilities_do_not_tie():
    P = np.array([[1.0, 1e-12, 1e-13, 1e-14, 1e-15], [1.0, 1e-15, 1e-15, 1e-15, 2e-15]])
    s = true_scores(P, [0, 0])
    assert s[0] > s[1] > 0  # 1 - p would round both to 0


def test_sets_and_metrics_on_a_known_example():
    P = np.array([[0.90, 0.05, 0.03, 0.01, 0.01],   # confident, right
                  [0.50, 0.45, 0.03, 0.01, 0.01],   # unsure, right label is option 1
                  [0.97, 0.01, 0.01, 0.005, 0.005]])  # confident, wrong (gold = 2)
    g = np.array([0, 1, 2])
    q = 0.55  # option j is in the set when sum of others <= 0.55, i.e. p_j >= 0.45
    sets = prediction_sets(P, q)
    assert sets.sum(1).tolist() == [1, 2, 1]
    m = evaluate(P, g, q)
    assert abs(m["coverage"] - 2 / 3) < 1e-12 and abs(m["mean_size"] - 4 / 3) < 1e-12
    assert abs(m["singleton_rate"] - 2 / 3) < 1e-12 and abs(m["singleton_wrong"] - 0.5) < 1e-12
    assert m["empty_rate"] == 0.0


def test_empty_sets_are_counted():
    P = np.array([[0.6, 0.4, 0.0, 0.0, 0.0]])
    m = evaluate(P, [0], qhat=0.1)  # nothing has p >= 0.9
    assert m["empty_rate"] == 1.0 and m["coverage"] == 0.0


def test_wilson_interval():
    lo, hi = wilson(450, 500)
    assert 0.87 < lo < 0.90 < hi < 0.93
    assert all_scores(np.eye(5)).shape == (5, 5)
