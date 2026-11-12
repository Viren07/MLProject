"""Split conformal prediction with the LAC score, plus the metrics we report.

Score of candidate option j = total probability of the OTHER options. This equals
1 - p_j but is computed as a sum of small positive numbers, so it stays precise when
the model is nearly certain (p close to 1). Computing 1 - p directly rounds to exactly
0 for saturated models, creating ties that distort coverage.
"""
from __future__ import annotations

import math

import numpy as np


def all_scores(probs) -> np.ndarray:
    """(n, K) probabilities -> (n, K) score of each candidate label."""
    P = np.asarray(probs, dtype=float)
    return np.stack([np.delete(P, j, axis=1).sum(axis=1) for j in range(P.shape[1])], axis=1)


def true_scores(probs, gold) -> np.ndarray:
    S = all_scores(probs)
    return S[np.arange(len(S)), np.asarray(gold)]


def calibrate(scores, alpha: float = 0.10) -> float:
    """Threshold = the ceil((n+1)(1-alpha))-th smallest calibration score (inf if n too small)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil(round((n + 1) * (1 - alpha), 9))
    return math.inf if k > n else float(s[k - 1])


def prediction_sets(probs, qhat: float) -> np.ndarray:
    """Boolean (n, K) matrix: option j is in the set when its score <= qhat."""
    return all_scores(probs) <= qhat


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def evaluate(probs, gold, qhat: float) -> dict:
    """Coverage splits into three parts: 1 - coverage = empty + multi-option misses + silent failures.
    Only silent failures (a single wrong answer) are unflagged errors; empty sets are abstentions."""
    P, g = np.asarray(probs, dtype=float), np.asarray(gold)
    n = len(g)
    sets = prediction_sets(P, qhat)
    size = sets.sum(axis=1)
    covered = sets[np.arange(n), g]
    single = size == 1
    pred = P.argmax(axis=1)
    lo, hi = wilson(int(covered.sum()), n)
    sw = float((sets[single].argmax(axis=1) != g[single]).mean()) if single.any() else math.nan
    return {"n": n, "coverage": float(covered.mean()), "cov_lo": lo, "cov_hi": hi,
            "mean_size": float(size.mean()), "singleton_rate": float(single.mean()),
            "empty_rate": float((size == 0).mean()), "multi_rate": float((size >= 2).mean()),
            "flag_rate": float((size != 1).mean()),  # empty or multi-option: the model does not commit
            "silent_failure": float((single & ~covered).mean()),  # committed to ONE answer and was wrong
            "accuracy": float((pred == g).mean()), "singleton_wrong": sw}
