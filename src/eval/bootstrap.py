"""Paired bootstrap confidence interval on an AUC delta between two conditions
scored on the same images. Tells you whether an AUC shift is real or just noise
at this sample size, which a point-estimate delta alone cannot.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(y, logits):
    return roc_auc_score(y, 1 / (1 + np.exp(-logits)))


def paired_auc_bootstrap_ci(y_true, scores_a, scores_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Delta is scores_b - scores_a, paired: the same resampled indices are
    applied to every condition on each iteration, so shared variance cancels.
    Returns (point_estimate_delta, ci_low, ci_high, significant). `significant`
    is False whenever the 95% CI contains 0, meaning the delta is not
    distinguishable from noise at this n regardless of its sign.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    point_delta = _auc(y_true, scores_b) - _auc(y_true, scores_a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():  # degenerate resample (single class), AUC undefined, skip
            continue
        deltas.append(_auc(yb, scores_b[idx]) - _auc(yb, scores_a[idx]))
    deltas = np.array(deltas)
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    significant = not (ci_low <= 0 <= ci_high)
    return point_delta, ci_low, ci_high, significant
