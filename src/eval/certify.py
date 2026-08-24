"""Clopper-Pearson lower confidence bound -- generic, not specific to any
model or defense. Used for randomized-smoothing certification (Cohen,
Rosenfeld, Kolter 2019): given N i.i.d. samples of a randomized classifier's
prediction and the count that agreed with the majority vote, this gives a
lower bound on the TRUE probability of that vote at the given confidence level.
"""
from scipy.stats import beta


def clopper_pearson_lower(n_success: int, n_total: int, alpha: float = 0.001) -> float:
    if n_success == 0:
        return 0.0
    return float(beta.ppf(alpha, n_success, n_total - n_success + 1))
