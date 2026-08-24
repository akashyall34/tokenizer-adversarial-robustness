"""AUC / attack-success-rate metrics shared across all attack-table evaluations."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y: np.ndarray, logits: np.ndarray) -> float:
    return roc_auc_score(y, 1 / (1 + np.exp(-np.clip(logits, -30, 30))))


def asr(clean_logits: np.ndarray, adv_logits: np.ndarray, y: np.ndarray) -> float:
    """% of originally-correct predictions flipped to incorrect by the attack."""
    correct = (clean_logits > 0).astype(int) == y
    flipped = (adv_logits > 0).astype(int) != y
    return 100 * (flipped & correct).sum() / max(correct.sum(), 1)
