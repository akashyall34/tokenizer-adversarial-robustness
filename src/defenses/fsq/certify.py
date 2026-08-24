"""Randomized-smoothing certified robustness for FSQ's winning SQS variant --
same role and same margin-adaptive-radius caveat as every other tokenizer's
certify.py.
"""
import numpy as np
import torch
from scipy.stats import norm

from src.defenses.fsq.sqs import dispatch_reconstruct
from src.eval.certify import clopper_pearson_lower
from src.models.densenet_classifier import logit as densenet_logit

RADIUS_VALID_VARIANTS = ("flat", "spatial")


def single_pass_params(sqs_params: dict) -> dict:
    p = dict(sqs_params)
    p["K"] = 1
    return p


@torch.no_grad()
def certify_sqs(tok, densenet, x: torch.Tensor, sqs_name: str, sqs_params: dict,
                 N: int = 100, alpha: float = 0.001):
    """x: (1,3,H,W). Returns (predicted_class, p_A_lower, certified_radius).
    certified_radius is NaN when sqs_name == "margin_adaptive"."""
    single_params = single_pass_params(sqs_params)
    radius_valid = sqs_name in RADIUS_VALID_VARIANTS
    cert_sigma = sqs_params.get("sigma")

    preds = []
    for _ in range(N):
        rec = dispatch_reconstruct(sqs_name, tok, x, single_params)
        preds.append(int(densenet_logit(densenet, rec).item() > 0))
    preds = np.array(preds)
    c_A = int(preds.mean() > 0.5)
    n_A = int((preds == c_A).sum())
    p_A_lower = clopper_pearson_lower(n_A, N, alpha)
    if p_A_lower <= 0.5 or not radius_valid:
        return c_A, p_A_lower, (0.0 if p_A_lower <= 0.5 else float("nan"))
    return c_A, p_A_lower, cert_sigma * float(norm.ppf(p_A_lower))
