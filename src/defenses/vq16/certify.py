"""Randomized-smoothing certified robustness (Cohen, Rosenfeld, Kolter 2019),
applied to whichever SQS variant won the sweep (src/defenses/vq16/sqs.py).

Every recovery number elsewhere is empirical, true only against the specific
attacks tested. A certified bound is a stronger, attack-independent claim: run N
independent single-pass (K=1) noisy defended forward passes through the full
pipeline (tokenizer and SQS, then decode, then classify), take the majority-vote
predicted class, and compute a Clopper-Pearson lower confidence bound p_A_lower
on the true probability that a pass returns that class. If p_A_lower > 0.5, the
prediction is certified robust to any perturbation of the pre-quantization latent
(the space the noise is injected in) within radius r = sigma * Phi^-1(p_A_lower).
That is a guarantee, not an observation against whatever is in the attack cache.

Caveat: Cohen et al.'s radius formula assumes one fixed, global noise scale. It
is not valid when sigma varies per token, which is what margin-adaptive noise
does. p_A_lower and the certified boolean stay meaningful for margin_adaptive
(they are pure vote agreement, distribution-agnostic), but the quantitative
radius would be unsound, so it is reported as NaN rather than a number that
overstates what is proven.
"""
import numpy as np
import torch
from scipy.stats import norm

from src.defenses.vq16.sqs import dispatch_reconstruct
from src.eval.certify import clopper_pearson_lower
from src.models.densenet_classifier import logit as densenet_logit

RADIUS_VALID_VARIANTS = ("flat", "spatial")


def single_pass_params(sqs_name: str, sqs_params: dict) -> dict:
    """Same params as the K-voted winner but forced to K=1. Certification needs N
    independent single noisy draws, not N K-voted draws."""
    p = dict(sqs_params)
    p["K"] = 1
    return p


@torch.no_grad()
def certify_sqs(model, densenet, x: torch.Tensor, sqs_name: str, sqs_params: dict,
                 N: int = 100, alpha: float = 0.001):
    """x: (1,3,H,W). Returns (predicted_class, p_A_lower, certified_radius).
    certified_radius is NaN when sqs_name == "margin_adaptive"."""
    single_params = single_pass_params(sqs_name, sqs_params)
    radius_valid = sqs_name in RADIUS_VALID_VARIANTS
    cert_sigma = sqs_params.get("sigma")  # only meaningful for flat/spatial

    preds = []
    for _ in range(N):
        rec = dispatch_reconstruct(sqs_name, model, x, single_params)
        preds.append(int(densenet_logit(densenet, rec).item() > 0))
    preds = np.array(preds)
    c_A = int(preds.mean() > 0.5)
    n_A = int((preds == c_A).sum())
    p_A_lower = clopper_pearson_lower(n_A, N, alpha)
    if p_A_lower <= 0.5 or not radius_valid:
        return c_A, p_A_lower, (0.0 if p_A_lower <= 0.5 else float("nan"))
    return c_A, p_A_lower, cert_sigma * float(norm.ppf(p_A_lower))
