"""Randomized-smoothing certified robustness (Cohen, Rosenfeld, Kolter 2019),
applied to whichever SQS variant won the sweep (src/defenses/vq16/sqs.py).

Every recovery number elsewhere is empirical -- true only against the
specific attacks tested. A certified bound is a stronger, attack-independent
claim: run N independent single-pass (K=1) noisy defended forward passes
through the full pipeline (tokenizer+SQS -> decode -> classify), take the
majority-vote predicted class, and compute a Clopper-Pearson lower confidence
bound p_A_lower on the TRUE probability that pass returns that class. If
p_A_lower > 0.5, the prediction is certified robust to ANY perturbation
(in the space noise is injected -- the pre-quantization latent) within
radius r = sigma * Phi^-1(p_A_lower) -- guaranteed, not just observed against
whatever's in the attack cache.

Caveat: Cohen et al.'s radius formula assumes ONE fixed, global noise scale.
It is NOT valid when sigma varies per token, which is exactly what
margin-adaptive noise does. p_A_lower and the certified boolean stay
meaningful (pure vote agreement, distribution-agnostic) for margin_adaptive,
but the quantitative radius would be mathematically unsound, so it's reported
as NaN rather than a number that overstates what's actually proven.
"""
import numpy as np
import torch
from scipy.stats import norm

from src.defenses.vq16.sqs import dispatch_reconstruct
from src.eval.certify import clopper_pearson_lower
from src.models.densenet_classifier import logit as densenet_logit

RADIUS_VALID_VARIANTS = ("flat", "spatial")


def single_pass_params(sqs_name: str, sqs_params: dict) -> dict:
    """Same params as the K-voted winner, but forced to K=1 -- certification
    needs N independent single noisy draws, not N K-voted draws."""
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
