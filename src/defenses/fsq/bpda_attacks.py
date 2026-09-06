"""Adaptive-attack constructors for FSQ: same role and same three attack
families as every other tokenizer's bpda_attacks.py, adapted to FSQ's
per-dimension scalar quantization. See src/defenses/vq16/bpda_attacks.py's
docstring for why the pure geometric attack is blind to decoder-side defenses
and why the blended geo+BCE attack can see them.
"""
import torch
import torch.nn.functional as F

from src.defenses.common.bpda import batched_pgd_attack, straight_through_quantize
from src.defenses.fsq.margin import clean_anchor, decode_only, fsq_margin, half_width
from src.defenses.fsq.sqs import dispatch_quantize
from src.models.densenet_classifier import logit as densenet_logit
from src.tokenizers.fsq import fsq_bound


def _plain_quantize(z: torch.Tensor) -> torch.Tensor:
    """Real, deterministic per-dim round-to-nearest-level (no noise, no voting),
    the undefended quantize step, for the "no SQS" adaptive-attack checks."""
    levels = fsq_bound(z).round()
    return levels / half_width(z.device)


def sqs_reconstruct_bpda(tok, x, sqs_name: str, sqs_params: dict):
    z = tok.encode_prequant(x)
    quant_t = straight_through_quantize(z, lambda zz: dispatch_quantize(sqs_name, tok, zz, sqs_params))
    return decode_only(tok.net, quant_t)


def plain_reconstruct_bpda(tok, x):
    z = tok.encode_prequant(x)
    quant_t = straight_through_quantize(z, _plain_quantize)
    return decode_only(tok.net, quant_t)


def bpda_sqs_attack(tok, densenet, x, y, eps, sqs_name, sqs_params, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        def loss_fn(xa):
            rec = sqs_reconstruct_bpda(tok, xa, sqs_name, sqs_params)
            return -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)


def bpda_plain_attack(tok, densenet, x, y, eps, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        def loss_fn(xa):
            rec = plain_reconstruct_bpda(tok, xa)
            return -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)


def geo_adaptive_attack(tok, x, eps, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        anchor = clean_anchor(tok, xb)

        def loss_fn(xa):
            return -fsq_margin(tok, xa, anchor, saturate=True)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=None)


def geo_bpda_attack(tok, densenet, x, y, eps, alpha, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        anchor = clean_anchor(tok, xb)

        def loss_fn(xa):
            geo_loss = -fsq_margin(tok, xa, anchor, saturate=True)
            rec = plain_reconstruct_bpda(tok, xa)
            bce_loss = -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
            return alpha * geo_loss + (1 - alpha) * bce_loss
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)
