"""Adaptive-attack constructors for MediTok/UniTok: same role and same three
attack families as src/defenses/vq16/bpda_attacks.py, adapted to the
multi-codebook cosine-similarity quantizer. See that module's docstring for why
the pure geometric attack is blind to decoder-side defenses and why the blended
geo+BCE attack can see them.
"""
import torch
import torch.nn.functional as F

from src.defenses.common.bpda import batched_pgd_attack, straight_through_quantize
from src.defenses.multicodebook.margin import decode_only, raw_margin
from src.defenses.multicodebook.sqs import dispatch_quantize
from src.models.densenet_classifier import logit as densenet_logit
from src.tokenizers.meditok_unitok import MultiCodebookTokenizer


def _plain_quantize(tok, z):
    """Real, deterministic per-codebook nearest-neighbor lookup (no noise, no
    voting), the undefended quantize step, for the "no SQS" adaptive-attack checks."""
    B, N, D = z.shape
    chunk_size = D // tok.num_codebooks
    chunks = z.split(chunk_size, dim=-1)
    code_chunks = []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f @ E.T)
        idx = d.argmin(-1)
        code_chunks.append(E[idx])
    return torch.cat(code_chunks, dim=-1)


def sqs_reconstruct_bpda(tok, x, sqs_name: str, sqs_params: dict):
    z = tok.encode_prequant(x)
    z_q = straight_through_quantize(z, lambda zz: dispatch_quantize(sqs_name, tok, zz, sqs_params))
    return decode_only(tok.net, z_q)


def plain_reconstruct_bpda(tok, x):
    z = tok.encode_prequant(x)
    z_q = straight_through_quantize(z, lambda zz: _plain_quantize(tok, zz))
    return decode_only(tok.net, z_q)


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
        z_clean = tok.encode_prequant(xb)
        k1_list = tok.clean_top1(z_clean)

        def loss_fn(xa):
            z = tok.encode_prequant(xa)
            return -raw_margin(tok, z, k1_list, saturate=True)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=None)


def geo_bpda_attack(tok, densenet, x, y, eps, alpha, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        z_clean = tok.encode_prequant(xb)
        k1_list = tok.clean_top1(z_clean)

        def loss_fn(xa):
            z = tok.encode_prequant(xa)
            geo_loss = -raw_margin(tok, z, k1_list, saturate=True)
            rec = plain_reconstruct_bpda(tok, xa)
            bce_loss = -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
            return alpha * geo_loss + (1 - alpha) * bce_loss
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)
