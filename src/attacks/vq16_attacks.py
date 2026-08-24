"""M0 / M1 / M3 attacks against the VQ-16 pipeline.

M0: raw image -> classifier, tokenizer-free baseline.
M1: image -> VQ-16 reconstruction -> classifier (attacks through reconstruction).
M3: geometric margin attack on the pre-quantization latent (pushes the nearest
    codebook entry to flip), optionally mixed with the M1 BCE loss via alpha.

`logit_fn` is any callable image[B,3,H,W] in [-1,1] -> [B] classifier logit,
so the same attack code works for DenseNet121 (Kermany) and the EfficientNet-B5
wrapper (APTOS) without caring about their different preprocessing.
"""
import torch
import torch.nn.functional as F

from src.attacks.pgd import run_pgd


def margin_loss(tok, z: torch.Tensor, k1: torch.Tensor, saturate: bool) -> torch.Tensor:
    """Margin between the clean code k1 and its nearest rival, per token.
    <0 = still assigned to k1, >=0 = flipped."""
    E = tok.codebook()
    d = z.pow(2).sum(-1, keepdim=True) - 2 * z @ E.T + E.pow(2).sum(-1)
    d_cur = d.gather(-1, k1.unsqueeze(-1)).squeeze(-1)
    vals, idxs = d.topk(2, dim=-1, largest=False)
    d_riv = torch.where(idxs[..., 0] == k1, vals[..., 1], vals[..., 0])
    m = d_cur - d_riv
    return -(m.clamp(max=0.0) if saturate else m).mean()


def make_attacks(tok, logit_fn, device: str, steps: int, lr: float):
    def M0(x, y, eps):
        y = y.float().to(device)
        return run_pgd(
            x, lambda xa: -F.binary_cross_entropy_with_logits(logit_fn(xa), y),
            eps, steps, lr, device,
        )

    def M1(x, y, eps):
        y = y.float().to(device)

        def loss_fn(xa):
            rec = tok.reconstruct(xa)
            return -F.binary_cross_entropy_with_logits(logit_fn(rec), y)

        return run_pgd(x, loss_fn, eps, steps, lr, device)

    def M3(x, y, eps, alpha: float, saturate: bool):
        y = y.float().to(device)
        with torch.no_grad():
            k1 = tok.clean_top1(tok.encode_prequant(x.to(device)))

        def loss_fn(xa):
            geo = margin_loss(tok, tok.encode_prequant(xa), k1, saturate)
            if alpha >= 1.0:
                return geo
            bce = -F.binary_cross_entropy_with_logits(logit_fn(tok.reconstruct(xa)), y)
            return alpha * geo + (1 - alpha) * bce

        return run_pgd(x, loss_fn, eps, steps, lr, device)

    return {
        "M0": lambda x, y, e: M0(x, y, e),
        "M1": lambda x, y, e: M1(x, y, e),
        "M3_alpha03": lambda x, y, e: M3(x, y, e, alpha=0.3, saturate=False),
        "M3_geo_sat": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=True),
        "M3_geo_unb": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=False),
    }
