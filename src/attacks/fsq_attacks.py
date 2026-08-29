"""M1 / M2 attacks against FSQ (paper attack names).

M1:     supervised task loss through the FSQ reconstruction.
M2-sat: geometric margin attack, saturating once a scalar dimension flips its bin.
M2-unb: same objective, unbounded.
M3_alpha03: non-paper alpha=0.3 blend of the M2 margin with M1's BCE loss.

FSQ has no learned codebook, so the "margin" is the distance from each scalar
latent dimension to its nearest rounding boundary. `margin_loss_anchored`
anchors the target bin to the clean sample once, before the PGD loop, rather
than recomputing the nearest bin from the perturbed latent every step -- this
keeps the attack pushing in one consistent direction as the perturbation
crosses a boundary, matching how the VQ-16 / multi-codebook margin losses
anchor to a fixed clean assignment.

`make_attacks_anchored` returns the four attacks above; each run script selects
the subset it needs (Kermany uses M1 / M2-sat / M2-unb; APTOS and ISIC add
M3_alpha03).

`logit_fn` is any callable image[B,3,H,W] in [-1,1] -> [B] classifier logit.
"""
import torch
import torch.nn.functional as F

from src.attacks.pgd import run_pgd
from src.tokenizers.fsq import fsq_bound


@torch.no_grad()
def clean_anchor(tok, x: torch.Tensor, device: str) -> torch.Tensor:
    """[B,N,5] fixed clean-bin anchor, computed once from the clean image."""
    z = tok.encode_prequant(x.to(device))
    return fsq_bound(z).round()


def margin_loss_anchored(z, anchor: torch.Tensor, saturate: bool):
    """Distance to the boundary of the FIXED clean anchor bin (not the
    current bin) -- keeps the attack pushing in one consistent direction."""
    u = fsq_bound(z)
    m_tok = (0.5 - (u - anchor).abs()).min(dim=-1).values
    return (m_tok.clamp(min=0.0) if saturate else m_tok).mean()


def make_attacks_anchored(tok, logit_fn, device: str, steps: int, lr: float):
    def M1(x, y, eps):
        y = y.float().to(device)
        return run_pgd(
            x, lambda xa: -F.binary_cross_entropy_with_logits(logit_fn(tok.reconstruct(xa)), y),
            eps, steps, lr, device,
        )

    def M3(x, y, eps, alpha: float, saturate: bool):
        y = y.float().to(device)
        anchor = clean_anchor(tok, x, device)

        def loss_fn(xa):
            geo = margin_loss_anchored(tok.encode_prequant(xa), anchor, saturate)
            if alpha >= 1.0:
                return geo
            bce = -F.binary_cross_entropy_with_logits(logit_fn(tok.reconstruct(xa)), y)
            return alpha * geo + (1 - alpha) * bce

        return run_pgd(x, loss_fn, eps, steps, lr, device)

    return {
        "M1": lambda x, y, e: M1(x, y, e),
        "M3_alpha03": lambda x, y, e: M3(x, y, e, alpha=0.3, saturate=False),
        "M2-sat": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=True),
        "M2-unb": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=False),
    }
