"""M1 / M2 attacks against the multi-codebook (cosine similarity) tokenizers,
MediTok and UniTok (paper attack names; M2-sat and M2-unb are the geometric
margin attack, saturating and unbounded). M3_alpha03 is an auxiliary alpha=0.3
blend not used in the paper. M0 (tokenizer-free) is shared across all tokenizers
and lives in src/attacks/vq16_attacks.py.

`logit_fn` is any callable mapping image[B,3,H,W] in [-1,1] to a [B] classifier
logit.
"""
import torch
import torch.nn.functional as F

from src.attacks.pgd import run_pgd


def margin_loss(tok, z: torch.Tensor, k1_list, saturate: bool) -> torch.Tensor:
    chunks = z.split(z.shape[-1] // tok.num_codebooks, dim=-1)
    margins = []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f @ E.T)
        k1 = k1_list[i]
        d_cur = d.gather(-1, k1.unsqueeze(-1)).squeeze(-1)
        vals, idxs = d.topk(2, dim=-1, largest=False)
        d_riv = torch.where(idxs[..., 0] == k1, vals[..., 1], vals[..., 0])
        margins.append(d_cur - d_riv)
    m = torch.stack(margins, dim=1)
    return -(m.clamp(max=0.0) if saturate else m).mean()


def make_attacks(tok, logit_fn, device: str, steps: int, lr: float):
    def M1(x, y, eps):
        y = y.float().to(device)
        return run_pgd(
            x, lambda xa: -F.binary_cross_entropy_with_logits(logit_fn(tok.reconstruct(xa)), y),
            eps, steps, lr, device,
        )

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
        "M1": lambda x, y, e: M1(x, y, e),
        "M3_alpha03": lambda x, y, e: M3(x, y, e, 0.3, False),
        "M2-sat": lambda x, y, e: M3(x, y, e, 1.0, True),
        "M2-unb": lambda x, y, e: M3(x, y, e, 1.0, False),
    }
