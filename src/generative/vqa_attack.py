"""M2-sat / M2-unb (and a matched-L_inf random control) on the MedITok vision
tower inside a frozen LLaVA-Med.

Same boundary-directed margin objective as
`src/attacks/meditok_unitok_attacks.py::margin_loss`, re-implemented here for one
structural reason: the VLM runs fp16, so the tower's encoder/quant_proj are cast
to fp32 for a stable backward and the codebook weights must be cast to match at
the matmul. The attack never sees the question, the answer, or any gradient from
the language model.
"""
import torch
import torch.nn.functional as F

from src.attacks.pgd import run_pgd


def _split(z, num_codebooks):
    return z.split(z.shape[-1] // num_codebooks, dim=-1)


def _top1(codebooks, z):
    """Per-codebook nearest-code id for the clean latent."""
    chunks = _split(z, len(codebooks))
    return [
        (F.normalize(chunks[i], dim=-1) @ F.normalize(cb.codebook.weight.float(), dim=1).T).argmax(-1)
        for i, cb in enumerate(codebooks)
    ]


def _margin_loss(codebooks, z, k1, saturate):
    chunks = _split(z, len(codebooks))
    margins = []
    for i, cb in enumerate(codebooks):
        f = F.normalize(chunks[i], dim=-1)
        E = F.normalize(cb.codebook.weight.float(), dim=1)
        d = 2.0 - 2.0 * (f @ E.T)
        k = k1[i]
        d_cur = d.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        vals, idxs = d.topk(2, dim=-1, largest=False)
        d_riv = torch.where(idxs[..., 0] == k, vals[..., 1], vals[..., 0])
        margins.append(d_cur - d_riv)
    m = torch.stack(margins, dim=1)
    return -(m.clamp(max=0.0) if saturate else m).mean()


def build_vqa_attacks(vision_tower, device: str, steps: int, lr: float):
    """Returns ({"M2-sat", "M2-unb", "random"}: (x, eps_native) -> x_adv,
    tok_stats(x_clean, x_adv) -> (token_flip_pct, embedding_disp)).

    Mutates `vision_tower` in place: its encoder/quant_proj are set to fp32 with
    grad enabled so the attack can backprop. The VLM's own `generate` runs under
    `inference_mode`, so this does not change the answers it produces."""
    quantizer = vision_tower.quantizer
    codebooks = quantizer.codebooks
    for p in vision_tower.encoder.parameters():
        p.requires_grad_(True)
    for p in vision_tower.quant_proj.parameters():
        p.requires_grad_(True)
    vision_tower.encoder.float()
    vision_tower.quant_proj.float()

    def encode(x):
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return vision_tower.quant_proj(vision_tower.encoder(x.to(device)))

    def geometric(x, eps, saturate):
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            k1 = _top1(codebooks, encode(x))
        return run_pgd(x, lambda xa: _margin_loss(codebooks, encode(xa), k1, saturate), eps, steps, lr, device)

    def random_attack(x, eps):
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return (x + (torch.rand_like(x) * 2 - 1) * eps).clamp(-1, 1)

    @torch.no_grad()
    def tok_stats(x_clean, x_adv):
        z0, za = encode(x_clean), encode(x_adv)
        flip = (quantizer.f_to_idx(z0) != quantizer.f_to_idx(za)).float().mean().item() * 100
        disp = (za - z0).norm(dim=-1).mean().item()
        return flip, disp

    return {
        "M2-sat": lambda x, e: geometric(x, e, True),
        "M2-unb": lambda x, e: geometric(x, e, False),
        "random": random_attack,
    }, tok_stats
