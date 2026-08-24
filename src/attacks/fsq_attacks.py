"""M1 / M3 attacks against FSQ.

One margin-loss variant: `margin_loss_anchored` / `make_attacks_anchored`,
used identically by all three datasets. It anchors the target quantization
bin to the clean sample once, before the PGD loop, and pushes away from that
fixed anchor for all steps -- consistent with how VQ-16/MediTok/UniTok's M3
margin losses already anchor to a fixed clean codebook assignment.

An earlier, un-anchored variant (recomputing the nearest bin from the
*current* perturbed latent every step, so the target drifted as the
perturbation crossed it) is what `VQ16.ipynb`'s embedded Cell FSQ-3
originally ran for Kermany. `Official-attack-runs.ipynb` -- the real,
further-along Kermany FSQ notebook (`DATA_ROOT=chest_xray`, DenseNet121;
despite lacking a "-Kermany" suffix, do not confuse it with the differently
named `Official-attack-runs-APTOS.ipynb`) -- ran that same un-anchored
version in its own Cell FSQ-3, then fixed it in Cell FSQ-3b ("FIXED FSQ
margin (anchored to clean bin) + re-run geo sat/unb", execution_count=33):
deleted the buggy cached `M3_geo_sat`/`M3_geo_unb` images and regenerated
them with the anchored margin, real output `"DONE — re-run with anchored
margin"`. The `fsq_adversarial_bundle.zip`/`fsq_attack_table.csv` this
notebook produced -- the real, final Kermany FSQ results -- already reflect
the anchored fix. `run_fsq_kermany.py` matches that: anchored margin, no
`M3_alpha03` (Kermany's `FSQ_ATTACKS` dict never had that variant -- Cell
FSQ-3b's fix only touched `M3_geo_sat`/`M3_geo_unb`, matching M1 unchanged).

`Official-attack-runs-APTOS.ipynb`'s own Cell FSQ-3c independently reaches
the same anchored formula for APTOS, and additionally adds `M3_alpha03`
(BCE-mixed) -- `run_fsq_aptos.py` uses that.

`ISIC-official-RUN.ipynb`'s own Cell 23 (execution_count=7, real output
including `"M1 eps0.5: saved 256"` and a real `isic_fsq_bundle.zip`) also
independently uses the anchored formula plus `M3_alpha03`, `meta.json`'s own
`margin` field literally reading `"anchored_signed"` -- run_fsq_isic.py
already matches this exactly.

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
        "M3_geo_sat": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=True),
        "M3_geo_unb": lambda x, y, e: M3(x, y, e, alpha=1.0, saturate=False),
    }
