"""Decoder-realism fine-tuning for MediTok/UniTok. Structurally simpler than
VQ-16's version (src/defenses/common/decoder_realism.py): trains only on the
corrupted/cached-adversarial codes as the single "fake" branch (real clean
image vs. reconstruction-from-corrupted-code), not VQ-16's double clean+
corrupted branch design -- this is the source notebook's actual recipe, not
a simplification made while porting. Reuses PatchDiscriminator as-is (same
architecture, no tokenizer-specific parts).
"""
import torch.nn.functional as F

from src.defenses.common.decoder_realism import PatchDiscriminator  # noqa: F401 (re-exported for callers)
from src.defenses.multicodebook.margin import decode_only


def train_step(tok, disc, opt_g, opt_d, densenet_features_fn, z_adv, xb_clean,
                perceptual_loss_weight: float = 0.1, adv_loss_weight: float = 0.01):
    """One (discriminator step, generator step) pair. z_adv is the cached
    adversarial code (already quantized), xb_clean is the TRUE clean image --
    the only reconstruction target and the only "real" sample the
    discriminator ever sees."""
    idx = tok.assign_indices(z_adv)
    zq = tok.quantizer.idx_to_f(idx)
    rec_adv = decode_only(tok.net, zq)

    opt_d.zero_grad()
    d_real = disc(xb_clean)
    d_fake = disc(rec_adv.detach())
    d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
    d_loss.backward()
    opt_d.step()

    opt_g.zero_grad()
    d_fake_g = disc(rec_adv)
    adv_loss = -d_fake_g.mean()
    recon_loss = F.l1_loss(rec_adv, xb_clean)
    perc_loss = F.l1_loss(densenet_features_fn(rec_adv), densenet_features_fn(xb_clean).detach())
    g_loss = recon_loss + perceptual_loss_weight * perc_loss + adv_loss_weight * adv_loss
    g_loss.backward()
    opt_g.step()

    return dict(g_loss=g_loss.item(), d_loss=d_loss.item(), recon_loss=recon_loss.item(),
                perc_loss=perc_loss.item(), adv_loss=adv_loss.item(),
                robust_recon_loss=recon_loss.item(), robust_perc_loss=perc_loss.item())
