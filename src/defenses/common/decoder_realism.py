"""Decoder-realism fine-tuning makes a tokenizer's decoder produce plausible,
in-domain reconstructions regardless of which codes it is handed, so
reconstruction quality holds up even when an attack has corrupted the codes.

This is a complementary lever to encoder fine-tuning (src/defenses/vq16/margin.py).
Encoder fine-tuning defends the boundary, making a token harder to flip. This
defends the decode, staying close to the natural image manifold even once tokens
have flipped. It trains on clean images plus the cached adversarial codes and
runs no new attack.

Not tokenizer-specific. `decode_fn(model, q)` is the tokenizer's real decode
path: a single `model.decode(q)` call for VQ-16, or `post_quant_proj(q)` then
`decoder(...)` for MediTok/UniTok, which have no combined method. See
`src/defenses/vq16/margin.py`'s `reconstruct` for VQ-16's `decode_fn`, or an
equivalent free function for a multi-codebook tokenizer. `decoder_attrs` names
the submodules that stay trainable during the run (decoder + post_quant_conv for
VQ-16; decoder + post_quant_proj for MediTok/UniTok). Everything else (encoder,
quant projection, codebook) stays frozen, the same discipline for every
tokenizer.
"""
import torch.nn as nn
import torch.nn.functional as F


def freeze_non_decoder(model, decoder_attrs=("decoder", "post_quant_conv"), verbose: bool = False) -> None:
    """Only the named decode-side submodules trainable; everything else frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    for attr in decoder_attrs:
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad_(True)
    if verbose:
        for name, sub in model.named_children():
            trainable = any(p.requires_grad for p in sub.parameters())
            print(f"  {name:15s} trainable={trainable}")


class PatchDiscriminator(nn.Module):
    """Small PatchGAN-style discriminator: real vs reconstructed at the patch
    level, the same family HiFiC and VQGAN use."""

    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()

        def block(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, norm=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_step(model, decode_fn, disc, opt_g, opt_d, densenet, q_clean, q_corrupt, xb,
                adv_loss_weight: float, perceptual_loss_weight: float,
                recon_loss_weight: float, robust_loss_weight: float):
    """One (discriminator step, generator step) pair. xb is the true clean
    image, the target for both the clean-code and corrupted-code branches."""
    # discriminator: sees clean recon AND corrupted recon as fake
    rec = decode_fn(model, q_clean)
    rec_corrupt = decode_fn(model, q_corrupt)
    opt_d.zero_grad()
    d_loss = (F.relu(1.0 - disc(xb)).mean()
              + 0.5 * F.relu(1.0 + disc(rec.detach())).mean()
              + 0.5 * F.relu(1.0 + disc(rec_corrupt.detach())).mean())
    d_loss.backward()
    opt_d.step()

    # generator: clean branch + a robustness branch on real cached-adversarial
    # codes, both targeting the TRUE clean image
    opt_g.zero_grad()
    rec = decode_fn(model, q_clean)
    rec_corrupt = decode_fn(model, q_corrupt)

    recon_loss = F.l1_loss(rec, xb)
    perc_loss = F.l1_loss(densenet.features(rec), densenet.features(xb))
    adv_loss = -disc(rec).mean()

    robust_recon_loss = F.l1_loss(rec_corrupt, xb)
    robust_perc_loss = F.l1_loss(densenet.features(rec_corrupt), densenet.features(xb))
    robust_adv_loss = -disc(rec_corrupt).mean()

    g_loss = (recon_loss_weight * recon_loss + perceptual_loss_weight * perc_loss + adv_loss_weight * adv_loss
              + robust_loss_weight * (robust_recon_loss + perceptual_loss_weight * robust_perc_loss
                                       + adv_loss_weight * robust_adv_loss))
    g_loss.backward()
    opt_g.step()

    return dict(g_loss=g_loss.item(), d_loss=d_loss.item(), recon_loss=recon_loss.item(),
                perc_loss=perc_loss.item(), adv_loss=adv_loss.item(),
                robust_recon_loss=robust_recon_loss.item(), robust_perc_loss=robust_perc_loss.item())
