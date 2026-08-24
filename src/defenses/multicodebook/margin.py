"""Freeze/decode helpers for MediTok/UniTok's multi-codebook architecture,
mirroring src/defenses/vq16/margin.py's role but for the encode_prequant ->
quant_proj -> quantizer -> post_quant_proj -> decoder pipeline shared by both
tokenizers (attribute names match MultiCodebookTokenizer.net exactly).

The margin/clean-top1 math itself is NOT duplicated here -- it's identical to
src/attacks/meditok_unitok_attacks.py's margin_loss and
MultiCodebookTokenizer.clean_top1(), confirmed by direct measurement in the
source notebook (per-sub-codebook agreement with the real quantizer's own
assignment, same discipline as VQ-16's verification, and unlike VQ-16,
MediTok/UniTok's attack-side margin math was already correctly normalized --
no bug to preserve here). `raw_margin` below just flips the sign back to what
the fine-tuning loop needs (minimize the raw margin to widen it), reusing
that exact function rather than reimplementing the math a second time.
"""
from src.attacks.meditok_unitok_attacks import margin_loss as _attack_margin_loss


def raw_margin(tok, z, k1_list, saturate: bool):
    """Non-negated margin for the fine-tuning loss (minimize -> widen the
    margin back out). src/attacks/meditok_unitok_attacks.margin_loss negates
    this same computation for attacker-loss-maximization framing."""
    return -_attack_margin_loss(tok, z, k1_list, saturate)


def freeze_non_encoder(tok, verbose: bool = False) -> None:
    """Only encoder + quant_proj trainable; quantizer + post_quant_proj + decoder frozen."""
    net = tok.net
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.encoder.parameters():
        p.requires_grad_(True)
    for p in net.quant_proj.parameters():
        p.requires_grad_(True)
    if verbose:
        _print_trainability(net)


def freeze_non_decoder(tok, verbose: bool = False) -> None:
    """Only decoder + post_quant_proj trainable; encoder/quant_proj/quantizer frozen."""
    net = tok.net
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.decoder.parameters():
        p.requires_grad_(True)
    for p in net.post_quant_proj.parameters():
        p.requires_grad_(True)
    if verbose:
        _print_trainability(net)


def _print_trainability(net) -> None:
    for name in ["encoder", "decoder", "quantizer", "quant_proj", "post_quant_proj"]:
        mod = getattr(net, name, None)
        trainable = mod is not None and any(p.requires_grad for p in mod.parameters())
        print(f"  {name:16s} trainable={trainable}")


def decode_only(net, q):
    """q: [B,N,D] quantized features -> reconstructed image, skipping encode/quantize."""
    t = net.post_quant_proj(q)
    return net.decoder(t).clamp(-1, 1)
