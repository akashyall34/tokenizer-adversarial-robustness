"""Freeze/decode helpers for MediTok/UniTok's multi-codebook architecture,
mirroring src/defenses/vq16/margin.py's role but for the encode_prequant ->
quant_proj -> quantizer -> post_quant_proj -> decoder pipeline shared by both
tokenizers (attribute names match MultiCodebookTokenizer.net exactly).

The margin and clean-top1 math is not duplicated here; it is imported from
src/attacks/meditok_unitok_attacks.py. MediTok and UniTok's attack-side margin is
already computed in the correctly normalized space (unlike VQ-16, see
docs/known_limitations.md), so there is no separate corrected version. `raw_margin`
below just flips the sign to what the fine-tuning loop needs, minimizing the raw
margin to widen it.
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
