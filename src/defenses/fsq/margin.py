"""Anchored signed margin for FSQ (Finite Scalar Quantization), and the
encoder/decoder freeze helpers its adversarial and decoder-realism fine-tuning
use. FSQ has no codebook. It quantizes 5 continuous scalar dimensions per token
independently to a fixed integer grid (tanh-bound, then round), so none of the
nearest-neighbor-in-embedding-space margin math from VQ-16, MediTok, or UniTok
applies. This is FSQ's own geometry, not an adaptation.

The sign convention matches every other tokenizer's margin function here:
negative means intact, non-negative means flipped, so `loss = fsq_margin(...)`
serves directly as both the attacker's and the defender's minimize-target.
`margin_loss_anchored` in `src/attacks/fsq_attacks.py` uses the opposite raw
convention (positive intact, non-positive flipped), and `fsq_margin` negates it
once. Negating a `min(dim=-1)` equals `max` of the negated per-dim values, so
the worst-dim ("weakest link") selection is preserved exactly, not just relabeled.
"""
import torch

from src.tokenizers.fsq import LEVELS, fsq_bound


def half_width(device, levels=LEVELS) -> torch.Tensor:
    """[l // 2 for l in levels], FSQ-pytorch's own normalization divisor, which
    turns an integer level back into the continuous scale net.dec expects. A
    5-element tensor, cheap enough to rebuild per call rather than cache."""
    return torch.tensor([l // 2 for l in levels], dtype=torch.float32, device=device)


@torch.no_grad()
def clean_anchor(tok, x: torch.Tensor) -> torch.Tensor:
    """Per-token, per-dim clean bin center: anchor = round(bound(encode(x))).
    Fixed once per attack or fine-tuning run against the clean image, never
    recomputed against a moving target, the same anchoring every other
    tokenizer's clean-top1 assignment uses."""
    z = tok.encode_prequant(x)
    return fsq_bound(z).round()


def fsq_margin(tok, x, anchor: torch.Tensor, saturate: bool):
    """m = -(0.5 - |u-anchor|).min(dim=-1). Raw signed margin: negative intact,
    non-negative flipped, meaning a level boundary was crossed in at least one of
    the 5 dims. The token is only as safe as its most fragile dimension."""
    z = tok.encode_prequant(x)
    u = fsq_bound(z)
    m_dim = 0.5 - (u - anchor).abs()  # [B,N,5], source convention: >0 intact per-dim
    m = -m_dim.min(dim=-1).values  # [B,N], negated: <0 intact, >=0 flipped
    if saturate:
        m = m.clamp(max=0.0)  # once flipped, stop pulling further
    return m.mean()


def freeze_non_encoder(tok, verbose: bool = False) -> None:
    """Only enc trainable; dec + quantize_t (parameter-free) frozen."""
    net = tok.net
    for p in net.parameters():
        p.requires_grad_(False)
    for p in net.enc.parameters():
        p.requires_grad_(True)
    if verbose:
        _print_trainability(net)


def _print_trainability(net) -> None:
    for name in ["enc", "dec", "quantize_t"]:
        mod = getattr(net, name, None)
        trainable = mod is not None and any(p.requires_grad for p in mod.parameters())
        print(f"  {name:12s} trainable={trainable}")


def unflatten_quant(quant_t: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    """Reshapes quant_t from [B,N,D] token-sequence form (used by encode_prequant,
    SQS voting, and the margin math) to the [B,D,Hp,Wp] spatial form net.dec's
    ConvTranspose2d layers require. FSQ's own quantize_t re-expands to spatial
    only when given a spatial 4D input; fed a flattened 3D tensor it stays
    flattened, so every caller that builds quant_t via encode_prequant must
    unflatten before decoding or net.dec fails on a channel-count mismatch."""
    B, N, D = quant_t.shape
    return quant_t.permute(0, 2, 1).reshape(B, D, Hp, Wp)


def decode_only(net, quant_t: torch.Tensor) -> torch.Tensor:
    """Decodes [B,N,D] flattened token-sequence quantized features to an image,
    skipping encode and quantize. N must be a perfect square (FSQ's real token
    grid, the same assumption spatial SQS relies on)."""
    B, N, D = quant_t.shape
    Hp = Wp = int(round(N ** 0.5))
    assert Hp * Wp == N, f"expected a square token grid, got N={N} (sqrt={N ** 0.5})"
    return net.dec(unflatten_quant(quant_t, Hp, Wp)).clamp(-1, 1)
