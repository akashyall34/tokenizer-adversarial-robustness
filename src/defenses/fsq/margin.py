"""Anchored signed margin for FSQ (Finite Scalar Quantization), and the
encoder/decoder freeze helpers used by its adversarial/decoder-realism
fine-tuning. FSQ has no codebook at all -- it quantizes 5 continuous scalar
dimensions per token independently to a fixed integer grid (tanh-bound then
round), so none of VQ-16/MediTok/UniTok's nearest-neighbor-in-embedding-space
margin math applies; this is FSQ's own geometry, not an adaptation.

Sign convention matches every other tokenizer's margin function in this
project (`<0 intact, >=0 flipped`, so `loss = fsq_margin(...)` directly
serves as both the attacker's and the defender's minimize-target) --
`src/attacks/fsq_attacks.py`'s `margin_loss_anchored` uses the opposite raw
convention (`>0 intact, <=0 flipped`); `fsq_margin` here negates it once.
Negating a `min(dim=-1)` is equivalent to `max` of the negated per-dim
values, so this is not just a label change -- the worst-dim ("weakest
link") selection is preserved exactly.
"""
import torch

from src.tokenizers.fsq import LEVELS, fsq_bound


def half_width(device, levels=LEVELS) -> torch.Tensor:
    """[l // 2 for l in levels] -- FSQ-pytorch's own normalization divisor,
    turning an integer level back into the continuous scale net.dec expects.
    Tiny (5-element) tensor, cheap enough to just rebuild per call rather
    than cache."""
    return torch.tensor([l // 2 for l in levels], dtype=torch.float32, device=device)


@torch.no_grad()
def clean_anchor(tok, x: torch.Tensor) -> torch.Tensor:
    """Per-token, per-dim clean bin center: anchor = round(bound(encode(x))).
    Fixed once per attack/fine-tuning run against the CLEAN image -- never
    recomputed against a moving target, same anchoring principle every other
    tokenizer's clean-top1 assignment uses."""
    z = tok.encode_prequant(x)
    return fsq_bound(z).round()


def fsq_margin(tok, x, anchor: torch.Tensor, saturate: bool):
    """m = -(0.5 - |u-anchor|).min(dim=-1) -- raw signed margin, <0 intact,
    >=0 flipped (crossed a level boundary in AT LEAST ONE of the 5 dims --
    the token is only as safe as its most fragile dimension)."""
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
    """quant_t: [B,N,D] (token-sequence form -- the format encode_prequant/SQS
    voting/margin math all use) -> [B,D,Hp,Wp] (spatial form net.dec's
    ConvTranspose2d layers require). FSQ's own quantize_t only re-expands to
    spatial when given a spatial 4D input; fed a flattened 3D tensor it stays
    flattened, so every caller that builds quant_t via encode_prequant must
    unflatten it before decoding, or net.dec crashes on a channel-count
    mismatch."""
    B, N, D = quant_t.shape
    return quant_t.permute(0, 2, 1).reshape(B, D, Hp, Wp)


def decode_only(net, quant_t: torch.Tensor) -> torch.Tensor:
    """quant_t: [B,N,D] flattened token-sequence quantized features -> image,
    skipping encode/quantize. N must be a perfect square (FSQ's real token
    grid, same assumption spatial SQS already relies on)."""
    B, N, D = quant_t.shape
    Hp = Wp = int(round(N ** 0.5))
    assert Hp * Wp == N, f"expected a square token grid, got N={N} (sqrt={N ** 0.5})"
    return net.dec(unflatten_quant(quant_t, Hp, Wp)).clamp(-1, 1)
