"""Corrected (L2-normalized) VQ-16 margin computation, used for adversarial
fine-tuning. It is kept separate from `margin_loss()` in
src/attacks/vq16_attacks.py, which computes the margin on unnormalized vectors;
docs/known_limitations.md explains why the two differ and why both exist.

VQ-16's `quantize` module L2-normalizes both the latent and the codebook before
its nearest-neighbor lookup. The unnormalized proxy agrees with that real
assignment only 51.93% of the time, measured directly against `model.encode()`
(see scripts/diagnostics/diagnose_vq16_margin_bug.py).
"""
import torch
import torch.nn.functional as F


def prequant(model, x: torch.Tensor) -> torch.Tensor:
    """Continuous latent before quantization. [B,3,H,W] -> [B,N,8]."""
    h = model.quant_conv(model.encoder(x))
    B, C, H, W = h.shape
    return h.permute(0, 2, 3, 1).reshape(B, H * W, C)


def l2_dist_matrix(z: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances from z to every codebook entry in E, in the same
    L2-normalized space the real quantizer uses. z, E unnormalized in ->
    normalized distances out."""
    z_n = F.normalize(z, p=2, dim=-1)
    E_n = F.normalize(E, p=2, dim=-1)
    return z_n.pow(2).sum(-1, keepdim=True) - 2 * z_n @ E_n.T + E_n.pow(2).sum(-1)


@torch.no_grad()
def clean_top1(model, x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,N] nearest codebook index per token, under model's own
    current weights, in the real (normalized) decision space."""
    z = prequant(model, x)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)
    return d.argmin(-1)


def margin(model, x: torch.Tensor, k1: torch.Tensor, saturate: bool) -> torch.Tensor:
    """m = d(z,k1) - d(z,rival); negative means intact, non-negative means
    flipped. Same sign convention as `margin_loss` in
    src/attacks/vq16_attacks.py, but computed in the correct normalized space."""
    z = prequant(model, x)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)  # [B,N,V]
    d_cur = d.gather(-1, k1.unsqueeze(-1)).squeeze(-1)
    vals, idxs = d.topk(2, dim=-1, largest=False)
    d_riv = torch.where(idxs[..., 0] == k1, vals[..., 1], vals[..., 0])
    m = d_cur - d_riv
    if saturate:
        m = m.clamp(max=0.0)
    return m.mean()


def freeze_non_encoder(model, verbose: bool = False) -> None:
    """Only encoder + quant_conv trainable; codebook + decoder frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    for p in model.quant_conv.parameters():
        p.requires_grad_(True)
    if verbose:
        for name, sub in model.named_children():
            trainable = any(p.requires_grad for p in sub.parameters())
            print(f"  {name:15s} trainable={trainable}")


def plain_quantize(model, h: torch.Tensor) -> torch.Tensor:
    """Maps an (B,C,Hp,Wp) pre-quant latent to z_q via the real, deterministic
    nearest-neighbor lookup (no noise, no voting). Used by the BPDA
    straight-through wrapper for "no SQS" adaptive-attack checks; the encoder and
    decoder pipeline is otherwise fully differentiable apart from this discrete
    argmin."""
    B, C, Hp, Wp = h.shape
    z = h.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    idx = l2_dist_matrix(z, E).argmin(-1)
    return model.quantize.get_codebook_entry(idx.view(-1), shape=(B, C, Hp, Wp), channel_first=True)


def reconstruct(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        quant, _, _ = model.encode(x)
        return model.decode(quant).clamp(-1, 1)


def probe_feat(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        quant, _, _ = model.encode(x)
        return quant.reshape(quant.shape[0], -1)
