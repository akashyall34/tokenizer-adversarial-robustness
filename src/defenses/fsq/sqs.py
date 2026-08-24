"""SQS for FSQ: same K-pass noisy-majority-vote mechanism as every other
tokenizer's sqs.py, but the vote is per SCALAR DIMENSION (independent across
the 5 dims), not a joint per-token codebook-index vote -- FSQ's own
quantizer already treats the 5 dims independently (`fsq_bound` then
`.round()` per-dim), so this mirrors the real quantizer's own decision
structure rather than inventing a joint one. Noise is injected directly in
z-space: `fsq_bound`'s own tanh IS the bounding step, so there's no separate
normalized lookup space to inject into the way VQ-16/MediTok's L2/cosine
spaces have.
"""
import torch
import torch.nn.functional as F

from src.defenses.fsq.margin import decode_only, half_width, unflatten_quant
from src.models.densenet_classifier import logit as densenet_logit
from src.tokenizers.fsq import fsq_bound, fsq_token_margins


def sqs_quantize(tok, z: torch.Tensor, K: int, sigma: float, seed: int = None):
    """z: [B,N,5] pre-quantization latent. Returns (voted_levels, quant_t)."""
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    z_rep = z.repeat(K, 1, 1)  # [K*B,N,D]
    z_noisy = z_rep + sigma * torch.randn_like(z_rep)
    levels = fsq_bound(z_noisy).round()  # [K*B,N,D] integer levels per dim
    stacked = levels.view(K, B, N, D)
    voted = torch.mode(stacked, dim=0).values  # [B,N,D] majority level, per dim, per token
    quant_t = voted / half_width(z.device)
    return voted, quant_t


def _token_grid_shape(N: int):
    hw = int(round(N ** 0.5))
    assert hw * hw == N, f"spatial-consensus SQS needs a square token grid, got N={N} (sqrt={N ** 0.5})"
    return hw, hw


def _token_grid_neighbor_mean(quant_t: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    """quant_t: [B,N,5] -> 4-connected spatial neighbor mean (edge tokens
    replicate-padded), adapted to FSQ's per-dim (not per-codebook-vector)
    features."""
    B, N, D = quant_t.shape
    g = quant_t.permute(0, 2, 1).reshape(B, D, Hp, Wp)
    pad = F.pad(g, (1, 1, 1, 1), mode="replicate")
    up, down = pad[:, :, 0:Hp, 1:Wp + 1], pad[:, :, 2:Hp + 2, 1:Wp + 1]
    left, right = pad[:, :, 1:Hp + 1, 0:Wp], pad[:, :, 1:Hp + 1, 2:Wp + 2]
    mean_g = (up + down + left + right) / 4.0
    return mean_g.reshape(B, D, N).permute(0, 2, 1)


def sqs_quantize_spatial(tok, z: torch.Tensor, K: int, sigma: float, spatial_weight: float,
                          agreement_thresh_pct: float, seed: int = None):
    """K-pass majority vote (per dim), then repair tokens flagged by EITHER
    (1) spatial disagreement with the neighbor mean exceeding this image's
    own median, or (2) low K-pass self-agreement (measured per-token across
    ALL 5 dims jointly -- a token only counts as "agreed" if every dim's
    voted level matches that pass's level, the all-or-nothing standard a
    joint codebook-index vote would apply)."""
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    Hp, Wp = _token_grid_shape(N)
    z_rep = z.repeat(K, 1, 1)
    z_noisy = z_rep + sigma * torch.randn_like(z_rep)
    levels = fsq_bound(z_noisy).round()
    stacked = levels.view(K, B, N, D)
    voted = torch.mode(stacked, dim=0).values
    agree_pct = (stacked == voted.unsqueeze(0)).all(dim=-1).float().mean(dim=0) * 100  # [B,N]
    hw = half_width(z.device)
    quant_t = voted / hw

    neighbor_mean = _token_grid_neighbor_mean(quant_t, Hp, Wp)
    disagreement = (quant_t - neighbor_mean).norm(dim=-1)  # [B,N]
    median_disagreement = disagreement.median(dim=1, keepdim=True).values

    is_spatial_outlier = disagreement > median_disagreement
    is_unstable = agree_pct < agreement_thresh_pct
    is_outlier = (is_spatial_outlier | is_unstable).unsqueeze(-1)

    blended = torch.where(is_outlier, (1 - spatial_weight) * quant_t + spatial_weight * neighbor_mean, quant_t)
    # re-snap the blend to a real level directly by rounding in level-space -- NOT by re-running
    # fsq_bound (that would re-apply the tanh compression to an ALREADY-bounded value, double-
    # bounding it; blended lives in the same normalized quant_t scale the real quantizer's OUTPUT
    # uses, not the raw pre-quantization z scale fsq_bound expects as input).
    voted_snap = (blended * hw).round()
    quant_t_snap = voted_snap / hw
    return voted_snap, quant_t_snap


def margin_adaptive_sigma_map(tok, z: torch.Tensor, sigma_base: float, margin_low: float,
                               margin_high: float, sigma_mult_max: float) -> torch.Tensor:
    """Per-token sigma: margin<=margin_low -> sigma_base*sigma_mult_max
    (fragile, protect hard); margin>=margin_high -> sigma_base (safe,
    preserve fidelity); linear between."""
    with torch.no_grad():
        margins = fsq_token_margins(z)  # [B,N], unanchored proximity-to-nearest-boundary
        t = ((margins - margin_low) / max(margin_high - margin_low, 1e-8)).clamp(0, 1)
        mult = sigma_mult_max - t * (sigma_mult_max - 1.0)
        return sigma_base * mult


def sqs_quantize_margin_adaptive(tok, z: torch.Tensor, K: int, sigma_map: torch.Tensor, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    z_rep = z.repeat(K, 1, 1)
    sigma_rep = sigma_map.repeat(K, 1).unsqueeze(-1)  # [K*B,N,1], broadcasts over the 5 dims
    z_noisy = z_rep + sigma_rep * torch.randn_like(z_rep)
    levels = fsq_bound(z_noisy).round()
    stacked = levels.view(K, B, N, D)
    voted = torch.mode(stacked, dim=0).values
    return voted / half_width(z.device)


@torch.no_grad()
def sqs_reconstruct(tok, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                     spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    z = tok.encode_prequant(x)
    _, quant_t = (sqs_quantize_spatial(tok, z, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
                  else sqs_quantize(tok, z, K, sigma, seed=seed))
    return decode_only(tok.net, quant_t)


@torch.no_grad()
def sqs_probe_feat(tok, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                    spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    z = tok.encode_prequant(x)
    _, quant_t = (sqs_quantize_spatial(tok, z, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
                  else sqs_quantize(tok, z, K, sigma, seed=seed))
    # unflatten to the SAME channel-first layout FSQTokenizer.probe_feat uses -- the real trained
    # probe was trained on that layout; a token-major flatten here would silently feed it a
    # scrambled permutation of the same values (wrong, not a crash).
    B, N, D = quant_t.shape
    Hp = Wp = int(round(N ** 0.5))
    quant_t_spatial = unflatten_quant(quant_t, Hp, Wp)
    return quant_t_spatial.reshape(quant_t_spatial.shape[0], -1)


@torch.no_grad()
def sqs_reconstruct_adaptive(tok, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                              margin_high: float, sigma_mult_max: float, seed: int = None):
    z = tok.encode_prequant(x)
    sigma_map = margin_adaptive_sigma_map(tok, z, sigma_base, margin_low, margin_high, sigma_mult_max)
    quant_t = sqs_quantize_margin_adaptive(tok, z, K, sigma_map, seed=seed)
    return decode_only(tok.net, quant_t)


@torch.no_grad()
def sqs_probe_feat_adaptive(tok, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                             margin_high: float, sigma_mult_max: float, seed: int = None):
    z = tok.encode_prequant(x)
    sigma_map = margin_adaptive_sigma_map(tok, z, sigma_base, margin_low, margin_high, sigma_mult_max)
    quant_t = sqs_quantize_margin_adaptive(tok, z, K, sigma_map, seed=seed)
    B, N, D = quant_t.shape
    Hp = Wp = int(round(N ** 0.5))
    quant_t_spatial = unflatten_quant(quant_t, Hp, Wp)
    return quant_t_spatial.reshape(quant_t_spatial.shape[0], -1)


def sqs_logit_ensemble_predict(tok, densenet, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    """A different smoothing POINT than the three variants above: K
    independent noisy re-quantizations, each decoded and classified
    separately (NOT voted first), then the K logits averaged (Cohen et al.
    2019's standard recipe)."""
    if seed is not None:
        torch.manual_seed(seed)
    z = tok.encode_prequant(x)
    B, N, D = z.shape
    z_rep = z.repeat(K, 1, 1)
    z_noisy = z_rep + sigma * torch.randn_like(z_rep)
    levels = fsq_bound(z_noisy).round()
    quant_t = levels / half_width(z.device)
    recon = decode_only(tok.net, quant_t)  # K*B separate reconstructions -- NOT voted first
    logits = densenet_logit(densenet, recon).view(K, B)
    return logits.mean(dim=0)


def dispatch_reconstruct(name: str, tok, x: torch.Tensor, params: dict):
    if name == "flat":
        return sqs_reconstruct(tok, x, params["K"], params["sigma"], spatial=False)
    if name == "spatial":
        return sqs_reconstruct(tok, x, params["K"], params["sigma"], spatial=True,
                                spatial_weight=params["spatial_weight"], agreement_thresh_pct=params["agreement_thresh_pct"])
    if name == "margin_adaptive":
        return sqs_reconstruct_adaptive(tok, x, params["K"], params["sigma_base"],
                                         params["margin_low"], params["margin_high"], params["sigma_mult_max"])
    raise ValueError(f"unknown SQS variant: {name}")


def dispatch_probe_feat(name: str, tok, x: torch.Tensor, params: dict):
    if name == "flat":
        return sqs_probe_feat(tok, x, params["K"], params["sigma"], spatial=False)
    if name == "spatial":
        return sqs_probe_feat(tok, x, params["K"], params["sigma"], spatial=True,
                               spatial_weight=params["spatial_weight"], agreement_thresh_pct=params["agreement_thresh_pct"])
    if name == "margin_adaptive":
        return sqs_probe_feat_adaptive(tok, x, params["K"], params["sigma_base"],
                                        params["margin_low"], params["margin_high"], params["sigma_mult_max"])
    raise ValueError(f"unknown SQS variant: {name}")


def dispatch_quantize(name: str, tok, z: torch.Tensor, params: dict) -> torch.Tensor:
    """z -> quant_t (unwraps the (voted, quant_t) tuple where relevant) --
    for the BPDA straight-through wrapper."""
    if name == "flat":
        _, quant_t = sqs_quantize(tok, z, params["K"], params["sigma"])
        return quant_t
    if name == "spatial":
        _, quant_t = sqs_quantize_spatial(tok, z, params["K"], params["sigma"],
                                           params["spatial_weight"], params["agreement_thresh_pct"])
        return quant_t
    if name == "margin_adaptive":
        sigma_map = margin_adaptive_sigma_map(tok, z, params["sigma_base"], params["margin_low"],
                                               params["margin_high"], params["sigma_mult_max"])
        return sqs_quantize_margin_adaptive(tok, z, params["K"], sigma_map)
    raise ValueError(f"unknown SQS variant: {name}")
