"""SQS (Selective Quantization Smoothing) injects noise into VQ-16's
pre-quantization latent, runs the noisy quantization K times in parallel, and
majority-votes the resulting token per position. The three token-voting variants
share that K-pass vote and differ only in what happens to the voted tokens; the
fourth does not vote on the token at all.

- flat: one global sigma for every token.
- spatial: after voting, a token that disagrees with its 4 spatial neighbors or
  had low self-agreement across its K draws is blended toward its neighbor mean
  and re-snapped to a real codebook entry.
- margin_adaptive: sigma varies per token, scaled by how close that token's
  clean latent sits to a codebook boundary, so fragile tokens get more noise.
- logit_ensemble: a different smoothing point rather than a different sigma. It
  decodes and classifies each of the K noisy draws separately, then averages the
  K classifier logits (standard randomized-smoothing aggregation, Cohen et al.
  2019). The other three variants smooth at the discrete-token level and are
  blind to a dense attack pattern spread across most tokens at once; this variant
  tests whether smoothing one level later, at the classifier decision, reaches
  what token-voting cannot.

Noise is injected in the same L2-normalized space VQ-16's real quantizer uses for
its nearest-neighbor lookup (see src/defenses/vq16/margin.py). Raw-scale noise
added before normalization would mostly wash out in the normalization step.
"""
import torch
import torch.nn.functional as F

from src.defenses.vq16.margin import l2_dist_matrix
from src.models.densenet_classifier import logit as densenet_logit


def quant_prequant(model, x: torch.Tensor) -> torch.Tensor:
    """Pre-quantization latent in its raw conv-grid shape (B,C,H,W), not the
    flattened (B,N,C) shape that `prequant()` in src/defenses/vq16/margin.py
    returns. That one is flattened for the margin/L2-distance computation; every
    quantize function here needs the grid shape to reshape z_q back into for
    model.decode()."""
    return model.quant_conv(model.encoder(x))


_quant_prequant = quant_prequant  # short alias used within this module


def sqs_quantize(model, h: torch.Tensor, K: int, sigma: float, seed: int = None):
    """h: (B,C,Hp,Wp) pre-quantization latent. Returns (voted_idx, z_q).

    Does its own nearest-neighbor lookup via l2_dist_matrix (a single normalize
    pass) instead of handing an already-normalized and noised tensor back to the
    model's own quantize(). That path would normalize a second time; although
    mathematically a no-op, float-precision differences right at a decision
    boundary can flip which code a token is assigned.
    """
    if seed is not None:
        torch.manual_seed(seed)
    B, C, Hp, Wp = h.shape
    h_rep = h.repeat(K, 1, 1, 1)
    h_perm = h_rep.permute(0, 2, 3, 1)
    h_norm = F.normalize(h_perm, p=2, dim=-1)
    h_noisy_norm = h_norm + sigma * torch.randn_like(h_norm)
    z = h_noisy_norm.reshape(K * B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)
    idx = d.argmin(-1)
    stacked = idx.view(K, B, Hp * Wp)
    voted_idx = torch.mode(stacked, dim=0).values
    z_q = model.quantize.get_codebook_entry(voted_idx.view(-1), shape=(B, C, Hp, Wp), channel_first=True)
    return voted_idx, z_q


def _token_grid_neighbor_mean(z_q: torch.Tensor) -> torch.Tensor:
    """z_q: (B,C,Hp,Wp). Mean of each token's 4-connected spatial neighbors
    (edge tokens replicate-padded so they still get a real 4-neighbor average)."""
    B, C, Hp, Wp = z_q.shape
    pad = F.pad(z_q, (1, 1, 1, 1), mode="replicate")
    up, down = pad[:, :, 0:Hp, 1:Wp + 1], pad[:, :, 2:Hp + 2, 1:Wp + 1]
    left, right = pad[:, :, 1:Hp + 1, 0:Wp], pad[:, :, 1:Hp + 1, 2:Wp + 2]
    return (up + down + left + right) / 4.0


def sqs_quantize_spatial(model, h: torch.Tensor, K: int, sigma: float,
                          spatial_weight: float, agreement_thresh_pct: float, seed: int = None):
    """K-pass majority vote, then repair tokens flagged by either (1) spatial
    disagreement with the neighbor mean exceeding this image's own median
    disagreement, or (2) low K-pass self-agreement."""
    if seed is not None:
        torch.manual_seed(seed)
    B, C, Hp, Wp = h.shape
    h_rep = h.repeat(K, 1, 1, 1)
    h_n = F.normalize(h_rep.permute(0, 2, 3, 1), p=2, dim=-1)
    h_noisy_norm = h_n + sigma * torch.randn_like(h_n)
    z = h_noisy_norm.reshape(K * B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)
    idx = d.argmin(-1)
    stacked = idx.view(K, B, Hp * Wp)
    voted_idx = torch.mode(stacked, dim=0).values

    agreement_pct = (stacked == voted_idx.unsqueeze(0)).float().mean(dim=0).view(B, Hp, Wp) * 100
    z_q = model.quantize.get_codebook_entry(voted_idx.view(-1), shape=(B, C, Hp, Wp), channel_first=True)

    neighbor_mean = _token_grid_neighbor_mean(z_q)
    z_n, neighbor_n = F.normalize(z_q, p=2, dim=1), F.normalize(neighbor_mean, p=2, dim=1)
    disagreement = (z_n - neighbor_n).norm(dim=1)  # (B,Hp,Wp)
    median_disagreement = disagreement.view(B, -1).median(dim=1).values.view(B, 1, 1)

    is_spatial_outlier = disagreement > median_disagreement
    is_unstable = agreement_pct < agreement_thresh_pct
    is_outlier = (is_spatial_outlier | is_unstable).unsqueeze(1)

    blended = torch.where(is_outlier, (1 - spatial_weight) * z_q + spatial_weight * neighbor_mean, z_q)
    return sqs_quantize(model, blended, K=1, sigma=0.0, seed=0)  # re-snap to a real code


@torch.no_grad()
def sqs_reconstruct(model, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                     spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    h = _quant_prequant(model, x)
    _, z_q = (sqs_quantize_spatial(model, h, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
              else sqs_quantize(model, h, K, sigma, seed=seed))
    return model.decode(z_q).clamp(-1, 1)


@torch.no_grad()
def sqs_probe_feat(model, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                    spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    h = _quant_prequant(model, x)
    _, z_q = (sqs_quantize_spatial(model, h, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
              else sqs_quantize(model, h, K, sigma, seed=seed))
    return z_q.reshape(z_q.shape[0], -1)


@torch.no_grad()
def token_margins_from_h(model, h: torch.Tensor) -> torch.Tensor:
    """h: (B,C,Hp,Wp) pre-quant latent. Per-token margin (>=0, larger=safer),
    in the same L2-normalized distance space l2_dist_matrix uses."""
    B, C, Hp, Wp = h.shape
    z = h.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)
    vals, _ = d.topk(2, dim=-1, largest=False)
    return (vals[..., 1] - vals[..., 0]).reshape(B, Hp, Wp)


def margin_adaptive_sigma_map(model, h: torch.Tensor, sigma_base: float, margin_low: float,
                               margin_high: float, sigma_mult_max: float) -> torch.Tensor:
    """Per-token sigma (B,Hp,Wp): margin<=margin_low -> sigma_base*sigma_mult_max
    (fragile, protect hard); margin>=margin_high -> sigma_base (safe, preserve
    fidelity); linear between."""
    margins = token_margins_from_h(model, h)
    t = ((margins - margin_low) / max(margin_high - margin_low, 1e-8)).clamp(0, 1)
    mult = sigma_mult_max - t * (sigma_mult_max - 1.0)
    return sigma_base * mult


def sqs_quantize_margin_adaptive(model, h: torch.Tensor, K: int, sigma_map: torch.Tensor, seed: int = None):
    """Same K-pass majority-vote mechanism as sqs_quantize, but sigma varies
    per token (sigma_map: (B,Hp,Wp)) instead of one global scalar."""
    if seed is not None:
        torch.manual_seed(seed)
    B, C, Hp, Wp = h.shape
    h_rep = h.repeat(K, 1, 1, 1)
    h_perm = h_rep.permute(0, 2, 3, 1)
    h_norm = F.normalize(h_perm, p=2, dim=-1)
    sigma_rep = sigma_map.repeat(K, 1, 1).unsqueeze(-1)  # (K*B,Hp,Wp,1), broadcasts over C
    h_noisy_norm = h_norm + sigma_rep * torch.randn_like(h_norm)
    z = h_noisy_norm.reshape(K * B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    d = l2_dist_matrix(z, E)
    idx = d.argmin(-1)
    stacked = idx.view(K, B, Hp * Wp)
    voted_idx = torch.mode(stacked, dim=0).values
    z_q = model.quantize.get_codebook_entry(voted_idx.view(-1), shape=(B, C, Hp, Wp), channel_first=True)
    return voted_idx, z_q


@torch.no_grad()
def sqs_reconstruct_adaptive(model, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                              margin_high: float, sigma_mult_max: float, seed: int = None):
    h = _quant_prequant(model, x)
    sigma_map = margin_adaptive_sigma_map(model, h, sigma_base, margin_low, margin_high, sigma_mult_max)
    _, z_q = sqs_quantize_margin_adaptive(model, h, K, sigma_map, seed=seed)
    return model.decode(z_q).clamp(-1, 1)


@torch.no_grad()
def sqs_probe_feat_adaptive(model, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                             margin_high: float, sigma_mult_max: float, seed: int = None):
    h = _quant_prequant(model, x)
    sigma_map = margin_adaptive_sigma_map(model, h, sigma_base, margin_low, margin_high, sigma_mult_max)
    _, z_q = sqs_quantize_margin_adaptive(model, h, K, sigma_map, seed=seed)
    return z_q.reshape(z_q.shape[0], -1)


def sqs_logit_ensemble_predict(model, densenet, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    """K independent noisy re-quantizations, EACH decoded and classified
    separately (no voting on the token), then the K logits are averaged."""
    if seed is not None:
        torch.manual_seed(seed)
    h = _quant_prequant(model, x)
    B, C, Hp, Wp = h.shape
    h_rep = h.repeat(K, 1, 1, 1)
    h_norm = F.normalize(h_rep.permute(0, 2, 3, 1), p=2, dim=-1)
    h_noisy_norm = h_norm + sigma * torch.randn_like(h_norm)
    z = h_noisy_norm.reshape(K * B, Hp * Wp, C)
    E = model.quantize.embedding.weight
    idx = l2_dist_matrix(z, E).argmin(-1)
    z_q = model.quantize.get_codebook_entry(idx.view(-1), shape=(K * B, C, Hp, Wp), channel_first=True)
    recon = model.decode(z_q)  # K*B separate reconstructions, not voted first
    logits = densenet_logit(densenet, recon).view(K, B)
    return logits.mean(dim=0)


def dispatch_reconstruct(name: str, model, x: torch.Tensor, params: dict):
    """Single dispatch point across the three token-voting variants, given
    the winning variant's name and its selected hyperparameters."""
    if name == "flat":
        return sqs_reconstruct(model, x, params["K"], params["sigma"], spatial=False)
    if name == "spatial":
        return sqs_reconstruct(model, x, params["K"], params["sigma"], spatial=True,
                                spatial_weight=params["spatial_weight"], agreement_thresh_pct=params["agreement_thresh_pct"])
    if name == "margin_adaptive":
        return sqs_reconstruct_adaptive(model, x, params["K"], params["sigma_base"],
                                         params["margin_low"], params["margin_high"], params["sigma_mult_max"])
    raise ValueError(f"unknown SQS variant: {name}")


def dispatch_quantize(name: str, model, h: torch.Tensor, params: dict) -> torch.Tensor:
    """Maps h to z_q, unwrapping the (idx, z_q) tuple every sqs_quantize*
    function returns. For the BPDA straight-through wrapper, which needs a single
    h-to-z_q dispatch point regardless of which SQS variant currently wins."""
    if name == "flat":
        _, z_q = sqs_quantize(model, h, params["K"], params["sigma"])
        return z_q
    if name == "spatial":
        _, z_q = sqs_quantize_spatial(model, h, params["K"], params["sigma"],
                                       params["spatial_weight"], params["agreement_thresh_pct"])
        return z_q
    if name == "margin_adaptive":
        sigma_map = margin_adaptive_sigma_map(model, h, params["sigma_base"], params["margin_low"],
                                               params["margin_high"], params["sigma_mult_max"])
        _, z_q = sqs_quantize_margin_adaptive(model, h, params["K"], sigma_map)
        return z_q
    raise ValueError(f"unknown SQS variant: {name}")


def dispatch_probe_feat(name: str, model, x: torch.Tensor, params: dict):
    if name == "flat":
        return sqs_probe_feat(model, x, params["K"], params["sigma"], spatial=False)
    if name == "spatial":
        return sqs_probe_feat(model, x, params["K"], params["sigma"], spatial=True,
                               spatial_weight=params["spatial_weight"], agreement_thresh_pct=params["agreement_thresh_pct"])
    if name == "margin_adaptive":
        return sqs_probe_feat_adaptive(model, x, params["K"], params["sigma_base"],
                                        params["margin_low"], params["margin_high"], params["sigma_mult_max"])
    raise ValueError(f"unknown SQS variant: {name}")
