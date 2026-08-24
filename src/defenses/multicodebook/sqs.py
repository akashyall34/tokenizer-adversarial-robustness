"""SQS for MediTok/UniTok's multi-codebook architecture -- same K-pass
noisy-majority-vote mechanism as src/defenses/vq16/sqs.py, but operating on
the flat [B,N,D] pre-quantization representation split into per-sub-codebook
chunks with cosine-similarity lookup, instead of VQ-16's single conv-grid
codebook with L2 lookup. Three variants, same shape as VQ-16's: flat,
spatial-consensus, margin-adaptive.

The spatial-consensus variant needs the N tokens to form a square grid to do
a 4-connected neighbor blend -- true for VQ-16 by construction (it's a conv
feature map), NOT assumed here. If a tokenizer's real token layout isn't
square, sqs_quantize_spatial raises rather than silently reshaping into
something wrong.
"""
import torch
import torch.nn.functional as F

from src.defenses.multicodebook.margin import decode_only
from src.models.densenet_classifier import logit as densenet_logit


def sqs_quantize(tok, z: torch.Tensor, K: int, sigma: float, seed: int = None):
    """z: [B,N,D] pre-quantization features. Returns (voted_idx_list, z_q) --
    voted_idx_list is a list of num_codebooks tensors [B,N], z_q is [B,N,D]
    reassembled from the voted per-chunk codes."""
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    chunk_size = D // tok.num_codebooks
    z_rep = z.repeat(K, 1, 1)  # [K*B,N,D]
    chunks = z_rep.split(chunk_size, dim=-1)
    voted_idx, code_chunks = [], []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        f_noisy = F.normalize(f + sigma * torch.randn_like(f), dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f_noisy @ E.T)
        idx = d.argmin(-1)
        stacked = idx.view(K, B, N)
        voted = torch.mode(stacked, dim=0).values
        voted_idx.append(voted)
        code_chunks.append(E[voted])
    z_q = torch.cat(code_chunks, dim=-1)
    return voted_idx, z_q


def _token_grid_neighbor_mean(z_q: torch.Tensor) -> torch.Tensor:
    """z_q: [B,N,D] -- reshapes N into a square H x W grid (asserted, not
    assumed) to do a 4-connected spatial blend."""
    B, N, D = z_q.shape
    H = int(round(N ** 0.5))
    assert H * H == N, (
        f"spatial-consensus SQS needs a square token grid, got N={N} which is not a perfect "
        f"square (H={H}) -- this tokenizer's real token layout doesn't match that assumption; "
        f"confirm the real grid shape and fix this reshape, or skip the spatial variant.")
    grid = z_q.view(B, H, H, D).permute(0, 3, 1, 2)  # [B,D,H,H]
    pad = F.pad(grid, (1, 1, 1, 1), mode="replicate")
    up, down = pad[:, :, 0:H, 1:H + 1], pad[:, :, 2:H + 2, 1:H + 1]
    left, right = pad[:, :, 1:H + 1, 0:H], pad[:, :, 1:H + 1, 2:H + 2]
    mean = (up + down + left + right) / 4.0
    return mean.permute(0, 2, 3, 1).reshape(B, N, D)


def sqs_quantize_spatial(tok, z: torch.Tensor, K: int, sigma: float, spatial_weight: float,
                          agreement_thresh_pct: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    chunk_size = D // tok.num_codebooks
    z_rep = z.repeat(K, 1, 1)
    chunks = z_rep.split(chunk_size, dim=-1)
    voted_idx, code_chunks, agreement_chunks = [], [], []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        f_noisy = F.normalize(f + sigma * torch.randn_like(f), dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f_noisy @ E.T)
        idx = d.argmin(-1)
        stacked = idx.view(K, B, N)
        voted = torch.mode(stacked, dim=0).values
        agree_pct = (stacked == voted.unsqueeze(0)).float().mean(dim=0) * 100
        voted_idx.append(voted)
        code_chunks.append(E[voted])
        agreement_chunks.append(agree_pct)
    z_q = torch.cat(code_chunks, dim=-1)
    agreement_pct = torch.stack(agreement_chunks, dim=-1).mean(dim=-1)  # [B,N] avg across chunks

    neighbor_mean = _token_grid_neighbor_mean(z_q)
    z_n, neighbor_n = F.normalize(z_q, dim=-1), F.normalize(neighbor_mean, dim=-1)
    disagreement = (z_n - neighbor_n).norm(dim=-1)  # [B,N]
    median_disagreement = disagreement.median(dim=-1, keepdim=True).values

    is_spatial_outlier = disagreement > median_disagreement
    is_unstable = agreement_pct < agreement_thresh_pct
    is_outlier = (is_spatial_outlier | is_unstable).unsqueeze(-1)

    blended = torch.where(is_outlier, (1 - spatial_weight) * z_q + spatial_weight * neighbor_mean, z_q)
    return sqs_quantize(tok, blended, K=1, sigma=0.0, seed=0)  # re-snap to real codes


def per_token_min_margin(tok, z: torch.Tensor) -> torch.Tensor:
    """Weakest sub-codebook margin per token: margin = d_cur - d_rival, always
    <= 0 since d_cur is the nearest code by construction (more negative =
    safer, closer to 0 = fragile). Shared by margin_adaptive_sigma_map_from_z
    and the SEL_ITEMS calibration step in the sweep script."""
    with torch.no_grad():
        chunk_size = z.shape[-1] // tok.num_codebooks
        chunks = z.split(chunk_size, dim=-1)
        margins = []
        for i, cb in enumerate(tok.codebooks):
            f = F.normalize(chunks[i], dim=-1)
            E = F.normalize(cb.codebook.weight, dim=1)
            d = 2.0 - 2.0 * (f @ E.T)
            k1 = d.argmin(-1)
            d_cur = d.gather(-1, k1.unsqueeze(-1)).squeeze(-1)
            vals, idxs = d.topk(2, dim=-1, largest=False)
            d_riv = torch.where(idxs[..., 0] == k1, vals[..., 1], vals[..., 0])
            margins.append(d_cur - d_riv)
        return torch.stack(margins, dim=-1).min(dim=-1).values


def margin_adaptive_sigma_map_from_z(tok, z: torch.Tensor, sigma_base: float, margin_low: float,
                                      margin_high: float, sigma_mult_max: float) -> torch.Tensor:
    """Per-token sigma, scaled by how close each token's WEAKEST chunk margin
    is to the boundary -- fragile tokens get more noise. Takes z directly
    (not x) so it composes cleanly with a BPDA straight-through wrapper."""
    with torch.no_grad():
        # frac must be 0 at margin_low (safest) and 1 at margin_high (least safe of the clean
        # distribution) so fragile tokens get MORE noise -- (margin - margin_low), not the reverse.
        margin = per_token_min_margin(tok, z)
        frac = ((margin - margin_low) / max(margin_high - margin_low, 1e-8)).clamp(0, 1)
        sigma_map = sigma_base * (1.0 + frac * (sigma_mult_max - 1.0))
    return sigma_map


def sqs_quantize_margin_adaptive(tok, z: torch.Tensor, K: int, sigma_map: torch.Tensor, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B, N, D = z.shape
    chunk_size = D // tok.num_codebooks
    z_rep = z.repeat(K, 1, 1)
    sigma_rep = sigma_map.repeat(K, 1).unsqueeze(-1)  # [K*B,N,1]
    chunks = z_rep.split(chunk_size, dim=-1)
    code_chunks = []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        f_noisy = F.normalize(f + sigma_rep * torch.randn_like(f), dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f_noisy @ E.T)
        idx = d.argmin(-1)
        stacked = idx.view(K, B, N)
        voted = torch.mode(stacked, dim=0).values
        code_chunks.append(E[voted])
    return torch.cat(code_chunks, dim=-1)


@torch.no_grad()
def sqs_reconstruct(tok, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                     spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    z = tok.encode_prequant(x)
    _, z_q = (sqs_quantize_spatial(tok, z, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
              else sqs_quantize(tok, z, K, sigma, seed=seed))
    return decode_only(tok.net, z_q)


@torch.no_grad()
def sqs_probe_feat(tok, x: torch.Tensor, K: int, sigma: float, spatial: bool = False, seed: int = None,
                    spatial_weight: float = 0.6, agreement_thresh_pct: float = 70):
    z = tok.encode_prequant(x)
    _, z_q = (sqs_quantize_spatial(tok, z, K, sigma, spatial_weight, agreement_thresh_pct, seed=seed) if spatial
              else sqs_quantize(tok, z, K, sigma, seed=seed))
    return z_q.reshape(z_q.shape[0], -1)


@torch.no_grad()
def sqs_reconstruct_adaptive(tok, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                              margin_high: float, sigma_mult_max: float, seed: int = None):
    z = tok.encode_prequant(x)
    sigma_map = margin_adaptive_sigma_map_from_z(tok, z, sigma_base, margin_low, margin_high, sigma_mult_max)
    z_q = sqs_quantize_margin_adaptive(tok, z, K, sigma_map, seed=seed)
    return decode_only(tok.net, z_q)


@torch.no_grad()
def sqs_probe_feat_adaptive(tok, x: torch.Tensor, K: int, sigma_base: float, margin_low: float,
                             margin_high: float, sigma_mult_max: float, seed: int = None):
    z = tok.encode_prequant(x)
    sigma_map = margin_adaptive_sigma_map_from_z(tok, z, sigma_base, margin_low, margin_high, sigma_mult_max)
    z_q = sqs_quantize_margin_adaptive(tok, z, K, sigma_map, seed=seed)
    return z_q.reshape(z_q.shape[0], -1)


def sqs_logit_ensemble_predict(tok, densenet, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    """A different smoothing POINT than the three variants above: K independent
    noisy re-quantizations, each decoded and classified separately, then the K
    logits averaged (Cohen et al. 2019's standard recipe) -- diagnostic for
    whether token-voting's failure against dense/redundant attacks is specific
    to voting on the token, or survives per-sample logit-averaging too."""
    if seed is not None:
        torch.manual_seed(seed)
    z = tok.encode_prequant(x)
    B, N, D = z.shape
    chunk_size = D // tok.num_codebooks
    z_rep = z.repeat(K, 1, 1)
    chunks = z_rep.split(chunk_size, dim=-1)
    code_chunks = []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        f_noisy = F.normalize(f + sigma * torch.randn_like(f), dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f_noisy @ E.T)
        idx = d.argmin(-1)  # NOT voted -- each of the K draws decoded separately
        code_chunks.append(E[idx])
    z_q = torch.cat(code_chunks, dim=-1)
    recon = decode_only(tok.net, z_q)
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
    """z -> z_q (unwraps the (idx, z_q) tuple where relevant) -- for the BPDA
    straight-through wrapper."""
    if name == "flat":
        _, z_q = sqs_quantize(tok, z, params["K"], params["sigma"])
        return z_q
    if name == "spatial":
        _, z_q = sqs_quantize_spatial(tok, z, params["K"], params["sigma"],
                                       params["spatial_weight"], params["agreement_thresh_pct"])
        return z_q
    if name == "margin_adaptive":
        sigma_map = margin_adaptive_sigma_map_from_z(tok, z, params["sigma_base"], params["margin_low"],
                                                       params["margin_high"], params["sigma_mult_max"])
        return sqs_quantize_margin_adaptive(tok, z, params["K"], sigma_map)
    raise ValueError(f"unknown SQS variant: {name}")
