"""Pixel-input noise baseline for MediTok/UniTok, the RQ2 comparison point, same
role as src/defenses/vq16/pixel_noise.py. Same K-pass majority-vote mechanism as
flat SQS, with noise at the pixel input instead of the pre-quantization latent.
VQ-16 gets token indices directly from a single net.encode(x) call; MediTok and
UniTok have no single-call encode-to-tokens path, so this re-derives indices the
way flat SQS does: encode, split into per-codebook chunks, cosine nearest
neighbor per chunk.
"""
import torch
import torch.nn.functional as F

from src.defenses.multicodebook.margin import decode_only


def pixel_noise_quantize(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B = x.shape[0]
    x_rep = (x.repeat(K, 1, 1, 1) + sigma * torch.randn(K * B, *x.shape[1:], device=x.device)).clamp(-1, 1)
    with torch.no_grad():
        z_rep = tok.encode_prequant(x_rep)
    N = z_rep.shape[1]
    chunk_size = z_rep.shape[-1] // tok.num_codebooks
    chunks = z_rep.split(chunk_size, dim=-1)
    voted_idx, code_chunks = [], []
    for i, cb in enumerate(tok.codebooks):
        f = F.normalize(chunks[i], dim=-1)
        E = F.normalize(cb.codebook.weight, dim=1)
        d = 2.0 - 2.0 * (f @ E.T)
        idx = d.argmin(-1)
        stacked = idx.view(K, B, N)
        voted = torch.mode(stacked, dim=0).values
        voted_idx.append(voted)
        code_chunks.append(E[voted])
    z_q = torch.cat(code_chunks, dim=-1)
    return voted_idx, z_q


@torch.no_grad()
def pixel_noise_reconstruct(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, z_q = pixel_noise_quantize(tok, x, K, sigma, seed=seed)
    return decode_only(tok.net, z_q)


@torch.no_grad()
def pixel_noise_probe_feat(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, z_q = pixel_noise_quantize(tok, x, K, sigma, seed=seed)
    return z_q.reshape(z_q.shape[0], -1)
