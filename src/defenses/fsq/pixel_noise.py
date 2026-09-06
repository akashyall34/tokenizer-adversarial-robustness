"""Pixel-input noise baseline for FSQ, the RQ2 comparison point, same role as
every other tokenizer's pixel_noise.py. Same K-pass majority-vote-per-dim
mechanism as flat SQS, with noise at the pixel input instead of the
pre-quantization latent.
"""
import torch

from src.defenses.fsq.margin import decode_only, half_width, unflatten_quant
from src.tokenizers.fsq import fsq_bound


def pixel_noise_quantize(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B = x.shape[0]
    x_rep = (x.repeat(K, 1, 1, 1) + sigma * torch.randn(K * B, *x.shape[1:], device=x.device)).clamp(-1, 1)
    with torch.no_grad():
        z = tok.encode_prequant(x_rep)
    _, N, D = z.shape
    levels = fsq_bound(z).round()
    stacked = levels.view(K, B, N, D)
    voted = torch.mode(stacked, dim=0).values
    quant_t = voted / half_width(z.device)
    return voted, quant_t


@torch.no_grad()
def pixel_noise_reconstruct(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, quant_t = pixel_noise_quantize(tok, x, K, sigma, seed=seed)
    return decode_only(tok.net, quant_t)


@torch.no_grad()
def pixel_noise_probe_feat(tok, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, quant_t = pixel_noise_quantize(tok, x, K, sigma, seed=seed)
    B, N, D = quant_t.shape
    Hp = Wp = int(round(N ** 0.5))
    quant_t_spatial = unflatten_quant(quant_t, Hp, Wp)
    return quant_t_spatial.reshape(quant_t_spatial.shape[0], -1)
