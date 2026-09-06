"""Pixel-input noise baseline, the direct RQ2 comparison point for SQS.

Same K-pass majority-vote mechanism as flat SQS (src/defenses/vq16/sqs.py),
but the noise goes onto the raw pixel input before encoding, not onto the
pre-quantization latent. Answers "does noise right before quantization beat
noise at the pixel input" with a real, fairly-swept baseline rather than an
assumed strawman.
"""
import torch


def pixel_noise_quantize(model, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    B = x.shape[0]
    x_rep = (x.repeat(K, 1, 1, 1) + sigma * torch.randn(K * B, *x.shape[1:], device=x.device)).clamp(-1, 1)
    with torch.no_grad():
        _, _, info = model.encode(x_rep)
        h_shape = model.quant_conv(model.encoder(x[:1])).shape  # (1,C,Hp,Wp), just for C/Hp/Wp
    _, C, Hp, Wp = h_shape
    voted_idx = torch.mode(info[2].view(K, B, Hp * Wp), dim=0).values
    z_q = model.quantize.get_codebook_entry(voted_idx.view(-1), shape=(B, C, Hp, Wp), channel_first=True)
    return voted_idx, z_q


@torch.no_grad()
def pixel_noise_reconstruct(model, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, z_q = pixel_noise_quantize(model, x, K, sigma, seed=seed)
    return model.decode(z_q).clamp(-1, 1)


@torch.no_grad()
def pixel_noise_probe_feat(model, x: torch.Tensor, K: int, sigma: float, seed: int = None):
    _, z_q = pixel_noise_quantize(model, x, K, sigma, seed=seed)
    return z_q.reshape(z_q.shape[0], -1)
