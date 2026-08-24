"""FSQ (Finite Scalar Quantization) tokenizer wrapper, 16k-codebook config.

Requires the FSQ-pytorch source tree on disk (see scripts/setup_third_party.sh,
which clones it into third_party/FSQ-pytorch) and the fsq_ckpt_16k.pt checkpoint.
"""
import sys
import types

import torch

LEVELS = [8, 8, 8, 6, 5]  # 16k-codebook config


def fsq_bound(z: torch.Tensor, levels=LEVELS, eps: float = 1e-3) -> torch.Tensor:
    L = torch.as_tensor(levels, dtype=torch.float32, device=z.device)
    half = (L - 1) * (1 - eps) / 2
    offset = torch.where(L % 2 == 0, torch.tensor(0.5, device=z.device), torch.tensor(0.0, device=z.device))
    shift = torch.tan(offset / half)
    return torch.tanh(z + shift) * half - offset


def fsq_token_margins(z: torch.Tensor, levels=LEVELS, eps: float = 1e-3) -> torch.Tensor:
    """z:[...,5] -> per-token margin = min over dims of (0.5 - |u-round(u)|). Low = wobbly."""
    u = fsq_bound(z, levels, eps)
    return (0.5 - (u - u.round()).abs()).min(dim=-1).values


class FSQTokenizer:
    name = "fsq"

    def __init__(self, fsq_repo: str, ckpt_path: str, device: str):
        if fsq_repo not in sys.path:
            sys.path.insert(0, fsq_repo)
        # avoid a module-name collision between FSQ-pytorch's `model` module
        # and any other package also named `model` already imported
        for m in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
            del sys.modules[m]
        from model import VQVAE

        args = types.SimpleNamespace(
            in_channel=3, channel=512, quantizer="fsq", levels=LEVELS,
            embed_dim=len(LEVELS), lfq_dim=10, entropy_loss_weight=0.0,
            codebook_loss_weight=0.0, n_embed=1024,
        )
        net = VQVAE(args).to(device).eval()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.device = device

    def encode_prequant(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B,N,5], differentiable."""
        z = self.net.enc(x)
        B, C, Hp, Wp = z.shape
        return z.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, C)

    @torch.no_grad()
    def assign_indices(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net.enc(x)
        _, idx = self.net.quantize_t(z)
        return idx.reshape(idx.shape[0], -1)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """encode -> quantize -> decode. Dodges a CUDA issue in the repo's own encode()."""
        z = self.net.enc(x)
        quant_t, _ = self.net.quantize_t(z)
        return self.net.decode(quant_t).clamp(-1, 1)

    @torch.no_grad()
    def probe_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened post-quant features: [B, 5*H'*W']."""
        z = self.net.enc(x)
        quant_t, _ = self.net.quantize_t(z)
        return quant_t.reshape(quant_t.shape[0], -1)
