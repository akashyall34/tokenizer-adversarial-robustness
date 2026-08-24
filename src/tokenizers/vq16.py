"""VQ-16 (LlamaGen) tokenizer wrapper.

Requires the LlamaGen source tree on the path (see scripts/setup_third_party.sh,
which clones it into third_party/LlamaGen) and the vq_ds16_c2i.pt checkpoint.
"""
import sys

import torch


def load_vq16_net(llamagen_repo: str, ckpt_path: str, device: str):
    """Loads a fresh, frozen (eval mode, requires_grad=False) VQ-16 net.
    Exposed standalone (not just inside VQ16Tokenizer) because the defense
    fine-tuning loop needs a new trainable copy per eps arm -- see
    src/defenses/vq16/margin.py."""
    if llamagen_repo not in sys.path:
        sys.path.insert(0, llamagen_repo)
    from tokenizer.tokenizer_image.vq_model import VQ_models

    model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    model.load_state_dict(sd)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class VQ16Tokenizer:
    name = "vq16"

    def __init__(self, llamagen_repo: str, ckpt_path: str, device: str):
        self.net = load_vq16_net(llamagen_repo, ckpt_path, device)
        self.device = device

    def codebook(self) -> torch.Tensor:
        return self.net.quantize.embedding.weight  # [16384, 8]

    def encode_prequant(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous latent before quantization, differentiable wrt x.
        [B,3,H,W] -> [B, N, 8]."""
        h = self.net.quant_conv(self.net.encoder(x))
        B, C, H, W = h.shape
        return h.permute(0, 2, 3, 1).reshape(B, H * W, C)

    @torch.no_grad()
    def assign_indices(self, x: torch.Tensor) -> torch.Tensor:
        _, _, info = self.net.encode(x)
        return info[2].reshape(x.shape[0], -1)  # LlamaGen: (perplexity, min_encodings, indices)

    @torch.no_grad()
    def clean_top1(self, z: torch.Tensor) -> torch.Tensor:
        """z:[B,N,D] -> [B,N] nearest codebook id (for margin attacks)."""
        E = self.codebook()
        d = z.pow(2).sum(-1, keepdim=True) - 2 * z @ E.T + E.pow(2).sum(-1)
        return d.argmin(-1)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """encode -> decode, in [-1,1]. Differentiable (for the M1 attack)."""
        quant, _, _ = self.net.encode(x)
        return self.net.decode(quant).clamp(-1, 1)

    @torch.no_grad()
    def probe_feat(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened post-quant features: [B,8,16,16] -> [B,2048]."""
        quant, _, _ = self.net.encode(x)
        return quant.reshape(quant.shape[0], -1)
