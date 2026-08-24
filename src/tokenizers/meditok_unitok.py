"""MediTok and UniTok tokenizer wrappers (multi-codebook, cosine-similarity quantizers).

Both repos define modules under similarly-named top-level packages (`models`,
`utils`, `quant`, ...), so `use_repo` swaps sys.path and purges those modules
around each import to avoid cross-contamination between the two.

Requires the MediTok and UniTok source trees on disk (see
scripts/setup_third_party.sh, which clones them into third_party/).
"""
import contextlib
import sys

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files

_PURGE = (
    "models", "utils", "open_clip", "layers", "local_openclip", "trainers",
    "utilities", "datasets", "evaluation", "quant", "vitamin", "vqvae",
    "dinov2", "meditok", "unitok", "wrapper", "discrim",
)


@contextlib.contextmanager
def use_repo(path: str):
    saved = list(sys.path)
    for n in [m for m in sys.modules if m.split(".")[0] in _PURGE]:
        del sys.modules[n]
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = saved
        for n in [m for m in sys.modules if m.split(".")[0] in _PURGE]:
            del sys.modules[n]


def _strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for p in ("_orig_mod.", "module."):
            if k.startswith(p):
                k = k[len(p):]
        if k.startswith("core."):
            k = k[len("core."):]
        out[k] = v
    return out


def _extract_meditok_state_dict(blob) -> dict:
    def looks_like_meditok(d):
        return isinstance(d, dict) and any("quantizer.codebooks.0.codebook.weight" in k for k in d)

    if looks_like_meditok(blob):
        return _strip_prefixes(blob)
    for v in blob.values():
        if looks_like_meditok(v):
            return _strip_prefixes(v)
        if isinstance(v, dict):
            for v2 in v.values():
                if looks_like_meditok(v2):
                    return _strip_prefixes(v2)
    raise RuntimeError("meditok weights not found in checkpoint")


def _reconstruct(net, x: torch.Tensor) -> torch.Tensor:
    """encode -> quantize -> post_quant_proj -> decoder, in [-1,1]. Differentiable."""
    t = net.encoder(x)
    t = net.quant_proj(t)
    t, _, _, _ = net.quantizer(t)
    t = net.post_quant_proj(t)
    return net.decoder(t).clamp(-1, 1)


class MultiCodebookTokenizer:
    def __init__(self, name: str, net):
        self.name = name
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.quantizer = net.quantizer
        self.num_codebooks = net.quantizer.num_codebooks
        self.codebooks = net.quantizer.codebooks

    def encode_prequant(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.quant_proj(self.net.encoder(x))  # [B,N,D]

    def assign_indices(self, z: torch.Tensor):
        return self.quantizer.f_to_idx(z)  # [B,C,N]

    @torch.no_grad()
    def probe_feat(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_prequant(x)
        zq = self.quantizer.idx_to_f(self.quantizer.f_to_idx(z))
        return zq.reshape(zq.shape[0], -1)  # flatten post-quant

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return _reconstruct(self.net, x)

    def clean_top1(self, z: torch.Tensor):
        """z:[B,N,D] -> per-codebook nearest id list (for margin attacks)."""
        chunks = z.split(z.shape[-1] // self.num_codebooks, dim=-1)
        return [
            (F.normalize(chunks[i], dim=-1) @ F.normalize(cb.codebook.weight, dim=1).T).argmax(-1)
            for i, cb in enumerate(self.codebooks)
        ]


def load_meditok(repo_dir: str, ckpt_dir: str, img_size: int, device: str) -> MultiCodebookTokenizer:
    with use_repo(repo_dir):
        from models.meditok import MedITok, get_meditok_args

        files = [f for f in list_repo_files("massaki75/meditok") if f.endswith((".pth", ".pt"))]
        ckpt_name = "meditok_ckpt_v1.pth" if "meditok_ckpt_v1.pth" in files else files[0]
        ck = hf_hub_download("massaki75/meditok", ckpt_name, local_dir=f"{ckpt_dir}/meditok")
        net = MedITok(get_meditok_args(img_size=img_size, grad_ckpt=False))
        net.load_state_dict(_extract_meditok_state_dict(torch.load(ck, map_location="cpu", weights_only=False)), strict=False)
    return MultiCodebookTokenizer("meditok", net.to(device))


def load_unitok(repo_dir: str, ckpt_dir: str, device: str) -> MultiCodebookTokenizer:
    with use_repo(repo_dir):
        from models.unitok import UniTok
        from utils.config import Args

        files = [f for f in list_repo_files("FoundationVision/unitok_tokenizer") if f.endswith((".pth", ".pt", ".bin"))]
        ck = hf_hub_download("FoundationVision/unitok_tokenizer", files[0], local_dir=f"{ckpt_dir}/unitok")
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = Args()
        cfg.load_state_dict(ckpt["args"])
        net = UniTok(cfg)
        sd = ckpt["trainer"]["unitok"] if ("trainer" in ckpt and "unitok" in ckpt["trainer"]) else ckpt.get("unitok", ckpt)
        net.load_state_dict(sd, strict=False)
    return MultiCodebookTokenizer("unitok", net.to(device))
