"""Reconstruction-fidelity metrics for the generative-downstream tables.

Each metric's input range is stated rather than assumed. LPIPS (AlexNet), PSNR,
and SSIM run on the repo's native [-1,1] images (PSNR and SSIM with
data_range=2.0). Only FID needs [0,1], converted by `to_unit` at the call site.
FID is not reported per (attack, eps) cell, since n=256 is below the sample size
it needs to be stable. Only a pooled rFID per attack is kept, and the paper
tables report LPIPS, PSNR, and SSIM only.
"""
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] to [0,1], clamped, on CPU. Used for FID only."""
    return ((x + 1) / 2).clamp(0, 1).cpu()


def new_fid():
    """A fresh FID accumulator (normalize=True, so it expects [0,1] inputs), on
    CPU. torchmetrics keeps a float64 covariance that MPS cannot hold, and the
    InceptionV3 features are deterministic, so the device does not affect the
    value. Returned raw so the caller can `.update()` it across an eps grid."""
    return FrechetInceptionDistance(feature=2048, normalize=True)


@torch.no_grad()
def lpips_psnr_ssim(a: torch.Tensor, b: torch.Tensor, device: str, bs: int = 16) -> dict:
    """LPIPS/PSNR/SSIM between two [-1,1] image tensors of identical shape. LPIPS
    is computed one image at a time, since torchmetrics reduces per call, so the
    result is a true mean over N rather than a mean of batch means."""
    psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    for i in range(0, a.shape[0], bs):
        ai, bi = a[i : i + bs].to(device), b[i : i + bs].to(device)
        psnr.update(ai, bi)
        ssim.update(ai, bi)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    per_image = [lpips(a[i : i + 1].to(device), b[i : i + 1].to(device)).item() for i in range(a.shape[0])]
    return dict(lpips=float(np.mean(per_image)), psnr=float(psnr.compute()), ssim=float(ssim.compute()))


@torch.no_grad()
def recon_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    """rFID between two [-1,1] image sets."""
    fid = new_fid()
    fid.update(to_unit(real), real=True)
    fid.update(to_unit(fake), real=False)
    return float(fid.compute())
