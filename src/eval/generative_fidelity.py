"""Reconstruction-fidelity grid: decode cached adversarial images through a
tokenizer (undefended or defended) and score LPIPS/PSNR/SSIM against both the
clean reconstruction and the original image, plus a pooled rFID per attack.

Shared body of the per-tokenizer generative-fidelity runs. It re-runs no attack:
it reads the cached adversarial pixels (`src/eval/attack_io.py`) and scores the
reconstruction, so the attack subset, seed, and preprocessing are exactly those
the adversarial `.pt` files were generated against.
"""
import os

import torch
from tqdm.auto import tqdm

from src.eval.attack_io import already_done, load_folder
from src.generative.fidelity import lpips_psnr_ssim, new_fid, recon_fid, to_unit

# paper attack ids in table order. M3_alpha03 is the auxiliary alpha=0.3 blend
# (not in the paper); it is generated for every tokenizer/dataset pair except
# FSQ/Kermany, so a pair that lacks it is simply skipped.
FIDELITY_ATTACKS = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]

# CSV columns. The `_DEFENSE` variants add a `defense` column (the fine-tune
# checkpoint id) right after `tokenizer`.
GRID_FIELDS = [
    "tokenizer", "attack", "eps255", "n", "fid_low_n",
    "lpips_vs_clean_recon", "psnr_vs_clean_recon", "ssim_vs_clean_recon",
    "lpips_vs_original", "psnr_vs_original", "ssim_vs_original",
]
CLEAN_FIELDS = [
    "tokenizer", "attack", "n", "fid_low_n",
    "lpips_recon_vs_original", "psnr_recon_vs_original",
    "ssim_recon_vs_original", "rfid_recon_vs_original",
]
RFID_FIELDS = ["tokenizer", "attack", "n_real", "n_fake", "rfid_pooled"]
GRID_FIELDS_DEFENSE = ["tokenizer", "defense"] + GRID_FIELDS[1:]
RFID_FIELDS_DEFENSE = ["tokenizer", "defense"] + RFID_FIELDS[1:]


def defended_checkpoint(ckpt_dir: str, tokenizer: str, defense: str, dataset: str) -> str:
    """Path of the fine-tune defense checkpoint, matching the names the
    `<tok>-defenses-<dataset>` pipeline writes (see scripts/defenses/*/run_advft*,
    run_decoder_realism*, merge_checkpoints*). Those scripts suffix the file with
    the dataset for APTOS and ISIC and leave Kermany unsuffixed. `defense` is a
    value from the fidelity CSV's `defense` column: encoder_ft_eps{0.5..8} |
    decoder_ft | combined. The combined checkpoint is always the eps=8 one,
    matching the attack budget."""
    suffix = "" if dataset == "kermany" else f"_{dataset}"
    if defense.startswith("encoder_ft_eps"):
        return os.path.join(ckpt_dir, f"{tokenizer}_advft_eps{defense[len('encoder_ft_eps'):]}{suffix}.pt")
    if defense == "decoder_ft":
        return os.path.join(ckpt_dir, f"{tokenizer}_decoder_realism_robust{suffix}.pt")
    if defense == "combined":
        return os.path.join(ckpt_dir, f"{tokenizer}_combined_eps8{suffix}.pt")
    raise ValueError(defense)


def batched_reconstruct(recon_fn, x: torch.Tensor, device: str, bs: int = 16) -> torch.Tensor:
    outs = []
    for i in range(0, x.shape[0], bs):
        with torch.no_grad():
            outs.append(recon_fn(x[i : i + bs].to(device)).clamp(-1, 1).cpu())
    return torch.cat(outs, 0)


def clean_baseline_row(tokenizer: str, Xc: torch.Tensor, Xc_recon: torch.Tensor, device: str) -> dict:
    """The undefended tokenizer's own reconstruction fidelity with no attack, the
    row the per-(attack, eps) grid has no place for. rFID here uses the original
    images as the real set, a different reference than the attack grid's pooled
    rFID (clean reconstruction), so the two are not directly differenceable."""
    m = lpips_psnr_ssim(Xc_recon, Xc, device)
    return dict(
        tokenizer=tokenizer, attack="clean_recon", n=len(Xc), fid_low_n=True,
        lpips_recon_vs_original=round(m["lpips"], 5),
        psnr_recon_vs_original=round(m["psnr"], 3),
        ssim_recon_vs_original=round(m["ssim"], 4),
        rfid_recon_vs_original=round(recon_fid(Xc, Xc_recon), 3),
    )


def fidelity_grid(tokenizer, recon_fn, adv_root, adv_tokenizer, Xc, Yc, eps_grid, device,
                  attacks=FIDELITY_ATTACKS):
    """Returns (grid_rows, rfid_rows). `adv_tokenizer` is the directory the cached
    adversarial images live under, always the undefended tokenizer's name, since
    the attack is generated once and every defense decodes the same pixels. An
    attack whose folder was never generated (FSQ/Kermany has no M3_alpha03) is
    skipped."""
    Xc_recon = batched_reconstruct(recon_fn, Xc, device)
    grid_rows, rfid_rows = [], []

    for attack in attacks:
        eps_present = [e for e in eps_grid if already_done(adv_root, adv_tokenizer, attack, e, len(Yc))]
        if not eps_present:
            continue

        fid_pooled = new_fid()
        fid_pooled.update(to_unit(Xc_recon), real=True)
        for eps in tqdm(eps_present, desc=f"{tokenizer}/{attack}", leave=False):
            Xa, Ya = load_folder(adv_root, adv_tokenizer, attack, eps)
            assert (Ya == Yc).all(), "cached adversarial order does not match the clean subset order"
            Xa_recon = batched_reconstruct(recon_fn, Xa, device)
            m_clean = lpips_psnr_ssim(Xa_recon, Xc_recon, device)
            m_orig = lpips_psnr_ssim(Xa_recon, Xc, device)
            fid_pooled.update(to_unit(Xa_recon), real=False)
            grid_rows.append(dict(
                tokenizer=tokenizer, attack=attack, eps255=eps, n=len(Ya), fid_low_n=True,
                lpips_vs_clean_recon=round(m_clean["lpips"], 5),
                psnr_vs_clean_recon=round(m_clean["psnr"], 3),
                ssim_vs_clean_recon=round(m_clean["ssim"], 4),
                lpips_vs_original=round(m_orig["lpips"], 5),
                psnr_vs_original=round(m_orig["psnr"], 3),
                ssim_vs_original=round(m_orig["ssim"], 4),
            ))

        rfid_rows.append(dict(
            tokenizer=tokenizer, attack=attack, n_real=len(Xc_recon),
            n_fake=len(Xc_recon) * len(eps_present),
            rfid_pooled=round(float(fid_pooled.compute()), 3),
        ))
        del fid_pooled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return grid_rows, rfid_rows
