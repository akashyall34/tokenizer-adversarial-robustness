"""Shared scoring for every SQS/pixel-noise hyperparameter sweep: clean AUC
plus mean adversarial AUC recovered across the four cached attacks, scored on
SEL_ITEMS only (hyperparameter selection never touches REPORT_ITEMS).
"""
import numpy as np
import torch

from src.eval.attack_io import load_folder
from src.eval.metrics import auc

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]


@torch.no_grad()
def batched(fn, X: torch.Tensor, device: str, bs: int):
    out = []
    for i in range(0, X.shape[0], bs):
        out.append(fn(X[i:i + bs].to(device)))
    return np.concatenate(out)


def score_config(recon_fn, densenet_logit_fn, Xc_sel, Yc_sel, adv_dir: str, tokenizer_name: str,
                  sweep_eps, sel_idx, device: str, bs: int):
    """recon_fn(x) -> reconstructed image. Returns (auc_clean, mean_auc_adv, per_attack_list)."""
    recon_c = batched(lambda x: densenet_logit_fn(recon_fn(x)).cpu().numpy(), Xc_sel, device, bs)
    auc_clean = auc(Yc_sel, recon_c)
    recoveries = []
    for attack in ATTACKS_TO_DEFEND:
        Xa, Ya = load_folder(adv_dir, tokenizer_name, attack, sweep_eps, idx_subset=sel_idx)
        assert (Ya == Yc_sel).all()
        recon_a = batched(lambda x: densenet_logit_fn(recon_fn(x)).cpu().numpy(), Xa, device, bs)
        recoveries.append(auc(Yc_sel, recon_a))
    return auc_clean, float(np.mean(recoveries)), recoveries


def score_config_logit_ensemble(predict_fn, Xc_sel, Yc_sel, adv_dir: str, tokenizer_name: str,
                                 sweep_eps, sel_idx, device: str, bs: int):
    """predict_fn(x) returns a logit directly, not a reconstructed image, for a
    mechanism that averages K logits rather than voting on a token first."""
    clean_lg = batched(lambda x: predict_fn(x).cpu().numpy(), Xc_sel, device, bs)
    auc_clean = auc(Yc_sel, clean_lg)
    recoveries = []
    for attack in ATTACKS_TO_DEFEND:
        Xa, Ya = load_folder(adv_dir, tokenizer_name, attack, sweep_eps, idx_subset=sel_idx)
        assert (Ya == Yc_sel).all()
        adv_lg = batched(lambda x: predict_fn(x).cpu().numpy(), Xa, device, bs)
        recoveries.append(auc(Yc_sel, adv_lg))
    return auc_clean, float(np.mean(recoveries)), recoveries
