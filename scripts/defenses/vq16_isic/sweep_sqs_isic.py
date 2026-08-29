#!/usr/bin/env python3
"""Full SQS defense sweep for VQ-16 on ISIC: one pipeline covering the
sanity check, the pixel-noise baseline (RQ2's comparison point), all three
token-voting SQS variants (flat, spatial-consensus, margin-adaptive), the
K x sigma check (is K worth varying, or is it inert like every other SQS
sweep suggested?), the logit-ensemble diagnostic (a different smoothing
POINT, not a different sigma -- see src/defenses/vq16/sqs.py), and finally
picking the one standalone winner used downstream for advft+SQS and the BPDA
adaptive-attack check.

Every hyperparameter is selected from a real measured tradeoff on SEL_ITEMS
only (never REPORT_ITEMS), at the worst-case eps in the config's eps_grid --
Part 3 (final reporting) verifies the chosen parameters across the full eps
grid separately, so selection and reporting never touch the same numbers.

Each sweep's CSV is cached under --out-dir; rerunning this script skips any
sweep whose CSV already exists rather than recomputing it.

Usage:
    python scripts/defenses/vq16_isic/sweep_sqs_isic.py --data-root /path/to/isic2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.vq16.margin import prequant
from src.defenses.vq16.margin import reconstruct as margin_reconstruct
from src.defenses.vq16.pixel_noise import pixel_noise_reconstruct
from src.defenses.vq16.sqs import (
    sqs_logit_ensemble_predict, sqs_reconstruct, sqs_reconstruct_adaptive, token_margins_from_h,
)
from src.eval.sel_report_split import stratified_half_split
from src.eval.sqs_sweep import score_config, score_config_logit_ensemble
from src.models.efficientnet_isic_classifier import EfficientNetB4Classifier, load_efficientnet_b4
from src.tokenizers.vq16 import load_vq16_net


def load_densenet(ckpt_path, device):
    """Alias: wrap EfficientNet-B4 into
    the same load_ckpt(path, device) -> classifier object interface this
    script's downstream code (margin/decoder-realism/certify helpers, all
    written against DenseNet's raw-nn.Module calling convention: net(x) for
    the logit, net.features(x) for perceptual features) expects, so none of
    that shared code needs touching."""
    net = load_efficientnet_b4(ckpt_path, device)
    return EfficientNetB4Classifier(net, device)


def densenet_logit(densenet, x):
    return densenet(x)

CLEAN_AUC_FLOOR = 0.85  # a defense that tanks clean performance to "win" doesn't count
K_SWEEP = 16
SIGMA_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
SIGMA_GRID_FLAT_EXTENDED = SIGMA_GRID + [0.05, 0.1]  # flat only -- confirms the plateau continues
K_GRID = [4, 8, 16, 32]
SPATIAL_WEIGHT_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
AGREEMENT_THRESH_GRID = [50, 60, 70, 80, 90]
SIGMA_MULT_GRID = [2.0, 4.0, 8.0]
LOGIT_ENSEMBLE_SIGMA_GRID = [0.005, 0.01, 0.02]  # top half of SIGMA_GRID -- sigma barely mattered
                                                   # for token-voting across its whole tested range


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--sweep-eps", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def pick_best(df, score_col, clean_col="auc_clean"):
    ok = df[df[clean_col] >= CLEAN_AUC_FLOOR]
    if len(ok):
        return ok.loc[ok[score_col].idxmax()]
    print(f"*** NOTHING clears the clean-AUC floor ({CLEAN_AUC_FLOOR}) -- picking best clean-preserving config instead ***")
    return df.loc[df[clean_col].idxmax()]


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    sweep_eps = args.sweep_eps if args.sweep_eps is not None else cfg["eps_grid"][-1]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, _ = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)
    bs = cfg["attack_batch"]

    def clf(x):
        return densenet_logit(densenet, x)

    def score(recon_fn):
        return score_config(recon_fn, clf, Xc_sel, Yc_sel, args.adv_dir, "vq16", sweep_eps, sel_idx, device, bs)

    # --- sanity check: sqs_quantize(sigma=0) should match the real quantizer's own reconstruct ---
    diff = (sqs_reconstruct(orig_net, Xc_sel[:16], K=1, sigma=0.0, spatial=False)
            - margin_reconstruct(orig_net, Xc_sel[:16])).abs().max().item()
    print(f"sqs_quantize(sigma=0) vs. real reconstruct: max abs diff = {diff:.6f}")
    if diff > 1e-2:
        print("*** MISMATCH exceeds the expected float32-decoder noise floor -- a REAL logic error. "
              "Needs fixing before trusting any SQS number. ***")
    else:
        print("within the expected float32-through-decoder noise floor (~0.004) -- safe to proceed.")

    # --- pixel-noise baseline (RQ2 comparison point) ---
    csv_pixel = os.path.join(args.out_dir, "vq16_pixel_noise_sweep.csv")
    if os.path.isfile(csv_pixel):
        print(f"already computed -> {csv_pixel}")
        pixel_df = pd.read_csv(csv_pixel)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID, desc="pixel-noise sweep"):
            auc_clean, mean_recovery, per_attack = score(lambda x, s=sigma: pixel_noise_reconstruct(orig_net, x, K_SWEEP, s))
            rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        pixel_df = pd.DataFrame(rows)
        pixel_df.to_csv(csv_pixel, index=False)
        print("wrote ->", csv_pixel)
    print(pixel_df.to_string(index=False))
    pixel_best = pick_best(pixel_df, "mean_auc_adv")
    print(f"selected pixel-noise baseline: sigma={pixel_best.sigma}, K={K_SWEEP}\n")

    # --- flat SQS sweep ---
    csv_flat = os.path.join(args.out_dir, "vq16_sqs_flat_sweep.csv")
    if os.path.isfile(csv_flat):
        print(f"already computed -> {csv_flat}")
        flat_df = pd.read_csv(csv_flat)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID_FLAT_EXTENDED, desc="flat SQS sweep"):
            auc_clean, mean_recovery, _ = score(lambda x, s=sigma: sqs_reconstruct(orig_net, x, K_SWEEP, s, spatial=False))
            rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        flat_df = pd.DataFrame(rows)
        flat_df.to_csv(csv_flat, index=False)
        print("wrote ->", csv_flat)
    print(flat_df.to_string(index=False))
    flat_best = pick_best(flat_df, "mean_auc_adv")
    sqs_sigma_flat = float(flat_best.sigma)
    print(f"selected flat SQS: sigma={sqs_sigma_flat}, K={K_SWEEP}\n")

    # --- K x sigma: is K worth varying, or inert? ---
    csv_k_sigma = os.path.join(args.out_dir, "vq16_sqs_k_sigma_grid.csv")
    if os.path.isfile(csv_k_sigma):
        print(f"already computed -> {csv_k_sigma}")
        k_sigma_df = pd.read_csv(csv_k_sigma)
    else:
        rows = []
        for k in tqdm(K_GRID, desc="K x sigma sweep"):
            for sigma in SIGMA_GRID:
                auc_clean, mean_recovery, _ = score(lambda x, kk=k, s=sigma: sqs_reconstruct(orig_net, x, kk, s, spatial=False))
                rows.append(dict(K=k, sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        k_sigma_df = pd.DataFrame(rows)
        k_sigma_df.to_csv(csv_k_sigma, index=False)
        print("wrote ->", csv_k_sigma)
    print(f"mean_auc_adv range across the K x sigma grid: {k_sigma_df.mean_auc_adv.min():.4f} - {k_sigma_df.mean_auc_adv.max():.4f}\n")

    # --- spatial-consensus SQS sweep ---
    csv_spatial = os.path.join(args.out_dir, "vq16_sqs_spatial_sweep.csv")
    if os.path.isfile(csv_spatial):
        print(f"already computed -> {csv_spatial}")
        spatial_df = pd.read_csv(csv_spatial)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID, desc="spatial SQS sweep"):
            for sw in SPATIAL_WEIGHT_GRID:
                for th in AGREEMENT_THRESH_GRID:
                    auc_clean, mean_recovery, _ = score(
                        lambda x, s=sigma, w=sw, t=th: sqs_reconstruct(
                            orig_net, x, K_SWEEP, s, spatial=True, spatial_weight=w, agreement_thresh_pct=t))
                    rows.append(dict(sigma=sigma, spatial_weight=sw, agreement_thresh_pct=th,
                                      auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        spatial_df = pd.DataFrame(rows)
        spatial_df.to_csv(csv_spatial, index=False)
        print(f"wrote full sweep -> {csv_spatial} ({len(spatial_df)} configs)")
    spatial_best = pick_best(spatial_df, "mean_auc_adv")
    sqs_sigma_spatial = float(spatial_best.sigma)
    sqs_spatial_weight = float(spatial_best.spatial_weight)
    sqs_agreement_thresh = float(spatial_best.agreement_thresh_pct)
    print(f"selected spatial SQS: sigma={sqs_sigma_spatial}, spatial_weight={sqs_spatial_weight}, "
          f"agreement_thresh_pct={sqs_agreement_thresh}, K={K_SWEEP}")
    print(f"clean AUC={spatial_best.auc_clean}, mean adversarial AUC={spatial_best.mean_auc_adv}\n")

    # --- margin-adaptive noise: calibrate margin_low/margin_high from the real clean margin distribution ---
    with torch.no_grad():
        h_clean = prequant(orig_net, Xc_sel)
        clean_margins = token_margins_from_h(orig_net, h_clean).flatten().cpu().numpy()
    margin_low = float(np.percentile(clean_margins, 10))
    margin_high = float(np.percentile(clean_margins, 90))
    print(f"clean margin distribution: p10={margin_low:.4f}  median={np.median(clean_margins):.4f}  p90={margin_high:.4f}")

    csv_adaptive = os.path.join(args.out_dir, "vq16_sqs_margin_adaptive_sweep.csv")
    if os.path.isfile(csv_adaptive):
        print(f"already computed -> {csv_adaptive}")
        adaptive_df = pd.read_csv(csv_adaptive)
    else:
        rows = []
        for sigma_base in tqdm(SIGMA_GRID, desc="margin-adaptive sweep"):
            for mult in SIGMA_MULT_GRID:
                auc_clean, mean_recovery, _ = score(
                    lambda x, s=sigma_base, m=mult: sqs_reconstruct_adaptive(orig_net, x, K_SWEEP, s, margin_low, margin_high, m))
                rows.append(dict(sigma_base=sigma_base, sigma_mult_max=mult,
                                  auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        adaptive_df = pd.DataFrame(rows)
        adaptive_df.to_csv(csv_adaptive, index=False)
        print("wrote ->", csv_adaptive)
    adaptive_best = pick_best(adaptive_df, "mean_auc_adv")
    sqs_sigma_adaptive_base = float(adaptive_best.sigma_base)
    sqs_sigma_mult_max = float(adaptive_best.sigma_mult_max)
    print(f"selected margin-adaptive SQS: sigma_base={sqs_sigma_adaptive_base}, sigma_mult_max={sqs_sigma_mult_max}, K={K_SWEEP}")
    print(f"clean AUC={adaptive_best.auc_clean}, mean adversarial AUC={adaptive_best.mean_auc_adv}\n")

    # --- logit-ensemble diagnostic: a different smoothing POINT, not a different sigma ---
    def score_logit(predict_fn):
        return score_config_logit_ensemble(predict_fn, Xc_sel, Yc_sel, args.adv_dir, "vq16", sweep_eps, sel_idx, device, bs)

    csv_logit = os.path.join(args.out_dir, "vq16_sqs_logit_ensemble_sweep.csv")
    rows = []
    for sigma in tqdm(LOGIT_ENSEMBLE_SIGMA_GRID, desc="logit-ensemble SQS sweep"):
        auc_clean, mean_recovery, per_attack = score_logit(
            lambda x, s=sigma: sqs_logit_ensemble_predict(orig_net, densenet, x, K_SWEEP, s))
        rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4),
                          auc_adv_M1=round(per_attack[0], 4), auc_adv_M3_alpha03=round(per_attack[1], 4)))
    logit_df = pd.DataFrame(rows)
    logit_df.to_csv(csv_logit, index=False)
    print("wrote ->", csv_logit)
    print(logit_df.to_string(index=False))
    print("Compare auc_adv_M1/auc_adv_M3_alpha03 here against flat/spatial/adaptive's same columns "
          "(all ~0.0 at every config). If still ~0.0, the dense/redundant attack pattern survives "
          "per-sample noise averaging too, not just per-token voting.\n")

    # --- pick the one standalone winner for advft+SQS and the BPDA check ---
    summary = pd.DataFrame([
        dict(variant="flat", auc_clean=flat_best.auc_clean, mean_auc_adv=flat_best.mean_auc_adv),
        dict(variant="spatial", auc_clean=spatial_best.auc_clean, mean_auc_adv=spatial_best.mean_auc_adv),
        dict(variant="margin_adaptive", auc_clean=adaptive_best.auc_clean, mean_auc_adv=adaptive_best.mean_auc_adv),
    ])
    print(summary.to_string(index=False))
    ok = summary[summary.auc_clean >= CLEAN_AUC_FLOOR]
    pool = ok if len(ok) else summary
    if not len(ok):
        print(f"*** NONE of the three variants clear the clean-AUC floor ({CLEAN_AUC_FLOOR}) -- "
              f"picking the best clean-preserving variant instead ***")
    best_name = pool.loc[pool.mean_auc_adv.idxmax(), "variant"] if len(ok) else pool.loc[pool.auc_clean.idxmax(), "variant"]
    print(f"\nWINNER: {best_name}")

    winner_params = {
        "flat": dict(K=K_SWEEP, sigma=sqs_sigma_flat),
        "spatial": dict(K=K_SWEEP, sigma=sqs_sigma_spatial, spatial_weight=sqs_spatial_weight,
                         agreement_thresh_pct=sqs_agreement_thresh),
        "margin_adaptive": dict(K=K_SWEEP, sigma_base=sqs_sigma_adaptive_base, margin_low=margin_low,
                                 margin_high=margin_high, sigma_mult_max=sqs_sigma_mult_max),
    }
    winner_config = dict(name=best_name, params=winner_params[best_name], all_variant_params=winner_params,
                          pixel_noise_sigma=float(pixel_best.sigma), pixel_noise_K=K_SWEEP)
    winner_path = os.path.join(args.out_dir, "vq16_sqs_winner.json")
    with open(winner_path, "w") as f:
        json.dump(winner_config, f, indent=2)
    print(f"wrote winner config -> {winner_path}")


if __name__ == "__main__":
    main()
