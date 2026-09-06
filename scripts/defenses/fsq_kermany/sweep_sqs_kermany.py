#!/usr/bin/env python3
"""Full SQS defense sweep for FSQ on Kermany: sanity check, pixel-noise
baseline, all three token-voting SQS variants (flat, spatial-consensus,
margin-adaptive), the K x sigma check, the logit-ensemble diagnostic, and
finally picking the one standalone winner used downstream for advft+SQS and
the BPDA check. Same role and SEL_ITEMS-only selection discipline as
scripts/defenses/fsq_aptos/sweep_sqs_aptos.py.

Every sigma/K value here starts from the same grid VQ-16/MediTok's sweeps
use (both cosine/unit-normalized spaces, plausibly similar order of
magnitude to FSQ's own bounded z-space) but is a new measured constant for
FSQ, not assumed equal; confirmed by this sweep against real data, same
rule every hyperparameter in this project follows.

Usage:
    python scripts/defenses/fsq_kermany/sweep_sqs_kermany.py \
        --data-root /path/to/chest_xray --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
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

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.defenses.fsq.pixel_noise import pixel_noise_reconstruct
from src.defenses.fsq.sqs import sqs_logit_ensemble_predict, sqs_reconstruct, sqs_reconstruct_adaptive
from src.eval.sel_report_split import stratified_half_split
from src.eval.sqs_sweep import score_config, score_config_logit_ensemble
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.fsq import FSQTokenizer, fsq_token_margins

CLEAN_AUC_FLOOR = 0.85
K_SWEEP = 16
SIGMA_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
K_GRID = [4, 8, 16, 32]
SPATIAL_WEIGHT_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
AGREEMENT_THRESH_GRID = [50, 60, 70, 80, 90]
SIGMA_MULT_GRID = [2.0, 4.0, 8.0]
LOGIT_ENSEMBLE_SIGMA_GRID = [0.005, 0.01, 0.02]


def load_densenet(ckpt_path, device):
    net = load_efficientnet(ckpt_path, device)
    return EfficientNetClassifier(net, device)


def densenet_logit(densenet, x):
    return densenet(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--fsq-repo", required=True)
    p.add_argument("--fsq-ckpt", required=True)
    p.add_argument("--densenet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--sweep-eps", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def pick_best(df, score_col, clean_col="auc_clean"):
    ok = df[df[clean_col] >= CLEAN_AUC_FLOOR]
    if len(ok):
        return ok.loc[ok[score_col].idxmax()]
    print(f"*** NOTHING clears the clean-AUC floor ({CLEAN_AUC_FLOOR}); picking best clean-preserving config instead ***")
    return df.loc[df[clean_col].idxmax()]


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    sweep_eps = args.sweep_eps if args.sweep_eps is not None else cfg["eps_grid"][-1]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, _ = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.densenet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)
    bs = cfg["attack_batch"]

    def clf(x):
        return densenet_logit(densenet, x)

    def score(recon_fn):
        return score_config(recon_fn, clf, Xc_sel, Yc_sel, args.adv_dir, "fsq", sweep_eps, sel_idx, device, bs)

    # --- sanity check: sqs_reconstruct(sigma=0) should match the real quantizer's own reconstruct ---
    diff = (sqs_reconstruct(orig_tok, Xc_sel[:16], K=1, sigma=0.0, spatial=False)
            - orig_tok.reconstruct(Xc_sel[:16])).abs().max().item()
    print(f"sqs_reconstruct(sigma=0) vs. real reconstruct: max abs diff = {diff:.6f}")
    if diff > 1e-2:
        print("*** MISMATCH exceeds the expected float32-decoder noise floor; a REAL logic error. "
              "Needs fixing before trusting any SQS number. ***")
    else:
        print("within the expected float32-through-decoder noise floor; safe to proceed.")

    # --- pixel-noise baseline ---
    csv_pixel = os.path.join(args.out_dir, "fsq_pixel_noise_sweep.csv")
    if os.path.isfile(csv_pixel):
        print(f"already computed -> {csv_pixel}")
        pixel_df = pd.read_csv(csv_pixel)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID, desc="pixel-noise sweep"):
            auc_clean, mean_recovery, _ = score(lambda x, s=sigma: pixel_noise_reconstruct(orig_tok, x, K_SWEEP, s))
            rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        pixel_df = pd.DataFrame(rows)
        pixel_df.to_csv(csv_pixel, index=False)
        print("wrote ->", csv_pixel)
    print(pixel_df.to_string(index=False))
    pixel_best = pick_best(pixel_df, "mean_auc_adv")
    print(f"selected pixel-noise baseline: sigma={pixel_best.sigma}, K={K_SWEEP}\n")

    # --- flat SQS sweep ---
    csv_flat = os.path.join(args.out_dir, "fsq_sqs_flat_sweep.csv")
    if os.path.isfile(csv_flat):
        print(f"already computed -> {csv_flat}")
        flat_df = pd.read_csv(csv_flat)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID, desc="flat SQS sweep"):
            auc_clean, mean_recovery, _ = score(lambda x, s=sigma: sqs_reconstruct(orig_tok, x, K_SWEEP, s, spatial=False))
            rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        flat_df = pd.DataFrame(rows)
        flat_df.to_csv(csv_flat, index=False)
        print("wrote ->", csv_flat)
    print(flat_df.to_string(index=False))
    flat_best = pick_best(flat_df, "mean_auc_adv")
    sqs_sigma_flat = float(flat_best.sigma)
    print(f"selected flat SQS: sigma={sqs_sigma_flat}, K={K_SWEEP}\n")

    # --- K x sigma: is K worth varying, or inert? ---
    csv_k_sigma = os.path.join(args.out_dir, "fsq_sqs_k_sigma_grid.csv")
    if os.path.isfile(csv_k_sigma):
        print(f"already computed -> {csv_k_sigma}")
        k_sigma_df = pd.read_csv(csv_k_sigma)
    else:
        rows = []
        for k in tqdm(K_GRID, desc="K x sigma sweep"):
            for sigma in SIGMA_GRID:
                auc_clean, mean_recovery, _ = score(lambda x, kk=k, s=sigma: sqs_reconstruct(orig_tok, x, kk, s, spatial=False))
                rows.append(dict(K=k, sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4)))
        k_sigma_df = pd.DataFrame(rows)
        k_sigma_df.to_csv(csv_k_sigma, index=False)
        print("wrote ->", csv_k_sigma)
    print(f"mean_auc_adv range across the K x sigma grid: {k_sigma_df.mean_auc_adv.min():.4f} - {k_sigma_df.mean_auc_adv.max():.4f}\n")

    # --- spatial-consensus SQS sweep ---
    csv_spatial = os.path.join(args.out_dir, "fsq_sqs_spatial_sweep.csv")
    if os.path.isfile(csv_spatial):
        print(f"already computed -> {csv_spatial}")
        spatial_df = pd.read_csv(csv_spatial)
    else:
        rows = []
        for sigma in tqdm(SIGMA_GRID[:6], desc="spatial SQS sweep"):
            for sw in SPATIAL_WEIGHT_GRID:
                for th in AGREEMENT_THRESH_GRID:
                    auc_clean, mean_recovery, _ = score(
                        lambda x, s=sigma, w=sw, t=th: sqs_reconstruct(
                            orig_tok, x, K_SWEEP, s, spatial=True, spatial_weight=w, agreement_thresh_pct=t))
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
        z_clean = orig_tok.encode_prequant(Xc_sel)
        clean_margins = fsq_token_margins(z_clean).flatten().cpu().numpy()
    margin_low = float(np.percentile(clean_margins, 10))
    margin_high = float(np.percentile(clean_margins, 90))
    print(f"clean margin distribution: p10={margin_low:.4f}  median={np.median(clean_margins):.4f}  p90={margin_high:.4f}")

    csv_adaptive = os.path.join(args.out_dir, "fsq_sqs_margin_adaptive_sweep.csv")
    if os.path.isfile(csv_adaptive):
        print(f"already computed -> {csv_adaptive}")
        adaptive_df = pd.read_csv(csv_adaptive)
    else:
        rows = []
        for sigma_base in tqdm(SIGMA_GRID[:6], desc="margin-adaptive sweep"):
            for mult in SIGMA_MULT_GRID:
                auc_clean, mean_recovery, _ = score(
                    lambda x, s=sigma_base, m=mult: sqs_reconstruct_adaptive(orig_tok, x, K_SWEEP, s, margin_low, margin_high, m))
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
        return score_config_logit_ensemble(predict_fn, Xc_sel, Yc_sel, args.adv_dir, "fsq", sweep_eps, sel_idx, device, bs)

    csv_logit = os.path.join(args.out_dir, "fsq_sqs_logit_ensemble_sweep.csv")
    rows = []
    for sigma in tqdm(LOGIT_ENSEMBLE_SIGMA_GRID, desc="logit-ensemble SQS sweep"):
        auc_clean, mean_recovery, per_attack = score_logit(
            lambda x, s=sigma: sqs_logit_ensemble_predict(orig_tok, densenet, x, K_SWEEP, s))
        rows.append(dict(sigma=sigma, auc_clean=round(auc_clean, 4), mean_auc_adv=round(mean_recovery, 4),
                          auc_adv_M1=round(per_attack[0], 4), auc_adv_M3_alpha03=round(per_attack[1], 4)))
    logit_df = pd.DataFrame(rows)
    logit_df.to_csv(csv_logit, index=False)
    print("wrote ->", csv_logit)
    print(logit_df.to_string(index=False))
    print("Compare auc_adv_M1/auc_adv_M3_alpha03 here against flat/spatial/adaptive's same columns "
          "-- if still ~0.0, the dense/redundant attack pattern survives per-sample logit averaging "
          "too, not just per-token voting.\n")

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
        print(f"*** NONE of the three variants clear the clean-AUC floor ({CLEAN_AUC_FLOOR}); "
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
    winner_path = os.path.join(args.out_dir, "fsq_sqs_winner.json")
    with open(winner_path, "w") as f:
        json.dump(winner_config, f, indent=2)
    print(f"wrote winner config -> {winner_path}")


if __name__ == "__main__":
    main()
