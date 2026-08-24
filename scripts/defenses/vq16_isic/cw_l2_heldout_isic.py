#!/usr/bin/env python3
"""Held-out generalization check: a Carlini & Wagner-style L2 margin/confidence
attack, never seen (cached or live) by any defense in this repo, against
advft (per-eps), decoder-realism-robust, and combined. See
src/defenses/common/cw_attack.py for why this is a genuine generalization test
rather than another remix of the same two loss primitives everything else here uses.

First calibrates the L2 budget against the UNDEFENDED tokenizer (same
"measure, don't assert" principle as every SQS hyperparameter) -- pick
--eps-l2 from that printout, or accept the default (the largest grid value)
if you haven't calibrated yet.

Usage:
    python scripts/defenses/vq16_isic/cw_l2_heldout_isic.py --data-root /path/to/isic2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import os
import sys

import pandas as pd
import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.common.cw_attack import cw_margin_attack
from src.defenses.vq16.bpda_attacks import plain_reconstruct_bpda
from src.defenses.vq16.margin import reconstruct
from src.eval.metrics import asr, auc
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_isic_classifier import EfficientNetB4Classifier, load_efficientnet_b4
from src.tokenizers.vq16 import load_vq16_net


def load_densenet(ckpt_path, device):
    """Alias, same trick the source notebook uses: wrap EfficientNet-B4 into
    the same load_ckpt(path, device) -> classifier object interface this
    script's downstream code (margin/decoder-realism/certify helpers, all
    written against DenseNet's raw-nn.Module calling convention: net(x) for
    the logit, net.features(x) for perceptual features) expects, so none of
    that shared code needs touching."""
    net = load_efficientnet_b4(ckpt_path, device)
    return EfficientNetB4Classifier(net, device)


def densenet_logit(densenet, x):
    return densenet(x)

CW_L2_EPS_GRID = [4.0, 8.0, 16.0]  # native pixel-space L2 radius, x in [-1,1], 256x256x3 image
CW_KAPPA = 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load -- must match whatever "
                         "merge_checkpoints_isic.py was run with (default: worst-case/largest in eps_grid)")
    p.add_argument("--eps-l2", type=float, default=None, help="default: calibrate and use the largest grid value")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]
    report_items = [attack_items[i] for i in report_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    # --- calibration: does eps_l2 actually break the undefended tokenizer? ---
    calib_items = sel_items[:16]
    calib_x, calib_y_np = load_xy(calib_items, cfg["img_size"], device)
    calib_y = torch.tensor(calib_y_np, dtype=torch.float32)

    print(f"--- CW-L2 calibration against UNDEFENDED tokenizer (n={len(calib_items)}) ---")
    for eps_l2 in CW_L2_EPS_GRID:
        x_cw = cw_margin_attack(orig_net, plain_reconstruct_bpda, lambda rec: densenet_logit(densenet, rec), calib_x, calib_y, eps_l2, CW_KAPPA, args.steps, args.lr, bs, device)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, calib_x).cpu().numpy()
            cw_lg = densenet_logit(densenet, reconstruct(orig_net, x_cw)).cpu().numpy()
        print(f"eps_l2={eps_l2:5.1f}: clean AUC={auc(calib_y_np, clean_lg):.4f}  "
              f"CW-attacked AUC={auc(calib_y_np, cw_lg):.4f}  ASR={asr(clean_lg, cw_lg, calib_y_np):.1f}%")
    eps_l2 = args.eps_l2 if args.eps_l2 is not None else CW_L2_EPS_GRID[-1]
    print(f"\nusing eps_l2={eps_l2} for the full sweep below "
          f"(pick the smallest value above that already drives the undefended tokenizer close to "
          f"complete failure, pass it via --eps-l2, if different from the default)\n")

    # --- full sweep against advft/decoder_realism/combined ---
    n = args.n if args.n is not None else len(report_items)
    items = report_items[:n]
    x, y_np = load_xy(items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)

    def run_cw_check(model, label):
        for p in model.parameters():
            p.requires_grad_(False)
        x_cw = cw_margin_attack(model, plain_reconstruct_bpda, lambda rec: densenet_logit(densenet, rec), x, y, eps_l2, CW_KAPPA, args.steps, args.lr, bs, device)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, x).cpu().numpy()
            static_lg = densenet_logit(densenet, reconstruct(model, x)).cpu().numpy()
            cw_lg = densenet_logit(densenet, reconstruct(model, x_cw)).cpu().numpy()
        row = dict(label=label, eps_l2=eps_l2, n=len(y_np),
                   clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg), cw_auc=auc(y_np, cw_lg),
                   cw_asr=asr(clean_lg, cw_lg, y_np))
        print(f"--- {label} -- HELD-OUT CW-L2 attack (n={len(y_np)}, eps_l2={eps_l2}) ---")
        print(f"clean AUC   : {row['clean_auc']:.4f}")
        print(f"CW-L2 AUC   : {row['cw_auc']:.4f}   (loss/norm this checkpoint was NEVER trained against)")
        print(f"CW-L2 ASR   : {row['cw_asr']:.1f}%\n")
        return row

    csv_path = os.path.join(args.out_dir, "vq16_cw_l2_heldout_full_grid.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning attacks -> {csv_path}")
        print(pd.read_csv(csv_path).to_string(index=False))
        return

    rows = []
    for eps255 in cfg["eps_grid"]:
        ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{eps255}_isic.pt")
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run run_advft_isic.py first"
        net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        net.eval()
        rows.append(run_cw_check(net, f"advft(eps{eps255}) ALONE, no SQS"))
        del net
        torch.cuda.empty_cache()

    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    for ckpt_name, label in [
        ("vq16_decoder_realism_robust_isic.pt", "decoder-realism-robust ALONE, no SQS"),
        (f"vq16_combined_eps{combined_eps}_isic.pt", "combined (advft+decoder-realism-robust) ALONE, no SQS"),
    ]:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run its training/merge script first"
        net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        net.eval()
        rows.append(run_cw_check(net, label))
        del net
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("wrote ->", csv_path)
    print("If AUC stays high / ASR stays low, that's real evidence these defenses generalize past the "
          "specific attacks they were built and evaluated against. If it collapses, that's the honest "
          "limitation to report: the defense works against the attack family it knows about, not "
          "attacks in general.")


if __name__ == "__main__":
    main()
