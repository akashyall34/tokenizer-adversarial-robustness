#!/usr/bin/env python3
"""Held-out generalization check: a Carlini & Wagner-style L2 margin/confidence
attack, never seen (cached or live) by any defense in this repo, against
MediTok advft (per-eps), decoder-realism-robust, and combined. Same
role as scripts/defenses/vq16_aptos/cw_l2_heldout_aptos.py.

First calibrates the L2 budget against each UNDEFENDED tokenizer -- pick
--eps-l2 from that printout, or accept the default (the largest grid value)
if you haven't calibrated yet.

Usage:
    python scripts/defenses/meditok_aptos/cw_l2_heldout_aptos.py \
        --data-root /path/to/aptos2019 --meditok-repo third_party/meditok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt path/to/effnet_ckpt
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

from src.datasets.aptos import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.common.cw_attack import cw_margin_attack
from src.defenses.multicodebook.bpda_attacks import plain_reconstruct_bpda
from src.eval.metrics import asr, auc
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet


def load_densenet(ckpt_path, device):
    """Alias: wrap EfficientNet-B5 into
    the same load_ckpt(path, device) -> classifier object interface this
    script's downstream code (margin/decoder-realism/certify helpers, all
    written against DenseNet's raw-nn.Module calling convention: net(x) for
    the logit, net.features(x) for perceptual features) expects, so none of
    that shared code needs touching."""
    net = load_efficientnet(ckpt_path, device)
    return EfficientNetClassifier(net, device)


def densenet_logit(densenet, x):
    return densenet(x)


from src.tokenizers.meditok_unitok import load_meditok

CW_L2_EPS_GRID = [4.0, 8.0, 16.0]  # native pixel-space L2 radius, x in [-1,1], 256x256x3 image
CW_KAPPA = 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--meditok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load -- must match whatever "
                         "merge_checkpoints_aptos.py was run with (default: worst-case/largest in eps_grid)")
    p.add_argument("--eps-l2", type=float, default=None, help="default: calibrate and use the largest grid value")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def run_one(tok_name, load_fresh, args, cfg, densenet, sel_items, report_items, bs, device):
    csv_path = os.path.join(args.out_dir, f"{tok_name}_cw_l2_heldout_full_grid.csv")
    if os.path.isfile(csv_path):
        print(f"{tok_name}: already computed, loading instead of rerunning attacks -> {csv_path}")
        print(pd.read_csv(csv_path).to_string(index=False))
        return

    orig_tok = load_fresh()

    calib_items = sel_items[:16]
    calib_x, calib_y_np = load_xy(calib_items, cfg["img_size"], device)
    calib_y = torch.tensor(calib_y_np, dtype=torch.float32)

    print(f"--- {tok_name} CW-L2 calibration against UNDEFENDED tokenizer (n={len(calib_items)}) ---")
    for eps_l2 in CW_L2_EPS_GRID:
        x_cw = cw_margin_attack(orig_tok, plain_reconstruct_bpda, lambda rec: densenet_logit(densenet, rec),
                                 calib_x, calib_y, eps_l2, CW_KAPPA, args.steps, args.lr, bs, device)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, calib_x).cpu().numpy()
            cw_lg = densenet_logit(densenet, orig_tok.reconstruct(x_cw)).cpu().numpy()
        print(f"eps_l2={eps_l2:5.1f}: clean AUC={auc(calib_y_np, clean_lg):.4f}  "
              f"CW-attacked AUC={auc(calib_y_np, cw_lg):.4f}  ASR={asr(clean_lg, cw_lg, calib_y_np):.1f}%")
    eps_l2 = args.eps_l2 if args.eps_l2 is not None else CW_L2_EPS_GRID[-1]
    print(f"\nusing eps_l2={eps_l2} for the full sweep below "
          f"(pick the smallest value above that already drives the undefended tokenizer close to "
          f"complete failure, pass it via --eps-l2, if different from the default)\n")
    del orig_tok
    torch.cuda.empty_cache()

    n = args.n if args.n is not None else len(report_items)
    items = report_items[:n]
    x, y_np = load_xy(items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)

    def run_cw_check(tok, label):
        for p in tok.net.parameters():
            p.requires_grad_(False)
        x_cw = cw_margin_attack(tok, plain_reconstruct_bpda, lambda rec: densenet_logit(densenet, rec),
                                 x, y, eps_l2, CW_KAPPA, args.steps, args.lr, bs, device)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, x).cpu().numpy()
            static_lg = densenet_logit(densenet, tok.reconstruct(x)).cpu().numpy()
            cw_lg = densenet_logit(densenet, tok.reconstruct(x_cw)).cpu().numpy()
        row = dict(label=label, eps_l2=eps_l2, n=len(y_np),
                   clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg), cw_auc=auc(y_np, cw_lg),
                   cw_asr=asr(clean_lg, cw_lg, y_np))
        print(f"--- {label} -- HELD-OUT CW-L2 attack (n={len(y_np)}, eps_l2={eps_l2}) ---")
        print(f"clean AUC   : {row['clean_auc']:.4f}")
        print(f"CW-L2 AUC   : {row['cw_auc']:.4f}   (loss/norm this checkpoint was NEVER trained against)")
        print(f"CW-L2 ASR   : {row['cw_asr']:.1f}%\n")
        return row

    rows = []
    for eps255 in cfg["eps_grid"]:
        ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_advft_eps{eps255}_aptos.pt")
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run run_advft_aptos.py first"
        tok = load_fresh()
        tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        tok.net.eval()
        rows.append(run_cw_check(tok, f"{tok_name} advft(eps{eps255}) ALONE, no SQS"))
        del tok
        torch.cuda.empty_cache()

    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    for ckpt_name, label in [
        (f"{tok_name}_decoder_realism_robust_aptos.pt", f"{tok_name} decoder-realism-robust ALONE, no SQS"),
        (f"{tok_name}_combined_eps{combined_eps}_aptos.pt", f"{tok_name} combined (advft+decoder-realism-robust) ALONE, no SQS"),
    ]:
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run its training/merge script first"
        tok = load_fresh()
        tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        tok.net.eval()
        rows.append(run_cw_check(tok, label))
        del tok
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"{tok_name}: wrote -> {csv_path}")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]
    report_items = [attack_items[i] for i in report_idx]

    densenet = load_densenet(args.effnet_ckpt, device)

    tok_name = "meditok"
    load_fresh = lambda: load_meditok(args.meditok_repo, args.tokenizer_ckpt_dir, cfg["img_size"], device)
    print(f"=== {tok_name} ===")
    run_one(tok_name, load_fresh, args, cfg, densenet, sel_items, report_items, bs, device)

print("If AUC stays high / ASR stays low, that's real evidence these defenses generalize past the "
      "specific attacks they were built and evaluated against. If it collapses, that's the honest "
      "limitation to report: the defense works against the attack family it knows about, not "
      "attacks in general.")


if __name__ == "__main__":
    main()
