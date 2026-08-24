#!/usr/bin/env python3
"""Randomized-smoothing certified robustness bound for the winning FSQ SQS
variant, against the original tokenizer and against advft+best_sqs (the
headline condition). See src/defenses/fsq/certify.py for the mechanism and
the margin-adaptive radius caveat.

The single most expensive check in this pipeline (N_IMAGES x N_PASSES full
pipeline forward passes, x2 models). Budget real wall-clock time; use
--n/--n-passes to shrink it while iterating.

Usage:
    python scripts/defenses/fsq_isic/certify_sqs_isic.py \
        --data-root /path/to/isic2019 --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import KermanyDS, seed_everything, stratified_subset
from src.defenses.fsq.certify import certify_sqs
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_isic_classifier import EfficientNetB4Classifier, load_efficientnet_b4
from src.tokenizers.fsq import FSQTokenizer


def load_densenet(ckpt_path, device):
    net = load_efficientnet(ckpt_path, device)
    return EfficientNetClassifier(net, device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--fsq-repo", required=True)
    p.add_argument("--fsq-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--n-passes", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.001)
    return p.parse_args()


def run_certification(tok, densenet, label, items, img_size, sqs_name, sqs_params, n_passes, alpha, device):
    rows = []
    for path, lbl in tqdm(items, desc=f"certifying: {label}"):
        x = KermanyDS([(path, lbl)], img_size)[0][0].unsqueeze(0).to(device)
        c_A, p_A_lower, radius = certify_sqs(tok, densenet, x, sqs_name, sqs_params, N=n_passes, alpha=alpha)
        rows.append(dict(model=label, label=lbl, pred=c_A, correct=int(c_A == lbl), p_A_lower=round(p_A_lower, 4),
                          certified=int(p_A_lower > 0.5), radius=(radius if radius == radius else round(radius, 5))))
    df = pd.DataFrame(rows)
    print(f"--- {label} ---")
    print(f"certified (p_A_lower>0.5): {df.certified.mean() * 100:.1f}% of images")
    cert_radii = df[df.certified == 1].radius
    print(f"median certified radius (certified images only): {cert_radii.median() if len(cert_radii) else float('nan'):.5f}")
    print(f"clean accuracy on this subset: {df.correct.mean() * 100:.1f}%\n")
    return df


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]

    winner_path = os.path.join(args.out_dir, "fsq_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path} -- run sweep_sqs_isic.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]
    if sqs_name == "margin_adaptive":
        print(f"BEST_SQS_NAME={sqs_name}: per-token sigma varies, so the certified RADIUS is not "
              f"well-defined -- only p_A_lower/certified will be reported, radius will be NaN.")

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    cert_items = report_items[:n]

    csv_path = os.path.join(args.out_dir, "fsq_sqs_certified_radii.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning certification -> {csv_path}")
        print(pd.read_csv(csv_path).to_string(index=False))
        return

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    df_orig = run_certification(orig_tok, densenet, "original tokenizer + best_sqs", cert_items,
                                 cfg["img_size"], sqs_name, sqs_params, args.n_passes, args.alpha, device)

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_advft_eps{advft_eps}_isic.pt")
    assert os.path.isfile(advft_ckpt_path), f"missing {advft_ckpt_path} -- run run_advft_isic.py first"
    ft_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    ft_tok.net.load_state_dict(torch.load(advft_ckpt_path, map_location=device, weights_only=False))
    ft_tok.net.eval()
    for p in ft_tok.net.parameters():
        p.requires_grad_(False)
    df_advft = run_certification(ft_tok, densenet, f"advft(eps{advft_eps})+best_sqs -- the headline condition",
                                  cert_items, cfg["img_size"], sqs_name, sqs_params, args.n_passes, args.alpha, device)

    df = pd.concat([df_orig, df_advft], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print("wrote ->", csv_path)


if __name__ == "__main__":
    main()
