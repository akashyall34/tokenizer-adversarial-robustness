#!/usr/bin/env python3
"""Randomized-smoothing certified robustness bound for the winning SQS
variant, against the original tokenizer and against advft+best_sqs (the
headline condition). See src/defenses/vq16/certify.py for the mechanism and
the margin-adaptive radius caveat.

The single most expensive check in this pipeline (N_IMAGES x N_PASSES full
pipeline forward passes, x2 models) -- at n=128, N=100 that's ~25,600 passes
total. Budget real wall-clock time; use --n/--n-passes to shrink it while iterating.

Usage:
    python scripts/defenses/vq16_aptos/certify_sqs_aptos.py --data-root /path/to/aptos2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b5_aptos.pt
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

from src.datasets.aptos import build_splits
from src.datasets.kermany import KermanyDS, seed_everything, stratified_subset
from src.defenses.vq16.certify import certify_sqs
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet
from src.tokenizers.vq16 import load_vq16_net


def load_densenet(ckpt_path, device):
    """Alias, same trick the source notebook uses: wrap EfficientNet-B5 into
    the same load_ckpt(path, device) -> classifier interface this script's
    downstream code (certify_sqs, duck-typed against any callable) expects."""
    net = load_efficientnet(ckpt_path, device)
    return EfficientNetClassifier(net, device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--n-passes", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.001)
    return p.parse_args()


def run_certification(model, densenet, label, items, img_size, sqs_name, sqs_params, n_passes, alpha, device):
    rows = []
    for path, lbl in tqdm(items, desc=f"certifying: {label}"):
        x = KermanyDS([(path, lbl)], img_size)[0][0].unsqueeze(0).to(device)
        c_A, p_A_lower, radius = certify_sqs(model, densenet, x, sqs_name, sqs_params, N=n_passes, alpha=alpha)
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

    winner_path = os.path.join(args.out_dir, "vq16_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path} -- run sweep_sqs_aptos.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]
    if sqs_name == "margin_adaptive":
        print(f"BEST_SQS_NAME={sqs_name}: per-token sigma varies, so the certified RADIUS is not "
              f"well-defined -- only p_A_lower/certified will be reported, radius will be NaN.")

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    cert_items = report_items[:n]

    csv_path = os.path.join(args.out_dir, "vq16_sqs_certified_radii.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning certification -> {csv_path}")
        print(pd.read_csv(csv_path).to_string(index=False))
        return

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    df_orig = run_certification(orig_net, densenet, "original tokenizer + best_sqs", cert_items,
                                 cfg["img_size"], sqs_name, sqs_params, args.n_passes, args.alpha, device)

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{advft_eps}_aptos.pt")
    assert os.path.isfile(advft_ckpt_path), f"missing {advft_ckpt_path} -- run run_advft_aptos.py first"
    ft_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    ft_net.load_state_dict(torch.load(advft_ckpt_path, map_location=device, weights_only=False))
    ft_net.eval()
    for p in ft_net.parameters():
        p.requires_grad_(False)
    df_advft = run_certification(ft_net, densenet, f"advft(eps{advft_eps})+best_sqs -- the headline condition",
                                  cert_items, cfg["img_size"], sqs_name, sqs_params, args.n_passes, args.alpha, device)

    df = pd.concat([df_orig, df_advft], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print("wrote ->", csv_path)


if __name__ == "__main__":
    main()
