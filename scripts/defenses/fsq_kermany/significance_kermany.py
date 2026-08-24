#!/usr/bin/env python3
"""Paired bootstrap significance test on clean-recon AUC across the four FSQ
Kermany conditions (original, advft, decoder-realism, combined). Same role as
scripts/defenses/fsq_aptos/significance_aptos.py.

Usage:
    python scripts/defenses/fsq_kermany/significance_kermany.py \
        --data-root /path/to/chest_xray --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
"""
import argparse
import os
import sys

import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.eval.bootstrap import paired_auc_bootstrap_ci
from src.eval.metrics import auc as auc_metric
from src.eval.sel_report_split import stratified_half_split
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.fsq import FSQTokenizer


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
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_advft_eps{advft_eps}.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, "fsq_decoder_realism_robust.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_combined_eps{advft_eps}.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path, combined_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p} -- run its training/merge script first"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.densenet_ckpt, device)
    x, y = load_xy(report_items, cfg["img_size"], device)

    def load_tok(ckpt_path):
        tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
        tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        tok.net.eval()
        return tok

    with torch.no_grad():
        lg_orig = densenet_logit(densenet, orig_tok.reconstruct(x)).cpu().numpy()
        lg_advft = densenet_logit(densenet, load_tok(advft_ckpt_path).reconstruct(x)).cpu().numpy()
        lg_decoder = densenet_logit(densenet, load_tok(decoder_ckpt_path).reconstruct(x)).cpu().numpy()
        lg_combined = densenet_logit(densenet, load_tok(combined_ckpt_path).reconstruct(x)).cpu().numpy()

    print(f"{'comparison':40s} {'AUC(a)':>8s} {'AUC(b)':>8s} {'delta':>8s} {'95% CI':>18s}  significant?")
    for label, lg_a, lg_b in [
        (f"advft(eps{advft_eps}) vs original", lg_orig, lg_advft),
        ("decoder-realism vs original", lg_orig, lg_decoder),
        ("combined vs original", lg_orig, lg_combined),
        ("combined vs decoder-realism", lg_decoder, lg_combined),
        ("combined vs advft alone", lg_advft, lg_combined),
    ]:
        d, lo, hi, sig = paired_auc_bootstrap_ci(y, lg_a, lg_b, seed=cfg["seed"])
        auc_a, auc_b = auc_metric(y, lg_a), auc_metric(y, lg_b)
        print(f"{label:40s} {auc_a:8.4f} {auc_b:8.4f} {d:+8.4f}   [{lo:+.4f}, {hi:+.4f}]  {'YES' if sig else 'no (within noise)'}")


if __name__ == "__main__":
    main()
