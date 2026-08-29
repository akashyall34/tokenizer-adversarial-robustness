#!/usr/bin/env python3
"""Paired bootstrap significance test on clean-recon AUC across the four VQ-16
ISIC conditions (original, advft, decoder-realism, combined). A point
estimate alone ("0.9570 -> 0.9638") doesn't say whether a gain is real at
this sample size -- see src/eval/bootstrap.py.

Requires run_advft_isic.py, run_decoder_realism_isic.py, and
merge_checkpoints_isic.py to have already produced their checkpoints.

Usage:
    python scripts/defenses/vq16_isic/significance_isic.py --data-root /path/to/isic2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.vq16.margin import reconstruct
from src.eval.bootstrap import paired_auc_bootstrap_ci
from src.eval.metrics import auc as auc_metric
from src.eval.sel_report_split import stratified_half_split
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None,
                    help="which advft/combined checkpoint pair to test (default: worst-case/largest in eps_grid)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{advft_eps}_isic.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust_isic.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_combined_eps{advft_eps}_isic.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path, combined_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p} -- run its training/merge script first"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    x, y = load_xy(report_items, cfg["img_size"], device)

    def load_net(ckpt_path):
        net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        net.eval()
        return net

    with torch.no_grad():
        lg_orig = densenet_logit(densenet, reconstruct(orig_net, x)).cpu().numpy()
        lg_advft = densenet_logit(densenet, reconstruct(load_net(advft_ckpt_path), x)).cpu().numpy()
        lg_decoder = densenet_logit(densenet, reconstruct(load_net(decoder_ckpt_path), x)).cpu().numpy()
        lg_combined = densenet_logit(densenet, reconstruct(load_net(combined_ckpt_path), x)).cpu().numpy()

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
