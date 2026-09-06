#!/usr/bin/env python3
"""Paired bootstrap significance test on clean-recon AUC across the four
conditions (original, advft, decoder-realism, combined), for both MediTok
and UniTok. Same role as
scripts/defenses/vq16_aptos/significance_aptos.py.

Usage:
    python scripts/defenses/unitok_aptos/significance_aptos.py \
        --data-root /path/to/aptos2019 --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt path/to/effnet_ckpt
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

from src.datasets.aptos import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.eval.bootstrap import paired_auc_bootstrap_ci
from src.eval.metrics import auc as auc_metric
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


from src.tokenizers.meditok_unitok import load_unitok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def run_one(tok_name, load_fresh, args, cfg, densenet, x, y, device):
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]
    advft_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_advft_eps{advft_eps}_aptos.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_decoder_realism_robust_aptos.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_combined_eps{advft_eps}_aptos.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path, combined_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p}; run its training/merge script first"

    def load_ckpt(ckpt_path):
        tok = load_fresh()
        tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        return tok

    orig_tok = load_fresh()
    with torch.no_grad():
        lg_orig = densenet_logit(densenet, orig_tok.reconstruct(x)).cpu().numpy()
        lg_advft = densenet_logit(densenet, load_ckpt(advft_ckpt_path).reconstruct(x)).cpu().numpy()
        lg_decoder = densenet_logit(densenet, load_ckpt(decoder_ckpt_path).reconstruct(x)).cpu().numpy()
        lg_combined = densenet_logit(densenet, load_ckpt(combined_ckpt_path).reconstruct(x)).cpu().numpy()

    print(f"\n=== {tok_name} ===")
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


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    densenet = load_densenet(args.effnet_ckpt, device)
    x, y = load_xy(report_items, cfg["img_size"], device)

    tok_name = "unitok"
    load_fresh = lambda: load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)
    run_one(tok_name, load_fresh, args, cfg, densenet, x, y, device)


if __name__ == "__main__":
    main()
