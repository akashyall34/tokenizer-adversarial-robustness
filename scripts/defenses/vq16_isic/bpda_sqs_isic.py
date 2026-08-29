#!/usr/bin/env python3
"""BPDA adaptive-attack check against the standalone SQS winner AND the
headline advft+best_sqs condition -- an attacker who knows SQS is there and
optimizes against it directly, as opposed to every recovery number in
sweep_sqs_isic.py (which measures recovery against attacks generated on
the UNDEFENDED tokenizer). Checking only the weaker standalone config would
leave the condition actually claimed as "best" without adaptive verification.

Run on REPORT_ITEMS (held out from hyperparameter selection) at the
worst-case eps. This is a confirmatory check, more expensive per image than
a static attack (K forward passes x many PGD steps), so it's worth running
at a smaller n first via --n if you're iterating.

Usage:
    python scripts/defenses/vq16_isic/bpda_sqs_isic.py --data-root /path/to/isic2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import json
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
from src.defenses.vq16.bpda_attacks import bpda_sqs_attack
from src.defenses.vq16.sqs import dispatch_reconstruct
from src.eval.metrics import asr, auc
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
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def run_bpda_check(model, label, densenet, x, y, y_np, sqs_name, sqs_params, eps255, steps, lr, bs, device):
    x_bpda = bpda_sqs_attack(model, densenet, x, y, eps255 / 255.0, sqs_name, sqs_params, steps, lr, bs, device)
    with torch.no_grad():
        clean_lg = densenet_logit(densenet, x).cpu().numpy()
        static_lg = densenet_logit(densenet, dispatch_reconstruct(sqs_name, model, x, sqs_params)).cpu().numpy()
        bpda_lg = densenet_logit(densenet, dispatch_reconstruct(sqs_name, model, x_bpda, sqs_params)).cpu().numpy()
    print(f"--- {label} (n={len(y_np)}, eps255={eps255}, variant={sqs_name}) ---")
    print(f"clean AUC         : {auc(y_np, clean_lg):.4f}")
    print(f"static-attack AUC : {auc(y_np, static_lg):.4f}   (attack generated on UNDEFENDED tokenizer, then defended)")
    print(f"BPDA-adaptive AUC : {auc(y_np, bpda_lg):.4f}   (attack generated WITH knowledge of the defense)")
    print(f"static-attack ASR : {asr(clean_lg, static_lg, y_np):.1f}%")
    print(f"BPDA-adaptive ASR : {asr(clean_lg, bpda_lg, y_np):.1f}%\n")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]
    sweep_eps = cfg["eps_grid"][-1]

    winner_path = os.path.join(args.out_dir, "vq16_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path} -- run sweep_sqs_isic.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    bpda_items = report_items[:n]

    x, y_np = load_xy(bpda_items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    run_bpda_check(orig_net, "original tokenizer + best_sqs", densenet, x, y, y_np, sqs_name, sqs_params,
                    sweep_eps, args.steps, args.lr, bs, device)

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{advft_eps}_isic.pt")
    assert os.path.isfile(advft_ckpt_path), f"missing {advft_ckpt_path} -- run run_advft_isic.py first"
    ft_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    ft_net.load_state_dict(torch.load(advft_ckpt_path, map_location=device, weights_only=False))
    ft_net.eval()
    run_bpda_check(ft_net, f"advft(eps{advft_eps})+best_sqs -- the actual headline condition",
                    densenet, x, y, y_np, sqs_name, sqs_params, sweep_eps, args.steps, args.lr, bs, device)

    print("If BPDA-adaptive AUC collapses back toward the undefended/static-attack AUC for EITHER "
          "model, the static recovery number was NOT real robustness -- report that honestly, "
          "especially if it happens for the advft+best_sqs headline condition specifically.")


if __name__ == "__main__":
    main()
