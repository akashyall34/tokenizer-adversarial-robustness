#!/usr/bin/env python3
"""BPDA adaptive-attack check against the standalone FSQ SQS winner and the
headline advft+best_sqs condition. Same role as
scripts/defenses/fsq_aptos/bpda_sqs_aptos.py.

Usage:
    python scripts/defenses/fsq_kermany/bpda_sqs_kermany.py \
        --data-root /path/to/chest_xray --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
"""
import argparse
import json
import os
import sys

import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.defenses.fsq.bpda_attacks import bpda_sqs_attack
from src.defenses.fsq.sqs import dispatch_reconstruct
from src.eval.metrics import asr, auc
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
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def run_bpda_check(tok, label, densenet, x, y, y_np, sqs_name, sqs_params, eps255, steps, lr, bs, device):
    x_bpda = bpda_sqs_attack(tok, densenet, x, y, eps255 / 255.0, sqs_name, sqs_params, steps, lr, bs, device)
    with torch.no_grad():
        clean_lg = densenet_logit(densenet, x).cpu().numpy()
        static_lg = densenet_logit(densenet, dispatch_reconstruct(sqs_name, tok, x, sqs_params)).cpu().numpy()
        bpda_lg = densenet_logit(densenet, dispatch_reconstruct(sqs_name, tok, x_bpda, sqs_params)).cpu().numpy()
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

    winner_path = os.path.join(args.out_dir, "fsq_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path}; run sweep_sqs_kermany.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    bpda_items = report_items[:n]

    x, y_np = load_xy(bpda_items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.densenet_ckpt, device)

    run_bpda_check(orig_tok, "original tokenizer + best_sqs", densenet, x, y, y_np, sqs_name, sqs_params,
                    sweep_eps, args.steps, args.lr, bs, device)

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_advft_eps{advft_eps}.pt")
    assert os.path.isfile(advft_ckpt_path), f"missing {advft_ckpt_path}; run run_advft_kermany.py first"
    ft_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    ft_tok.net.load_state_dict(torch.load(advft_ckpt_path, map_location=device, weights_only=False))
    ft_tok.net.eval()
    run_bpda_check(ft_tok, f"advft(eps{advft_eps})+best_sqs; the actual headline condition",
                    densenet, x, y, y_np, sqs_name, sqs_params, sweep_eps, args.steps, args.lr, bs, device)

    print("If BPDA-adaptive AUC collapses back toward the undefended/static-attack AUC for EITHER "
          "model, the static recovery number was NOT real robustness; report that honestly, "
          "especially if it happens for the advft+best_sqs headline condition specifically.")


if __name__ == "__main__":
    main()
