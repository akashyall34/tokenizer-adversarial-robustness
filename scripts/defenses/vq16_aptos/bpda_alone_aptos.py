#!/usr/bin/env python3
"""BPDA adaptive attack (BCE-through-plain-quantize, no SQS at all) against
every VQ-16 defense checkpoint on its own: advft (per-eps checkpoints,
each attacked at its own matching eps255), decoder-realism-robust, and the
combined checkpoint (each a single fixed checkpoint, attacked across the
full eps grid as the attack budget rather than which checkpoint is loaded).

Isolates whether each defense survives adaptive attack without SQS's help.
Every BPDA result in bpda_sqs_aptos.py included SQS in the pipeline, and
flat SQS is the mechanism that collapsed under adaptive attack there, so
this answers a real open question rather than restating that result.

One CSV, all three checkpoint families; compare them directly: whichever
shows the smallest BPDA-adaptive AUC drop from its own clean AUC is the one
that actually survives an adaptive attacker best.

Usage:
    python scripts/defenses/vq16_aptos/bpda_alone_aptos.py --data-root /path/to/aptos2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b5_aptos.pt
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
from src.defenses.vq16.bpda_attacks import bpda_plain_attack
from src.defenses.vq16.margin import reconstruct
from src.eval.metrics import asr, auc
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet
from src.tokenizers.vq16 import load_vq16_net


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load; must match whatever "
                         "merge_checkpoints_aptos.py was run with (default: worst-case/largest in eps_grid)")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def run_bpda_alone_check(model, label, densenet, x, y, y_np, eps255, steps, lr, bs, device):
    for p in model.parameters():
        p.requires_grad_(False)
    x_bpda = bpda_plain_attack(model, densenet, x, y, eps255 / 255.0, steps, lr, bs, device)

    with torch.no_grad():
        clean_lg = densenet_logit(densenet, x).cpu().numpy()
        static_lg = densenet_logit(densenet, reconstruct(model, x)).cpu().numpy()
        bpda_lg = densenet_logit(densenet, reconstruct(model, x_bpda)).cpu().numpy()

    row = dict(label=label, eps255=eps255, n=len(y_np),
               clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg), bpda_auc=auc(y_np, bpda_lg),
               static_asr=asr(clean_lg, static_lg, y_np), bpda_asr=asr(clean_lg, bpda_lg, y_np))
    print(f"--- {label} (n={row['n']}, eval eps255={eps255}) ---")
    print(f"clean AUC         : {row['clean_auc']:.4f}")
    print(f"static-attack AUC : {row['static_auc']:.4f}")
    print(f"BPDA-adaptive AUC : {row['bpda_auc']:.4f}")
    print(f"static-attack ASR : {row['static_asr']:.1f}%")
    print(f"BPDA-adaptive ASR : {row['bpda_asr']:.1f}%\n")
    return row


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    bpda_items = report_items[:n]

    x, y_np = load_xy(bpda_items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)
    densenet = load_densenet(args.effnet_ckpt, device)

    csv_path = os.path.join(args.out_dir, "vq16_bpda_alone_full_grid.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning attacks -> {csv_path}")
        print(pd.read_csv(csv_path).to_string(index=False))
        return

    rows = []
    for eps255 in cfg["eps_grid"]:
        ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{eps255}_aptos.pt")
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path}; run run_advft_aptos.py first"
        net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        net.eval()
        rows.append(run_bpda_alone_check(net, f"advft(eps{eps255}) ALONE, no SQS", densenet, x, y, y_np,
                                          eps255, args.steps, args.lr, bs, device))
        del net
        torch.cuda.empty_cache()

    decoder_ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust_aptos.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_combined_eps{combined_eps}_aptos.pt")
    for ckpt_path, label in [
        (decoder_ckpt_path, "decoder-realism-robust ALONE, no SQS"),
        (combined_ckpt_path, "combined (advft+decoder-realism-robust) ALONE, no SQS"),
    ]:
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path}; run its training/merge script first"
        for eps255 in cfg["eps_grid"]:
            net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
            net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            net.eval()
            rows.append(run_bpda_alone_check(net, label, densenet, x, y, y_np, eps255, args.steps, args.lr, bs, device))
            del net
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("wrote ->", csv_path)
    print("\nCompare all three families: whichever shows the smallest BPDA-adaptive AUC drop from its "
          "own clean AUC, at the eps that matters, is the one that actually survives an adaptive "
          "attacker best; that comparison, not any single number in isolation, is what should decide "
          "which checkpoint becomes the headline defense.")


if __name__ == "__main__":
    main()
