#!/usr/bin/env python3
"""Sanity check for src/defenses/vq16/margin.py: does the L2-normalized proxy's
predicted nearest code actually match what the real VQ-16 quantizer assigns?
Run this before trusting any fine-tuning or margin-adaptive-noise result that
depends on src/defenses/vq16/margin.py -- if this doesn't read ~100%, something about the
real quantizer's internals has changed and nothing downstream should be trusted.

Usage:
    python scripts/diagnostics/verify_vq16_margin_fix.py --data-root /path/to/chest_xray \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt
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
from src.defenses.vq16.margin import clean_top1
from src.tokenizers.vq16 import load_vq16_net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--n", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    items = stratified_subset(test_items, args.n, cfg["seed"])
    x, _ = load_xy(items, cfg["img_size"], device)

    net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    with torch.no_grad():
        _, _, info = net.encode(x)
        real_idx = info[2].reshape(x.shape[0], -1)
        proxy_idx = clean_top1(net, x)

    agreement = (real_idx == proxy_idx).float().mean().item()
    print(f"normalized proxy vs. real quantizer token agreement (n={len(items)}): {agreement * 100:.2f}%")
    if agreement < 0.999:
        print(f"*** STILL MISMATCHED -- {(1 - agreement) * 100:.2f}% disagreement remains. "
              f"Do not trust the fine-tuning loop or margin-adaptive noise until this is resolved. ***")
        sys.exit(1)
    print("agreement is effectively 1.0 -- src/defenses/vq16/margin.py measures the real boundary. Safe to proceed.")


if __name__ == "__main__":
    main()
