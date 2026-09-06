#!/usr/bin/env python3
"""Quantifies the footprint of the unnormalized VQ-16 attack margin (see
docs/known_limitations.md) on the cached M2 / M3_alpha03 attack images.

The perturbations are not regenerated; the mistargeting is measured directly:
of the tokens that actually flipped (judged by the real quantizer, not the
unnormalized proxy), what fraction landed on the code that was truly the
nearest rival at the clean point, versus some other, non-nearest code?

Usage:
    python scripts/diagnostics/diagnose_vq16_margin_bug.py --data-root /path/to/chest_xray \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt
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

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.defenses.vq16.margin import l2_dist_matrix, prequant, clean_top1
from src.eval.attack_io import load_folder
from src.eval.sel_report_split import stratified_half_split
from src.tokenizers.vq16 import load_vq16_net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--eps255", type=int, default=8, help="worst-case eps to diagnose")
    return p.parse_args()


@torch.no_grad()
def diagnose_geo_targeting_quality(net, items_sub, idx_subset, adv_dir, attack, eps255, img_size, device):
    """Of the tokens that actually flipped (real quantizer), what fraction
    flipped to the TRUE nearest-rival code (also computed via the correct,
    normalized math) vs. some other, non-nearest code?"""
    Xc_sub, _ = load_xy(items_sub, img_size, device)
    Xa_sub, _ = load_folder(adv_dir, "vq16", attack, eps255, idx_subset=idx_subset)
    Xa_sub = Xa_sub.to(device)

    z_clean = prequant(net, Xc_sub)
    E = net.quantize.embedding.weight
    d_clean = l2_dist_matrix(z_clean, E)
    k1 = d_clean.argmin(-1)
    vals, idxs = d_clean.topk(2, dim=-1, largest=False)
    true_rival = torch.where(idxs[..., 0] == k1, idxs[..., 1], idxs[..., 0])

    adv_idx = clean_top1(net, Xa_sub)

    flipped = adv_idx != k1
    n_flipped = flipped.sum().item()
    flipped_to_true_rival = flipped & (adv_idx == true_rival)
    flip_rate = flipped.float().mean().item()
    hit_rate = (flipped_to_true_rival.sum().float() / max(n_flipped, 1)).item() if n_flipped else float("nan")
    print(f"  {attack} eps{eps255}: real flip rate={flip_rate * 100:5.1f}%  |  of flipped tokens, "
          f"{hit_rate * 100:5.1f}% landed on the TRUE nearest rival "
          f"({100 - hit_rate * 100:.1f}% landed on some other, non-nearest code)")
    return flip_rate, hit_rate


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)

    csv_path = os.path.join(args.out_dir, "vq16_geo_targeting_diagnostic.csv")
    rows = []
    print(f"Real impact of the unnormalized-margin bug, measured on REPORT_ITEMS (held-out, eps={args.eps255}):")
    for attack in ["M3_alpha03", "M2-sat", "M2-unb"]:
        fr, hr = diagnose_geo_targeting_quality(
            net, report_items, report_idx, args.adv_dir, attack, args.eps255, cfg["img_size"], device,
        )
        rows.append(dict(attack=attack, eps255=args.eps255, flip_rate=round(fr, 4), true_rival_hit_rate=round(hr, 4)))

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote -> {csv_path}; cite this table directly in the paper's limitations section.")


if __name__ == "__main__":
    main()
