#!/usr/bin/env python3
"""Reconstruction / generative fidelity: decode the cached adversarial images
through each tokenizer (undefended, or with --defended each fine-tune defense
checkpoint) and score LPIPS/PSNR/SSIM plus pooled rFID against the clean
reconstruction and the original image.

Reads results/adv_images/<tok>/<attack>/eps<n>/ (produced by attacks-<dataset>)
and, for --defended, ckpts/defenses/ (produced by <tok>-defenses-<dataset>).
Re-runs no attack.

Usage:
    python scripts/generative/run_fidelity.py --dataset kermany \
        --data-root /path/to/chest_xray \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --fsq-repo third_party/FSQ-pytorch --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --meditok-repo third_party/meditok --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt
"""
import argparse
import csv
import os
import sys

import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.eval.generative_fidelity import (
    CLEAN_FIELDS, GRID_FIELDS, GRID_FIELDS_DEFENSE, RFID_FIELDS, RFID_FIELDS_DEFENSE,
    batched_reconstruct, clean_baseline_row, defended_checkpoint, fidelity_grid,
)

TOKENIZERS = ("vq16", "fsq", "meditok", "unitok")
DEFENSES = ["encoder_ft_eps0.5", "encoder_ft_eps1", "encoder_ft_eps2",
            "encoder_ft_eps4", "encoder_ft_eps8", "decoder_ft", "combined"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["kermany", "aptos", "isic"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--tokenizers", default=",".join(TOKENIZERS),
                   help="comma-separated subset to score (default: all four)")
    p.add_argument("--defended", action="store_true", help="score the fine-tune defense checkpoints")
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--fsq-repo", required=True)
    p.add_argument("--fsq-ckpt", required=True)
    p.add_argument("--meditok-repo", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--out-dir", default="results/generative_fidelity")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--config", default=None, help="default: configs/<dataset>_attack.yaml")
    return p.parse_args()


def test_split(dataset, data_root, out_dir, cfg):
    """The three dataset modules do not share a build_splits signature."""
    if dataset == "kermany":
        from src.datasets.kermany import build_splits

        return build_splits(data_root, out_dir, cfg["seed"], cfg["val_frac"])[2]
    if dataset == "aptos":
        from src.datasets.aptos import build_splits

        return build_splits(data_root, out_dir)[2]
    from src.datasets.isic import build_splits

    return build_splits(data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])[2]


def make_tokenizer(name, args, device, img_size):
    if name == "vq16":
        from src.tokenizers.vq16 import VQ16Tokenizer

        return VQ16Tokenizer(args.llamagen_repo, args.vq16_ckpt, device)
    if name == "fsq":
        from src.tokenizers.fsq import FSQTokenizer

        return FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    if name == "meditok":
        from src.tokenizers.meditok_unitok import load_meditok

        return load_meditok(args.meditok_repo, args.tokenizer_ckpt_dir, img_size, device)
    if name == "unitok":
        from src.tokenizers.meditok_unitok import load_unitok

        return load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)
    raise ValueError(f"unknown tokenizer {name!r} (expected one of {TOKENIZERS})")


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", path)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config or f"configs/{args.dataset}_attack.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(cfg["seed"])
    test_items = test_split(args.dataset, args.data_root, args.out_dir, cfg)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    Xc, Yc = load_xy(attack_items, cfg["img_size"], "cpu")

    for name in [t.strip() for t in args.tokenizers.split(",") if t.strip()]:
        suffix = "_defense" if args.defended else ""
        grid_csv = os.path.join(args.out_dir, f"{name}_{args.dataset}{suffix}.csv")
        rfid_csv = f"{grid_csv[:-4]}_rfid_pooled.csv"
        clean_csv = os.path.join(args.out_dir, f"{name}_{args.dataset}_clean_baseline.csv")
        # the last file each branch writes; skip only when it is already there
        last = grid_csv if args.defended else clean_csv
        if os.path.isfile(last):
            print(f"{name}: already computed -> {last}")
            continue

        if args.defended:
            grid, rfid = [], []
            for defense in DEFENSES:
                ckpt = defended_checkpoint(args.ckpt_dir, name, defense, args.dataset)
                assert os.path.isfile(ckpt), f"missing {ckpt}; run {name}-defenses-{args.dataset} first"
                tok = make_tokenizer(name, args, device, cfg["img_size"])
                tok.net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
                tok.net.eval()
                g, r = fidelity_grid(name, tok.reconstruct, args.adv_dir, name, Xc, Yc, cfg["eps_grid"], device)
                for row in g + r:
                    row["defense"] = defense
                grid += g
                rfid += r
                del tok
                if device == "cuda":
                    torch.cuda.empty_cache()
            write_csv(grid_csv, GRID_FIELDS_DEFENSE, grid)
            write_csv(rfid_csv, RFID_FIELDS_DEFENSE, rfid)
        else:
            tok = make_tokenizer(name, args, device, cfg["img_size"])
            grid, rfid = fidelity_grid(name, tok.reconstruct, args.adv_dir, name, Xc, Yc, cfg["eps_grid"], device)
            write_csv(grid_csv, GRID_FIELDS, grid)
            write_csv(rfid_csv, RFID_FIELDS, rfid)
            Xc_recon = batched_reconstruct(tok.reconstruct, Xc, device)
            write_csv(clean_csv, CLEAN_FIELDS, [clean_baseline_row(name, Xc, Xc_recon, device)])
            del tok
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
