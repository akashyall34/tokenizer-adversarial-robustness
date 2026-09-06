#!/usr/bin/env python3
"""Merges FSQ's advft (encoder-side) checkpoint with its decoder-realism
(decoder-side) checkpoint into one combined defense. Same hard-stop-on-
collapse discipline as scripts/defenses/fsq_aptos/merge_checkpoints_aptos.py.

FSQ's quantize_t is parameter-free (no codebook, no learnable buffers), so
every frozen key in both checkpoints is expected to be byte-identical, with
zero diagnostic exceptions needed (unlike MediTok/UniTok's vocab_usage EMA
buffer).

Usage:
    python scripts/defenses/fsq_isic/merge_checkpoints_isic.py \
        --data-root /path/to/isic2019 --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.common.checkpoint_merge import merge_state_dicts
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_isic_classifier import EfficientNetB4Classifier, load_efficientnet_b4
from src.tokenizers.fsq import FSQTokenizer

ENCODER_PREFIXES = ("enc.",)
DECODER_PREFIXES = ("dec.",)
AUC_FLOOR = 0.55


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
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_advft_eps{advft_eps}_isic.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, "fsq_decoder_realism_robust_isic.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_combined_eps{advft_eps}_isic.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p}; run its training script first"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    if os.path.isfile(combined_ckpt_path) and os.path.getsize(combined_ckpt_path) > 0:
        print(f"combined checkpoint already exists, skip merge -> {combined_ckpt_path}")
        combined = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
        combined.net.load_state_dict(torch.load(combined_ckpt_path, map_location=device, weights_only=False))
    else:
        sd_advft = torch.load(advft_ckpt_path, map_location=device, weights_only=False)
        sd_decoder = torch.load(decoder_ckpt_path, map_location=device, weights_only=False)
        merged = merge_state_dicts(sd_advft, sd_decoder, ENCODER_PREFIXES, DECODER_PREFIXES)
        combined = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
        combined.net.load_state_dict(merged)
        torch.save(combined.net.state_dict(), combined_ckpt_path)
        assert os.path.getsize(combined_ckpt_path) > 0, f"torch.save produced a 0-byte file at {combined_ckpt_path}"
        print(f"combined checkpoint saved -> {combined_ckpt_path}")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(drift_x)).cpu().numpy())))
        auc_comb = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, combined.reconstruct(drift_x)).cpu().numpy())))
    print(f"clean recon AUC: original={auc_orig:.4f}  combined(encoder+decoder)={auc_comb:.4f}  "
          f"(drop={auc_orig - auc_comb:.4f}){'  *** LARGE DROP ***' if (auc_orig - auc_comb) > 0.03 else ''}")

    assert auc_comb > AUC_FLOOR, (
        f"combined checkpoint's clean AUC ({auc_comb:.4f}) is at/near chance; broken checkpoint, "
        f"not a real defense result. Delete {advft_ckpt_path} and {combined_ckpt_path}, then rerun "
        f"run_advft_isic.py for this eps.")
    print(f"\nDONE. combined encoder+decoder model ready at {combined_ckpt_path} "
          f"(advft eps={advft_eps} + decoder_realism)")


if __name__ == "__main__":
    main()
