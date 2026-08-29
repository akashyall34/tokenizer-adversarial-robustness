#!/usr/bin/env python3
"""Merges each of UniTok's advft (encoder-side) checkpoint with its
decoder-realism (decoder-side) checkpoint into one combined defense. Same
hard-stop-on-collapse discipline as
scripts/defenses/unitok_aptos/merge_checkpoints_aptos.py.

One extra wrinkle vs. VQ-16: UniTok's quantizer has a `vocab_usage`
buffer -- an EMA/usage-counter stat updated on every forward pass (including
plain eval-mode reconstruct() calls), not a learned parameter, and it never
feeds back into quantization. It legitimately diverges between any two
checkpoints even from the same starting weights, so it's excluded from the
merge's identity check rather than made to fail on a false positive.

Usage:
    python scripts/defenses/unitok_isic/merge_checkpoints_isic.py \
        --data-root /path/to/isic2019 --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt path/to/effnet_ckpt
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


from src.tokenizers.meditok_unitok import load_unitok

ENCODER_PREFIXES = ("encoder.", "quant_proj.")
DECODER_PREFIXES = ("decoder.", "post_quant_proj.")
DIAGNOSTIC_ONLY_SUFFIXES = ("vocab_usage",)
AUC_FLOOR = 0.55


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    return p.parse_args()


def merge_one(tok_name, load_fresh, args, cfg, densenet, report_items, device):
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]
    advft_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_advft_eps{advft_eps}_isic.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_decoder_realism_robust_isic.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_combined_eps{advft_eps}_isic.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p} -- run its training script first"

    orig_tok = load_fresh()

    if os.path.isfile(combined_ckpt_path) and os.path.getsize(combined_ckpt_path) > 0:
        print(f"{tok_name}: combined checkpoint already exists, skip merge -> {combined_ckpt_path}")
        combined = load_fresh()
        combined.net.load_state_dict(torch.load(combined_ckpt_path, map_location=device, weights_only=False))
    else:
        sd_advft = torch.load(advft_ckpt_path, map_location=device, weights_only=False)
        sd_decoder = torch.load(decoder_ckpt_path, map_location=device, weights_only=False)
        merged = merge_state_dicts(sd_advft, sd_decoder, ENCODER_PREFIXES, DECODER_PREFIXES,
                                    diagnostic_only_suffixes=DIAGNOSTIC_ONLY_SUFFIXES)
        combined = load_fresh()
        combined.net.load_state_dict(merged)
        torch.save(combined.net.state_dict(), combined_ckpt_path)
        assert os.path.getsize(combined_ckpt_path) > 0, f"torch.save produced a 0-byte file at {combined_ckpt_path}"
        print(f"{tok_name}: combined checkpoint saved -> {combined_ckpt_path}")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(drift_x)).cpu().numpy())))
        auc_comb = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, combined.reconstruct(drift_x)).cpu().numpy())))
    print(f"{tok_name}: clean recon AUC -- original={auc_orig:.4f}  combined={auc_comb:.4f}  "
          f"(drop={auc_orig - auc_comb:.4f}){'  *** LARGE DROP ***' if (auc_orig - auc_comb) > 0.03 else ''}")

    assert auc_comb > AUC_FLOOR, (
        f"{tok_name} combined checkpoint's clean AUC ({auc_comb:.4f}) is at/near chance -- broken "
        f"checkpoint, not a real defense result. Delete {advft_ckpt_path} and {combined_ckpt_path}, "
        f"then rerun this eps's fine-tuning.")
    print(f"{tok_name}: DONE. combined model ready at {combined_ckpt_path}\n")
    del orig_tok, combined
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    densenet = load_densenet(args.effnet_ckpt, device)

    tok_name = "unitok"
    load_fresh = lambda: load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)
    print(f"=== {tok_name} ===")
    merge_one(tok_name, load_fresh, args, cfg, densenet, report_items, device)


if __name__ == "__main__":
    main()
