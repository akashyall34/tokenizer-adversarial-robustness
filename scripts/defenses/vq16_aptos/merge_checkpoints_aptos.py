#!/usr/bin/env python3
"""Merges one adversarial-fine-tuned (encoder-side) checkpoint with the
decoder-realism (decoder-side) checkpoint into a single combined defense.
Valid as a state-dict merge, not a joint training run, because the two runs
touch disjoint parameter subsets and never move the frozen codebook -- see
src/defenses/common/checkpoint_merge.py and the caveat in this script's docstring below.

Caveat, stated plainly: this is not provably identical to a hypothetical
joint training run. The decoder-realism run trained against the ORIGINAL
encoder's clean-image codes, not the advft-fine-tuned encoder's codes. In
practice this should be a non-issue -- advft only widens the margin around
each clean image's existing top-1 code, it doesn't change which code clean
images get assigned to (confirmed by both checkpoints' independent clean-AUC
drift checks) -- so the merged decoder should see ~identical quant indices at
inference time to what it was trained on. But this is an assumption the
merge relies on, not something proven by construction.

The AUC-floor assertion below is a HARD stop, not just a print: a collapsed
combined checkpoint reportedly reached the final results table as real
numbers once (all-0.5 AUC rows) because that check used to be a warning
only. If you hit this assertion, delete the encoder checkpoint for this eps
and rerun run_advft_aptos.py for it before merging again.

Usage:
    python scripts/defenses/vq16_aptos/merge_checkpoints_aptos.py --data-root /path/to/aptos2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b5_aptos.pt
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

from src.datasets.aptos import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.common.checkpoint_merge import merge_state_dicts
from src.defenses.vq16.margin import reconstruct
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet
from src.tokenizers.vq16 import load_vq16_net


def load_densenet(ckpt_path, device):
    """Alias, same trick the source notebook uses: wrap EfficientNet-B5 into
    the same load_ckpt(path, device) -> classifier object interface this
    script's downstream code (margin/decoder-realism/certify helpers, all
    written against DenseNet's raw-nn.Module calling convention: net(x) for
    the logit, net.features(x) for perceptual features) expects, so none of
    that shared code needs touching."""
    net = load_efficientnet(ckpt_path, device)
    return EfficientNetClassifier(net, device)


def densenet_logit(densenet, x):
    return densenet(x)

ENCODER_PREFIXES = ("encoder.", "quant_conv.")
DECODER_PREFIXES = ("decoder.", "post_quant_conv.")
AUC_FLOOR = 0.55  # near chance for this binary task -- a checkpoint at/below this is broken, not a real result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--advft-eps255", type=int, default=None,
                    help="which advft checkpoint's encoder to merge (default: worst-case/largest in eps_grid)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    advft_eps = args.advft_eps255 if args.advft_eps255 is not None else cfg["eps_grid"][-1]

    advft_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{advft_eps}_aptos.pt")
    decoder_ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust_aptos.pt")
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_combined_eps{advft_eps}_aptos.pt")
    for p in (advft_ckpt_path, decoder_ckpt_path):
        assert os.path.isfile(p), f"missing checkpoint: {p} -- run its training script first"

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)

    if os.path.isfile(combined_ckpt_path) and os.path.getsize(combined_ckpt_path) > 0:
        print(f"combined checkpoint already exists, skip merge -> {combined_ckpt_path}")
        combined = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        combined.load_state_dict(torch.load(combined_ckpt_path, map_location=device, weights_only=False))
        combined.eval()
    else:
        sd_advft = torch.load(advft_ckpt_path, map_location=device, weights_only=False)
        sd_decoder = torch.load(decoder_ckpt_path, map_location=device, weights_only=False)
        merged = merge_state_dicts(sd_advft, sd_decoder, ENCODER_PREFIXES, DECODER_PREFIXES)
        combined = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        combined.load_state_dict(merged)
        torch.save(combined.state_dict(), combined_ckpt_path)
        assert os.path.getsize(combined_ckpt_path) > 0, f"torch.save produced a 0-byte file at {combined_ckpt_path}"
        print(f"combined checkpoint saved -> {combined_ckpt_path}")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(orig_net, drift_x)).cpu().numpy())))
        auc_comb = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(combined, drift_x)).cpu().numpy())))
    print(f"clean recon AUC -- original={auc_orig:.4f}  combined(encoder+decoder)={auc_comb:.4f}  "
          f"(drop={auc_orig - auc_comb:.4f}){'  *** LARGE DROP -- possible collapse ***' if (auc_orig - auc_comb) > 0.03 else ''}")

    assert auc_comb > AUC_FLOOR, (
        f"combined checkpoint's clean AUC ({auc_comb:.4f}) is at/near chance -- this is a broken "
        f"checkpoint, not a real defense result. Do not let it into the final results table. Delete "
        f"{advft_ckpt_path} and {combined_ckpt_path}, then rerun run_advft_aptos.py for this eps.")
    print(f"\nDONE. combined encoder+decoder model ready at {combined_ckpt_path} "
          f"(advft eps={advft_eps} + decoder_realism)")


if __name__ == "__main__":
    main()
