#!/usr/bin/env python3
"""Two geometric adaptive-attack checks against every VQ-16 defense
checkpoint, run in one pass per checkpoint (the notebook this was ported
from reloaded each checkpoint twice, once per attack; this loads once):

- geo (pure margin): maximizes the CORRECT margin against the model's own
  current boundary. Fully differentiable, never touches decode() or the
  classifier.
- geo_bpda (blended): geo margin + BCE-through-decode, live-adaptive
  counterpart to VQ16.ipynb's cached M3_alpha03 attack.

IMPORTANT, read before trusting geo's numbers for decoder-realism/combined:
the pure geo attack's gradient only ever touches encoder+quant_conv+quantize
-- it never reaches decode() or the classifier. decoder-realism's encoder is
frozen and byte-identical to the original tokenizer's (only decoder/
post_quant_conv were trained), so geo computes the EXACT SAME perturbation
against decoder-realism as it would against the untouched original -- it has
no gradient line-of-sight into the one thing decoder-realism actually
changed. A high geo-adaptive AUC for decoder-realism/combined is NOT evidence
of real robustness; it's evidence this attack never had the ability to see
the defense. Trust geo_bpda's numbers for decoder-realism/combined instead
(it routes through decode(), so it can actually see what changed) -- for
advft, both are meaningful since advft's encoder is what actually changed.
See docs/known_limitations.md.

Usage:
    python scripts/defenses/vq16_aptos/geo_adaptive_aptos.py --data-root /path/to/aptos2019 \
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
from src.defenses.vq16.bpda_attacks import geo_adaptive_attack, geo_bpda_attack
from src.defenses.vq16.margin import reconstruct
from src.eval.metrics import asr, auc
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

GEO_BPDA_ALPHA = 0.3  # same 30%-geometric/70%-BCE blend VQ16.ipynb's M3_alpha03 uses


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
                    help="which combined checkpoint to load -- must match whatever "
                         "merge_checkpoints_aptos.py was run with (default: worst-case/largest in eps_grid)")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


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
    items = report_items[:n]

    x, y_np = load_xy(items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)
    densenet = load_densenet(args.effnet_ckpt, device)

    geo_csv = os.path.join(args.out_dir, "vq16_geo_adaptive_full_grid.csv")
    geo_bpda_csv = os.path.join(args.out_dir, "vq16_geo_bpda_full_grid.csv")
    if os.path.isfile(geo_csv) and os.path.isfile(geo_bpda_csv):
        print(f"already computed, loading instead of rerunning attacks -> {geo_csv}, {geo_bpda_csv}")
        print(pd.read_csv(geo_csv).to_string(index=False))
        print(pd.read_csv(geo_bpda_csv).to_string(index=False))
        return

    def evaluate(model, x_attacked):
        with torch.no_grad():
            return densenet_logit(densenet, reconstruct(model, x_attacked)).cpu().numpy()

    geo_rows, geo_bpda_rows = [], []

    def run_both(model, label, eps255):
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, x).cpu().numpy()
            static_lg = densenet_logit(densenet, reconstruct(model, x)).cpu().numpy()

        x_geo = geo_adaptive_attack(model, x, eps255 / 255.0, args.steps, args.lr, bs, device)
        geo_lg = evaluate(model, x_geo)
        geo_rows.append(dict(label=label, eps255=eps255, n=len(y_np),
                              clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg), geo_auc=auc(y_np, geo_lg),
                              geo_asr=asr(clean_lg, geo_lg, y_np)))
        print(f"--- {label} -- GEOMETRIC adaptive attack (n={len(y_np)}, eval eps255={eps255}) ---")
        print(f"clean AUC          : {geo_rows[-1]['clean_auc']:.4f}")
        print(f"static-attack AUC  : {geo_rows[-1]['static_auc']:.4f}")
        print(f"geo-adaptive AUC   : {geo_rows[-1]['geo_auc']:.4f}")
        print(f"geo-adaptive ASR   : {geo_rows[-1]['geo_asr']:.1f}%\n")

        x_gb = geo_bpda_attack(model, densenet, x, y, eps255 / 255.0, GEO_BPDA_ALPHA, args.steps, args.lr, bs, device)
        gb_lg = evaluate(model, x_gb)
        geo_bpda_rows.append(dict(label=label, eps255=eps255, n=len(y_np),
                                   clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg),
                                   geo_bpda_auc=auc(y_np, gb_lg), geo_bpda_asr=asr(clean_lg, gb_lg, y_np)))
        print(f"--- {label} -- BLENDED live-adaptive attack (n={len(y_np)}, eval eps255={eps255}) ---")
        print(f"geo+BCE-adaptive AUC: {geo_bpda_rows[-1]['geo_bpda_auc']:.4f}")
        print(f"geo+BCE-adaptive ASR: {geo_bpda_rows[-1]['geo_bpda_asr']:.1f}%\n")

    for eps255 in cfg["eps_grid"]:
        ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{eps255}_aptos.pt")
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run run_advft_aptos.py first"
        net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        net.eval()
        run_both(net, f"advft(eps{eps255}) ALONE, no SQS", eps255)
        del net
        torch.cuda.empty_cache()

    decoder_ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust_aptos.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_combined_eps{combined_eps}_aptos.pt")
    for ckpt_path, label in [
        (decoder_ckpt_path, "decoder-realism-robust ALONE, no SQS"),
        (combined_ckpt_path, "combined (advft+decoder-realism-robust) ALONE, no SQS"),
    ]:
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path} -- run its training/merge script first"
        for eps255 in cfg["eps_grid"]:
            net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
            net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            net.eval()
            run_both(net, label, eps255)
            del net
            torch.cuda.empty_cache()

    pd.DataFrame(geo_rows).to_csv(geo_csv, index=False)
    pd.DataFrame(geo_bpda_rows).to_csv(geo_bpda_csv, index=False)
    print("wrote ->", geo_csv)
    print("wrote ->", geo_bpda_csv)
    print("\nFor decoder-realism/combined, trust geo_bpda_auc, not geo_auc -- see this script's "
          "module docstring for why the pure geo attack is blind to decoder-side defenses.")


if __name__ == "__main__":
    main()
