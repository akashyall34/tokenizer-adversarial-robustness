#!/usr/bin/env python3
"""Two geometric adaptive-attack checks against every UniTok defense
checkpoint, one pass per checkpoint. Same role and same geo/geo_bpda pairing
as scripts/defenses/unitok_aptos/geo_adaptive_aptos.py:

- geo (pure margin): maximizes the multi-codebook cosine margin against the
  model's own current boundary. Fully differentiable, never touches decode()
  or the classifier.
- geo_bpda (blended): geo margin + BCE-through-decode, live-adaptive
  counterpart to the cached M3_alpha03 attack.

Same blind spot as VQ-16's version: pure geo's gradient never reaches
decode() or the classifier, so decoder-realism/combined's geo_auc is not
evidence of real robustness against that attack; trust geo_bpda_auc for
those two conditions instead. See docs/known_limitations.md.

Usage:
    python scripts/defenses/unitok_isic/geo_adaptive_isic.py \
        --data-root /path/to/isic2019 --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt path/to/effnet_ckpt
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

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.multicodebook.bpda_attacks import geo_adaptive_attack, geo_bpda_attack
from src.eval.metrics import asr, auc
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

GEO_BPDA_ALPHA = 0.3  # same 30%-geometric/70%-BCE blend M3_alpha03 uses


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--attack-batch", type=int, default=4,
                    help="overrides the config's attack_batch (8, sized for VQ-16); "
                         "UniTok runs heavier: decoder + B4 classifier")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load; must match whatever "
                         "merge_checkpoints_isic.py was run with (default: worst-case/largest in eps_grid)")
    p.add_argument("--n", type=int, default=None, help="default: full REPORT_ITEMS")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    return p.parse_args()


def run_one(tok_name, load_fresh, args, cfg, densenet, x, y, y_np, bs, device):
    geo_csv = os.path.join(args.out_dir, f"{tok_name}_geo_adaptive_full_grid.csv")
    geo_bpda_csv = os.path.join(args.out_dir, f"{tok_name}_geo_bpda_full_grid.csv")
    if os.path.isfile(geo_csv) and os.path.isfile(geo_bpda_csv):
        print(f"{tok_name}: already computed, loading instead of rerunning attacks -> {geo_csv}, {geo_bpda_csv}")
        print(pd.read_csv(geo_csv).to_string(index=False))
        print(pd.read_csv(geo_bpda_csv).to_string(index=False))
        return

    def evaluate(tok, x_attacked):
        with torch.no_grad():
            return densenet_logit(densenet, tok.reconstruct(x_attacked)).cpu().numpy()

    geo_rows, geo_bpda_rows = [], []

    def run_both(tok, label, eps255):
        for p in tok.net.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            clean_lg = densenet_logit(densenet, x).cpu().numpy()
            static_lg = densenet_logit(densenet, tok.reconstruct(x)).cpu().numpy()

        x_geo = geo_adaptive_attack(tok, x, eps255 / 255.0, args.steps, args.lr, bs, device)
        geo_lg = evaluate(tok, x_geo)
        geo_rows.append(dict(label=label, eps255=eps255, n=len(y_np),
                              clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg), geo_auc=auc(y_np, geo_lg),
                              geo_asr=asr(clean_lg, geo_lg, y_np)))
        print(f"--- {label}: GEOMETRIC adaptive attack (n={len(y_np)}, eval eps255={eps255}) ---")
        print(f"clean AUC          : {geo_rows[-1]['clean_auc']:.4f}")
        print(f"static-attack AUC  : {geo_rows[-1]['static_auc']:.4f}")
        print(f"geo-adaptive AUC   : {geo_rows[-1]['geo_auc']:.4f}")
        print(f"geo-adaptive ASR   : {geo_rows[-1]['geo_asr']:.1f}%\n")

        x_gb = geo_bpda_attack(tok, densenet, x, y, eps255 / 255.0, GEO_BPDA_ALPHA, args.steps, args.lr, bs, device)
        gb_lg = evaluate(tok, x_gb)
        geo_bpda_rows.append(dict(label=label, eps255=eps255, n=len(y_np),
                                   clean_auc=auc(y_np, clean_lg), static_auc=auc(y_np, static_lg),
                                   geo_bpda_auc=auc(y_np, gb_lg), geo_bpda_asr=asr(clean_lg, gb_lg, y_np)))
        print(f"--- {label}: BLENDED live-adaptive attack (n={len(y_np)}, eval eps255={eps255}) ---")
        print(f"geo+BCE-adaptive AUC: {geo_bpda_rows[-1]['geo_bpda_auc']:.4f}")
        print(f"geo+BCE-adaptive ASR: {geo_bpda_rows[-1]['geo_bpda_asr']:.1f}%\n")

    for eps255 in cfg["eps_grid"]:
        ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_advft_eps{eps255}_isic.pt")
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path}; run run_advft_isic.py first"
        tok = load_fresh()
        tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        tok.net.eval()
        run_both(tok, f"{tok_name} advft(eps{eps255}) ALONE, no SQS", eps255)
        del tok
        torch.cuda.empty_cache()

    decoder_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_decoder_realism_robust_isic.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_combined_eps{combined_eps}_isic.pt")
    for ckpt_path, label in [
        (decoder_ckpt_path, f"{tok_name} decoder-realism-robust ALONE, no SQS"),
        (combined_ckpt_path, f"{tok_name} combined (advft+decoder-realism-robust) ALONE, no SQS"),
    ]:
        assert os.path.isfile(ckpt_path), f"missing {ckpt_path}; run its training/merge script first"
        for eps255 in cfg["eps_grid"]:
            tok = load_fresh()
            tok.net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            tok.net.eval()
            run_both(tok, label, eps255)
            del tok
            torch.cuda.empty_cache()

    pd.DataFrame(geo_rows).to_csv(geo_csv, index=False)
    pd.DataFrame(geo_bpda_rows).to_csv(geo_bpda_csv, index=False)
    print(f"{tok_name}: wrote -> {geo_csv}")
    print(f"{tok_name}: wrote -> {geo_bpda_csv}")
    print(f"\n{tok_name}: for decoder-realism/combined, trust geo_bpda_auc, not geo_auc; see this "
          f"script's module docstring for why the pure geo attack is blind to decoder-side defenses.\n")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = args.attack_batch

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]
    n = args.n if args.n is not None else len(report_items)
    items = report_items[:n]

    x, y_np = load_xy(items, cfg["img_size"], device)
    y = torch.tensor(y_np, dtype=torch.float32)
    densenet = load_densenet(args.effnet_ckpt, device)

    tok_name = "unitok"
    load_fresh = lambda: load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)
    print(f"=== {tok_name} ===")
    run_one(tok_name, load_fresh, args, cfg, densenet, x, y, y_np, bs, device)


if __name__ == "__main__":
    main()
