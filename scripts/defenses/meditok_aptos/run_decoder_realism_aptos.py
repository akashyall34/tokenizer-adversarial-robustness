#!/usr/bin/env python3
"""Decoder-realism fine-tuning for MediTok on APTOS: single-branch
recipe (real clean image vs. reconstruction-from-corrupted-code as the only
"fake" the discriminator sees) -- see
src/defenses/multicodebook/decoder_realism.py for why this differs from
VQ-16's double clean+corrupted branch design. Runs meditok in one pass.

Usage:
    python scripts/defenses/meditok_aptos/run_decoder_realism_aptos.py \
        --data-root /path/to/aptos2019 --meditok-repo third_party/meditok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt path/to/effnet_ckpt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.aptos import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.common.decoder_realism import PatchDiscriminator
from src.defenses.multicodebook.decoder_realism import train_step
from src.defenses.multicodebook.margin import freeze_non_decoder
from src.eval.attack_io import load_folder
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet


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


from src.tokenizers.meditok_unitok import load_meditok

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--meditok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--perceptual-loss-weight", type=float, default=0.1)
    p.add_argument("--adv-loss-weight", type=float, default=0.01)
    p.add_argument("--early-stop-min-epochs", type=int, default=3)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-rel-thresh", type=float, default=0.10)
    return p.parse_args()


def densenet_features(densenet, x):
    feats = densenet.features(x)
    return F.adaptive_avg_pool2d(feats, 1).flatten(1)


def train_one(tok_name, load_fresh, args, cfg, densenet, sel_items, report_items, Xc_sel, Yc_sel, sel_idx, device):
    ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_decoder_realism_robust_aptos.pt")
    log_path = os.path.join(args.out_dir, f"{tok_name}_decoder_realism_robust_loss_log.csv")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"{tok_name} decoder-realism checkpoint already exists, skip -> {ckpt_path}")
        return

    eps255 = args.eps255 if args.eps255 is not None else cfg["eps_grid"][-1]
    orig_tok = load_fresh()
    tok = load_fresh()
    print(f"{tok_name} decoder-realism: submodule trainability check --")
    freeze_non_decoder(tok, verbose=True)
    disc = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam([p for p in tok.net.parameters() if p.requires_grad], lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))

    with torch.no_grad():
        def encode_batched(X, bs=32):
            return torch.cat([tok.encode_prequant(X[i:i + bs].to(device)) for i in range(0, X.shape[0], bs)], dim=0)

        adv_list, clean_idx_list = [], []
        for attack in ATTACKS_TO_DEFEND:
            advs, ys = load_folder(args.adv_dir, tok_name, attack, eps255, idx_subset=sel_idx)
            assert (ys == Yc_sel).all(), f"label mismatch loading {attack} eps{eps255} vs SEL_ITEMS"
            adv_list.append(encode_batched(advs))
            clean_idx_list.append(np.arange(len(sel_items)))
        Z_adv_train = torch.cat(adv_list, dim=0)
        clean_idx_for_adv = np.concatenate(clean_idx_list)

    n = Z_adv_train.shape[0]
    print(f"{tok_name} decoder-realism: training on {n} cached-adversarial code pairs (eps={eps255})")

    epoch_log = []
    first_delta, patience_ctr, prev_loss = None, 0, None
    for epoch in range(args.max_epochs):
        perm = np.random.permutation(n)
        pbar = tqdm(range(0, n, args.batch), desc=f"{tok_name} decoder-realism epoch{epoch}", leave=False)
        sums = dict(g_loss=0.0, d_loss=0.0, recon_loss=0.0, perc_loss=0.0, adv_loss=0.0,
                    robust_recon_loss=0.0, robust_perc_loss=0.0)
        n_seen = 0
        for bi in pbar:
            batch_pos = perm[bi:bi + args.batch]
            z_adv = Z_adv_train[batch_pos].to(device)
            xb_clean = Xc_sel[clean_idx_for_adv[batch_pos]].to(device)
            step = train_step(tok, disc, opt_g, opt_d, lambda x: densenet_features(densenet, x), z_adv, xb_clean,
                               args.perceptual_loss_weight, args.adv_loss_weight)
            pbar.set_postfix(g=step["g_loss"], d=step["d_loss"])
            bs = xb_clean.shape[0]
            for k in sums:
                sums[k] += step[k] * bs
            n_seen += bs

        row = dict(epoch=epoch, **{k: v / n_seen for k, v in sums.items()})
        epoch_log.append(row)
        print(f"  {tok_name} epoch {epoch}: g={row['g_loss']:.4f} d={row['d_loss']:.4f} recon={row['recon_loss']:.4f}")
        pd.DataFrame(epoch_log).to_csv(log_path, index=False)
        torch.save(tok.net.state_dict(), ckpt_path)

        if prev_loss is not None:
            delta = prev_loss - row["robust_recon_loss"]
            if first_delta is None:
                first_delta = delta
            elif first_delta > 0 and delta < args.early_stop_rel_thresh * first_delta:
                patience_ctr += 1
            else:
                patience_ctr = 0
            if epoch + 1 >= args.early_stop_min_epochs and patience_ctr >= args.early_stop_patience:
                print(f"  {tok_name}: early stop at epoch {epoch} -- robust_recon_loss plateaued")
                break
        prev_loss = row["robust_recon_loss"]

    assert os.path.getsize(ckpt_path) > 0, f"torch.save produced a 0-byte file at {ckpt_path}"
    print(f"saved -> {ckpt_path}")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(drift_x)).cpu().numpy())))
        auc_dec = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, tok.reconstruct(drift_x)).cpu().numpy())))
    drop = auc_orig - auc_dec
    print(f"{tok_name} clean recon AUC (REPORT_ITEMS) -- original={auc_orig:.4f}  decoder-realism={auc_dec:.4f}  "
          f"(drop={drop:.4f}){'  *** LARGE DROP ***' if drop > 0.03 else ''}")
    del tok, orig_tok
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]
    report_items = [attack_items[i] for i in report_idx]

    densenet = load_densenet(args.effnet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)

    tok_name = "meditok"
    load_fresh = lambda: load_meditok(args.meditok_repo, args.tokenizer_ckpt_dir, cfg["img_size"], device)
    print(f"=== {tok_name} ===")
    seed_everything(cfg["seed"])
    train_one(tok_name, load_fresh, args, cfg, densenet, sel_items, report_items, Xc_sel, Yc_sel, sel_idx, device)


if __name__ == "__main__":
    main()
