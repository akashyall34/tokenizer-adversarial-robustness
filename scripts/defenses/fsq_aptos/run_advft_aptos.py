#!/usr/bin/env python3
"""Adversarial fine-tuning of the FSQ encoder against its own M3 boundary
attack, on APTOS. One checkpoint per eps255 in the config's eps_grid. Same
mechanism as scripts/defenses/vq16_aptos/run_advft_aptos.py, adapted to
FSQ's per-dimension anchored scalar margin (src/defenses/fsq/margin.py) --
FSQ has no codebook, so this rebuilds the margin geometry from scratch
rather than adapting VQ-16/MediTok's nearest-neighbor math.

Trains directly against the already-generated cached attack images -- no
attack is generated here. Trains ONLY on SEL_ITEMS; Part 3/BPDA/certification
score against REPORT_ITEMS.

Usage:
    python scripts/defenses/fsq_aptos/run_advft_aptos.py \
        --data-root /path/to/aptos2019 --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b5_aptos.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.aptos import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.fsq.margin import clean_anchor, freeze_non_encoder, fsq_margin
from src.eval.attack_io import load_folder
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet
from src.tokenizers.fsq import FSQTokenizer

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M3_geo_sat", "M3_geo_unb"]


def load_densenet(ckpt_path, device):
    """Alias, same trick the source notebook uses: wrap EfficientNet-B5 into
    the same load_ckpt(path, device) -> classifier object interface this
    script's downstream code expects."""
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
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--early-stop-min-epochs", type=int, default=3)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-rel-thresh", type=float, default=0.10)
    p.add_argument("--clean-auc-collapse-drop", type=float, default=0.10)
    return p.parse_args()


def build_cached_train_set(adv_dir, sel_idx, sel_items, eps255):
    adv_list, clean_idx_list = [], []
    clean_ys = np.array([l for _, l in sel_items])
    for attack in ATTACKS_TO_DEFEND:
        advs, ys = load_folder(adv_dir, "fsq", attack, eps255, idx_subset=sel_idx)
        assert (ys == clean_ys).all(), f"label mismatch loading {attack} eps{eps255} vs SEL_ITEMS"
        adv_list.append(advs)
        clean_idx_list.append(np.arange(len(sel_items)))
    return torch.cat(adv_list, dim=0), np.concatenate(clean_idx_list)


def finetune_one(eps255, args, orig_tok, densenet, sel_items, report_items, Xc_sel, Yc_sel, sel_idx, img_size, device):
    ckpt_path = os.path.join(args.ckpt_dir, f"fsq_advft_eps{eps255}_aptos.pt")
    log_path = os.path.join(args.out_dir, f"fsq_advft_eps{eps255}_loss_log.csv")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"  eps{eps255}: checkpoint already exists, skip -> {ckpt_path}")
        return ckpt_path

    X_adv_train, clean_idx_train = build_cached_train_set(args.adv_dir, sel_idx, sel_items, eps255)
    print(f"eps{eps255}: training set = {X_adv_train.shape[0]} cached (image, attack-variant) pairs")

    tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    print(f"eps{eps255}: submodule trainability check --")
    freeze_non_encoder(tok, verbose=True)
    opt = torch.optim.Adam([p for p in tok.net.parameters() if p.requires_grad], lr=args.lr)

    with torch.no_grad():
        clean_auc_baseline = roc_auc_score(
            Yc_sel, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(Xc_sel)).cpu().numpy())))
    print(f"  eps{eps255}: baseline clean AUC on SEL_ITEMS = {clean_auc_baseline:.4f}")

    epoch_log = []
    first_delta, patience_ctr, prev_loss = None, 0, None
    best_auc, best_state = -1.0, None
    n = X_adv_train.shape[0]
    for epoch in range(args.max_epochs):
        perm = np.random.permutation(n)
        pbar = tqdm(range(0, n, args.batch), desc=f"advft eps{eps255} epoch{epoch}", leave=False)
        loss_sum, n_seen = 0.0, 0
        for bi in pbar:
            batch_pos = perm[bi:bi + args.batch]
            xb_adv = X_adv_train[batch_pos].to(device)
            xb_clean = Xc_sel[clean_idx_train[batch_pos]].to(device)
            with torch.no_grad():
                anchor = clean_anchor(tok, xb_clean)
            opt.zero_grad()
            loss = fsq_margin(tok, xb_adv, anchor, saturate=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
            bs = xb_adv.shape[0]
            loss_sum += loss.item() * bs
            n_seen += bs

        with torch.no_grad():
            clean_auc = roc_auc_score(
                Yc_sel, 1 / (1 + np.exp(-densenet_logit(densenet, tok.reconstruct(Xc_sel)).cpu().numpy())))

        row = dict(epoch=epoch, margin_loss=loss_sum / n_seen, clean_auc=clean_auc)
        epoch_log.append(row)
        print(f"  eps{eps255} epoch {epoch}: margin_loss={row['margin_loss']:.4f}  clean_auc={clean_auc:.4f}")
        pd.DataFrame(epoch_log).to_csv(log_path, index=False)

        if clean_auc > best_auc:
            best_auc = clean_auc
            best_state = {k: v.detach().clone().cpu() for k, v in tok.net.state_dict().items()}

        if clean_auc < clean_auc_baseline - args.clean_auc_collapse_drop:
            print(f"  eps{eps255}: *** CLEAN AUC COLLAPSE at epoch {epoch} -- stopping, "
                  f"reverting to best epoch (clean_auc={best_auc:.4f}) ***")
            break

        if prev_loss is not None:
            delta = prev_loss - row["margin_loss"]
            if first_delta is None:
                first_delta = delta
            elif first_delta > 0 and delta < args.early_stop_rel_thresh * first_delta:
                patience_ctr += 1
            else:
                patience_ctr = 0
            if epoch + 1 >= args.early_stop_min_epochs and patience_ctr >= args.early_stop_patience:
                print(f"  eps{eps255}: early stop at epoch {epoch} -- improvement plateaued")
                break
        prev_loss = row["margin_loss"]

    assert best_state is not None, f"eps{eps255}: no epoch ever completed -- nothing to save"
    tok.net.load_state_dict(best_state)
    torch.save(tok.net.state_dict(), ckpt_path)
    assert os.path.getsize(ckpt_path) > 0, f"eps{eps255}: torch.save produced a 0-byte file"
    print(f"  eps{eps255}: saved BEST epoch (clean_auc={best_auc:.4f}) -> {ckpt_path}")

    drift_x, drift_y = load_xy(report_items, img_size, device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(drift_x)).cpu().numpy())))
        auc_ft = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, tok.reconstruct(drift_x)).cpu().numpy())))
    drop = auc_orig - auc_ft
    print(f"  eps{eps255}: clean recon AUC (REPORT_ITEMS) -- original={auc_orig:.4f}  "
          f"fine-tuned={auc_ft:.4f}  (drop={drop:.4f})"
          f"{'  *** LARGE DROP -- inspect before trusting this checkpoint ***' if drop > 0.03 else ''}")
    del tok, X_adv_train
    torch.cuda.empty_cache()
    return ckpt_path


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

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)

    seed_everything(cfg["seed"])
    ckpts = {}
    for eps255 in cfg["eps_grid"]:
        ckpts[eps255] = finetune_one(eps255, args, orig_tok, densenet, sel_items, report_items,
                                      Xc_sel, Yc_sel, sel_idx, cfg["img_size"], device)
    print("\nDONE. fine-tuned checkpoints:", ckpts)


if __name__ == "__main__":
    main()
