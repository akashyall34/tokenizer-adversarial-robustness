#!/usr/bin/env python3
"""Adversarial fine-tuning of the VQ-16 encoder against its own M2 boundary
attack, on APTOS. One checkpoint per eps255 in the config's eps_grid.

Trains directly against the already-generated cached attack images (see
run_vq16_aptos.py); no attack is generated here. Trains only on SEL_ITEMS
(half of the fixed 256-image attack subset); Part 3/BPDA/certification score
against REPORT_ITEMS, the untouched other half, so nothing reported was also
used to pick its own hyperparameters or train the model being reported on.

The inner "attack" step finds the worst-case perturbation that maximizes the
margin against the current model (M2-sat's mechanism, pointed at the
trainable model); the outer "defense" step takes a gradient step on the
encoder + quant_conv only (codebook, decoder, classifier frozen) to minimize
that same worst-case margin. No labels: the target code k1 is read off the
model's own clean forward pass each step.

Uses the corrected, L2-normalized margin (src/defenses/vq16/margin.py), not
the buggy unnormalized one in src/attacks/vq16_attacks.py; see
docs/known_limitations.md.

Usage:
    python scripts/defenses/vq16_aptos/run_advft_aptos.py --data-root /path/to/aptos2019 \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
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
from src.defenses.vq16.margin import clean_top1, freeze_non_encoder, margin, reconstruct
from src.eval.attack_io import load_folder
from src.eval.sel_report_split import stratified_half_split
from src.models.efficientnet_classifier import EfficientNetClassifier, load_efficientnet
from src.tokenizers.vq16 import load_vq16_net


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


ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    p.add_argument("--max-epochs", type=int, default=40, help="ceiling, not a target; watch the loss log")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--early-stop-min-epochs", type=int, default=3,
                    help="never stop before this many epochs; early deltas aren't a reliable plateau signal")
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-rel-thresh", type=float, default=0.10,
                    help="an epoch's improvement must be >= this fraction of the arm's own first "
                         "epoch-over-epoch improvement to still count as improving")
    p.add_argument("--clean-auc-collapse-drop", type=float, default=0.10,
                    help="stop immediately if clean AUC falls this far below the pretrained baseline; "
                         "margin loss alone keeps improving smoothly even through a clean-AUC collapse, "
                         "so it can't be trusted as the only stopping signal")
    return p.parse_args()


def build_cached_train_set(adv_dir, sel_idx, sel_items, eps255, device):
    """SEL_ITEMS' cached adversarial images across every attack in
    ATTACKS_TO_DEFEND, for this eps255."""
    adv_list, clean_idx_list = [], []
    for attack in ATTACKS_TO_DEFEND:
        advs, ys = load_folder(adv_dir, "vq16", attack, eps255, idx_subset=sel_idx)
        clean_ys = np.array([l for _, l in sel_items])
        assert (ys == clean_ys).all(), f"label mismatch loading {attack} eps{eps255} cached images vs SEL_ITEMS"
        adv_list.append(advs)
        clean_idx_list.append(np.arange(len(sel_items)))
    return torch.cat(adv_list, dim=0), np.concatenate(clean_idx_list)


def finetune_one(eps255, args, orig_net, densenet, sel_items, report_items, Xc_sel, Yc_sel, sel_idx, img_size, device):
    ckpt_path = os.path.join(args.ckpt_dir, f"vq16_advft_eps{eps255}_aptos.pt")
    log_path = os.path.join(args.out_dir, f"vq16_advft_eps{eps255}_loss_log.csv")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"  eps{eps255}: checkpoint already exists, skip -> {ckpt_path}")
        return ckpt_path

    X_adv_train, clean_idx_train = build_cached_train_set(args.adv_dir, sel_idx, sel_items, eps255, device)
    print(f"eps{eps255}: training set = {X_adv_train.shape[0]} cached (image, attack-variant) pairs "
          f"from SEL_ITEMS x {ATTACKS_TO_DEFEND}")

    model = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    print(f"eps{eps255}: submodule trainability check:")
    freeze_non_encoder(model, verbose=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    with torch.no_grad():
        clean_auc_baseline = roc_auc_score(
            Yc_sel, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(orig_net, Xc_sel)).cpu().numpy())))
    print(f"  eps{eps255}: baseline (pretrained) clean AUC on SEL_ITEMS = {clean_auc_baseline:.4f}")

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
                k1 = clean_top1(model, xb_clean)
            opt.zero_grad()
            loss = margin(model, xb_adv, k1, saturate=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
            bs = xb_adv.shape[0]
            loss_sum += loss.item() * bs
            n_seen += bs

        with torch.no_grad():
            clean_auc = roc_auc_score(
                Yc_sel, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(model, Xc_sel)).cpu().numpy())))

        row = dict(epoch=epoch, margin_loss=loss_sum / n_seen, clean_auc=clean_auc)
        epoch_log.append(row)
        print(f"  eps{eps255} epoch {epoch}: margin_loss={row['margin_loss']:.4f}  clean_auc={clean_auc:.4f} "
              f"(baseline={clean_auc_baseline:.4f})")
        pd.DataFrame(epoch_log).to_csv(log_path, index=False)

        if clean_auc > best_auc:
            best_auc = clean_auc
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

        if clean_auc < clean_auc_baseline - args.clean_auc_collapse_drop:
            print(f"  eps{eps255}: *** CLEAN AUC COLLAPSE at epoch {epoch} "
                  f"({clean_auc:.4f} vs baseline {clean_auc_baseline:.4f}); stopping now, "
                  f"reverting to best epoch (clean_auc={best_auc:.4f}) instead of this one ***")
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
                print(f"  eps{eps255}: early stop at epoch {epoch}; improvement plateaued "
                      f"(this epoch's delta={delta:.4f} vs first-epoch delta={first_delta:.4f})")
                break
        prev_loss = row["margin_loss"]

    assert best_state is not None, f"eps{eps255}: no epoch ever completed; nothing to save"
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    assert os.path.getsize(ckpt_path) > 0, f"eps{eps255}: torch.save produced a 0-byte file"
    print(f"  eps{eps255}: saved BEST epoch (clean_auc={best_auc:.4f}, baseline={clean_auc_baseline:.4f}) -> {ckpt_path}")

    drift_x, drift_y = load_xy(report_items, img_size, device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(orig_net, drift_x)).cpu().numpy())))
        auc_ft = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(model, drift_x)).cpu().numpy())))
    drop = auc_orig - auc_ft
    print(f"  eps{eps255}: clean recon AUC (REPORT_ITEMS, held-out): original={auc_orig:.4f}  "
          f"fine-tuned={auc_ft:.4f}  (drop={drop:.4f})"
          f"{'  *** LARGE DROP; possible collapse, inspect before trusting this checkpoint ***' if drop > 0.03 else ''}")
    del model
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
    print(f"SEL_ITEMS: n={len(sel_items)}")
    print(f"REPORT_ITEMS: n={len(report_items)}")

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)

    seed_everything(cfg["seed"])
    ckpts = {}
    for eps255 in cfg["eps_grid"]:
        ckpts[eps255] = finetune_one(
            eps255, args, orig_net, densenet, sel_items, report_items, Xc_sel, Yc_sel, sel_idx, cfg["img_size"], device,
        )
    print("\nDONE. fine-tuned checkpoints:", ckpts)


if __name__ == "__main__":
    main()
