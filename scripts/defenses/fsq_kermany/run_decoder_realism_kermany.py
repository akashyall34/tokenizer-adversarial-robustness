#!/usr/bin/env python3
"""Decoder-realism fine-tuning for FSQ on Kermany: GAN (hinge loss) + perceptual
(EfficientNet-B5 feature-matching) + L1, decoder-only. Reuses
src/defenses/common/decoder_realism.py directly; FSQ's decode path is a
single net.dec(q) call on the spatial quantized tensor (same shape as
VQ-16's model.decode(q)), so nothing tokenizer-specific needs reimplementing
here, unlike MediTok/UniTok's two-call decode path.

Corrupted codes come from the cached .pt images (encoded through the frozen
encoder), not a synthetic flip simulation.

Usage:
    python scripts/defenses/fsq_kermany/run_decoder_realism_kermany.py \
        --data-root /path/to/chest_xray --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
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

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.defenses.common.decoder_realism import PatchDiscriminator, freeze_non_decoder, train_step
from src.eval.attack_io import load_folder
from src.eval.sel_report_split import stratified_half_split
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.fsq import FSQTokenizer

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]


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
    p.add_argument("--densenet-ckpt", required=True)
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--eps255", type=int, default=None, help="default: worst-case/largest in eps_grid")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--adv-loss-weight", type=float, default=0.1)
    p.add_argument("--perceptual-loss-weight", type=float, default=1.0)
    p.add_argument("--recon-loss-weight", type=float, default=1.0)
    p.add_argument("--robust-loss-weight", type=float, default=1.0)
    p.add_argument("--early-stop-min-epochs", type=int, default=3)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--early-stop-rel-thresh", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    eps255 = args.eps255 if args.eps255 is not None else cfg["eps_grid"][-1]

    ckpt_path = os.path.join(args.ckpt_dir, "fsq_decoder_realism_robust.pt")
    log_path = os.path.join(args.out_dir, "fsq_decoder_realism_robust_loss_log.csv")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"decoder-realism (robust) checkpoint already exists, skip -> {ckpt_path}")
        return

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]
    report_items = [attack_items[i] for i in report_idx]

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.densenet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)

    tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    print("decoder-realism (robust): submodule trainability check:")
    freeze_non_decoder(tok.net, decoder_attrs=("dec",), verbose=True)
    disc = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam([p for p in tok.net.parameters() if p.requires_grad], lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))

    with torch.no_grad():
        def encode_batched(X, bs=32):
            outs = []
            for i in range(0, X.shape[0], bs):
                z = tok.net.enc(X[i:i + bs].to(device))
                quant_t, _ = tok.net.quantize_t(z)
                outs.append(quant_t)
            return torch.cat(outs, dim=0)

        quant_clean_sel = encode_batched(Xc_sel)
        quant_adv_list, clean_idx_list = [], []
        for attack in ATTACKS_TO_DEFEND:
            advs, ys = load_folder(args.adv_dir, "fsq", attack, eps255, idx_subset=sel_idx)
            assert (ys == Yc_sel).all(), f"label mismatch loading {attack} eps{eps255} vs SEL_ITEMS"
            quant_adv_list.append(encode_batched(advs))
            clean_idx_list.append(np.arange(len(sel_items)))
        quant_adv_sel = torch.cat(quant_adv_list, dim=0)
        clean_idx_for_adv = np.concatenate(clean_idx_list)

    n = quant_adv_sel.shape[0]
    print(f"decoder-realism (robust): training on {n} cached-adversarial code pairs (eps={eps255})")

    decode_fn = lambda m, q: m.dec(q)  # noqa: E731; FSQ's decode is a single net.dec(q) call

    epoch_log = []
    first_delta, patience_ctr, prev_loss = None, 0, None
    for epoch in range(args.max_epochs):
        perm = np.random.permutation(n)
        pbar = tqdm(range(0, n, args.batch), desc=f"decoder-realism-robust epoch{epoch}", leave=False)
        sums = dict(g_loss=0.0, d_loss=0.0, recon_loss=0.0, perc_loss=0.0, adv_loss=0.0,
                    robust_recon_loss=0.0, robust_perc_loss=0.0)
        n_seen = 0
        for bi in pbar:
            batch_pos = perm[bi:bi + args.batch]
            q_corrupt = quant_adv_sel[batch_pos]
            c_idx = clean_idx_for_adv[batch_pos]
            q_clean = quant_clean_sel[c_idx]
            xb = Xc_sel[c_idx].to(device)

            step = train_step(tok.net, decode_fn, disc, opt_g, opt_d, densenet, q_clean, q_corrupt, xb,
                               args.adv_loss_weight, args.perceptual_loss_weight,
                               args.recon_loss_weight, args.robust_loss_weight)
            pbar.set_postfix(g=step["g_loss"], d=step["d_loss"], rrec=step["robust_recon_loss"])
            bs = xb.shape[0]
            for k in sums:
                sums[k] += step[k] * bs
            n_seen += bs

        row = dict(epoch=epoch, **{k: v / n_seen for k, v in sums.items()})
        epoch_log.append(row)
        print(f"  epoch {epoch}: g={row['g_loss']:.4f} d={row['d_loss']:.4f} "
              f"robust_recon={row['robust_recon_loss']:.4f}")
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
                print(f"  early stop at epoch {epoch}; robust_recon_loss plateaued")
                break
        prev_loss = row["robust_recon_loss"]

    assert os.path.getsize(ckpt_path) > 0, f"torch.save produced a 0-byte file at {ckpt_path}"
    print(f"saved -> {ckpt_path}")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, orig_tok.reconstruct(drift_x)).cpu().numpy())))
        auc_dec = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, tok.reconstruct(drift_x)).cpu().numpy())))
    drop = auc_orig - auc_dec
    print(f"clean recon AUC (REPORT_ITEMS): original={auc_orig:.4f}  decoder-realism={auc_dec:.4f}  "
          f"(drop={drop:.4f}){'  *** LARGE DROP ***' if drop > 0.03 else ''}")


if __name__ == "__main__":
    main()
