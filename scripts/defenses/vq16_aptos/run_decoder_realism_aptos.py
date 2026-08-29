#!/usr/bin/env python3
"""Decoder-realism fine-tuning for VQ-16 on APTOS: trains the decoder (+
post_quant_conv) to produce plausible, in-domain reconstructions regardless
of which codes it's handed, via a perceptual + adversarial (GAN) objective
plus a robustness branch on real cached-adversarial codes. See
src/defenses/common/decoder_realism.py for the mechanism and why this is a
complementary lever to encoder fine-tuning (run_advft_aptos.py), not a
replacement for it.

Trained on CLEAN images only, using cached codes -- no new attack is
generated here. Precomputes every code once up front (encoder is frozen for
the whole run, so codes never go stale) instead of re-encoding every epoch.

Usage:
    python scripts/defenses/vq16_aptos/run_decoder_realism_aptos.py --data-root /path/to/aptos2019 \
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
from src.defenses.common.decoder_realism import PatchDiscriminator, freeze_non_decoder, train_step
from src.defenses.vq16.margin import reconstruct
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
    p.add_argument("--eps255", type=int, default=None,
                    help="which cached-attack eps to train the robustness branch against "
                         "(default: the worst-case / largest value in the config's eps_grid)")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8, help="roughly doubles per-batch memory vs. a clean-only "
                                                          "loop (two decode + two densenet.features + three "
                                                          "disc calls) -- halve first if you hit OOM")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--adv-loss-weight", type=float, default=0.1)
    p.add_argument("--perceptual-loss-weight", type=float, default=1.0)
    p.add_argument("--recon-loss-weight", type=float, default=1.0)
    p.add_argument("--robust-loss-weight", type=float, default=1.0,
                    help="weight on the corrupted-code branch -- same magnitude as the clean branch "
                         "so the decoder can't just optimize the easier clean objective and ignore the corrupted one")
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

    ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust_aptos.pt")
    log_path = os.path.join(args.out_dir, "vq16_decoder_realism_robust_loss_log.csv")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"decoder-realism (robust) checkpoint already exists, skip -> {ckpt_path}")
        return

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    sel_idx, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    sel_items = [attack_items[i] for i in sel_idx]
    report_items = [attack_items[i] for i in report_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    Xc_sel, Yc_sel = load_xy(sel_items, cfg["img_size"], device)

    model = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    print("decoder-realism (robust): submodule trainability check --")
    freeze_non_decoder(model, verbose=True)
    disc = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.999))

    with torch.no_grad():
        def encode_batched(X, bs=32):
            outs = []
            for i in range(0, X.shape[0], bs):
                q, _, _ = model.encode(X[i:i + bs].to(device))
                outs.append(q)
            return torch.cat(outs, dim=0)

        quant_clean_sel = encode_batched(Xc_sel)
        quant_adv_list, clean_idx_list = [], []
        for attack in ATTACKS_TO_DEFEND:
            advs, ys = load_folder(args.adv_dir, "vq16", attack, eps255, idx_subset=sel_idx)
            assert (ys == Yc_sel).all(), f"label mismatch loading {attack} eps{eps255} vs SEL_ITEMS"
            quant_adv_list.append(encode_batched(advs))
            clean_idx_list.append(np.arange(len(sel_items)))
        quant_adv_sel = torch.cat(quant_adv_list, dim=0)
        clean_idx_for_adv = np.concatenate(clean_idx_list)

    n = quant_adv_sel.shape[0]
    print(f"decoder-realism (robust): training on {n} cached-adversarial code pairs "
          f"(SEL_ITEMS x {ATTACKS_TO_DEFEND}, eps={eps255}) + their {len(sel_items)} matching clean images")

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

            step = train_step(model, lambda m, q: m.decode(q), disc, opt_g, opt_d, densenet, q_clean, q_corrupt, xb,
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
              f"(recon={row['recon_loss']:.4f} perc={row['perc_loss']:.4f} adv={row['adv_loss']:.4f}  "
              f"robust_recon={row['robust_recon_loss']:.4f} robust_perc={row['robust_perc_loss']:.4f})")
        pd.DataFrame(epoch_log).to_csv(log_path, index=False)
        torch.save(model.state_dict(), ckpt_path)  # checkpoint every epoch

        # early stopping on robust_recon_loss specifically -- NOT g_loss/d_loss, which oscillate
        # around an adversarial equilibrium by construction and don't have a meaningful plateau
        if prev_loss is not None:
            delta = prev_loss - row["robust_recon_loss"]
            if first_delta is None:
                first_delta = delta
            elif first_delta > 0 and delta < args.early_stop_rel_thresh * first_delta:
                patience_ctr += 1
            else:
                patience_ctr = 0
            if epoch + 1 >= args.early_stop_min_epochs and patience_ctr >= args.early_stop_patience:
                print(f"  early stop at epoch {epoch} -- robust_recon_loss plateaued "
                      f"(this epoch's delta={delta:.4f} vs first-epoch delta={first_delta:.4f})")
                break
        prev_loss = row["robust_recon_loss"]

    assert os.path.getsize(ckpt_path) > 0, f"torch.save produced a 0-byte file at {ckpt_path} -- write failed"
    print(f"saved -> {ckpt_path}")
    print("Check the log: g_loss/d_loss should settle into a stable back-and-forth (D neither collapsing "
          "to ~0 nor blowing out) -- early stopping above is based on robust_recon_loss only, so a "
          "still-unstable GAN loss at the stopping point is worth a manual look even if reconstruction "
          "loss itself plateaued.")

    drift_x, drift_y = load_xy(report_items, cfg["img_size"], device)
    with torch.no_grad():
        auc_orig = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(orig_net, drift_x)).cpu().numpy())))
        auc_dec = roc_auc_score(drift_y, 1 / (1 + np.exp(-densenet_logit(densenet, reconstruct(model, drift_x)).cpu().numpy())))
    drop = auc_orig - auc_dec
    print(f"clean recon AUC (REPORT_ITEMS, held-out) -- original={auc_orig:.4f}  "
          f"decoder-realism-robust={auc_dec:.4f}  (drop={drop:.4f})"
          f"{'  *** LARGE DROP -- possible collapse ***' if drop > 0.03 else ''}")


if __name__ == "__main__":
    main()
