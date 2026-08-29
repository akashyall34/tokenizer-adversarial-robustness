#!/usr/bin/env python3
"""MediTok + UniTok ISIC pipeline. Same attack/probe logic as
run_meditok_unitok_kermany.py, against ISIC 2019 + EfficientNet-B4 (timm).

Two things differ from the Kermany/APTOS versions:
  - attack_batch defaults to 4 here (not the config's 8, which is for the
    VQ-16 run) -- the decoder plus the B4 classifier need the extra memory
    headroom.
  - The per-tokenizer eval table does NOT include M0 rows. M0 is
    tokenizer-free and already covered by vq16_attack_table.csv (see
    run_vq16_isic.py, which saves it under <adv-dir>/m0/).

Usage:
    python scripts/attacks/run_meditok_unitok_isic.py --data-root /path/to/isic2019 \
        --meditok-repo third_party/meditok --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.attacks.meditok_unitok_attacks import make_attacks
from src.datasets.isic import build_splits
from src.datasets.kermany import make_loader, seed_everything, stratified_subset
from src.eval.attack_io import already_done, save_batch, save_dir, load_folder, write_meta
from src.eval.metrics import asr, auc
from src.eval.probes import train_linear_probe
from src.models.efficientnet_isic_classifier import load_efficientnet_b4, make_binary_logit_fn
from src.tokenizers.meditok_unitok import load_meditok, load_unitok

FIELDS = ["tokenizer", "attack", "eps255", "eps_pixel", "n", "steps",
          "auc_raw_clean", "auc_raw_adv", "drop_raw", "asr_raw",
          "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
          "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe",
          "token_flip_rate_pct", "emb_disp_pure_mean"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--meditok-repo", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--probe-out-dir", default="ckpts/linear_probes")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--attack-batch", type=int, default=4,
                    help="overrides the config's attack_batch (8, sized for VQ-16) -- "
                         "MediTok/UniTok need a smaller batch here (decoder + B4 classifier)")
    return p.parse_args()


def run_tokenizer(tok, logit_fn, cfg, args, train_items, val_items, test_items, attack_items, loader, device):
    attacks = make_attacks(tok, logit_fn, device, cfg["steps"], cfg["attack_lr"])
    jobs = [(a, e) for a in ["M1", "M3_alpha03", "M2-sat", "M2-unb"] for e in cfg["eps_grid"]]

    seed_everything(cfg["seed"])
    for attack, eps in jobs:
        if already_done(args.adv_dir, tok.name, attack, eps, len(attack_items)):
            print(f"  {tok.name} {attack} eps{eps}: skip")
            continue
        d = save_dir(args.adv_dir, tok.name, attack, eps)
        gidx = 0
        for xb, yb in tqdm(loader(attack_items, args.attack_batch), desc=f"{tok.name} {attack} eps{eps}", leave=False):
            xadv = attacks[attack](xb.to(device), yb, eps / 255.0).detach().cpu()
            gidx += save_batch(d, xadv, attack_items, gidx)
        write_meta(d, attack=attack, eps255=eps, eps_pixel=eps / 2, tokenizer=tok.name, n=gidx,
                   seed=cfg["seed"], steps=cfg["steps"], eps_scale="internal[-1,1]", img_size=cfg["img_size"])
        print(f"  {tok.name} {attack} eps{eps}: saved {gidx}")
    torch.cuda.empty_cache()

    probe_out = f"{args.probe_out_dir}/probe_{tok.name}_isic.pt"
    probe, val_auc, test_auc = train_linear_probe(
        tok.probe_feat, loader(train_items, cfg["batch"]), loader(val_items, cfg["batch"]),
        loader(test_items, cfg["batch"]), device, probe_out,
    )
    print(f"{tok.name} probe: val AUC={val_auc:.4f}  clean test AUC={test_auc:.4f}")

    def clean_refs():
        xs, ys = [], []
        for p, l in attack_items:
            im = Image.open(p).convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.LANCZOS)
            xs.append(torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0 * 2 - 1)
            ys.append(l)
        X = torch.stack(xs)
        Y = np.array(ys)

        def batched(fn, bs=args.attack_batch):
            return np.concatenate([fn(X[i:i + bs].to(device)) for i in range(0, X.shape[0], bs)])

        raw = batched(lambda x: logit_fn(x).cpu().numpy())
        recon = batched(lambda x: logit_fn(tok.reconstruct(x)).cpu().numpy())
        prb = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy())
        z0 = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy())
        idx0 = batched(lambda x: tok.assign_indices(tok.encode_prequant(x)).reshape(x.shape[0], -1).cpu().numpy().astype(np.int64))
        return Y, raw, recon, prb, z0, idx0

    print(f"=== {tok.name}: clean refs ===")
    Yc, raw_c, recon_c, prb_c, z0, idx0 = clean_refs()
    auc_raw_c, auc_recon_c, auc_prb_c = auc(Yc, raw_c), auc(Yc, recon_c), auc(Yc, prb_c)
    print(f"  clean: raw={auc_raw_c:.4f} recon={auc_recon_c:.4f} probe={auc_prb_c:.4f}")

    # no M0 here -- it's tokenizer-free and already fully covered by vq16_attack_table.csv
    rows = []
    for attack, eps in tqdm(jobs, desc=f"{tok.name} eval"):
        Xa, Ya = load_folder(args.adv_dir, tok.name, attack, eps)
        assert (Ya == Yc).all(), f"{attack} eps{eps}: order mismatch"
        row = dict(tokenizer=tok.name, attack=attack, eps255=eps, eps_pixel=eps / 2, n=len(Ya), steps=cfg["steps"],
                   auc_raw_clean=round(auc_raw_c, 4), auc_recon_clean=round(auc_recon_c, 4),
                   auc_probe_clean=round(auc_prb_c, 4))

        def batched(fn, bs=args.attack_batch):
            return np.concatenate([fn(Xa[i:i + bs].to(device)) for i in range(0, Xa.shape[0], bs)])

        raw_a = batched(lambda x: logit_fn(x).cpu().numpy())
        rec_a = batched(lambda x: logit_fn(tok.reconstruct(x)).cpu().numpy())
        prb_a = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy())
        za = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy())
        idxa = batched(lambda x: tok.assign_indices(tok.encode_prequant(x)).reshape(x.shape[0], -1).cpu().numpy().astype(np.int64))
        row.update(auc_raw_adv=round(auc(Yc, raw_a), 4), drop_raw=round(auc_raw_c - auc(Yc, raw_a), 4),
                   asr_raw=round(asr(raw_c, raw_a, Yc), 1),
                   auc_recon_adv=round(auc(Yc, rec_a), 4), drop_recon=round(auc_recon_c - auc(Yc, rec_a), 4),
                   asr_recon=round(asr(recon_c, rec_a, Yc), 1),
                   auc_probe_adv=round(auc(Yc, prb_a), 4), drop_probe=round(auc_prb_c - auc(Yc, prb_a), 4),
                   asr_probe=round(asr(prb_c, prb_a, Yc), 1),
                   token_flip_rate_pct=round(float((idx0 != idxa).mean() * 100), 2),
                   emb_disp_pure_mean=round(float(np.linalg.norm(za - z0, axis=-1).mean()), 4))
        rows.append(row)

    csv_path = f"{args.out_dir}/{tok.name}_attack_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("  wrote", csv_path)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.adv_dir, exist_ok=True)
    os.makedirs(args.probe_out_dir, exist_ok=True)

    seed_everything(cfg["seed"])
    train_items, val_items, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])

    def loader(items, bs, shuffle=False):
        return make_loader(items, cfg["img_size"], bs, cfg["num_workers"], device, shuffle)

    effnet = load_efficientnet_b4(args.effnet_ckpt, device)
    logit_fn = make_binary_logit_fn(effnet, device)

    print("loading meditok...")
    meditok = load_meditok(args.meditok_repo, args.tokenizer_ckpt_dir, cfg["img_size"], device)
    print("loading unitok...")
    unitok = load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)

    for tok in (meditok, unitok):
        print(f"=== {tok.name} ===")
        run_tokenizer(tok, logit_fn, cfg, args, train_items, val_items, test_items, attack_items, loader, device)


if __name__ == "__main__":
    main()
