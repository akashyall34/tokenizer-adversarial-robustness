#!/usr/bin/env python3
"""FSQ APTOS pipeline. Uses the anchored margin loss and the full M0/M1/M3_alpha03/
M2-sat/M2-unb attack set (see src/attacks/fsq_attacks.py's module docstring
for why this differs from run_fsq_kermany.py's FSQ attacks). M0 is reused from the
VQ-16 APTOS run (see run_vq16_aptos.py, which saves it under <adv-dir>/m0/).

Usage:
    python scripts/attacks/run_fsq_aptos.py --data-root /path/to/aptos2019 \
        --fsq-repo third_party/FSQ-pytorch --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b5_aptos.pt
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

from src.attacks.fsq_attacks import make_attacks_anchored
from src.datasets.aptos import build_splits
from src.datasets.kermany import make_loader, seed_everything, stratified_subset
from src.eval.attack_io import already_done, save_batch, save_dir, load_folder, write_meta
from src.eval.metrics import asr, auc
from src.eval.probes import train_linear_probe
from src.models.efficientnet_classifier import load_efficientnet, make_binary_logit_fn
from src.tokenizers.fsq import FSQTokenizer

FIELDS = ["tokenizer", "attack", "eps255", "eps_pixel", "n", "steps",
          "auc_raw_clean", "auc_raw_adv", "drop_raw", "asr_raw",
          "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
          "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe",
          "token_flip_rate_pct", "emb_disp_pure_mean"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--fsq-repo", required=True)
    p.add_argument("--fsq-ckpt", required=True)
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--probe-out", default="ckpts/linear_probes/probe_fsq_aptos.pt")
    p.add_argument("--config", default="configs/aptos_attack.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.adv_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.probe_out), exist_ok=True)

    seed_everything(cfg["seed"])
    train_items, val_items, test_items = build_splits(args.data_root, args.out_dir)
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])

    def loader(items, bs, shuffle=False):
        return make_loader(items, cfg["img_size"], bs, cfg["num_workers"], device, shuffle)

    effnet = load_efficientnet(args.effnet_ckpt, device)
    logit_fn = make_binary_logit_fn(effnet, device)
    tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)

    attacks = make_attacks_anchored(tok, logit_fn, device, cfg["steps"], cfg["attack_lr"])
    jobs = [(a, e) for a in ["M1", "M3_alpha03", "M2-sat", "M2-unb"] for e in cfg["eps_grid"]]

    seed_everything(cfg["seed"])
    for attack, eps in jobs:
        if already_done(args.adv_dir, tok.name, attack, eps, len(attack_items)):
            print(f"  fsq {attack} eps{eps}: skip")
            continue
        d = save_dir(args.adv_dir, tok.name, attack, eps)
        gidx = 0
        for xb, yb in tqdm(loader(attack_items, cfg["attack_batch"]), desc=f"fsq {attack} eps{eps}", leave=False):
            xadv = attacks[attack](xb.to(device), yb, eps / 255.0).detach().cpu()
            gidx += save_batch(d, xadv, attack_items, gidx)
        write_meta(d, attack=attack, eps255=eps, eps_pixel=eps / 2, tokenizer="fsq", n=gidx,
                   seed=cfg["seed"], steps=cfg["steps"], margin="anchored_signed", eps_scale="internal[-1,1]")
        print(f"  fsq {attack} eps{eps}: saved {gidx}")
    torch.cuda.empty_cache()

    probe, val_auc, test_auc = train_linear_probe(
        tok.probe_feat, loader(train_items, cfg["batch"]), loader(val_items, cfg["batch"]),
        loader(test_items, cfg["batch"]), device, args.probe_out,
    )
    print(f"fsq probe: val AUC={val_auc:.4f}  clean test AUC={test_auc:.4f}")

    xs, ys = [], []
    for p, l in attack_items:
        im = Image.open(p).convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.LANCZOS)
        xs.append(torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0 * 2 - 1)
        ys.append(l)
    Xc = torch.stack(xs)
    Yc = np.array(ys)

    def batched(fn, X, bs=cfg["attack_batch"]):
        return np.concatenate([fn(X[i:i + bs].to(device)) for i in range(0, X.shape[0], bs)])

    raw_c = batched(lambda x: logit_fn(x).cpu().numpy(), Xc)
    recon_c = batched(lambda x: logit_fn(tok.reconstruct(x)).cpu().numpy(), Xc)
    prb_c = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy(), Xc)
    z0 = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy(), Xc)
    idx0 = batched(lambda x: tok.assign_indices(x).cpu().numpy().astype(np.int64), Xc)
    auc_raw_c, auc_recon_c, auc_prb_c = auc(Yc, raw_c), auc(Yc, recon_c), auc(Yc, prb_c)
    print(f"clean: raw={auc_raw_c:.4f} recon={auc_recon_c:.4f} probe={auc_prb_c:.4f}")

    m0_jobs = [("M0", e) for e in cfg["eps_grid"]]
    rows = []
    for attack, eps in tqdm(m0_jobs + jobs, desc="fsq eval"):
        Xa, Ya = load_folder(args.adv_dir, "m0" if attack == "M0" else "fsq", attack, eps)
        row = dict(tokenizer="fsq", attack=attack, eps255=eps, eps_pixel=eps / 2, n=len(Ya), steps=cfg["steps"],
                   auc_raw_clean=round(auc_raw_c, 4), auc_recon_clean=round(auc_recon_c, 4),
                   auc_probe_clean=round(auc_prb_c, 4))
        raw_a = batched(lambda x: logit_fn(x).cpu().numpy(), Xa)
        row.update(auc_raw_adv=round(auc(Yc, raw_a), 4), drop_raw=round(auc_raw_c - auc(Yc, raw_a), 4),
                   asr_raw=round(asr(raw_c, raw_a, Yc), 1))
        if attack != "M0":
            rec_a = batched(lambda x: logit_fn(tok.reconstruct(x)).cpu().numpy(), Xa)
            prb_a = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy(), Xa)
            za = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy(), Xa)
            idxa = batched(lambda x: tok.assign_indices(x).cpu().numpy().astype(np.int64), Xa)
            row.update(auc_recon_adv=round(auc(Yc, rec_a), 4), drop_recon=round(auc_recon_c - auc(Yc, rec_a), 4),
                       asr_recon=round(asr(recon_c, rec_a, Yc), 1),
                       auc_probe_adv=round(auc(Yc, prb_a), 4), drop_probe=round(auc_prb_c - auc(Yc, prb_a), 4),
                       asr_probe=round(asr(prb_c, prb_a, Yc), 1),
                       token_flip_rate_pct=round(float((idx0 != idxa).mean() * 100), 2),
                       emb_disp_pure_mean=round(float(np.linalg.norm(za - z0, axis=-1).mean()), 4))
        rows.append(row)

    csv_path = f"{args.out_dir}/fsq_attack_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
