#!/usr/bin/env python3
"""VQ-16 Kermany pipeline: build splits, run M0/M1/M3 attacks and save adversarial
images, train a linear probe, and evaluate raw/reconstruction/probe paths into
outputs/vq16_attack_table.csv.

Usage:
    python scripts/attacks/run_vq16_kermany.py --data-root /path/to/chest_xray \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
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

from src.attacks.vq16_attacks import make_attacks
from src.datasets.kermany import build_splits, make_loader, seed_everything, stratified_subset
from src.eval.attack_io import already_done, save_batch, save_dir, load_folder, write_meta
from src.eval.metrics import asr, auc
from src.eval.probes import train_linear_probe
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.vq16 import VQ16Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Kermany chest_xray/ directory (train/, test/)")
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--densenet-ckpt", required=True)
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--probe-out", default="ckpts/linear_probes/probe_vq16_kermany.pt")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.adv_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.probe_out), exist_ok=True)

    seed_everything(cfg["seed"])
    train_items, val_items, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    print(f"attack subset: {len(attack_items)}")

    def loader(items, bs, shuffle=False):
        return make_loader(items, cfg["img_size"], bs, cfg["num_workers"], device, shuffle)

    densenet = load_densenet(args.densenet_ckpt, device)
    tok = VQ16Tokenizer(args.llamagen_repo, args.vq16_ckpt, device)

    logit_fn = lambda x: densenet_logit(densenet, x)
    attacks = make_attacks(tok, logit_fn, device, cfg["steps"], cfg["attack_lr"])
    jobs = [("M0", e) for e in cfg["eps_grid"]] + \
           [(a, e) for a in ["M1", "M3_alpha03", "M3_geo_sat", "M3_geo_unb"] for e in cfg["eps_grid"]]

    seed_everything(cfg["seed"])
    for attack, eps in jobs:
        # M0 is tokenizer-free, so it's saved under a shared "m0" folder rather
        # than "vq16" -- MediTok/UniTok/FSQ runs reuse it instead of regenerating it.
        save_name = "m0" if attack == "M0" else tok.name
        if already_done(args.adv_dir, save_name, attack, eps, len(attack_items)):
            print(f"  {attack} eps{eps}: already saved, skip")
            continue
        d = save_dir(args.adv_dir, save_name, attack, eps)
        fn = attacks[attack]
        gidx = 0
        for xb, yb in tqdm(loader(attack_items, cfg["attack_batch"]), desc=f"{attack} eps{eps}", leave=False):
            xadv = fn(xb, yb, eps / 255.0).detach().cpu()
            gidx += save_batch(d, xadv, attack_items, gidx)
        write_meta(d, attack=attack, eps255=eps, eps_pixel=eps / 2, tokenizer=("m0" if attack == "M0" else tok.name),
                   n=gidx, seed=cfg["seed"], steps=cfg["steps"], attack_lr=cfg["attack_lr"],
                   eps_scale="internal[-1,1]", img_range="[-1,1]", img_size=cfg["img_size"])
        print(f"  {attack} eps{eps}: saved {gidx} imgs -> {d}")
    torch.cuda.empty_cache()

    def probe_feat(x):
        return tok.probe_feat(x)

    probe, val_auc, test_auc = train_linear_probe(
        probe_feat, loader(train_items, cfg["batch"]), loader(val_items, cfg["batch"]),
        loader(test_items, cfg["batch"]), device, args.probe_out,
    )
    print(f"probe: val AUC={val_auc:.4f}  clean test AUC={test_auc:.4f}")

    def clean_refs():
        xs, ys = [], []
        for p, l in attack_items:
            im = Image.open(p).convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.LANCZOS)
            xs.append(torch.from_numpy(np.array(im)).float().permute(2, 0, 1) / 255.0 * 2 - 1)
            ys.append(l)
        X = torch.stack(xs)
        Y = np.array(ys)

        def batched(fn, bs=cfg["attack_batch"]):
            return np.concatenate([fn(X[i:i + bs].to(device)) for i in range(0, X.shape[0], bs)])

        raw = batched(lambda x: densenet_logit(densenet, x).cpu().numpy())
        recon = batched(lambda x: densenet_logit(densenet, tok.reconstruct(x)).cpu().numpy())
        prb = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy())
        z0 = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy())
        k0 = batched(lambda x: tok.clean_top1(tok.encode_prequant(x)).cpu().numpy())
        return Y, raw, recon, prb, z0, k0

    print("computing clean references...")
    Yc, raw_c, recon_c, prb_c, z0, k0 = clean_refs()
    auc_raw_c, auc_recon_c, auc_prb_c = auc(Yc, raw_c), auc(Yc, recon_c), auc(Yc, prb_c)
    print(f"clean: raw={auc_raw_c:.4f} recon={auc_recon_c:.4f} probe={auc_prb_c:.4f}")

    fields = ["tokenizer", "attack", "eps255", "eps_pixel", "n", "steps",
              "auc_raw_clean", "auc_raw_adv", "drop_raw", "asr_raw",
              "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
              "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe",
              "token_flip_rate_pct", "emb_disp_pure_mean"]
    rows = []
    for attack, eps in tqdm(jobs, desc="eval"):
        Xa, Ya = load_folder(args.adv_dir, "m0" if attack == "M0" else tok.name, attack, eps)
        assert (Ya == Yc).all(), f"{attack} eps{eps}: order mismatch vs clean refs"
        row = dict(tokenizer=tok.name, attack=attack, eps255=eps, eps_pixel=eps / 2, n=len(Ya), steps=cfg["steps"],
                   auc_raw_clean=round(auc_raw_c, 4), auc_recon_clean=round(auc_recon_c, 4),
                   auc_probe_clean=round(auc_prb_c, 4))

        def batched(fn, bs=cfg["attack_batch"]):
            return np.concatenate([fn(Xa[i:i + bs].to(device)) for i in range(0, Xa.shape[0], bs)])

        raw_a = batched(lambda x: densenet_logit(densenet, x).cpu().numpy())
        row.update(auc_raw_adv=round(auc(Yc, raw_a), 4), drop_raw=round(auc_raw_c - auc(Yc, raw_a), 4),
                   asr_raw=round(asr(raw_c, raw_a, Yc), 1))
        if attack != "M0":
            recon_a = batched(lambda x: densenet_logit(densenet, tok.reconstruct(x)).cpu().numpy())
            prb_a = batched(lambda x: probe(tok.probe_feat(x)).cpu().numpy())
            row.update(auc_recon_adv=round(auc(Yc, recon_a), 4), drop_recon=round(auc_recon_c - auc(Yc, recon_a), 4),
                       asr_recon=round(asr(recon_c, recon_a, Yc), 1),
                       auc_probe_adv=round(auc(Yc, prb_a), 4), drop_probe=round(auc_prb_c - auc(Yc, prb_a), 4),
                       asr_probe=round(asr(prb_c, prb_a, Yc), 1))
            za = batched(lambda x: tok.encode_prequant(x).detach().cpu().numpy())
            ka = batched(lambda x: tok.clean_top1(tok.encode_prequant(x)).cpu().numpy())
            row.update(token_flip_rate_pct=round(float((ka != k0).mean() * 100), 2),
                       emb_disp_pure_mean=round(float(np.linalg.norm(za - z0, axis=-1).mean()), 4))
        rows.append(row)

    csv_path = f"{args.out_dir}/vq16_attack_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
