#!/usr/bin/env python3
"""LLaVA-Med (MedITok) VQA red-team: the label-free M2-sat / M2-unb attack on the
frozen VLM's vision tower, plus a matched-L_inf random control, recorded together
with the model's free-text answers.

M2-sat, M2-unb and the random control run in one pass so they share the same
deterministic per-image seed. One --dataset per invocation; rows are appended to
the shared CSVs (resumable). CUDA and fp16 only, ~15 GB of weights. Run
setup_third_party.sh with MEDITOK_VLM=1 first and install the [vqa] dependency
extra in its own env (see docs/checkpoints.md).

Outputs under --out-dir (default results/vqa/):
    vqa_sweep_seeded_m2sat.csv   raw answers; schema matches results/paper_inputs/vqa/
    vqa_sweep_seeded_m2unb.csv
    vqa_clinical_summary.csv     diagnosis-flip summary (score_clinical_csv)

Usage:
    python scripts/generative/run_vqa_redteam.py --dataset kermany \
        --data-root /path/to/chest_xray --meditok-repo third_party/meditok
"""
import argparse
import csv
import hashlib
import os
import sys

import pandas as pd
import torch
import yaml

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.kermany import seed_everything, stratified_subset
from src.generative.llavamed import QUESTIONS, load_llavamed, swap_vision_tower
from src.generative.vqa_attack import build_vqa_attacks
from src.generative.vqa_scoring import score_clinical_csv

# the true_label string written to the CSV for each dataset's integer label
TRUE_LABEL = {
    "kermany": {0: "normal", 1: "pneumonia"},
    "aptos": {0: "non-referable DR", 1: "referable DR"},
    "isic": {0: "benign", 1: "malignant"},
}
CSV_FIELDS = ["dataset", "image", "true_label", "question", "eps",
              "adv_flip_pct", "adv_disp", "rnd_flip_pct", "rnd_disp", "clean", "adv", "random"]


def det_seed(dataset, image, eps):
    return int(hashlib.md5(f"{dataset}|{image}|{eps}".encode()).hexdigest(), 16) % (2 ** 31)


def test_split(dataset, data_root, out_dir, cfg):
    """The three dataset modules do not share a build_splits signature."""
    if dataset == "kermany":
        from src.datasets.kermany import build_splits

        return build_splits(data_root, out_dir, cfg["seed"], cfg["val_frac"])[2]
    if dataset == "aptos":
        from src.datasets.aptos import build_splits

        return build_splits(data_root, out_dir)[2]
    from src.datasets.isic import build_splits

    return build_splits(data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["kermany", "aptos", "isic"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--meditok-repo", required=True)
    p.add_argument("--vlm-weights-dir", default=None, help="default: <meditok-repo>/weights")
    p.add_argument("--defended-ckpt", default=None, help="beyond-paper: swap in a defended vision tower")
    p.add_argument("--out-dir", default="results/vqa")
    p.add_argument("--config", default="configs/vqa.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "the LLaVA-Med stack needs CUDA (fp16 7B model)"
    os.makedirs(args.out_dir, exist_ok=True)
    ds = args.dataset

    seed_everything(cfg["seed"])
    test_items = test_split(ds, args.data_root, args.out_dir, cfg)
    sample = stratified_subset(test_items, cfg["n_per_dataset"], cfg["seed"])

    vlm = load_llavamed(args.meditok_repo, args.vlm_weights_dir, device)
    if args.defended_ckpt:
        swap_vision_tower(vlm, args.defended_ckpt)
    attacks, tok_stats = build_vqa_attacks(vlm.vision_tower_core, device, cfg["attack_steps"], cfg["attack_lr"])

    paths = {"M2-sat": os.path.join(args.out_dir, "vqa_sweep_seeded_m2sat.csv"),
             "M2-unb": os.path.join(args.out_dir, "vqa_sweep_seeded_m2unb.csv")}
    done, writers, handles = {}, {}, {}
    for a, path in paths.items():
        done[a] = set()
        if os.path.isfile(path):
            with open(path) as f:
                done[a] = {(r["dataset"], r["image"], r["eps"]) for r in csv.DictReader(f)}
        handles[a] = open(path, "a", newline="")
        writers[a] = csv.DictWriter(handles[a], fieldnames=CSV_FIELDS)
        if os.path.getsize(path) == 0:
            writers[a].writeheader()

    questions = QUESTIONS[ds]
    for img_path, label in sample:
        image = os.path.basename(img_path)
        if all((ds, image, str(e)) in done[a] for a in paths for e in cfg["eps_grid"]):
            continue
        clean = vlm.load_image(img_path)
        clean_ans = {q: vlm.ask(clean, q) for q in questions}
        true_label = TRUE_LABEL[ds][int(label)]

        for eps in cfg["eps_grid"]:
            torch.manual_seed(det_seed(ds, image, eps))
            rnd = attacks["random"](clean, eps / 255).squeeze(0).cpu().to(clean.dtype)
            rnd_ans = {q: vlm.ask(rnd, q) for q in questions}
            rf, rd = tok_stats(clean, rnd)

            for a in paths:
                if (ds, image, str(eps)) in done[a]:
                    continue
                torch.manual_seed(det_seed(ds, image, eps))
                adv = attacks[a](clean, eps / 255).squeeze(0).cpu().to(clean.dtype)
                af, ad = tok_stats(clean, adv)
                adv_ans = {q: vlm.ask(adv, q) for q in questions}
                for q in questions:
                    writers[a].writerow({
                        "dataset": ds, "image": image, "true_label": true_label, "question": q, "eps": eps,
                        "adv_flip_pct": round(af, 1), "adv_disp": round(ad, 2),
                        "rnd_flip_pct": round(rf, 1), "rnd_disp": round(rd, 2),
                        "clean": clean_ans[q], "adv": adv_ans[q], "random": rnd_ans[q],
                    })
                handles[a].flush()
        print(f"  {ds}/{image}: done")

    for f in handles.values():
        f.close()

    summary = []
    for a, path in paths.items():
        summary += score_clinical_csv(pd.read_csv(path), a)
    if summary:
        out = os.path.join(args.out_dir, "vqa_clinical_summary.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        print("wrote", out)


if __name__ == "__main__":
    main()
