#!/usr/bin/env python3
"""Part 3: the main reported defense table for FSQ on ISIC. Seven
conditions, using the parameters selected by sweep_sqs_isic.py, evaluated
on REPORT_ITEMS across the full eps grid. Same role as
scripts/defenses/fsq_aptos/defense_eval_table_aptos.py.

Conditions:
  none              undefended baseline
  pixel_noise       original tokenizer + noise at pixel input (RQ2's named comparison point)
  best_sqs          original tokenizer + whichever latent-space SQS variant won the sweep
  advft             fine-tuned tokenizer alone (per-eps255 checkpoint)
  advft+best_sqs    fine-tuned tokenizer + best_sqs
  decoder_realism   decoder-realism-robust checkpoint alone (single checkpoint, not per-eps)
  combined          advft(worst-case eps) + decoder-realism-robust merged (single checkpoint)

Static numbers only; BPDA is what decides which of these is real.

Usage:
    python scripts/defenses/fsq_isic/defense_eval_table_isic.py \
        --data-root /path/to/isic2019 --fsq-repo third_party/FSQ-pytorch \
        --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --effnet-ckpt ckpts/classifiers/effnet_b4_isic.pt \
        --probe-ckpt ckpts/linear_probes/probe_fsq_isic.pt
"""
import argparse
import csv
import json
import os
import sys

import torch
import yaml
from tqdm.auto import tqdm

_root = os.path.abspath(os.path.dirname(__file__))
while not os.path.isfile(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from src.datasets.isic import build_splits
from src.datasets.kermany import load_xy, seed_everything, stratified_subset
from src.defenses.fsq.pixel_noise import pixel_noise_probe_feat, pixel_noise_reconstruct
from src.defenses.fsq.sqs import dispatch_probe_feat, dispatch_reconstruct
from src.eval.attack_io import load_folder
from src.eval.metrics import asr, auc
from src.eval.probes import LinearProbe
from src.eval.sel_report_split import stratified_half_split
from src.eval.sqs_sweep import batched
from src.models.efficientnet_isic_classifier import EfficientNetB4Classifier, load_efficientnet_b4
from src.tokenizers.fsq import FSQTokenizer

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]
DEFENSES = ["none", "pixel_noise", "best_sqs", "advft", "advft+best_sqs", "decoder_realism", "combined"]
FIELDS = ["defense", "finetune_eps255", "attack", "eval_eps255", "eval_eps_pixel", "n",
          "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
          "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe"]


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
    p.add_argument("--effnet-ckpt", required=True)
    p.add_argument("--probe-ckpt", required=True, help="the linear probe trained in run_fsq_isic.py")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/isic_attack.yaml")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load; must match whatever "
                         "merge_checkpoints_isic.py was run with (default: worst-case/largest in eps_grid)")
    return p.parse_args()


def load_probe(ckpt_path, dim, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    probe = LinearProbe(dim, state["mu"], state["sigma"]).to(device)
    probe.load_state_dict(state)
    probe.eval()
    return probe


def recon_fn_for(defense, tok, pixel_sigma, pixel_K, sqs_name, sqs_params):
    if defense in ("none", "advft", "decoder_realism", "combined"):
        return lambda x: tok.reconstruct(x)
    if defense == "pixel_noise":
        return lambda x: pixel_noise_reconstruct(tok, x, pixel_K, pixel_sigma)
    if defense in ("best_sqs", "advft+best_sqs"):
        return lambda x: dispatch_reconstruct(sqs_name, tok, x, sqs_params)
    raise ValueError(defense)


def probe_fn_for(defense, tok, pixel_sigma, pixel_K, sqs_name, sqs_params):
    if defense in ("none", "advft", "decoder_realism", "combined"):
        return lambda x: tok.probe_feat(x)
    if defense == "pixel_noise":
        return lambda x: pixel_noise_probe_feat(tok, x, pixel_K, pixel_sigma)
    if defense in ("best_sqs", "advft+best_sqs"):
        return lambda x: dispatch_probe_feat(sqs_name, tok, x, sqs_params)
    raise ValueError(defense)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]
    os.makedirs(args.out_dir, exist_ok=True)

    winner_path = os.path.join(args.out_dir, "fsq_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path}; run sweep_sqs_isic.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]
    pixel_sigma, pixel_K = winner["pixel_noise_sigma"], winner["pixel_noise_K"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, cfg["seed"], cfg["val_frac"], cfg["test_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_tok = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
    densenet = load_densenet(args.effnet_ckpt, device)
    probe_dim = orig_tok.probe_feat(load_xy(report_items[:1], cfg["img_size"], device)[0]).shape[1]
    probe = load_probe(args.probe_ckpt, probe_dim, device)

    Xc_report, Yc_report = load_xy(report_items, cfg["img_size"], device)

    csv_path = os.path.join(args.out_dir, "fsq_defense_table.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning the full eval -> {csv_path}")
        return

    decoder_ckpt_path = os.path.join(args.ckpt_dir, "fsq_decoder_realism_robust_isic.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"fsq_combined_eps{combined_eps}_isic.pt")

    def model_for(defense, ft_eps):
        if defense.startswith("advft"):
            m = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
            m.net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"fsq_advft_eps{ft_eps}_isic.pt"),
                                              map_location=device, weights_only=False))
            m.net.eval()
            for p in m.net.parameters():
                p.requires_grad_(False)
            return m
        if defense == "decoder_realism":
            m = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
            m.net.load_state_dict(torch.load(decoder_ckpt_path, map_location=device, weights_only=False))
            m.net.eval()
            return m
        if defense == "combined":
            m = FSQTokenizer(args.fsq_repo, args.fsq_ckpt, device)
            m.net.load_state_dict(torch.load(combined_ckpt_path, map_location=device, weights_only=False))
            m.net.eval()
            return m
        return orig_tok

    rows = []
    for defense in DEFENSES:
        ft_eps_list = cfg["eps_grid"] if defense.startswith("advft") else [None]
        for ft_eps in ft_eps_list:
            tok = model_for(defense, ft_eps)
            recon_fn = recon_fn_for(defense, tok, pixel_sigma, pixel_K, sqs_name, sqs_params)
            probe_fn = probe_fn_for(defense, tok, pixel_sigma, pixel_K, sqs_name, sqs_params)

            recon_c = batched(lambda x: densenet_logit(densenet, recon_fn(x)).cpu().numpy(), Xc_report, device, bs)
            prb_c = batched(lambda x: probe(probe_fn(x)).cpu().numpy(), Xc_report, device, bs)
            auc_recon_clean, auc_probe_clean = auc(Yc_report, recon_c), auc(Yc_report, prb_c)

            for attack in tqdm(ATTACKS_TO_DEFEND, desc=f"{defense} ft_eps{ft_eps}"):
                for eval_eps in cfg["eps_grid"]:
                    Xa, Ya = load_folder(args.adv_dir, "fsq", attack, eval_eps, idx_subset=report_idx)
                    assert (Ya == Yc_report).all()
                    recon_a = batched(lambda x: densenet_logit(densenet, recon_fn(x)).cpu().numpy(), Xa, device, bs)
                    prb_a = batched(lambda x: probe(probe_fn(x)).cpu().numpy(), Xa, device, bs)
                    rows.append(dict(
                        defense=defense, finetune_eps255=ft_eps, attack=attack,
                        eval_eps255=eval_eps, eval_eps_pixel=eval_eps / 2, n=len(Ya),
                        auc_recon_clean=round(auc_recon_clean, 4), auc_recon_adv=round(auc(Yc_report, recon_a), 4),
                        drop_recon=round(auc_recon_clean - auc(Yc_report, recon_a), 4),
                        asr_recon=round(asr(recon_c, recon_a, Yc_report), 1),
                        auc_probe_clean=round(auc_probe_clean, 4), auc_probe_adv=round(auc(Yc_report, prb_a), 4),
                        drop_probe=round(auc_probe_clean - auc(Yc_report, prb_a), 4),
                        asr_probe=round(asr(prb_c, prb_a, Yc_report), 1),
                    ))
            if defense.startswith("advft"):
                del tok
                torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
