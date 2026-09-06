#!/usr/bin/env python3
"""Part 3: the main reported defense table for MediTok and UniTok. Same
seven conditions, same REPORT_ITEMS x full-eps-grid discipline, as
scripts/defenses/vq16_kermany/defense_eval_table_kermany.py. Runs both
tokenizers in one pass, one CSV per tokenizer.

Conditions:
  none              undefended baseline
  pixel_noise       original tokenizer + noise at pixel input (RQ2's named comparison point)
  best_sqs          original tokenizer + whichever latent-space SQS variant won the sweep
  advft             fine-tuned tokenizer alone (per-eps255 checkpoint)
  advft+best_sqs    fine-tuned tokenizer + best_sqs
  decoder_realism   decoder-realism-robust checkpoint alone (single checkpoint, not per-eps)
  combined          advft(worst-case eps) + decoder-realism-robust merged (single checkpoint)

Static numbers only; BPDA is what decides which of these is real. Cross-
reference bpda_alone_kermany.py / bpda_sqs_kermany.py before trusting any row
here as a genuine robustness claim.

Usage:
    python scripts/defenses/unitok_kermany/defense_eval_table_kermany.py \
        --data-root /path/to/chest_xray --unitok-repo third_party/unitok \
        --tokenizer-ckpt-dir third_party/ckpt --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt
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

from src.datasets.kermany import build_splits, load_xy, seed_everything, stratified_subset
from src.defenses.multicodebook.pixel_noise import pixel_noise_probe_feat, pixel_noise_reconstruct
from src.defenses.multicodebook.sqs import dispatch_probe_feat, dispatch_reconstruct
from src.eval.attack_io import load_folder
from src.eval.metrics import asr, auc
from src.eval.probes import LinearProbe
from src.eval.sel_report_split import stratified_half_split
from src.eval.sqs_sweep import batched
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.meditok_unitok import load_unitok

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]
DEFENSES = ["none", "pixel_noise", "best_sqs", "advft", "advft+best_sqs", "decoder_realism", "combined"]
FIELDS = ["defense", "finetune_eps255", "attack", "eval_eps255", "eval_eps_pixel", "n",
          "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
          "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--unitok-repo", required=True)
    p.add_argument("--tokenizer-ckpt-dir", required=True)
    p.add_argument("--densenet-ckpt", required=True)
    p.add_argument("--probe-ckpt-dir", default="ckpts/linear_probes",
                   help="dir holding probe_meditok_kermany.pt / probe_unitok_kermany.pt from run_meditok_unitok_kermany.py")
    p.add_argument("--out-dir", default="results/outputs")
    p.add_argument("--adv-dir", default="results/adv_images")
    p.add_argument("--ckpt-dir", default="ckpts/defenses")
    p.add_argument("--config", default="configs/kermany_attack.yaml")
    p.add_argument("--combined-eps255", type=int, default=None,
                    help="which combined checkpoint to load; must match whatever "
                         "merge_checkpoints_kermany.py was run with (default: worst-case/largest in eps_grid)")
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


def run_one(tok_name, load_fresh, args, cfg, densenet, report_items, report_idx, bs, device):
    winner_path = os.path.join(args.out_dir, f"{tok_name}_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path}; run sweep_sqs_kermany.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]
    pixel_sigma, pixel_K = winner["pixel_noise_sigma"], winner["pixel_noise_K"]

    csv_path = os.path.join(args.out_dir, f"{tok_name}_defense_table.csv")
    if os.path.isfile(csv_path):
        print(f"{tok_name}: already computed, loading instead of rerunning the full eval -> {csv_path}")
        return

    orig_tok = load_fresh()
    probe_ckpt = os.path.join(args.probe_ckpt_dir, f"probe_{tok_name}_kermany.pt")
    assert os.path.isfile(probe_ckpt), f"missing {probe_ckpt}; run run_meditok_unitok_kermany.py first"
    probe_dim = orig_tok.probe_feat(load_xy(report_items[:1], cfg["img_size"], device)[0]).shape[1]
    probe = load_probe(probe_ckpt, probe_dim, device)

    Xc_report, Yc_report = load_xy(report_items, cfg["img_size"], device)

    decoder_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_decoder_realism_robust.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"{tok_name}_combined_eps{combined_eps}.pt")

    def model_for(defense, ft_eps):
        if defense.startswith("advft"):
            m = load_fresh()
            m.net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{tok_name}_advft_eps{ft_eps}.pt"),
                                              map_location=device, weights_only=False))
            m.net.eval()
            for p in m.net.parameters():
                p.requires_grad_(False)
            return m
        if defense == "decoder_realism":
            m = load_fresh()
            m.net.load_state_dict(torch.load(decoder_ckpt_path, map_location=device, weights_only=False))
            m.net.eval()
            return m
        if defense == "combined":
            m = load_fresh()
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

            for attack in tqdm(ATTACKS_TO_DEFEND, desc=f"{tok_name} {defense} ft_eps{ft_eps}"):
                for eval_eps in cfg["eps_grid"]:
                    Xa, Ya = load_folder(args.adv_dir, tok_name, attack, eval_eps, idx_subset=report_idx)
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
    print(f"{tok_name}: wrote {csv_path}")
    del orig_tok
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    densenet = load_densenet(args.densenet_ckpt, device)

    tok_name = "unitok"
    load_fresh = lambda: load_unitok(args.unitok_repo, args.tokenizer_ckpt_dir, device)
    print(f"=== {tok_name} ===")
    run_one(tok_name, load_fresh, args, cfg, densenet, report_items, report_idx, bs, device)


if __name__ == "__main__":
    main()
