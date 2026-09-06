#!/usr/bin/env python3
"""Part 3: the main reported defense table. Seven conditions, using the
parameters selected by sweep_sqs_kermany.py, evaluated on REPORT_ITEMS (the
half of the fixed attack subset none of the sweeps/hyperparameter selection
ever saw) across the full eps grid; not just the worst-case eps used for
selection, so the eps=8 row is a genuinely independent number, not a
restatement of the selection criterion.

Conditions:
  none              undefended baseline (should reproduce vq16_attack_table.csv's recon/probe columns)
  pixel_noise       original tokenizer + noise at pixel input (RQ2's named comparison point)
  best_sqs          original tokenizer + whichever latent-space SQS variant won the sweep
  advft             fine-tuned tokenizer alone (per-eps255 checkpoint)
  advft+best_sqs    fine-tuned tokenizer + best_sqs
  decoder_realism   decoder-realism-robust checkpoint alone (single checkpoint, not per-eps)
  combined          advft(worst-case eps) + decoder-realism-robust merged (single checkpoint)

These are static numbers only; BPDA decides which of them is real. SQS's
static recovery here collapses to ~0.0 AUC under BPDA (bpda_sqs_kermany.py), and a
high static number for decoder_realism or combined is not sufficient either, so
cross-reference bpda_alone_kermany.py before trusting any row here as a robustness
claim.

M0 is excluded; it never routes through the tokenizer, so no tokenizer-side
defense can affect it (see vq16_attack_table.csv for M0's own numbers).

Usage:
    python scripts/defenses/vq16_kermany/defense_eval_table_kermany.py --data-root /path/to/chest_xray \
        --llamagen-repo third_party/LlamaGen --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt \
        --probe-ckpt ckpts/linear_probes/probe_vq16_kermany.pt
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
from src.defenses.vq16.margin import probe_feat as plain_probe_feat, reconstruct as plain_reconstruct
from src.defenses.vq16.pixel_noise import pixel_noise_probe_feat, pixel_noise_reconstruct
from src.defenses.vq16.sqs import dispatch_probe_feat, dispatch_reconstruct
from src.eval.attack_io import load_folder
from src.eval.metrics import asr, auc
from src.eval.probes import LinearProbe
from src.eval.sel_report_split import stratified_half_split
from src.eval.sqs_sweep import batched
from src.models.densenet_classifier import load_densenet, logit as densenet_logit
from src.tokenizers.vq16 import load_vq16_net

ATTACKS_TO_DEFEND = ["M1", "M3_alpha03", "M2-sat", "M2-unb"]
DEFENSES = ["none", "pixel_noise", "best_sqs", "advft", "advft+best_sqs", "decoder_realism", "combined"]
FIELDS = ["defense", "finetune_eps255", "attack", "eval_eps255", "eval_eps_pixel", "n",
          "auc_recon_clean", "auc_recon_adv", "drop_recon", "asr_recon",
          "auc_probe_clean", "auc_probe_adv", "drop_probe", "asr_probe"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--llamagen-repo", required=True)
    p.add_argument("--vq16-ckpt", required=True)
    p.add_argument("--densenet-ckpt", required=True)
    p.add_argument("--probe-ckpt", required=True, help="the linear probe trained in run_vq16_kermany.py")
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


def recon_fn_for(defense, model, pixel_sigma, pixel_K, sqs_name, sqs_params):
    if defense in ("none", "advft", "decoder_realism", "combined"):
        return lambda x: plain_reconstruct(model, x)
    if defense == "pixel_noise":
        return lambda x: pixel_noise_reconstruct(model, x, pixel_K, pixel_sigma)
    if defense in ("best_sqs", "advft+best_sqs"):
        return lambda x: dispatch_reconstruct(sqs_name, model, x, sqs_params)
    raise ValueError(defense)


def probe_fn_for(defense, model, pixel_sigma, pixel_K, sqs_name, sqs_params):
    if defense in ("none", "advft", "decoder_realism", "combined"):
        return lambda x: plain_probe_feat(model, x)
    if defense == "pixel_noise":
        return lambda x: pixel_noise_probe_feat(model, x, pixel_K, pixel_sigma)
    if defense in ("best_sqs", "advft+best_sqs"):
        return lambda x: dispatch_probe_feat(sqs_name, model, x, sqs_params)
    raise ValueError(defense)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = cfg["attack_batch"]
    os.makedirs(args.out_dir, exist_ok=True)

    winner_path = os.path.join(args.out_dir, "vq16_sqs_winner.json")
    assert os.path.isfile(winner_path), f"missing {winner_path}; run sweep_sqs_kermany.py first"
    winner = json.load(open(winner_path))
    sqs_name, sqs_params = winner["name"], winner["params"]
    pixel_sigma, pixel_K = winner["pixel_noise_sigma"], winner["pixel_noise_K"]

    seed_everything(cfg["seed"])
    _, _, test_items = build_splits(args.data_root, args.out_dir, cfg["seed"], cfg["val_frac"])
    attack_items = stratified_subset(test_items, cfg["n_attack"], cfg["seed"])
    _, report_idx = stratified_half_split(attack_items, cfg["seed"] + 999)
    report_items = [attack_items[i] for i in report_idx]

    orig_net = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
    densenet = load_densenet(args.densenet_ckpt, device)
    probe_dim = plain_probe_feat(orig_net, load_xy(report_items[:1], cfg["img_size"], device)[0]).shape[1]
    probe = load_probe(args.probe_ckpt, probe_dim, device)

    Xc_report, Yc_report = load_xy(report_items, cfg["img_size"], device)

    csv_path = os.path.join(args.out_dir, "vq16_defense_table.csv")
    if os.path.isfile(csv_path):
        print(f"already computed, loading instead of rerunning the full eval -> {csv_path}")
        return

    decoder_ckpt_path = os.path.join(args.ckpt_dir, "vq16_decoder_realism_robust.pt")
    combined_eps = args.combined_eps255 if args.combined_eps255 is not None else cfg["eps_grid"][-1]
    combined_ckpt_path = os.path.join(args.ckpt_dir, f"vq16_combined_eps{combined_eps}.pt")

    def model_for(defense, ft_eps):
        if defense.startswith("advft"):
            m = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
            m.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"vq16_advft_eps{ft_eps}.pt"),
                                          map_location=device, weights_only=False))
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m
        if defense == "decoder_realism":
            m = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
            m.load_state_dict(torch.load(decoder_ckpt_path, map_location=device, weights_only=False))
            m.eval()
            return m
        if defense == "combined":
            m = load_vq16_net(args.llamagen_repo, args.vq16_ckpt, device)
            m.load_state_dict(torch.load(combined_ckpt_path, map_location=device, weights_only=False))
            m.eval()
            return m
        return orig_net

    rows = []
    for defense in DEFENSES:
        ft_eps_list = cfg["eps_grid"] if defense.startswith("advft") else [None]
        for ft_eps in ft_eps_list:
            model = model_for(defense, ft_eps)
            recon_fn = recon_fn_for(defense, model, pixel_sigma, pixel_K, sqs_name, sqs_params)
            probe_fn = probe_fn_for(defense, model, pixel_sigma, pixel_K, sqs_name, sqs_params)

            recon_c = batched(lambda x: densenet_logit(densenet, recon_fn(x)).cpu().numpy(), Xc_report, device, bs)
            prb_c = batched(lambda x: probe(probe_fn(x)).cpu().numpy(), Xc_report, device, bs)
            auc_recon_clean, auc_probe_clean = auc(Yc_report, recon_c), auc(Yc_report, prb_c)

            for attack in tqdm(ATTACKS_TO_DEFEND, desc=f"{defense} ft_eps{ft_eps}"):
                for eval_eps in cfg["eps_grid"]:
                    Xa, Ya = load_folder(args.adv_dir, "vq16", attack, eval_eps, idx_subset=report_idx)
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
                del model
                torch.cuda.empty_cache()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
