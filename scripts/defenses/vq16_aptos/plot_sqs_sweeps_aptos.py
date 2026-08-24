#!/usr/bin/env python3
"""Plots the three SQS sweeps produced by sweep_sqs_aptos.py: flat (1D line),
spatial-consensus (sigma x spatial_weight, faceted by agreement_thresh_pct),
and margin-adaptive (sigma_base x sigma_mult_max). Color/height = mean
adversarial AUC recovered, the quantity every sweep maximized. Reads the CSVs
and the winner config from --out-dir; run sweep_sqs_aptos.py first.

Usage:
    python scripts/defenses/vq16_aptos/plot_sqs_sweeps_aptos.py --out-dir results/outputs
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/outputs")
    return p.parse_args()


def main():
    args = parse_args()
    flat_df = pd.read_csv(os.path.join(args.out_dir, "vq16_sqs_flat_sweep.csv"))
    spatial_df = pd.read_csv(os.path.join(args.out_dir, "vq16_sqs_spatial_sweep.csv"))
    adaptive_df = pd.read_csv(os.path.join(args.out_dir, "vq16_sqs_margin_adaptive_sweep.csv"))
    k_sigma_df = pd.read_csv(os.path.join(args.out_dir, "vq16_sqs_k_sigma_grid.csv"))
    winner = json.load(open(os.path.join(args.out_dir, "vq16_sqs_winner.json")))
    params = winner["all_variant_params"]
    sigma_flat = params["flat"]["sigma"]
    sigma_spatial, spatial_weight, agreement_thresh = (
        params["spatial"]["sigma"], params["spatial"]["spatial_weight"], params["spatial"]["agreement_thresh_pct"])
    sigma_base, sigma_mult_max = params["margin_adaptive"]["sigma_base"], params["margin_adaptive"]["sigma_mult_max"]
    k_flat = params["flat"]["K"]

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    pivot_k = k_sigma_df.pivot(index="K", columns="sigma", values="mean_auc_adv")
    im3 = ax3.imshow(pivot_k.values, aspect="auto", origin="lower", cmap="viridis")
    ax3.set_xticks(range(len(pivot_k.columns))); ax3.set_xticklabels(pivot_k.columns, rotation=45)
    ax3.set_yticks(range(len(pivot_k.index))); ax3.set_yticklabels(pivot_k.index)
    ax3.set_xlabel("sigma"); ax3.set_ylabel("K (noisy passes)")
    ax3.set_title("flat SQS: K x sigma grid -- mean_auc_adv")
    if k_flat in pivot_k.index and sigma_flat in pivot_k.columns:
        ki, si2 = list(pivot_k.index).index(k_flat), list(pivot_k.columns).index(sigma_flat)
        ax3.scatter([si2], [ki], marker="*", s=300, c="red", edgecolors="white", label="flat SQS winner")
        ax3.legend(fontsize=8)
    fig3.colorbar(im3, ax=ax3, label="mean_auc_adv")
    fig3.tight_layout()
    png_k_sigma = os.path.join(args.out_dir, "vq16_sqs_k_sigma_grid.png")
    fig3.savefig(png_k_sigma, dpi=150, bbox_inches="tight")
    print("saved ->", png_k_sigma)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    ax = axes[0]
    ax.plot(flat_df.sigma, flat_df.mean_auc_adv, marker="o", label="mean_auc_adv")
    ax.plot(flat_df.sigma, flat_df.auc_clean, marker="s", linestyle="--", color="gray", label="auc_clean")
    ax.axvline(sigma_flat, color="red", linestyle=":", label=f"selected sigma={sigma_flat}")
    ax.set_xscale("log")
    ax.set_xlabel("sigma"); ax.set_ylabel("AUC"); ax.set_title("flat SQS (1D sweep)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    sub = spatial_df[spatial_df.agreement_thresh_pct == agreement_thresh]
    pivot = sub.pivot(index="sigma", columns="spatial_weight", values="mean_auc_adv")
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("spatial_weight"); ax.set_ylabel("sigma")
    ax.set_title(f"spatial SQS, agreement_thresh={agreement_thresh}")
    wi, si = list(pivot.columns).index(spatial_weight), list(pivot.index).index(sigma_spatial)
    ax.scatter([wi], [si], marker="*", s=300, c="red", edgecolors="white", label="winner")
    ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax, label="mean_auc_adv")

    ax = axes[2]
    pivot2 = adaptive_df.pivot(index="sigma_base", columns="sigma_mult_max", values="mean_auc_adv")
    im2 = ax.imshow(pivot2.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot2.columns))); ax.set_xticklabels(pivot2.columns)
    ax.set_yticks(range(len(pivot2.index))); ax.set_yticklabels(pivot2.index)
    ax.set_xlabel("sigma_mult_max"); ax.set_ylabel("sigma_base")
    ax.set_title("margin-adaptive SQS")
    mi, bi = list(pivot2.columns).index(sigma_mult_max), list(pivot2.index).index(sigma_base)
    ax.scatter([mi], [bi], marker="*", s=300, c="red", edgecolors="white", label="winner")
    ax.legend(fontsize=8)
    fig.colorbar(im2, ax=ax, label="mean_auc_adv")

    fig.suptitle("SQS hyperparameter sweeps (SEL_ITEMS, worst-case eps) -- color/height = mean adversarial AUC recovered", y=1.03)
    fig.tight_layout()
    png_main = os.path.join(args.out_dir, "vq16_sqs_grid_plots.png")
    fig.savefig(png_main, dpi=150, bbox_inches="tight")
    print("saved ->", png_main)

    thresholds = sorted(spatial_df.agreement_thresh_pct.unique())
    fig2, axes2 = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4), sharey=True)
    for ax, th in zip(axes2, thresholds):
        sub = spatial_df[spatial_df.agreement_thresh_pct == th]
        piv = sub.pivot(index="sigma", columns="spatial_weight", values="mean_auc_adv")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis",
                        vmin=spatial_df.mean_auc_adv.min(), vmax=spatial_df.mean_auc_adv.max())
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel("spatial_weight"); ax.set_title(f"agreement_thresh={th}")
        if th == agreement_thresh:
            wi, si = list(piv.columns).index(spatial_weight), list(piv.index).index(sigma_spatial)
            ax.scatter([wi], [si], marker="*", s=250, c="red", edgecolors="white")
    axes2[0].set_ylabel("sigma")
    fig2.colorbar(im, ax=axes2, label="mean_auc_adv", fraction=0.02)
    fig2.suptitle("spatial SQS: full sigma x spatial_weight grid, faceted by agreement_thresh_pct", y=1.05)
    png_spatial = os.path.join(args.out_dir, "vq16_sqs_spatial_grid_facets.png")
    fig2.savefig(png_spatial, dpi=150, bbox_inches="tight")
    print("saved ->", png_spatial)


if __name__ == "__main__":
    main()
