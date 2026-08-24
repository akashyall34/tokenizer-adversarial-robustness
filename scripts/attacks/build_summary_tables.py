#!/usr/bin/env python3
"""Pivots the per-tokenizer *_attack_table.csv files (written by the
run_*_kermany.py scripts) into tokenizer x eps summary tables: AUC drop with
and without reconstruction, token-flip rate, embedding displacement, and a
clean-AUC reference table. Run after the tokenizer runs you care about.

Usage:
    python scripts/attacks/build_summary_tables.py --out-dir results/outputs
"""
import argparse
import os

import pandas as pd

ORDER = ["vq16", "meditok", "unitok", "fsq"]  # row order; skips any not present
WITH_RECON_ATTACKS = [
    ("M1", "M1 (recon attack)"),
    ("M3_alpha03", "M3 alpha=0.3"),
    ("M3_geo_sat", "M3 geo saturated"),
    ("M3_geo_unb", "M3 geo unbounded"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/outputs", help="where *_attack_table.csv live")
    return p.parse_args()


def make_table(df, order, eps_values, attack, value, out_dir, fname, title):
    sub = df[df["attack"] == attack]
    p = sub.pivot_table(index="tokenizer", columns="eps_pixel", values=value).reindex(index=order, columns=eps_values)
    p.to_csv(f"{out_dir}/{fname}.csv")
    print(f"\n### {title}   (rows = tokenizer, cols = eps pixel /255)")
    print(p.round(3).to_string())
    return p


def main():
    args = parse_args()
    out_dir = args.out_dir
    tbl_dir = f"{out_dir}/simple_tables"
    os.makedirs(tbl_dir, exist_ok=True)

    frames = []
    for name in ORDER:
        f = f"{out_dir}/{name}_attack_table.csv"
        if os.path.isfile(f):
            frames.append(pd.read_csv(f))
        else:
            print("missing (skipped):", f)
    if not frames:
        raise SystemExit(f"no *_attack_table.csv found under {out_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["eps_pixel"] = df["eps_pixel"].astype(float)
    eps_values = sorted(df["eps_pixel"].unique())
    order = [t for t in ORDER if t in df["tokenizer"].unique()]

    for atk, lbl in WITH_RECON_ATTACKS:
        make_table(df, order, eps_values, atk, "drop_recon", tbl_dir,
                   f"withRecon_AUCdrop_{atk}", f"WITH reconstruction -- AUC drop -- {lbl}")
    for atk, lbl in WITH_RECON_ATTACKS:
        make_table(df, order, eps_values, atk, "drop_probe", tbl_dir,
                   f"withoutRecon_AUCdrop_{atk}", f"WITHOUT reconstruction (probe) -- AUC drop -- {lbl}")

    make_table(df, order, eps_values, "M3_geo_unb", "token_flip_rate_pct", tbl_dir,
               "flip_rate_M3_geo_unb", "Token flip rate % -- M3 geo unbounded")
    make_table(df, order, eps_values, "M3_geo_unb", "emb_disp_pure_mean", tbl_dir,
               "displacement_M3_geo_unb", "Embedding displacement -- M3 geo unbounded")

    clean = df.groupby("tokenizer")[["auc_raw_clean", "auc_recon_clean", "auc_probe_clean"]].first().reindex(order)
    clean.to_csv(f"{tbl_dir}/clean_AUC_reference.csv")
    print("\n### Clean AUC (no attack) -- reference")
    print(clean.round(3).to_string())

    print("\nsaved", len(os.listdir(tbl_dir)), "tables to", tbl_dir)


if __name__ == "__main__":
    main()
