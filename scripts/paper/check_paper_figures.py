#!/usr/bin/env python3
"""Check the regenerated numbers against the ones printed in the paper.

The constants below are transcribed from the paper's inline TikZ / tabular source
(neurips_2026.tex, appendix_extra_data.tex). Each value is recomputed from
results/paper_inputs/ with the helpers build_paper_figures.py uses and compared
to the paper. Coverage is the headline cells of each float, not every entry --
enough to catch an aggregation regression. Exits non-zero on any mismatch.

Usage:
    python scripts/paper/check_paper_figures.py [--tol 0.012] [--inputs results/paper_inputs]
"""
import argparse
import os

import numpy as np
import pandas as pd

from build_paper_figures import (
    BY_DATASET_KEYS, MEAN_FLOATS, TOKS, amean, dmean,
    load_attack, load_defense, report_clean, sqs_winner_bar,
)

# tab:main -- {tok: {pipeline attack: (auc_recon, auc_probe, flip_pct)}}, eps=2/255
TAB_MAIN = {
    "vq16":    {"M1": (0.005, 0.762, 85.0), "M2-sat": (0.940, 0.864, 96.7), "M2-unb": (0.901, 0.836, 98.0)},
    "meditok": {"M1": (0.000, 0.744, 81.3), "M2-sat": (0.951, 0.868, 99.3), "M2-unb": (0.923, 0.239, 98.6)},
    "unitok":  {"M1": (0.000, 0.764, 83.4), "M2-sat": (0.939, 0.886, 99.0), "M2-unb": (0.921, 0.721, 99.0)},
    "fsq":     {"M1": (0.000, 0.780, 43.7), "M2-sat": (0.842, 0.764, 96.5), "M2-unb": (0.850, 0.769, 80.6)},
}
# tab:flip_vs_damage -- {tok: (cleanP, flip_sat, flip_unb, disp_sat, disp_unb, probe_sat, probe_unb)}, eps=2/255
TAB_FLIP = {
    "vq16":    (0.863, 96.73, 97.99, 35.88, 75.31, 0.864, 0.836),
    "meditok": (0.919, 99.29, 98.58, 21.61, 50.24, 0.868, 0.239),
    "unitok":  (0.907, 99.04, 99.03, 11.34, 22.82, 0.886, 0.721),
    "fsq":     (0.855, 96.50, 80.62,  1.66,  1.82, 0.764, 0.769),
}
# tab:defense_recon_all -- {tok: {"recon"/"probe": [None, Pixel-noise, SQS, Encoder-FT, Decoder-FT, Enc+Dec-FT]}}
TAB_DEF = {
    "vq16":    {"recon": [0.760, 0.766, 0.661, 0.740, 0.896, 0.890], "probe": [0.728, 0.738, 0.724, 0.756, 0.728, 0.756]},
    "meditok": {"recon": [0.697, 0.889, 0.719, 0.720, 0.869, 0.870], "probe": [0.104, 0.828, 0.105, 0.125, 0.104, 0.125]},
    "unitok":  {"recon": [0.856, 0.902, 0.843, 0.847, 0.890, 0.900], "probe": [0.409, 0.835, 0.410, 0.653, 0.409, 0.653]},
    "fsq":     {"recon": [0.775, 0.773, 0.761, 0.756, 0.881, 0.840], "probe": [0.596, 0.597, 0.605, 0.630, 0.596, 0.630]},
}
TAB_DEF_IDS = ["none", "pixel_noise", "best_sqs", "advft", "decoder_realism", "combined"]
# fig:sweep -- {tok: {"probe"/"recon": [5 eps]}}, M2-unb adversarial AUC
FIG_SWEEP = {
    "vq16":    {"probe": [0.869, 0.869, 0.836, 0.809, 0.709], "recon": [0.942, 0.937, 0.901, 0.832, 0.752]},
    "meditok": {"probe": [0.827, 0.601, 0.239, 0.126, 0.097], "recon": [0.950, 0.943, 0.923, 0.804, 0.705]},
    "unitok":  {"probe": [0.869, 0.830, 0.721, 0.521, 0.431], "recon": [0.947, 0.936, 0.921, 0.878, 0.849]},
    "fsq":     {"probe": [0.844, 0.826, 0.769, 0.688, 0.653], "recon": [0.846, 0.850, 0.850, 0.833, 0.785]},
}
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
# tab:displacement_all, Kermany rows -- emb_disp_pure_mean, interleaved (M2-sat, M2-unb) per eps
TAB_DISP_KERMANY = {
    "vq16":    [12.21, 20.84, 20.33, 37.08, 30.35, 62.47, 40.62, 93.83, 49.96, 120.39],
    "meditok": [7.28, 20.38, 13.67, 35.98, 22.11, 52.03, 30.39, 65.21, 37.21, 73.01],
    "unitok":  [5.30, 10.03, 7.41, 15.47, 9.56, 20.51, 11.48, 24.81, 13.61, 27.82],
    "fsq":     [0.30, 0.31, 0.57, 0.61, 1.12, 1.23, 2.08, 2.32, 3.12, 4.07],
}
# fig:spine / fig:sweep clean-baseline lines -- REPORT split, {tok: (recon, probe)}.
# fig:spine prints FSQ's line as 0.85 (a rounding slip); every other clean baseline uses these.
CLEAN_REPORT = {"vq16": (0.942, 0.871), "meditok": (0.947, 0.913),
                "unitok": (0.937, 0.903), "fsq": (0.831, 0.786)}
# fig:sqs-attack-dependence -- {tok: (M1, M2-sat, M2-unb)} sweep-winner recon AUC
FIG_SQS = {"vq16": (0.000, 0.919, 0.753), "meditok": (0.000, 0.939, 0.728),
           "unitok": (0.000, 0.935, 0.861), "fsq": (0.000, 0.847, 0.819)}
FIG_SQS_CLEAN = {"vq16": 0.942, "meditok": 0.947, "unitok": 0.937, "fsq": 0.831}
SQS_COLS = ["auc_adv_M1", "auc_adv_M2-sat", "auc_adv_M2-unb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.012, help="absolute tolerance in AUC units")
    ap.add_argument("--inputs", default="results/paper_inputs")
    ap.add_argument("--figures-dir", default="results/paper_figures",
                    help="where build_paper_figures.py wrote its output; used for the "
                         "<name>_by_dataset.csv self-consistency check")
    args = ap.parse_args()

    att = load_attack(args.inputs)
    dfn = load_defense(args.inputs)
    fails = []

    def chk(name, got, paper, tol):
        ok = abs(got - paper) <= tol
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:44s} got {got:8.3f}  paper {paper:8.3f}  d {got - paper:+.3f}")
        if not ok:
            fails.append(name)

    print("tab:main, eps=2/255")
    for tok, per in TAB_MAIN.items():
        for a, (rec, prb, flip) in per.items():
            chk(f"{tok}/{a} recon", amean(att, tok, a, 2.0, "auc_recon_adv"), rec, args.tol)
            chk(f"{tok}/{a} probe", amean(att, tok, a, 2.0, "auc_probe_adv"), prb, args.tol)
            chk(f"{tok}/{a} flip%", amean(att, tok, a, 2.0, "token_flip_rate_pct"), flip, args.tol * 20)

    print("\ntab:flip_vs_damage, eps=2/255")
    for tok, (cp, fs, fu, ds, du, ps, pu) in TAB_FLIP.items():
        chk(f"{tok} clean probe", amean(att, tok, "M2-unb", 2.0, "auc_probe_clean"), cp, args.tol)
        chk(f"{tok} flip sat", amean(att, tok, "M2-sat", 2.0, "token_flip_rate_pct"), fs, args.tol * 20)
        chk(f"{tok} flip unb", amean(att, tok, "M2-unb", 2.0, "token_flip_rate_pct"), fu, args.tol * 20)
        chk(f"{tok} disp sat", amean(att, tok, "M2-sat", 2.0, "emb_disp_pure_mean"), ds, 0.2)
        chk(f"{tok} disp unb", amean(att, tok, "M2-unb", 2.0, "emb_disp_pure_mean"), du, 0.2)
        chk(f"{tok} probe sat", amean(att, tok, "M2-sat", 2.0, "auc_probe_adv"), ps, args.tol)
        chk(f"{tok} probe unb", amean(att, tok, "M2-unb", 2.0, "auc_probe_adv"), pu, args.tol)

    print("\nfig:sweep, M2-unb budget sweep")
    for tok, per in FIG_SWEEP.items():
        for path, col in [("probe", "auc_probe_adv"), ("recon", "auc_recon_adv")]:
            for eps, paper in zip(EPS_GRID, per[path]):
                chk(f"{tok}/{path}/eps{eps:g}", amean(att, tok, "M2-unb", eps, col), paper, args.tol)

    print("\ntab:displacement_all, Kermany rows (eps-major: M2-sat, M2-unb per eps)")
    kerm = att[att.dataset == "kermany"]
    for tok, want in TAB_DISP_KERMANY.items():
        got = []
        for eps in EPS_GRID:
            for a in ("M2-sat", "M2-unb"):
                v = kerm[(kerm.tokenizer == tok) & (kerm.attack == a) & (kerm.eps255 == eps)]
                got.append(float(v["emb_disp_pure_mean"].iloc[0]))
        for j, (g, w) in enumerate(zip(got, want)):
            chk(f"{tok} disp col{j + 1}", g, w, 0.02)

    print("\nfig:spine / fig:sweep clean baselines (REPORT split)")
    clean = report_clean(dfn)
    for tok, (rec, prb) in CLEAN_REPORT.items():
        chk(f"{tok} clean recon", clean[tok, "recon"], rec, args.tol)
        chk(f"{tok} clean probe", clean[tok, "probe"], prb, args.tol)

    print("\ntab:defense_recon_all, M2-unb eps=8/255")
    for tok, per in TAB_DEF.items():
        for path, col in [("recon", "auc_recon_adv"), ("probe", "auc_probe_adv")]:
            for d, paper in zip(TAB_DEF_IDS, per[path]):
                chk(f"{tok}/{path}/{d}", dmean(dfn, tok, d, "M2-unb", 8.0, col), paper, args.tol)

    print("\nfig:sqs-attack-dependence, sweep winner, mean over datasets")
    for tok in TOKS:
        chk(f"{tok} clean diamond",
            dmean(dfn, tok, "none", "M2-unb", 8.0, "auc_recon_clean"), FIG_SQS_CLEAN[tok], args.tol)
        for col, paper in zip(SQS_COLS, FIG_SQS[tok]):
            chk(f"{tok} bar {col[8:]}", sqs_winner_bar(args.inputs, tok, col), paper, args.tol)

    print("\n<name>_by_dataset.csv: mean over the 3 datasets == headline .csv")
    fig_dir = args.figures_dir
    if not os.path.isdir(fig_dir) or not os.path.isfile(f"{fig_dir}/fig_spine_by_dataset.csv"):
        print("  (skipped -- run build_paper_figures.py first)")
    else:
        for name in MEAN_FLOATS:
            comp = pd.read_csv(f"{fig_dir}/{name}_by_dataset.csv")
            head = pd.read_csv(f"{fig_dir}/{name}.csv")
            keys = BY_DATASET_KEYS[name]
            vcols = [c for c in head.columns if c not in keys and c != "dataset"]
            got = comp.groupby(keys, sort=False)[vcols].mean().reset_index().merge(
                head[keys + vcols], on=keys, suffixes=("_c", "_h"))
            bad = [c for c in vcols if not np.allclose(got[f"{c}_c"], got[f"{c}_h"], atol=1e-3)]
            print(f"  [{'ok  ' if not bad else 'FAIL'}] {name:36s} ({len(comp)} per-dataset rows)")
            if bad:
                fails.append(f"{name}:{bad}")

    print()
    if fails:
        raise SystemExit(f"{len(fails)} mismatch(es): {fails}")
    print("all checks match the paper within tolerance")


if __name__ == "__main__":
    main()
