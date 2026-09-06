#!/usr/bin/env python3
"""Check the regenerated numbers against the ones printed in the paper.

The constants below are transcribed from the paper's inline TikZ / tabular source
(neurips_2026.tex, appendix_extra_data.tex). Each value is recomputed from
results/paper_inputs/ with the helpers build_paper_figures.py uses and compared
to the paper. Coverage is the headline cells of each float, not every entry, which is
enough to catch an aggregation regression. Exits non-zero on any mismatch.

Usage:
    python scripts/paper/check_paper_figures.py [--tol 0.012] [--inputs results/paper_inputs]
"""
import argparse
import os

import numpy as np
import pandas as pd

from build_paper_figures import (
    BY_DATASET_KEYS, MEAN_FLOATS, TOKS, amean, direction_counts, dmean, eps_pooled,
    load_attack, load_defense, load_gen_fidelity, load_gen_fidelity_clean,
    load_gen_fidelity_defense, load_vqa, report_clean, sqs_winner_bar,
)

# tab:main: {tok: {pipeline attack: (auc_recon, auc_probe, flip_pct)}}, eps=2/255
TAB_MAIN = {
    "vq16":    {"M1": (0.005, 0.762, 85.0), "M2-sat": (0.940, 0.864, 96.7), "M2-unb": (0.901, 0.836, 98.0)},
    "meditok": {"M1": (0.000, 0.744, 81.3), "M2-sat": (0.951, 0.868, 99.3), "M2-unb": (0.923, 0.239, 98.6)},
    "unitok":  {"M1": (0.000, 0.764, 83.4), "M2-sat": (0.939, 0.886, 99.0), "M2-unb": (0.921, 0.721, 99.0)},
    "fsq":     {"M1": (0.000, 0.780, 43.7), "M2-sat": (0.842, 0.764, 96.5), "M2-unb": (0.850, 0.769, 80.6)},
}
# tab:flip_vs_damage: {tok: (cleanP, flip_sat, flip_unb, disp_sat, disp_unb, probe_sat, probe_unb)}, eps=2/255
TAB_FLIP = {
    "vq16":    (0.863, 96.73, 97.99, 35.88, 75.31, 0.864, 0.836),
    "meditok": (0.919, 99.29, 98.58, 21.61, 50.24, 0.868, 0.239),
    "unitok":  (0.907, 99.04, 99.03, 11.34, 22.82, 0.886, 0.721),
    "fsq":     (0.855, 96.50, 80.62,  1.66,  1.82, 0.764, 0.769),
}
# tab:defense_recon_all: {tok: {"recon"/"probe": [None, Pixel-noise, SQS, Encoder-FT, Decoder-FT, Enc+Dec-FT]}}
TAB_DEF = {
    "vq16":    {"recon": [0.760, 0.766, 0.661, 0.740, 0.896, 0.890], "probe": [0.728, 0.738, 0.724, 0.756, 0.728, 0.756]},
    "meditok": {"recon": [0.697, 0.889, 0.719, 0.720, 0.869, 0.870], "probe": [0.104, 0.828, 0.105, 0.125, 0.104, 0.125]},
    "unitok":  {"recon": [0.856, 0.902, 0.843, 0.847, 0.890, 0.900], "probe": [0.409, 0.835, 0.410, 0.653, 0.409, 0.653]},
    "fsq":     {"recon": [0.775, 0.773, 0.761, 0.756, 0.881, 0.840], "probe": [0.596, 0.597, 0.605, 0.630, 0.596, 0.630]},
}
TAB_DEF_IDS = ["none", "pixel_noise", "best_sqs", "advft", "decoder_realism", "combined"]
# fig:sweep: {tok: {"probe"/"recon": [5 eps]}}, M2-unb adversarial AUC
FIG_SWEEP = {
    "vq16":    {"probe": [0.869, 0.869, 0.836, 0.809, 0.709], "recon": [0.942, 0.937, 0.901, 0.832, 0.752]},
    "meditok": {"probe": [0.827, 0.601, 0.239, 0.126, 0.097], "recon": [0.950, 0.943, 0.923, 0.804, 0.705]},
    "unitok":  {"probe": [0.869, 0.830, 0.721, 0.521, 0.431], "recon": [0.947, 0.936, 0.921, 0.878, 0.849]},
    "fsq":     {"probe": [0.844, 0.826, 0.769, 0.688, 0.653], "recon": [0.846, 0.850, 0.850, 0.833, 0.785]},
}
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
# tab:displacement_all, Kermany rows: emb_disp_pure_mean, interleaved (M2-sat, M2-unb) per eps
TAB_DISP_KERMANY = {
    "vq16":    [12.21, 20.84, 20.33, 37.08, 30.35, 62.47, 40.62, 93.83, 49.96, 120.39],
    "meditok": [7.28, 20.38, 13.67, 35.98, 22.11, 52.03, 30.39, 65.21, 37.21, 73.01],
    "unitok":  [5.30, 10.03, 7.41, 15.47, 9.56, 20.51, 11.48, 24.81, 13.61, 27.82],
    "fsq":     [0.30, 0.31, 0.57, 0.61, 1.12, 1.23, 2.08, 2.32, 3.12, 4.07],
}
# fig:spine / fig:sweep clean-baseline lines: REPORT split, {tok: (recon, probe)}.
# fig:spine prints FSQ's line as 0.85 (a rounding slip); every other clean baseline uses these.
CLEAN_REPORT = {"vq16": (0.942, 0.871), "meditok": (0.947, 0.913),
                "unitok": (0.937, 0.903), "fsq": (0.831, 0.786)}
# fig:sqs-attack-dependence: {tok: (M1, M2-sat, M2-unb)} sweep-winner recon AUC
FIG_SQS = {"vq16": (0.000, 0.919, 0.753), "meditok": (0.000, 0.939, 0.728),
           "unitok": (0.000, 0.935, 0.861), "fsq": (0.000, 0.847, 0.819)}
FIG_SQS_CLEAN = {"vq16": 0.942, "meditok": 0.947, "unitok": 0.937, "fsq": 0.831}
SQS_COLS = ["auc_adv_M1", "auc_adv_M2-sat", "auc_adv_M2-unb"]

# tab:gen_fidelity_appendix_attack: (LPIPS, PSNR) vs original at eps=8/255,
# {tok: {ds: {attack: (lpips8, psnr8)}}}
GENFID_ATTACK = {
    "vq16": {"kermany": {"M2-sat": (0.213, 24.72), "M2-unb": (0.485, 22.47)},
             "aptos": {"M2-sat": (0.200, 29.14), "M2-unb": (0.363, 27.32)},
             "isic": {"M2-sat": (0.239, 26.92), "M2-unb": (0.473, 24.86)}},
    "fsq": {"kermany": {"M2-sat": (0.256, 27.11), "M2-unb": (0.321, 26.28)},
            "aptos": {"M2-sat": (0.209, 28.87), "M2-unb": (0.298, 27.98)},
            "isic": {"M2-sat": (0.234, 27.99), "M2-unb": (0.360, 26.78)}},
    "meditok": {"kermany": {"M2-sat": (0.158, 29.62), "M2-unb": (0.423, 24.08)},
                "aptos": {"M2-sat": (0.148, 32.57), "M2-unb": (0.479, 16.26)},
                "isic": {"M2-sat": (0.184, 30.23), "M2-unb": (0.514, 19.62)}},
    "unitok": {"kermany": {"M2-sat": (0.154, 29.12), "M2-unb": (0.347, 25.28)},
               "aptos": {"M2-sat": (0.137, 32.84), "M2-unb": (0.275, 27.95)},
               "isic": {"M2-sat": (0.165, 30.17), "M2-unb": (0.342, 27.11)}},
}
# tab:gen_fidelity_defense_appendix: (LPIPS, PSNR, SSIM), M2-unb eps=8,
# {tok: {defense: {ds: (lpips, psnr, ssim)}}}. Defense keys: Clean/None/encoder_ft_eps8/decoder_ft/combined.
GENFID_DEFENSE = {
    "vq16": {"Clean": {"kermany": (0.103, 25.60, 0.675), "aptos": (0.070, 30.35, 0.842), "isic": (0.109, 28.07, 0.725)},
             "None": {"kermany": (0.485, 22.47, 0.417), "aptos": (0.363, 27.32, 0.657), "isic": (0.473, 24.86, 0.503)},
             "decoder_ft": {"kermany": (0.207, 23.47, 0.539), "aptos": (0.171, 28.69, 0.794), "isic": (0.307, 26.08, 0.607)},
             "combined": {"kermany": (0.185, 23.69, 0.569), "aptos": (0.169, 28.09, 0.788), "isic": (0.273, 26.05, 0.626)}},
    "unitok": {"Clean": {"kermany": (0.042, 30.48, 0.832), "aptos": (0.033, 34.63, 0.891), "isic": (0.053, 31.78, 0.817)},
               "decoder_ft": {"kermany": (0.127, 27.39, 0.727), "aptos": (0.122, 32.44, 0.864), "isic": (0.179, 29.25, 0.751)},
               "combined": {"kermany": (0.111, 27.72, 0.740), "aptos": (0.109, 32.03, 0.871), "isic": (0.156, 29.23, 0.760)}},
    "meditok": {"decoder_ft": {"kermany": (0.169, 26.32, 0.710), "aptos": (0.147, 31.13, 0.850), "isic": (0.272, 26.61, 0.685)}},
}


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

    print("\ntab:gen_fidelity_appendix_attack, LPIPS/PSNR vs original at eps=8/255")
    gf = load_gen_fidelity(args.inputs)
    for tok, per_ds in GENFID_ATTACK.items():
        for ds, per_a in per_ds.items():
            sub = gf[(gf.tokenizer == tok) & (gf.dataset == ds)].set_index(["attack", "eps255"])
            for a, (lp, ps) in per_a.items():
                chk(f"{tok}/{ds}/{a} LPIPS8", float(sub.loc[(a, 8.0), "lpips_vs_original"]), lp, 0.002)
                chk(f"{tok}/{ds}/{a} PSNR8", float(sub.loc[(a, 8.0), "psnr_vs_original"]), ps, 0.02)

    print("\ntab:gen_fidelity_defense_appendix, LPIPS/PSNR/SSIM under M2-unb eps=8/255")
    gfd, gfc = load_gen_fidelity_defense(args.inputs), load_gen_fidelity_clean(args.inputs)
    for tok, per_def in GENFID_DEFENSE.items():
        for dkey, per_ds in per_def.items():
            for ds, (lp, ps, ss) in per_ds.items():
                if dkey == "Clean":
                    r = gfc[(gfc.tokenizer == tok) & (gfc.dataset == ds)].iloc[0]
                    got = (r["lpips"], r["psnr"], r["ssim"])
                elif dkey == "None":
                    r = gf[(gf.tokenizer == tok) & (gf.dataset == ds) & (gf.attack == "M2-unb")
                           ].set_index("eps255").loc[8.0]
                    got = (r["lpips_vs_original"], r["psnr_vs_original"], r["ssim_vs_original"])
                else:
                    r = gfd[(gfd.tokenizer == tok) & (gfd.dataset == ds) & (gfd.defense == dkey)
                            & (gfd.attack == "M2-unb") & (gfd.eps255 == 8.0)].iloc[0]
                    got = (r["lpips_vs_original"], r["psnr_vs_original"], r["ssim_vs_original"])
                for metric, g, w, t in zip(("LPIPS", "PSNR", "SSIM"), got, (lp, ps, ss), (0.002, 0.02, 0.002)):
                    chk(f"{tok}/{dkey}/{ds} {metric}", float(g), w, t)

    # --- generative downstream: LLaVA-Med VQA -------------------------------
    # The paper notes greedy-decode wording shifts an individual image's breaking
    # eps by a step between runs, so the pooled tables are checked tightly and the
    # single-eps tab:vqa-full cells loosely.
    vqa = load_vqa(args.inputs)
    ep_u, ep_s = eps_pooled(vqa["m2unb"]), eps_pooled(vqa["m2sat"])
    # tab:vqa-eps: {eps: (unb flip, unb disp, unb Correct%, sat Correct%, rnd Correct%)}
    TAB_VQA_EPS = {2: (98.7, 50.6, 59.0, 96.0, 99.0),
                   4: (99.8, 65.1, 13.0, 97.0, 99.0),
                   8: (100.0, 71.3, 1.0, 98.0, 98.0)}
    print("\ntab:vqa-eps: pooled flip/disp/Correct% (n=99 clean-correct cells)")
    for i, eps in enumerate((2, 4, 8)):
        u, s = ep_u.iloc[i], ep_s.iloc[i]
        fl, di, cu, cs, cr = TAB_VQA_EPS[eps]
        chk(f"eps{eps} M2-unb flip%", float(u["adv_flip_pct"]), fl, 0.5)
        chk(f"eps{eps} M2-unb disp", float(u["adv_disp"]), di, 0.3)
        chk(f"eps{eps} M2-unb Correct%", float(u["adv_correct_pct"]), cu, 3.0)
        chk(f"eps{eps} M2-sat Correct%", float(s["adv_correct_pct"]), cs, 3.0)
        chk(f"eps{eps} Random Correct%", float(u["rnd_correct_pct"]), cr, 3.0)

    # tab:vqa-direction: exact counts, {(ds, eps): (erased, erased_of, invented, invented_of)}
    TAB_VQA_DIR = {("aptos", 2): (4, 10, 6, 7), ("aptos", 4): (6, 10, 7, 7), ("aptos", 8): (3, 10, 7, 7),
                   ("isic", 2): (6, 6, 1, 10), ("isic", 4): (6, 6, 0, 10), ("isic", 8): (6, 6, 2, 10)}
    dc = direction_counts(vqa["m2unb"]).set_index(["dataset", "eps"])
    print("\ntab:vqa-direction: error-direction counts under M2-unb (exact)")
    for (ds, eps), (er, erof, inv, invof) in TAB_VQA_DIR.items():
        row = dc.loc[(ds, eps)]
        chk(f"{ds} eps{eps} erased", float(row["erased"]), er, 0.5)
        chk(f"{ds} eps{eps} erased_of", float(row["erased_of"]), erof, 0.5)
        chk(f"{ds} eps{eps} invented", float(row["invented"]), inv, 0.5)
        chk(f"{ds} eps{eps} invented_of", float(row["invented_of"]), invof, 0.5)

    # tab:vqa-full: clean and M2-sat cells are stable; M2-unb single-eps cells
    # carry the run-to-run wording variance, so check them at +-12 pp.
    from src.generative.vqa_scoring import full_accuracy

    fc, fu, fs = (full_accuracy(vqa["m2unb"], "clean"), full_accuracy(vqa["m2unb"], "adv"),
                  full_accuracy(vqa["m2sat"], "adv"))
    Q = {"modality": "What modality is used to take this image?",
         "region": "Which part of the body does this image belong to?"}
    TAB_VQA_FULL = {  # (dataset, key): (clean, [unb 2/4/8], [sat 2/4/8])
        ("kermany", "modality"): (100.0, [75.0, 5.0, 0.0], [100.0, 100.0, 100.0]),
        ("kermany", "region"): (100.0, [90.0, 60.0, 5.0], [100.0, 100.0, 100.0]),
        ("aptos", "modality"): (100.0, [55.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        ("aptos", "region"): (100.0, [65.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        ("isic", "modality"): (95.0, [10.0, 0.0, 0.0], [80.0, 85.0, 90.0]),
    }
    print("\ntab:vqa-full: modality/body-region accuracy (clean and M2-sat tight, M2-unb loose)")
    for (ds, key), (cln, unb, sat) in TAB_VQA_FULL.items():
        q = Q[key]
        chk(f"{ds}/{key} clean", float(fc[(fc.dataset == ds) & (fc.question == q)].acc.iloc[0]), cln, 0.1)
        for eps, wu, ws in zip((2, 4, 8), unb, sat):
            gu = float(fu[(fu.dataset == ds) & (fu.question == q) & (fu.eps == eps)].acc.iloc[0])
            gs = float(fs[(fs.dataset == ds) & (fs.question == q) & (fs.eps == eps)].acc.iloc[0])
            chk(f"{ds}/{key} M2-unb eps{eps}", gu, wu, 12.0)
            chk(f"{ds}/{key} M2-sat eps{eps}", gs, ws, 3.0)

    print("\n<name>_by_dataset.csv: mean over the 3 datasets == headline .csv")
    fig_dir = args.figures_dir
    if not os.path.isdir(fig_dir) or not os.path.isfile(f"{fig_dir}/fig_spine_by_dataset.csv"):
        print("  (skipped: run build_paper_figures.py first)")
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
