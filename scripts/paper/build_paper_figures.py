#!/usr/bin/env python3
"""Regenerate the workshop paper's data figures and results tables from the
pipeline CSVs.

For each float this writes three files to --out-dir:

  <name>.csv   the plotted / tabulated numbers, one tidy table
  <name>.tex   the complete tikzpicture / tabular, ready to drop into the paper
  <name>.pdf   that .tex compiled standalone (needs pdflatex; --no-pdf skips it)

The .tex scaffolding (groupplot option blocks, table headers, the vizone..vizmauve
colours) is lifted verbatim from the paper's source so a compiled PDF matches the
figure as printed; only the data coordinates come from this script. fig_defense_pareto
is the one exception -- the paper hand-places its point labels and Pareto stars, so
the regenerated version uses plain numbered labels and data-driven axis limits.

Outputs are named by the paper's \\label (fig:spine -> fig_spine). Threat-model and
pipeline-overview are hand-drawn and have no data, so they are not covered.

Values are a mean over the three datasets unless the float is per-dataset. The
adversarial curves come from the 256-image attack subset; the clean-baseline lines
in fig_spine / fig_sweep come from the 128-image REPORT split (the defense tables'
"none" rows), matching the paper. eps255 is the paper's eps/255 budget.

Attack ids (M0, M1, M2-sat, M2-unb) match the paper. M0 and the auxiliary
non-paper M3_alpha03 blend are not plotted.

Aggregation touches no model, GPU, or checkpoint.

Usage:
    python scripts/paper/build_paper_figures.py \
        --inputs results/paper_inputs --out-dir results/paper_figures
"""
import argparse
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PREAMBLE = os.path.join(HERE, "figure_preamble.tex")

TOKS = ["vq16", "meditok", "unitok", "fsq"]
TOK_LABEL = {"vq16": "LlamaGen VQ-16", "meditok": "MedITok", "unitok": "UniTok", "fsq": "FSQ"}
TOK_SHORT = {"vq16": "VQ-16", "meditok": "MedITok", "unitok": "UniTok", "fsq": "FSQ"}
DATASETS = ["kermany", "aptos", "isic"]
DS_LABEL = {"kermany": "Kermany", "aptos": "APTOS", "isic": "ISIC"}
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]

# defense_table.csv id -> paper column header, in paper column order
DEFENSE_COLS = [
    ("none", "None"), ("pixel_noise", "Pixel-noise"), ("best_sqs", "SQS"),
    ("advft", "Encoder-FT"), ("advft+best_sqs", "Encoder-FT+SQS"),
    ("decoder_realism", "Decoder-FT"), ("combined", "Encoder-FT+Decoder-FT"),
]
FT_EPS = 8.0  # advft / advft+best_sqs read at the finetune budget matching the attack

# per-tokenizer line style for fig_sweep, paper order
SWEEP_STYLE = [("vizone", "*", "1.8pt"), ("viztwo", "square*", "1.8pt"),
               ("vizthree", "triangle*", "2.2pt"), ("vizmauve", "diamond*", "2.2pt")]


# --- inputs -------------------------------------------------------------------
def load_attack(inputs):
    frames = []
    for tok in TOKS:
        for ds in DATASETS:
            df = pd.read_csv(f"{inputs}/attack-results/{tok}_attack_table_{ds}.csv")
            df["dataset"] = ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_defense(inputs):
    frames = []
    for tok in TOKS:
        for ds in DATASETS:
            d = f"{inputs}/defense-results/{tok}_defense_outputs_{ds}"
            df = pd.read_csv(f"{d}/{tok}_defense_table.csv")
            df["tokenizer"], df["dataset"] = tok, ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- aggregation over the three datasets ------------------------------------
# The *vals helpers return the per-dataset Series (indexed by DATASETS); the
# *mean helpers are just its mean. build_by_dataset() writes the Series out so
# every averaged number in the paper is the mean of three committed rows.
def avals(att, tok, attack, eps, col):
    m = (att.tokenizer == tok) & (att.attack == attack) & (att.eps255 == eps)
    v = att.loc[m].set_index("dataset")[col].reindex(DATASETS)
    if v.isna().any():
        raise KeyError(f"missing attack rows: {tok}/{attack}/eps{eps}/{col}")
    return v.astype(float)


def dvals(dfn, tok, defense, attack, eps, col):
    m = ((dfn.tokenizer == tok) & (dfn.defense == defense)
         & (dfn.attack == attack) & (dfn.eval_eps255 == eps))
    if defense in ("advft", "advft+best_sqs"):
        m = m & (dfn.finetune_eps255 == FT_EPS)
    v = dfn.loc[m].set_index("dataset")[col].reindex(DATASETS)
    if v.isna().any():
        raise KeyError(f"missing defense rows: {tok}/{defense}/{attack}/eval{eps}/{col}")
    return v.astype(float)


def sqs_winner_vals(inputs, tok, col):
    """Per-dataset value of one auc_adv_* column at the SQS selection-sweep winner
    (largest mean_auc_adv). Uses the K x sigma grid, or the flat sigma sweep for
    FSQ, which has no K x sigma grid."""
    out = {}
    for ds in DATASETS:
        d = f"{inputs}/defense-results/{tok}_defense_outputs_{ds}"
        path = f"{d}/{tok}_sqs_k_sigma_grid.csv"
        if not os.path.isfile(path):
            path = f"{d}/{tok}_sqs_flat_sweep.csv"
        s = pd.read_csv(path)
        out[ds] = s.loc[s.mean_auc_adv.idxmax(), col]
    return pd.Series(out).reindex(DATASETS).astype(float)


def amean(att, tok, attack, eps, col):
    return float(avals(att, tok, attack, eps, col).mean())


def dmean(dfn, tok, defense, attack, eps, col):
    return float(dvals(dfn, tok, defense, attack, eps, col).mean())


def sqs_winner_bar(inputs, tok, col):
    return float(sqs_winner_vals(inputs, tok, col).mean())


def report_clean(dfn):
    """Undefended clean AUC on the 128-image REPORT split (defense-table "none"
    rows), mean over datasets. {(tok, "recon"|"probe"): auc}."""
    out = {}
    for tok in TOKS:
        out[tok, "recon"] = dmean(dfn, tok, "none", "M1", 8.0, "auc_recon_clean")
        out[tok, "probe"] = dmean(dfn, tok, "none", "M1", 8.0, "auc_probe_clean")
    return out


# --- output -----------------------------------------------------------------
def write_csv(df, path):
    df.round(4).to_csv(path, index=False)
    print("wrote", path)


def write_tex(lines, path):
    with open(path, "w") as f:
        f.write("% generated by scripts/paper/build_paper_figures.py -- do not edit\n")
        f.write("\n".join(lines).rstrip() + "\n")
    print("wrote", path)


def compile_pdf(name, out_dir):
    """Compile <out_dir>/<name>.tex standalone into <out_dir>/<name>.pdf."""
    work = os.path.join(out_dir, "_latex")
    os.makedirs(work, exist_ok=True)
    doc = os.path.join(work, f"{name}.tex")
    with open(doc, "w") as f:
        f.write("\\documentclass[border=3pt]{standalone}\n"
                f"\\input{{{PREAMBLE}}}\n"
                "\\begin{document}\n"
                f"\\input{{{os.path.abspath(os.path.join(out_dir, name + '.tex'))}}}\n"
                "\\end{document}\n")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", work, doc]
    proc = None
    for _ in range(2):  # group plots need two passes to size correctly
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"pdflatex failed for {name}; see {work}/{name}.log\n{proc.stdout[-1500:]}")
    shutil.move(os.path.join(work, f"{name}.pdf"), os.path.join(out_dir, f"{name}.pdf"))
    print("wrote", os.path.join(out_dir, f"{name}.pdf"))


def coords(pts):
    """pgfplots "coordinates {...}" literal. Each point is (x, y), or (x, y, label)
    for the nodes-near-coords text used by the bar charts."""
    out = [f"({p[0]},{p[1]})" + (f"[{p[2]}]" if len(p) == 3 else "") for p in pts]
    return "coordinates {" + " ".join(out) + "}"


def texrow(cells):
    return " & ".join(cells) + r" \\"


def tikz(*parts):
    return ["\\begin{tikzpicture}", *parts, "\\end{tikzpicture}"]


# --- verbatim scaffolding from neurips_2026.tex / appendix_extra_data.tex ----
SPINE_GROUPPLOT = r"""\begin{groupplot}[
    group style={group size=4 by 1, horizontal sep=0.4cm,
        ylabels at=edge left, yticklabels at=edge left},
    width=3.5cm, height=4.4cm,
    ymin=0, ymax=1.3, ytick={0,0.2,0.4,0.6,0.8,1}, yticklabels={0,0.2,0.4,0.6,0.8,1},
    symbolic x coords={M1,M2-unb}, xtick=data,
    xticklabels={Task\\(M1), Bound.\\(M2-unb)},
    tick label style={font=\footnotesize, align=center}, label style={font=\small},
    title style={font=\small\bfseries},
    ymajorgrids, grid style={gray!12},
    axis lines*=left, axis line style={gray!50},
    enlarge x limits=0.5, ybar,
    every axis/.append style={bar width=10pt},
    every axis plot/.append style={draw=none},
    nodes near coords, point meta=explicit symbolic,
    nodes near coords style={font=\tiny, rotate=90, anchor=west, xshift=1.5pt,
        /pgf/number format/fixed, /pgf/number format/precision=2,
        /pgf/number format/zerofill},
    ]"""

DEFGRID_GROUPPLOT = r"""\begin{groupplot}[
    group style={group size=4 by 1, horizontal sep=0.4cm,
        ylabels at=edge left, yticklabels at=edge left},
    width=3.6cm, height=4.4cm,
    ymin=0, ymax=1.3, ytick={0,0.2,0.4,0.6,0.8,1}, yticklabels={0,0.2,0.4,0.6,0.8,1},
    symbolic x coords={N,PN,DFT}, xtick=data,
    tick label style={font=\footnotesize}, label style={font=\small},
    title style={font=\small\bfseries},
    ymajorgrids, grid style={gray!12},
    axis lines*=left, axis line style={gray!50},
    enlarge x limits=0.28, ybar,
    every axis/.append style={bar width=7pt},
    every axis plot/.append style={draw=none},
    nodes near coords, point meta=explicit symbolic,
    nodes near coords style={font=\tiny, rotate=90, anchor=west, xshift=1.5pt,
        /pgf/number format/fixed, /pgf/number format/precision=2,
        /pgf/number format/zerofill},
    ]"""

CLEAN_DRAW = (r"\draw[vizthree, line width=1pt]" "\n"
              r"    ({rel axis cs:0,0}|-{axis cs:@X@,@V@}) -- ({rel axis cs:1,0}|-{axis cs:@X@,@V@});")
CLEAN_NODE = r"\node[anchor=north west, font=\tiny, text=vizthree] at (rel axis cs:0.04,1.0) {Clean $@V2@$};"

SQS_AXIS = r"""\begin{axis}[
    ybar,
    bar width=11pt,
    width=0.62\linewidth,
    height=5.0cm,
    ymin=0, ymax=1.12, ytick={0,0.2,0.4,0.6,0.8,1},
    ylabel={AUC},
    symbolic x coords={VQ-16,MedITok,UniTok,FSQ},
    xtick=data,
    enlarge x limits=0.15,
    axis lines*=left,
    ymajorgrids,
    grid style={gray!25},
    tick label style={font=\normalsize},
    label style={font=\normalsize},
    legend style={at={(0.5,-0.22)},anchor=north,legend columns=-1,draw=none,font=\normalsize},
    ]"""

FLIPS_GROUPPLOT = r"""\begin{groupplot}[
    group style={group size=2 by 1, horizontal sep=1.8cm},
    width=5.7cm, height=4.2cm,
    symbolic x coords={VQ-16,MedITok,UniTok,FSQ},
    xtick=data,
    x tick label style={rotate=25, anchor=east, font=\small},
    enlarge x limits=0.18,
    axis lines*=left,
    ymajorgrids, grid style={gray!25},
    tick label style={font=\small},
    label style={font=\small},
    title style={font=\small},
    ]"""
FLIPS_P1 = r"\nextgroupplot[title={Token flip rate}, ylabel={Token flip rate (\%)}, ymin=0, ymax=112, bar width=14pt, ybar]"
FLIPS_P2 = (r"\nextgroupplot[title={Probe AUC under attack}, ylabel={Probe AUC (adv)}, ymin=0, ymax=1.12, "
            r"ytick={0,0.2,0.4,0.6,0.8,1}, bar width=14pt, ybar," "\n"
            r"    legend style={at={(-0.15,1.25)},anchor=south,legend columns=-1,draw=none,font=\small}]")
FLIPS_DIAMOND = (r"\addplot[only marks, mark=diamond*, mark size=2.5pt, color=gray," "\n"
                 r"    legend image code/.code={\node[diamond, fill=gray, scale=0.55] at (0.3cm,0cm) {};}]" "\n"
                 r"    ")

SWEEP_GROUPPLOT = r"""\begin{groupplot}[
    group style={group size=2 by 1, horizontal sep=2.0cm},
    width=6.2cm, height=4.8cm,
    xmode=log, log basis x=2,
    xtick={0.5,1,2,4,8}, xticklabels={0.5,1,2,4,8},
    ymin=0, ymax=1.05, ytick={0,0.2,0.4,0.6,0.8,1},
    axis lines*=left,
    ymajorgrids, grid style={gray!25},
    tick label style={font=\normalsize},
    label style={font=\normalsize},
    title style={font=\normalsize},
    xlabel={$\ell_\infty$ budget $\epsilon$ (/255)},
    ylabel={AUC},
    legend style={at={(1.05,-0.30)},anchor=north,legend columns=-1,draw=none,font=\normalsize},
    ]"""

PARETO_GROUPPLOT = r"""\begin{groupplot}[
    group style={group size=2 by 2, horizontal sep=1.3cm, vertical sep=1.5cm},
    width=4.8cm, height=3.8cm,
    xlabel={Clean recon.\ AUC}, ylabel={Adv.\ recon.\ AUC},
    label style={font=\small}, tick label style={font=\small},
    title style={font=\small},
    ymajorgrids, grid style={gray!25},
    ]"""

# per-panel axis limits, verbatim from appendix_extra_data.tex
PARETO_LIMITS = {
    "vq16": "xmin=0.895, xmax=0.965, ymin=0.58, ymax=0.945",
    "meditok": "xmin=0.895, xmax=0.960, ymin=0.65, ymax=0.925",
    "unitok": "xmin=0.905, xmax=0.960, ymin=0.795, ymax=0.930",
    "fsq": "xmin=0.70, xmax=0.97, ymin=0.70, ymax=0.925",
}
# per-panel point labels and leader lines, verbatim from appendix_extra_data.tex
# (hand-tuned anti-overlap placement; the marker coordinates and the circle/star
#  split are regenerated from the data)
PARETO_LABELS = {
    "vq16": r"""\node[anchor=north west, font=\small] at (axis cs:0.942,0.760) {1};
\node[anchor=south west, font=\small] at (axis cs:0.938,0.766) {2};
\node[anchor=south, font=\small] at (axis cs:0.914,0.660) {3};
\node[anchor=north east, font=\small] at (axis cs:0.933,0.740) {4};
\node[anchor=north, font=\small] at (axis cs:0.911,0.634) {5};
\node[anchor=south east, font=\small] at (axis cs:0.944,0.890) {7};
\node[anchor=south west, font=\small] at (axis cs:0.951,0.895) {6};""",
    "meditok": r"""\node[anchor=north west, font=\small] at (axis cs:0.947,0.697) {1};
\node[anchor=south east, font=\small] at (axis cs:0.919,0.889) {2};
\draw[gray, line width=1pt] (axis cs:0.942,0.719) -- (axis cs:0.954,0.700);
\node[anchor=west, font=\small] at (axis cs:0.954,0.700) {3};
\draw[gray, line width=1pt] (axis cs:0.939,0.720) -- (axis cs:0.925,0.703);
\node[anchor=east, font=\small] at (axis cs:0.925,0.703) {4};
\node[anchor=south west, font=\small] at (axis cs:0.944,0.732) {5};
\draw[gray, line width=1pt] (axis cs:0.937,0.869) -- (axis cs:0.920,0.847);
\node[anchor=east, font=\small] at (axis cs:0.920,0.847) {6};
\draw[gray, line width=1pt] (axis cs:0.936,0.870) -- (axis cs:0.950,0.853);
\node[anchor=west, font=\small] at (axis cs:0.950,0.853) {7};""",
    "unitok": r"""\node[anchor=south, font=\small] at (axis cs:0.937,0.856) {1};
\node[anchor=north east, font=\small] at (axis cs:0.924,0.902) {2};
\node[anchor=north west, font=\small] at (axis cs:0.945,0.843) {3};
\node[anchor=south east, font=\small] at (axis cs:0.933,0.847) {4};
\node[anchor=south, font=\small] at (axis cs:0.942,0.840) {5};
\node[anchor=west, font=\small] at (axis cs:0.943,0.890) {6};
\node[anchor=south west, font=\small] at (axis cs:0.938,0.900) {7};""",
    "fsq": r"""\node[anchor=south east, font=\small] at (axis cs:0.831,0.775) {1};
\node[anchor=north west, font=\small] at (axis cs:0.838,0.773) {2};
\node[anchor=south west, font=\small] at (axis cs:0.837,0.761) {3};
\node[anchor=south east, font=\small] at (axis cs:0.778,0.756) {4};
\node[anchor=north west, font=\small] at (axis cs:0.784,0.747) {5};
\node[anchor=south east, font=\small] at (axis cs:0.930,0.880) {6};
\node[anchor=north west, font=\small] at (axis cs:0.865,0.841) {7};""",
}

# table headers, verbatim; @BODY@ is replaced with the generated rows
TAB_MAIN_TPL = r"""\begin{tabular}{llccc}
\toprule
Tokenizer & Attack & AUC$_{\text{recon}}\downarrow$ & AUC$_{\text{probe}}\downarrow$ & Flip\%$\uparrow$ \\
\midrule
@BODY@
\bottomrule
\end{tabular}"""

TAB_DEFENSE_TPL = r"""\begin{tabular}{llccccccc}
\toprule
Tokenizer & Path & Clean & None & Pixel-noise & SQS & Encoder-FT & Decoder-FT & +Decoder-FT \\
\midrule
@BODY@
\bottomrule
\end{tabular}"""

TAB_FLIP_TPL = r"""\begin{tabular}{lccccccc}
\toprule
\multirow{2}{*}{Tokenizer} & \multirow{2}{*}{Clean AUC$_p$} & \multicolumn{2}{c}{Flip rate (\%)} & \multicolumn{2}{c}{Emb.\ displ.} & \multicolumn{2}{c}{Probe AUC (adv)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
 & & M2-sat & M2-unb & M2-sat & M2-unb & M2-sat & M2-unb \\
\midrule
@BODY@
\bottomrule
\end{tabular}"""

TAB_DISP_TPL = r"""\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccccccccc}
\toprule
& \multirow{2}{*}{Tokenizer} & \multicolumn{2}{c}{$\nicefrac{0.5}{255}$} & \multicolumn{2}{c}{$\nicefrac{1}{255}$} & \multicolumn{2}{c}{$\nicefrac{2}{255}$} & \multicolumn{2}{c}{$\nicefrac{4}{255}$} & \multicolumn{2}{c}{$\nicefrac{8}{255}$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
& & M2-sat & M2-unb & M2-sat & M2-unb & M2-sat & M2-unb & M2-sat & M2-unb & M2-sat & M2-unb \\
\midrule
@BODY@
\bottomrule
\end{tabular}%
}"""

TAB_GRID_TPL = r"""\begin{tabular}{llcccccc}
\toprule
& Tokenizer & clean & $\nicefrac{0.5}{255}$ & $\nicefrac{1}{255}$ & $\nicefrac{2}{255}$ & $\nicefrac{4}{255}$ & $\nicefrac{8}{255}$ \\
\midrule
@BODY@
\bottomrule
\end{tabular}"""


# --- attack-driven floats -------------------------------------------------
def fig_spine(att, clean, out):
    """fig:spine -- reconstruction- vs probe-path AUC under M1 and M2-unb, eps=2/255."""
    eps, attacks = 2.0, ["M1", "M2-unb"]
    rows, body = [], []
    for i, tok in enumerate(TOKS):
        rec, prb = [], []
        for a in attacks:
            r = amean(att, tok, a, eps, "auc_recon_adv")
            p = amean(att, tok, a, eps, "auc_probe_adv")
            rows.append(dict(tokenizer=TOK_LABEL[tok], attack=a,
                             auc_recon_adv=r, auc_probe_adv=p, clean_recon_auc=clean[tok, "recon"]))
            rec.append((a, f"{r:.3f}", f"{r:.2f}"))
            prb.append((a, f"{p:.3f}", f"{p:.2f}"))
        cv = clean[tok, "recon"]
        head = f"\\nextgroupplot[title={{{TOK_SHORT[tok]}}}, ylabel={{AUC}}]" if i == 0 \
            else f"\\nextgroupplot[title={{{TOK_SHORT[tok]}}}]"
        body += [head,
                 r"\addplot[fill=vizone] " + coords(rec) + ";",
                 r"\addplot[fill=vizmauve] " + coords(prb) + ";",
                 CLEAN_DRAW.replace("@X@", "M1").replace("@V@", f"{cv:.3f}"),
                 CLEAN_NODE.replace("@V2@", f"{cv:.2f}")]
    write_csv(pd.DataFrame(rows), f"{out}/fig_spine.csv")
    write_tex(tikz(SPINE_GROUPPLOT, *body, r"\end{groupplot}"), f"{out}/fig_spine.tex")


def fig_flips(att, out):
    """fig:flips -- token flip rate and probe AUC under M2-sat vs M2-unb, eps=4/255."""
    eps, attacks = 4.0, ["M2-sat", "M2-unb"]
    rows = []
    for tok in TOKS:
        for a in attacks:
            rows.append(dict(
                tokenizer=TOK_SHORT[tok], attack=a,
                token_flip_rate_pct=amean(att, tok, a, eps, "token_flip_rate_pct"),
                auc_probe_adv=amean(att, tok, a, eps, "auc_probe_adv"),
                clean_probe_auc=amean(att, tok, a, eps, "auc_probe_clean")))
    write_csv(pd.DataFrame(rows), f"{out}/fig_flips.csv")

    def bars(col, fmt):
        out_lines = []
        for a, fill, shift in [(attacks[0], "viztwo", "-8pt"), (attacks[1], "vizthree", "8pt")]:
            pts = [(TOK_SHORT[t], format(amean(att, t, a, eps, col), fmt)) for t in TOKS]
            out_lines.append(f"\\addplot[fill={fill}, draw=none, barlegend={fill}, bar shift={shift}] " + coords(pts) + ";")
        return out_lines

    diamond = [(TOK_SHORT[t], f"{amean(att, t, 'M2-unb', eps, 'auc_probe_clean'):.3f}") for t in TOKS]
    body = [FLIPS_P1, *bars("token_flip_rate_pct", ".2f"),
            FLIPS_P2, *bars("auc_probe_adv", ".3f"),
            FLIPS_DIAMOND + coords(diamond) + ";",
            r"\legend{M2-sat,M2-unb,Clean}"]
    write_tex(tikz(FLIPS_GROUPPLOT, *body, r"\end{groupplot}"), f"{out}/fig_flips.tex")


def fig_sweep(att, clean, out):
    """fig:sweep -- M2-unb adversarial AUC vs eps on the probe and recon paths."""
    a = "M2-unb"
    rows = []
    for tok in TOKS:
        for eps in EPS_GRID:
            rows.append(dict(
                tokenizer=TOK_SHORT[tok], eps255=eps,
                auc_probe_adv=amean(att, tok, a, eps, "auc_probe_adv"),
                auc_recon_adv=amean(att, tok, a, eps, "auc_recon_adv"),
                clean_probe_auc=clean[tok, "probe"], clean_recon_auc=clean[tok, "recon"]))
    write_csv(pd.DataFrame(rows), f"{out}/fig_sweep.csv")

    body = []
    for title, col, path in [("Token (probe) path", "auc_probe_adv", "probe"),
                             ("Reconstruction path", "auc_recon_adv", "recon")]:
        body.append(f"\\nextgroupplot[title={{{title}}}]")
        for tok, (color, mark, ms) in zip(TOKS, SWEEP_STYLE):
            pts = [(f"{eps:g}", f"{amean(att, tok, a, eps, col):.3f}") for eps in EPS_GRID]
            body.append(f"\\addplot[{color}, line width=1.5pt, mark={mark}, mark size={ms}] " + coords(pts) + ";")
        for tok, (color, _, _) in zip(TOKS, SWEEP_STYLE):
            cv = f"{clean[tok, path]:.3f}"
            body.append(f"\\draw[{color}, dashed, line width=1.5pt] (axis cs:0.5,{cv}) -- (axis cs:8,{cv});")
        if path == "probe":
            body.append(r"\legend{VQ-16, MedITok, UniTok, FSQ}")
    write_tex(tikz(SWEEP_GROUPPLOT, *body, r"\end{groupplot}"), f"{out}/fig_sweep.tex")


def tab_main(att, out):
    """tab:main -- reconstruction/probe adversarial AUC and flip rate, eps=2/255."""
    eps = 2.0
    label = {"M1": "M1 (supervised)", "M2-sat": "M2-sat (geom., bounded)",
             "M2-unb": "M2-unb (geom., unbounded)"}
    attacks = list(label)
    rows, body = [], []
    for tok in TOKS:
        for i, a in enumerate(attacks):
            rec = amean(att, tok, a, eps, "auc_recon_adv")
            prb = amean(att, tok, a, eps, "auc_probe_adv")
            flip = amean(att, tok, a, eps, "token_flip_rate_pct")
            rows.append(dict(tokenizer=TOK_LABEL[tok], attack=a,
                             auc_recon=rec, auc_probe=prb, flip_pct=flip))
            head = rf"\multirow{{3}}{{*}}{{{TOK_LABEL[tok]}}}" if i == 0 else ""
            body.append(texrow([head, label[a], f"{rec:.3f}", f"{prb:.3f}", f"{flip:.1f}"]))
        if tok != TOKS[-1]:
            body.append(r"\midrule")
    write_csv(pd.DataFrame(rows), f"{out}/tab_main.csv")
    write_tex([TAB_MAIN_TPL.replace("@BODY@", "\n".join(body))], f"{out}/tab_main.tex")


def tab_flip_vs_damage(att, out):
    """tab:flip_vs_damage -- matched flip rate, unmatched displacement and harm, eps=2/255."""
    eps = 2.0
    rows, body = [], []
    for tok in TOKS:
        cp = amean(att, tok, "M2-unb", eps, "auc_probe_clean")
        fs = amean(att, tok, "M2-sat", eps, "token_flip_rate_pct")
        fu = amean(att, tok, "M2-unb", eps, "token_flip_rate_pct")
        ds = amean(att, tok, "M2-sat", eps, "emb_disp_pure_mean")
        du = amean(att, tok, "M2-unb", eps, "emb_disp_pure_mean")
        ps = amean(att, tok, "M2-sat", eps, "auc_probe_adv")
        pu = amean(att, tok, "M2-unb", eps, "auc_probe_adv")
        rows.append(dict(tokenizer=TOK_LABEL[tok], clean_probe_auc=cp,
                         flip_pct_sat=fs, flip_pct_unb=fu, emb_disp_sat=ds, emb_disp_unb=du,
                         probe_auc_sat=ps, probe_auc_unb=pu))
        body.append(texrow([TOK_LABEL[tok], f"{cp:.3f}", f"{fs:.2f}", f"{fu:.2f}",
                            f"{ds:.2f}", f"{du:.2f}", f"{ps:.3f}", f"{pu:.3f}"]))
    write_csv(pd.DataFrame(rows), f"{out}/tab_flip_vs_damage.csv")
    write_tex([TAB_FLIP_TPL.replace("@BODY@", "\n".join(body))], f"{out}/tab_flip_vs_damage.tex")


def per_dataset_grid(att, out, name, template, attacks, value_col, clean_col, prec):
    """tab:displacement_all / tab:probe_m3unb_all / tab:recon_m3unb_all -- one row
    per (dataset, tokenizer): an optional clean value then the eps grid. When
    there are two attacks the columns are grouped by eps then attack
    (eps0.5: M2-sat, M2-unb; eps1: ...), matching the paper header. Values are
    read straight from the per-dataset CSV, no averaging."""
    def key(a, eps):
        return f"{a}_eps{eps:g}" if len(attacks) > 1 else f"eps{eps:g}"

    order = [(eps, a) for eps in EPS_GRID for a in attacks]  # eps-major, then attack

    rows = []
    for ds in DATASETS:
        for tok in TOKS:
            sub = att[(att.tokenizer == tok) & (att.dataset == ds)]
            r = dict(dataset=DS_LABEL[ds], tokenizer=TOK_LABEL[tok])
            if clean_col:
                r["clean"] = float(sub[sub.attack == attacks[0]][clean_col].iloc[0])
            for eps, a in order:
                s = sub[sub.attack == a].set_index("eps255")[value_col]
                r[key(a, eps)] = float(s.loc[eps])
            rows.append(r)
    df = pd.DataFrame(rows)
    write_csv(df, f"{out}/{name}.csv")

    body = []
    for ds in DATASETS:
        for i, tok in enumerate(TOKS):
            r = df[(df.dataset == DS_LABEL[ds]) & (df.tokenizer == TOK_LABEL[tok])].iloc[0]
            head = rf"\multirow{{4}}{{*}}{{{DS_LABEL[ds]}}}" if i == 0 else ""
            cells = [head, TOK_LABEL[tok]]
            if clean_col:
                cells.append(f"{r['clean']:.{prec}f}")
            cells += [f"{r[key(a, eps)]:.{prec}f}" for eps, a in order]
            body.append(texrow(cells))
        if ds != DATASETS[-1]:
            body.append(r"\midrule")
    write_tex([template.replace("@BODY@", "\n".join(body))], f"{out}/{name}.tex")


# --- defense-driven floats ----------------------------------------------
def tab_defense_recon_all(dfn, out):
    """tab:defense_recon_all -- reconstruction- vs probe-path AUC per defense,
    M2-unb at eps=8/255. Omits the Encoder-FT+SQS column."""
    cols = [c for c in DEFENSE_COLS if c[0] != "advft+best_sqs"]
    rows, body = [], []
    for tok in TOKS:
        for i, (path, adv, cl) in enumerate([("Recon", "auc_recon_adv", "auc_recon_clean"),
                                             ("Probe", "auc_probe_adv", "auc_probe_clean")]):
            clean = dmean(dfn, tok, "none", "M2-unb", 8.0, cl)
            adv_vals = [dmean(dfn, tok, d, "M2-unb", 8.0, adv) for d, _ in cols]
            rows.append(dict(tokenizer=TOK_LABEL[tok], path=path, clean=clean,
                             **{lab: v for (_, lab), v in zip(cols, adv_vals)}))
            head = rf"\multirow{{2}}{{*}}{{{TOK_LABEL[tok]}}}" if i == 0 else ""
            body.append(texrow([head, path, f"{clean:.3f}"] + [f"{v:.3f}" for v in adv_vals]))
        if tok != TOKS[-1]:
            body.append(r"\midrule")
    write_csv(pd.DataFrame(rows), f"{out}/tab_defense_recon_all.csv")
    write_tex([TAB_DEFENSE_TPL.replace("@BODY@", "\n".join(body))], f"{out}/tab_defense_recon_all.tex")


def fig_defense_grid(dfn, out):
    """fig:defense-grid -- None (N) / Pixel-noise (PN) / Decoder-FT (DFT), both
    paths, M2-unb at eps=8/255."""
    cols = [("none", "N"), ("pixel_noise", "PN"), ("decoder_realism", "DFT")]
    rows, body = [], []
    for i, tok in enumerate(TOKS):
        cv = dmean(dfn, tok, "none", "M2-unb", 8.0, "auc_recon_clean")
        rec, prb = [], []
        for d, tag in cols:
            r = dmean(dfn, tok, d, "M2-unb", 8.0, "auc_recon_adv")
            p = dmean(dfn, tok, d, "M2-unb", 8.0, "auc_probe_adv")
            rows.append(dict(tokenizer=TOK_SHORT[tok], defense=tag,
                             auc_recon_adv=r, auc_probe_adv=p, clean_recon_auc=cv))
            rec.append((tag, f"{r:.3f}", f"{r:.2f}"))
            prb.append((tag, f"{p:.3f}", f"{p:.2f}"))
        head = f"\\nextgroupplot[title={{{TOK_SHORT[tok]}}}, ylabel={{AUC}}]" if i == 0 \
            else f"\\nextgroupplot[title={{{TOK_SHORT[tok]}}}]"
        body += [head,
                 r"\addplot[fill=vizone, bar shift=-4pt] " + coords(rec) + ";",
                 r"\addplot[fill=vizmauve, bar shift=4pt] " + coords(prb) + ";",
                 CLEAN_DRAW.replace("@X@", "N").replace("@V@", f"{cv:.3f}"),
                 CLEAN_NODE.replace("@V2@", f"{cv:.2f}")]
    write_csv(pd.DataFrame(rows), f"{out}/fig_defense_grid.csv")
    write_tex(tikz(DEFGRID_GROUPPLOT, *body, r"\end{groupplot}"), f"{out}/fig_defense_grid.tex")


def fig_sqs_attack_dependence(inputs, dfn, out):
    """fig:sqs-attack-dependence -- SQS reconstruction AUC by attack, eps=8/255.

    The M1 / M2-sat / M2-unb bars are the SQS selection sweep's winning row per
    dataset (auc_adv_*), averaged over datasets -- not the best_sqs rows in
    defense_table.csv, which are a later re-evaluation on a different split/seed
    and read 0.02-0.15 AUC lower. The sweep scores SQS against each attack at
    eps=8/255, matching the caption. The clean diamonds are the undefended clean
    reconstruction AUC from defense_table.csv."""
    bars = [("auc_adv_M1", "vizone", "M1 (supervised)"),
            ("auc_adv_M2-sat", "viztwo", "M2-sat (geometric)"),
            ("auc_adv_M2-unb", "vizthree", "M2-unb (geometric)")]
    rows, body = [], []
    for tok in TOKS:
        r = dict(tokenizer=TOK_SHORT[tok],
                 clean_undefended_auc=dmean(dfn, tok, "none", "M2-unb", 8.0, "auc_recon_clean"))
        for col, _, lab in bars:
            r[lab] = sqs_winner_bar(inputs, tok, col)
        rows.append(r)
    for col, fill, _ in bars:
        pts = [(TOK_SHORT[t], f"{sqs_winner_bar(inputs, t, col):.3f}") for t in TOKS]
        body.append(f"\\addplot[fill={fill}, draw=none] " + coords(pts) + ";")
    clean = [(TOK_SHORT[t], f"{dmean(dfn, t, 'none', 'M2-unb', 8.0, 'auc_recon_clean'):.3f}") for t in TOKS]
    body += [r"\addplot[only marks, mark=diamond*, mark size=3pt, color=gray]" "\n    " + coords(clean) + ";",
             r"\legend{M1 (supervised), M2-sat (geometric), M2-unb (geometric), Clean (undefended)}"]
    write_csv(pd.DataFrame(rows), f"{out}/fig_sqs_attack_dependence.csv")
    write_tex(tikz(SQS_AXIS, *body, r"\end{axis}"), f"{out}/fig_sqs_attack_dependence.tex")


def fig_defense_pareto(dfn, out):
    """fig:defense-pareto -- clean vs adversarial reconstruction AUC, one point
    per defense, M2-unb at eps=8/255.

    Axis limits and point labels/leader lines are the paper's verbatim
    (hand-tuned placement); the marker coordinates and the circle-vs-star split
    are regenerated. Stars mark the globally Pareto-optimal points, computed here
    across all four panels -- the result matches the paper's three starred points
    (VQ-16/Decoder-FT, UniTok/Pixel-noise, UniTok/Encoder-FT+Decoder-FT)."""
    pts = {}  # (tok, idx) -> (clean, adv)
    rows = []
    for tok in TOKS:
        for i, (d, lab) in enumerate(DEFENSE_COLS, start=1):
            x = dmean(dfn, tok, d, "M2-unb", 8.0, "auc_recon_clean")
            y = dmean(dfn, tok, d, "M2-unb", 8.0, "auc_recon_adv")
            pts[tok, i] = (x, y)
            rows.append(dict(tokenizer=TOK_SHORT[tok], idx=i, defense=lab,
                             clean_recon_auc=x, adv_recon_auc=y))
    every = list(pts.values())
    star = {k for k, (x, y) in pts.items()
            if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for x2, y2 in every)}

    body = []
    for tok in TOKS:
        circ = [(f"{pts[tok, i][0]:.3f}", f"{pts[tok, i][1]:.3f}")
                for i in range(1, 8) if (tok, i) not in star]
        stars = [(f"{pts[tok, i][0]:.3f}", f"{pts[tok, i][1]:.3f}")
                 for i in range(1, 8) if (tok, i) in star]
        body.append(f"\\nextgroupplot[title={{{TOK_SHORT[tok]}}}, {PARETO_LIMITS[tok]}, only marks]")
        body.append(r"\addplot[mark=*, mark size=2pt, vizone] " + coords(circ) + ";")
        if stars:
            body.append(r"\addplot[mark=star, mark size=4pt, viztwo] " + coords(stars) + ";")
        body.append(PARETO_LABELS[tok])
    write_csv(pd.DataFrame(rows), f"{out}/fig_defense_pareto.csv")
    write_tex(tikz(PARETO_GROUPPLOT, *body, r"\end{groupplot}"), f"{out}/fig_defense_pareto.tex")


NAMES = ["fig_spine", "fig_flips", "fig_sweep", "tab_main", "tab_flip_vs_damage",
         "tab_displacement_all", "tab_probe_m3unb_all", "tab_recon_m3unb_all",
         "tab_defense_recon_all", "fig_defense_grid", "fig_sqs_attack_dependence",
         "fig_defense_pareto"]

# floats whose headline is a mean over the three datasets -> get a _by_dataset.csv,
# and the non-dataset key columns to group on when averaging it back
BY_DATASET_KEYS = {
    "fig_spine": ["tokenizer", "attack"],
    "fig_flips": ["tokenizer", "attack"],
    "fig_sweep": ["tokenizer", "eps255"],
    "tab_main": ["tokenizer", "attack"],
    "tab_flip_vs_damage": ["tokenizer"],
    "tab_defense_recon_all": ["tokenizer", "path"],
    "fig_defense_grid": ["tokenizer", "defense"],
    "fig_sqs_attack_dependence": ["tokenizer"],
    "fig_defense_pareto": ["tokenizer", "idx", "defense"],
}
MEAN_FLOATS = list(BY_DATASET_KEYS)


def build_by_dataset(att, dfn, inputs, out):
    """For each mean-over-datasets float, write <name>_by_dataset.csv: the same
    table with a `dataset` column and no averaging, so the headline number is
    visibly the mean of three committed per-dataset rows. Asserts that
    groupby(mean) of the components reproduces the headline .csv."""
    E2, E4, E8, GU = 2.0, 4.0, 8.0, "M2-unb"
    frames = {}

    rows = []
    for tok in TOKS:
        clean = dvals(dfn, tok, "none", "M1", E8, "auc_recon_clean")
        for a in ["M1", GU]:
            r, p = avals(att, tok, a, E2, "auc_recon_adv"), avals(att, tok, a, E2, "auc_probe_adv")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_LABEL[tok], attack=a, dataset=DS_LABEL[ds],
                                 auc_recon_adv=r[ds], auc_probe_adv=p[ds], clean_recon_auc=clean[ds]))
    frames["fig_spine"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        for a in ["M2-sat", GU]:
            f = avals(att, tok, a, E4, "token_flip_rate_pct")
            p = avals(att, tok, a, E4, "auc_probe_adv")
            c = avals(att, tok, a, E4, "auc_probe_clean")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_SHORT[tok], attack=a, dataset=DS_LABEL[ds],
                                 token_flip_rate_pct=f[ds], auc_probe_adv=p[ds], clean_probe_auc=c[ds]))
    frames["fig_flips"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        cp = dvals(dfn, tok, "none", "M1", E8, "auc_probe_clean")
        cr = dvals(dfn, tok, "none", "M1", E8, "auc_recon_clean")
        for eps in EPS_GRID:
            pa = avals(att, tok, GU, eps, "auc_probe_adv")
            ra = avals(att, tok, GU, eps, "auc_recon_adv")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_SHORT[tok], eps255=eps, dataset=DS_LABEL[ds],
                                 auc_probe_adv=pa[ds], auc_recon_adv=ra[ds],
                                 clean_probe_auc=cp[ds], clean_recon_auc=cr[ds]))
    frames["fig_sweep"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        for a in ["M1", "M2-sat", GU]:
            r = avals(att, tok, a, E2, "auc_recon_adv")
            p = avals(att, tok, a, E2, "auc_probe_adv")
            f = avals(att, tok, a, E2, "token_flip_rate_pct")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_LABEL[tok], attack=a, dataset=DS_LABEL[ds],
                                 auc_recon=r[ds], auc_probe=p[ds], flip_pct=f[ds]))
    frames["tab_main"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        cp = avals(att, tok, GU, E2, "auc_probe_clean")
        cols = {f"{m}_{tag}": avals(att, tok, a, E2, c)
                for tag, a in [("sat", "M2-sat"), ("unb", GU)]
                for m, c in [("flip_pct", "token_flip_rate_pct"),
                             ("emb_disp", "emb_disp_pure_mean"), ("probe_auc", "auc_probe_adv")]}
        for ds in DATASETS:
            rows.append(dict(tokenizer=TOK_LABEL[tok], dataset=DS_LABEL[ds],
                             clean_probe_auc=cp[ds], **{k: v[ds] for k, v in cols.items()}))
    frames["tab_flip_vs_damage"] = pd.DataFrame(rows)

    defs = [c for c in DEFENSE_COLS if c[0] != "advft+best_sqs"]
    rows = []
    for tok in TOKS:
        for path, adv, cl in [("Recon", "auc_recon_adv", "auc_recon_clean"),
                              ("Probe", "auc_probe_adv", "auc_probe_clean")]:
            clean = dvals(dfn, tok, "none", GU, E8, cl)
            per = {lab: dvals(dfn, tok, d, GU, E8, adv) for d, lab in defs}
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_LABEL[tok], path=path, dataset=DS_LABEL[ds],
                                 clean=clean[ds], **{lab: per[lab][ds] for _, lab in defs}))
    frames["tab_defense_recon_all"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        clean = dvals(dfn, tok, "none", GU, E8, "auc_recon_clean")
        for d, tag in [("none", "N"), ("pixel_noise", "PN"), ("decoder_realism", "DFT")]:
            r = dvals(dfn, tok, d, GU, E8, "auc_recon_adv")
            p = dvals(dfn, tok, d, GU, E8, "auc_probe_adv")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_SHORT[tok], defense=tag, dataset=DS_LABEL[ds],
                                 auc_recon_adv=r[ds], auc_probe_adv=p[ds], clean_recon_auc=clean[ds]))
    frames["fig_defense_grid"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        c = dvals(dfn, tok, "none", GU, E8, "auc_recon_clean")
        m1 = sqs_winner_vals(inputs, tok, "auc_adv_M1")
        ms = sqs_winner_vals(inputs, tok, "auc_adv_M2-sat")
        mu = sqs_winner_vals(inputs, tok, "auc_adv_M2-unb")
        for ds in DATASETS:
            rows.append({"tokenizer": TOK_SHORT[tok], "dataset": DS_LABEL[ds],
                         "M1 (supervised)": m1[ds], "M2-sat (geometric)": ms[ds],
                         "M2-unb (geometric)": mu[ds], "clean_undefended_auc": c[ds]})
    frames["fig_sqs_attack_dependence"] = pd.DataFrame(rows)

    rows = []
    for tok in TOKS:
        for i, (d, lab) in enumerate(DEFENSE_COLS, start=1):
            x = dvals(dfn, tok, d, GU, E8, "auc_recon_clean")
            y = dvals(dfn, tok, d, GU, E8, "auc_recon_adv")
            for ds in DATASETS:
                rows.append(dict(tokenizer=TOK_SHORT[tok], idx=i, defense=lab, dataset=DS_LABEL[ds],
                                 clean_recon_auc=x[ds], adv_recon_auc=y[ds]))
    frames["fig_defense_pareto"] = pd.DataFrame(rows)

    for name, comp in frames.items():
        write_csv(comp, f"{out}/{name}_by_dataset.csv")
        keys = BY_DATASET_KEYS[name]
        head = pd.read_csv(f"{out}/{name}.csv")
        vcols = [c for c in head.columns if c not in keys and c != "dataset"]
        got = comp.groupby(keys, sort=False)[vcols].mean().reset_index().merge(
            head[keys + vcols], on=keys, suffixes=("_c", "_h"))
        for c in vcols:
            if not np.allclose(got[f"{c}_c"], got[f"{c}_h"], atol=1e-3):
                raise SystemExit(f"{name}: {name}_by_dataset.csv mean of '{c}' != headline {name}.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", default="results/paper_inputs",
                   help="dir holding attack-results/ and defense-results/ (vendored pipeline CSVs)")
    p.add_argument("--out-dir", default="results/paper_figures")
    p.add_argument("--no-pdf", action="store_true", help="write .csv and .tex only, skip pdflatex")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    att = load_attack(args.inputs)
    dfn = load_defense(args.inputs)
    clean = report_clean(dfn)

    fig_spine(att, clean, args.out_dir)
    fig_flips(att, args.out_dir)
    fig_sweep(att, clean, args.out_dir)
    tab_main(att, args.out_dir)
    tab_flip_vs_damage(att, args.out_dir)
    per_dataset_grid(att, args.out_dir, "tab_displacement_all", TAB_DISP_TPL,
                     ["M2-sat", "M2-unb"], "emb_disp_pure_mean", None, 2)
    per_dataset_grid(att, args.out_dir, "tab_probe_m3unb_all", TAB_GRID_TPL,
                     ["M2-unb"], "auc_probe_adv", "auc_probe_clean", 3)
    per_dataset_grid(att, args.out_dir, "tab_recon_m3unb_all", TAB_GRID_TPL,
                     ["M2-unb"], "auc_recon_adv", "auc_recon_clean", 3)
    tab_defense_recon_all(dfn, args.out_dir)
    fig_defense_grid(dfn, args.out_dir)
    fig_sqs_attack_dependence(args.inputs, dfn, args.out_dir)
    fig_defense_pareto(dfn, args.out_dir)

    build_by_dataset(att, dfn, args.inputs, args.out_dir)

    if args.no_pdf:
        return
    if shutil.which("pdflatex") is None:
        print("pdflatex not found -- skipping PDF compilation (rerun without --no-pdf once TeX is installed)")
        return
    for name in NAMES:
        compile_pdf(name, args.out_dir)
    shutil.rmtree(os.path.join(args.out_dir, "_latex"), ignore_errors=True)


if __name__ == "__main__":
    main()
