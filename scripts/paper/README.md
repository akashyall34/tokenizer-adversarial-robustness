# `scripts/paper/`

Regenerate the workshop paper's data figures and results tables from the pipeline
CSVs, so every number in the paper traces back to the run output.

```bash
python scripts/paper/build_paper_figures.py    # -> results/paper_figures/<name>.{csv,tex,pdf}
python scripts/paper/check_paper_figures.py     # recompute headline cells, compare to the paper
```

`build_paper_figures.py` reads `results/paper_inputs/` (pipeline CSVs vendored
from the paper's `attack-results/` and `defense-results/`) and, per float, writes:

- `<name>.csv` -- the plotted / tabulated numbers.
- `<name>.tex` -- the complete `tikzpicture` / `tabular`, ready to drop into the
  paper. The scaffolding (groupplot option blocks, table headers, the
  `vizone`..`vizmauve` colours) is lifted verbatim from the paper source; only
  the data coordinates come from the script.
- `<name>.pdf` -- that `.tex` compiled standalone with `pdflatex`, using the
  shared `figure_preamble.tex`. `--no-pdf` skips this step; a missing `pdflatex`
  is reported and skipped, not fatal.
- `<name>_by_dataset.csv` -- for the floats whose headline is a mean over the
  three datasets: the same table with a `dataset` column and no averaging, so
  each paper number is visibly `mean(Kermany, APTOS, ISIC)` of three committed
  rows. The build asserts `groupby(...).mean()` of this file reproduces
  `<name>.csv`; `check_paper_figures.py` re-checks it.

`check_paper_figures.py` recomputes the headline cells against values transcribed
from the paper source and exits non-zero on any drift past `--tol` (default 0.012
AUC). It is CSV aggregation only -- no model, GPU, or checkpoint.

For `fig_defense_pareto` the axis limits and the point labels / leader lines are
the paper's verbatim (hand-tuned placement); the marker coordinates and the
circle-vs-star split are regenerated. Stars mark the globally Pareto-optimal
points, computed across all four panels -- the result matches the paper's three
starred points.

## Float coverage

Outputs are named by the paper's `\label`. Values are a mean over the three
datasets unless the row says *per dataset*.

| `\label` | output | source | notes |
|---|---|---|---|
| `fig:spine` | `fig_spine` | attack tables | `eps255=2`, M1 & M2-unb; clean line from REPORT split |
| `fig:flips` | `fig_flips` | attack tables | `eps255=4`, M2-sat vs M2-unb |
| `fig:sweep` | `fig_sweep` | attack tables | M2-unb, full ε grid; clean lines from REPORT split |
| `tab:main` | `tab_main` | attack tables | `eps255=2` |
| `tab:flip_vs_damage` | `tab_flip_vs_damage` | attack tables | `eps255=2` |
| `tab:displacement_all` | `tab_displacement_all` | attack tables | *per dataset*, full ε grid |
| `tab:probe_m3unb_all` | `tab_probe_m3unb_all` | attack tables | *per dataset*, M2-unb, full ε grid |
| `tab:recon_m3unb_all` | `tab_recon_m3unb_all` | attack tables | *per dataset*, M2-unb, full ε grid |
| `tab:defense_recon_all` | `tab_defense_recon_all` | defense tables | M2-unb, `eval_eps255=8`, finetune eps 8 |
| `fig:defense-grid` | `fig_defense_grid` | defense tables | subset: None / Pixel-noise / Decoder-FT |
| `fig:sqs-attack-dependence` | `fig_sqs_attack_dependence` | SQS sweeps + defense tables | see below |
| `fig:defense-pareto` | `fig_defense_pareto` | defense tables | 7 defenses, clean vs adversarial recon AUC |

Not covered: `fig:threat-model` and the pipeline-overview diagram -- hand-drawn,
no data.

## Names

Attack ids match the paper: `M0`, `M1`, `M2-sat`, `M2-unb`. `M0` and
`M3_alpha03` (the auxiliary alpha=0.3 blend, not used in the paper)
are not plotted.

Defense ids (`defense_table.csv` -> paper column): `none`, `pixel_noise`,
`best_sqs`, `advft` (Encoder-FT), `advft+best_sqs`, `decoder_realism`
(Decoder-FT), `combined` (Encoder-FT+Decoder-FT). `advft` and `advft+best_sqs`
carry a finetune-budget sweep and are read at `finetune_eps255 = 8`.

## `fig:sqs-attack-dependence`

Its M1 / M2-sat / M2-unb bars are the SQS **selection sweep's** winning row per
dataset (`<tok>_sqs_k_sigma_grid.csv`, or `<tok>_sqs_flat_sweep.csv` for FSQ,
which has no K x sigma grid), averaged over datasets -- not the `best_sqs` rows
in `defense_table.csv`, which are a later re-evaluation on a different split/seed
and read 0.02-0.15 AUC lower. The sweep scores SQS against each attack at
`eps=8/255`, matching the caption. The clean diamonds are the undefended clean
reconstruction AUC from `defense_table.csv`.

## Averages over the three datasets

Every dataset-averaged number in the paper is backed by three committed
per-dataset CSVs in `results/paper_inputs/` (Kermany, APTOS, and ISIC -- not
Kermany only). `build_paper_figures.py` averages them; the `<name>_by_dataset.csv`
artifacts expose the three components, and both the build and
`check_paper_figures.py` assert that their mean equals the headline number.
Re-deriving the per-dataset CSVs from raw images is a separate step (datasets +
GPU + `run_pipeline.py`); see the repo-root README and `docs/known_limitations.md`.

## Inputs

`results/paper_inputs/` holds the CSVs the scripts read -- the 12
`*_attack_table_<dataset>.csv`, the 12 `<tok>_defense_outputs_<ds>/<tok>_defense_table.csv`,
and the SQS selection sweeps -- vendored from the paper's `attack-results/` and
`defense-results/` and committed (`results/` is otherwise gitignored).
`results/paper_figures/` (the `.csv` / `.tex` / `.pdf` output) is not committed;
`figure_preamble.tex` is.
