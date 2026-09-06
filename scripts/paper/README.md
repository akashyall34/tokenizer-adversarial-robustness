# `scripts/paper/`

Regenerate the workshop paper's data figures and results tables from the pipeline
CSVs, so every number in the paper traces back to run output.

```bash
python scripts/paper/build_paper_figures.py    # -> results/paper_figures/<name>.{csv,tex,pdf}
python scripts/paper/check_paper_figures.py     # recompute headline cells, compare to the paper
```

`build_paper_figures.py` reads `results/paper_inputs/` (pipeline CSVs vendored
from the paper's `attack-results/` and `defense-results/`) and writes four files
per float:

- `<name>.csv`: the plotted or tabulated numbers.
- `<name>.tex`: the complete `tikzpicture` or `tabular`, ready to drop into the
  paper. The scaffolding (groupplot option blocks, table headers, the
  `vizone`..`vizmauve` colours) is copied from the paper source; only the data
  coordinates come from the script.
- `<name>.pdf`: that `.tex` compiled standalone with `pdflatex` and the shared
  `figure_preamble.tex`. `--no-pdf` skips this step. A missing `pdflatex` is
  reported and skipped rather than treated as an error.
- `<name>_by_dataset.csv`: for floats whose headline is a mean over the three
  datasets, the same table with a `dataset` column and no averaging, so each
  paper number is visibly `mean(Kermany, APTOS, ISIC)` of three committed rows.
  The build asserts that `groupby(...).mean()` of this file reproduces
  `<name>.csv`, and `check_paper_figures.py` re-checks it.

`check_paper_figures.py` recomputes the headline cells against values transcribed
from the paper source and exits non-zero on any drift past `--tol` (default 0.012
AUC). It does CSV aggregation only, with no model, GPU, or checkpoint.

For `fig_defense_pareto` the axis limits and the point labels and leader lines
are taken from the paper (the placement is hand-tuned); the marker coordinates
and the circle-vs-star split are regenerated. Stars mark the globally
Pareto-optimal points across all four panels, which matches the paper's three
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
| `tab:gen_fidelity_appendix_attack` | `tab_gen_fidelity_appendix_attack` | `generative-fidelity/<tok>_<ds>.csv` | *per dataset*, LPIPS/PSNR vs original, M2-sat & M2-unb, full ε grid |
| `tab:gen_fidelity_defense_appendix` | `tab_gen_fidelity_defense_appendix` | `generative-fidelity/` (attack grid + `_defense` + `_clean_baseline`) | *per dataset*, LPIPS/PSNR/SSIM, M2-unb `ε=8`; Encoder-FT = the `ε=8` fine-tune checkpoint |
| `tab:vqa-eps` | `tab_vqa_eps` | `vqa/vqa_sweep_seeded_m2{sat,unb}.csv` | pooled flip/disp/Correct% over the `n=99` clean-correct modality & body-region cells |
| `tab:vqa-full` | `tab_vqa_full` | same | answer accuracy by (dataset, question), `n=20` |
| `tab:vqa-direction` | `tab_vqa_direction` | same | error-direction counts under M2-unb |

VQA "Correct%" comes from `is_correct` in `src/generative/vqa_scoring.py`. For
the modality and body-region questions it reads the first sentence of the answer
and takes whichever of the expected or distractor vocabulary appears first; the
vocabulary lists are enumerated from the distinct answer openings in the data.
`check_paper_figures.py` validates `tab:vqa-eps` and `tab:vqa-direction` tightly.
Two `tab:vqa-full` cells (Kermany and ISIC modality, M2-unb at `eps=2`) land one
to two images off the printed values and are checked at ±12 pp; these sit on the
boundary where the smallest budget flips an answer, and the effect is the
decode variance described in `docs/known_limitations.md`.

Not covered: `fig:threat-model` and the pipeline-overview diagram (hand-drawn,
no data), and `tab:vqa-progression` (the five verbatim hero-image answers in
Figure `fig:vqa`, transcribed by hand from one run).

## Names

Attack ids match the paper: `M0`, `M1`, `M2-sat`, `M2-unb`. `M0` and
`M3_alpha03` (the auxiliary alpha=0.3 blend, not used in the paper) are not
plotted.

Defense ids (`defense_table.csv` column -> paper column): `none`, `pixel_noise`,
`best_sqs`, `advft` (Encoder-FT), `advft+best_sqs`, `decoder_realism`
(Decoder-FT), `combined` (Encoder-FT+Decoder-FT). `advft` and `advft+best_sqs`
carry a finetune-budget sweep and are read at `finetune_eps255 = 8`.

## `fig:sqs-attack-dependence`

Its M1 / M2-sat / M2-unb bars are the winning row per dataset from the SQS
**selection sweep** (`<tok>_sqs_k_sigma_grid.csv`, or `<tok>_sqs_flat_sweep.csv`
for FSQ, which has no K x sigma grid), averaged over datasets. They are not the
`best_sqs` rows in `defense_table.csv`, which come from a later re-evaluation on
a different split and seed and read 0.02 to 0.15 AUC lower. The sweep scores SQS
against each attack at `eps=8/255`, matching the caption. The clean diamonds are
the undefended clean reconstruction AUC from `defense_table.csv`.

## Averages over the three datasets

Every dataset-averaged number in the paper is backed by three committed
per-dataset CSVs in `results/paper_inputs/` (Kermany, APTOS, and ISIC, not
Kermany alone). `build_paper_figures.py` averages them, the
`<name>_by_dataset.csv` artifacts expose the three components, and both the build
and `check_paper_figures.py` assert that their mean equals the headline number.
Re-deriving the per-dataset CSVs from raw images is a separate step that needs
the datasets, a GPU, and `run_pipeline.py`; see the repo-root README and
`docs/known_limitations.md`.

## Inputs

`results/paper_inputs/` holds the CSVs the scripts read: the 12
`*_attack_table_<dataset>.csv`, the 12
`<tok>_defense_outputs_<ds>/<tok>_defense_table.csv`, the SQS selection sweeps,
the `generative-fidelity/` CSVs (LPIPS/PSNR/SSIM/rFID per
`<tok>_<dataset>[ _defense | _clean_baseline | _rfid_pooled ]`), and the raw
LLaVA-Med VQA answers under `vqa/`. All are vendored from run output and
committed (`results/` is otherwise gitignored). The `.csv` / `.tex` / `.pdf`
output under `results/paper_figures/` is not committed; `figure_preamble.tex` is.
