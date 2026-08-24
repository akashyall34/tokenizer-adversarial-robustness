# Adversarial Robustness of Neural Image Tokenizers

Code for studying adversarial attacks and defenses against neural image tokenizers
(VQ-16, FSQ, and the domain-paired ViT-based MediTok/UniTok) placed ahead of a
medical-image classifier.

## Research questions

1. Does a boundary-targeted (Voronoi/grid-wall) attack on the tokenizer's codebook
   beat a global-displacement attack?
2. Does noise injected right before quantization defend better than noise at the
   pixel input?
3. Does token-flip translate to clinically meaningful misclassification, and does
   this generalize across tokenizers and datasets?
4. Are domain-specific tokenizers inherently more robust than general-domain ones?

## Structure

```
src/
  attacks/            attack implementations (per tokenizer): vq16_attacks.py,
                       meditok_unitok_attacks.py, fsq_attacks.py, pgd.py
  defenses/
    common/           generic mechanisms: BPDA, checkpoint merging, CW-L2 attack,
                       decoder-realism training loop -- not tied to any one tokenizer
    vq16/             VQ-16-specific margin/SQS/certification math
    multicodebook/    MediTok/UniTok-specific margin/SQS math (shared codebook geometry)
    fsq/              FSQ-specific margin/SQS/certification math (no codebook --
                       per-dimension scalar quantization, its own geometry throughout)
  tokenizers/         tokenizer loading + reconstruct wrappers (vq16.py, fsq.py,
                       meditok_unitok.py)
  datasets/           dataset loading + splits (kermany.py, aptos.py, isic.py)
  models/             downstream classifier wrappers (DenseNet121, EfficientNet-B4/B5)
  eval/               metrics, probes, bootstrap significance, SEL/REPORT split
scripts/
  attacks/            one orchestration script per (tokenizer, dataset) pair
  defenses/           one directory per (tokenizer, dataset) pair, each holding
                       the full advft/decoder-realism/merge/SQS-sweep/BPDA/geo/CW-L2/
                       certification/final-table pipeline for that pair -- run via
                       run_pipeline.py, not by hand. 4 tokenizers x 3 datasets = 12:
                         vq16_kermany/     vq16_aptos/     vq16_isic/
                         meditok_kermany/  meditok_aptos/  meditok_isic/
                         unitok_kermany/   unitok_aptos/   unitok_isic/
                         fsq_kermany/      fsq_aptos/      fsq_isic/
                       (attack generation stays combined per scripts/attacks/
                       run_meditok_unitok_{dataset}.py -- MediTok and UniTok share one
                       attack-generation pass since the source notebooks do, but the
                       defense notebooks split into separate meditok_adv_finetune_*.ipynb
                       / unitok_adv_finetune_*.ipynb, so the defense scripts split too.)
  diagnostics/        one-off verification/measurement scripts, not part of the main pipeline
  run_pipeline.py      single entry point that runs a named stage in the correct order
  setup_third_party.sh clones LlamaGen/MediTok/UniTok/FSQ-pytorch + fetches their checkpoints
configs/              hyperparameters (eps grid, sigma values, etc.), one YAML per dataset
ckpts/                classifier + linear-probe checkpoints (Git LFS)
docs/                 data acquisition, checkpoint layout, known limitations
```

**Coverage** (tokenizer x dataset): all four tokenizers (VQ-16, MediTok,
UniTok, FSQ) have a full defense pipeline on all three datasets (Kermany,
APTOS, ISIC) -- 12/12. Attack generation stays combined per tokenizer family
(`scripts/attacks/run_meditok_unitok_{dataset}.py` produces both MediTok's
and UniTok's attack images in one pass, matching the source notebooks) --
only the defense pipelines split into separate `meditok_{dataset}/` and
`unitok_{dataset}/` directories, matching the source repo's now-separate
`meditok_adv_finetune_*.ipynb` / `unitok_adv_finetune_*.ipynb` notebooks.

Not every combination was ported from an already-executed notebook -- some
were built by wiring a tokenizer's already-verified, dataset-agnostic
defense math (`src/defenses/{vq16,fsq,multicodebook}/`) to a dataset's
existing loader/classifier, the same code paths every other pipeline already
exercises. Execution status (which notebooks have actually run, how far, and
with what results) changes frequently upstream -- see
`docs/known_limitations.md`'s execution-status table for the current state
rather than treating "code-complete" as "verified."

## Setup

```bash
pip install -e .
scripts/setup_third_party.sh   # clones LlamaGen/MediTok/UniTok/FSQ-pytorch + fetches their checkpoints
```

`torch`/`torchvision` install their default (CUDA) wheels via pip; if you need a
specific CUDA/CPU build, install those two first from the
[PyTorch index](https://pytorch.org/get-started/locally/) before `pip install -e .`.

Datasets (Kermany chest X-ray, APTOS, ISIC 2019) are not included; see
`docs/data.md` for acquisition instructions.

## Pretrained checkpoints

Classifier and linear-probe checkpoints are committed in `ckpts/` via Git LFS.
Install Git LFS before cloning (`git lfs install`) so the real weights come
down instead of pointer files — see `docs/checkpoints.md` for the layout.

## Reproducing results

Run `scripts/run_pipeline.py --stage <name>` rather than invoking individual
scripts by hand -- it runs each stage's scripts in the order they actually
depend on each other (e.g. `advft` and `decoder_realism` must both finish
before `merge`). See `python scripts/run_pipeline.py --help` for the list of
stages. Split indices aren't pre-checked-in files; each script deterministically
re-derives the same split from the dataset's config (`seed`, `val_frac`, ...),
so nothing needs to ship separately for the split to be reproducible.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.
