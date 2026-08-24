# Pretrained checkpoints

Checkpoints are committed directly in `ckpts/`, tracked via [Git LFS](https://git-lfs.com)
(`ckpts/**/*.pt`, see `.gitattributes`). A regular `git clone` pulls them
automatically as long as Git LFS is installed:

```bash
git lfs install   # once per machine
git clone <repo-url>
```

If you cloned before installing Git LFS, run `git lfs pull` afterward to fetch
the real file contents (otherwise `ckpts/` will contain small pointer files).

## Layout

```
ckpts/
  classifiers/                        # per-dataset downstream classifier
    densenet121_kermany.pt
    effnet_b4_isic.pt
    effnet_b5_aptos.pt
  linear_probes/                       # per-(tokenizer, dataset) linear probe on frozen codes
    probe_fsq_aptos.pt
    probe_fsq_kermany.pt
    probe_meditok_aptos.pt
    probe_meditok_kermany.pt
    probe_unitok_aptos.pt
    probe_unitok_kermany.pt
    probe_vq16_aptos.pt
    probe_vq16_isic.pt
    probe_vq16_kermany.pt
    probe_meditok_isic.pt
    probe_unitok_isic.pt
    probe_fsq_isic.pt                  # not yet present -- run_fsq_isic.py regenerates it
```

**Pipeline coverage is complete (12/12 tokenizer x dataset combinations)** --
4 tokenizers (VQ-16, MediTok, UniTok, FSQ) x 3 datasets (Kermany, APTOS,
ISIC). MediTok and UniTok defenses used to share one combined directory per
dataset; they're now split into separate `meditok_{dataset}/` and
`unitok_{dataset}/` pipelines, matching the source repo's now-separate
`meditok_adv_finetune_*.ipynb` / `unitok_adv_finetune_*.ipynb` notebooks.
Not every combination had an already-executed upstream notebook to port from
when it was built -- some were built by wiring a tokenizer's
already-verified, dataset-agnostic defense code (`src/defenses/{vq16,fsq,
multicodebook}/`) to a dataset's existing loader/classifier, not new math.
See the top-level README's Coverage note and `docs/known_limitations.md`'s
execution-status table for which notebooks have actually run.

**Not yet run in this environment:** no GPU has actually executed any of
these pipelines here, so `probe_fsq_isic.pt` above doesn't exist on disk yet,
and none of the `*_attack_table.csv` / `*_defense_table.csv` outputs exist
either -- the code is complete and verified (imports resolve, every script
compiles, checkpoint naming is consistent end to end), but no real numbers
have been produced yet. Code-complete is not the same as run-complete.

Total footprint: ~205MB (classifiers dominate at ~200MB; probes are tens–hundreds
of KB each), well within the LFS free tier.

## Why LFS instead of plain git

GitHub hard-rejects any single file over 100MB pushed as a regular blob;
`effnet_b5_aptos.pt` is ~114MB. LFS stores the large binary content
separately and keeps the main repo history small. The free tier is 1GB
storage / 1GB bandwidth per month per account — enough for this repo's
checkpoints, but be mindful that every clone/pull by a reviewer counts
against that quota.
