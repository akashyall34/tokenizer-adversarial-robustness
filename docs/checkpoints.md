# Pretrained checkpoints

Checkpoints are committed in `ckpts/`, tracked via [Git LFS](https://git-lfs.com)
(`ckpts/**/*.pt`, see `.gitattributes`). A regular `git clone` pulls them as
long as Git LFS is installed:

```bash
git lfs install   # once per machine
git clone <repo-url>
```

If you cloned before installing Git LFS, run `git lfs pull` to fetch the real
contents (otherwise `ckpts/` holds small pointer files).

## Layout

```
ckpts/
  classifiers/                      # per-dataset downstream classifier
    densenet121_kermany.pt
    effnet_b4_isic.pt
    effnet_b5_aptos.pt
  linear_probes/                    # per-(tokenizer, dataset) linear probe on frozen codes
    probe_{vq16,meditok,unitok,fsq}_{kermany,aptos,isic}.pt   # 12 files
```

All 15 checkpoints (3 classifiers, 12 probes) cover the full 4 tokenizers x
3 datasets grid. The tokenizer weights themselves are not vendored here --
`scripts/setup_third_party.sh` fetches them from the upstream repos.

Total footprint is ~205 MB (the classifiers dominate at ~200 MB; each probe is
tens to hundreds of KB), within the Git LFS free tier.

## Why LFS instead of plain git

GitHub rejects any single file over 100 MB pushed as a regular blob, and
`effnet_b5_aptos.pt` is ~114 MB. LFS stores the large binary content
separately and keeps the main history small.
