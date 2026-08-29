# Pretrained checkpoints

All checkpoints are committed directly in `ckpts/` -- a plain `git clone` gets
them, no Git LFS needed. One file ships split (see below).

## Layout

```
ckpts/
  classifiers/                      # per-dataset downstream classifier
    densenet121_kermany.pt          # 28 MB
    effnet_b4_isic.pt               # 71 MB
    effnet_b5_aptos.pt.part00       # \ 114 MB total, split because it exceeds
    effnet_b5_aptos.pt.part01       # / GitHub's 100 MB single-file limit
  linear_probes/                    # per-(tokenizer, dataset) linear probe on frozen codes
    probe_{vq16,meditok,unitok,fsq}_{kermany,aptos,isic}.pt   # 12 files, KB each
```

After cloning, reassemble the split file once:

```bash
python scripts/fetch_checkpoints.py
```

That concatenates `effnet_b5_aptos.pt.part00` + `.part01` into
`ckpts/classifiers/effnet_b5_aptos.pt` and verifies its SHA-256. It is
idempotent and the assembled file is gitignored.

All 15 checkpoints (3 classifiers, 12 probes) cover the full 4 tokenizers x
3 datasets grid. Tokenizer weights are not vendored here --
`scripts/setup_third_party.sh` fetches them from the upstream repos.
