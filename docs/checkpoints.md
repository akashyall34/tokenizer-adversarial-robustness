# Pretrained checkpoints

All checkpoints are committed directly in `ckpts/`, so a plain `git clone` gets
them with no Git LFS needed. One file ships split (see below).

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
3 datasets grid. Tokenizer weights are not vendored here;
`scripts/setup_third_party.sh` fetches them from the upstream repos.

## Generative downstream: LLaVA-Med (MedITok) VQA

The VQA red-team (`scripts/generative/run_vqa_redteam.py`) needs the MedITok
paper's released downstream VLM, which is **not** vendored (~15 GB, CUDA-only):

```
third_party/meditok/weights/
  llavamed_meditok/                 # the LLaVA-Mistral-7B VLM (config.json + shards)
  meditok/meditok_simple_v1.pth     # the MedITok vision-tower weights it loads
```

Fetch both from the `massaki75/meditok` Hugging Face repo:

```bash
MEDITOK_VLM=1 scripts/setup_third_party.sh
```

That also patches one API break in the meditok "understanding" fork's
`builder.py` (a removed `use_flash_attention_2=` kwarg on modern
`transformers`). The fork pins `transformers==4.36.2`, which conflicts with the
main environment, so install the `[vqa]` extra in a separate venv:

```bash
pip install -e ".[vqa]"
```

The generative-**fidelity** runs (`scripts/generative/run_fidelity.py`) reuse the
ordinary tokenizer + defense checkpoints above and need nothing extra.
