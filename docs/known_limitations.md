# Known limitations

## VQ-16 attack margin uses an unnormalized distance

`clean_top1()` in `src/tokenizers/vq16.py` and `margin_loss()` in
`src/attacks/vq16_attacks.py` measure the nearest-codebook distance on the raw
latent and codebook vectors. VQ-16's own `quantize` module L2-normalizes both
before its nearest-neighbour lookup, so the two disagree.

On real images the unnormalized proxy picks the same nearest code as the
quantizer only 51.93% of the time, which is no better than chance. The VQ-16
geometric attacks (`M2-sat`, `M2-unb`, `M3_alpha03`, all three datasets) are
therefore often pushing the latent away from the wrong rival code.

This is a VQ-16 issue only. The MedITok and UniTok margin normalizes correctly
(`F.normalize` in `src/attacks/meditok_unitok_attacks.py`), and FSQ's margin is
a distance to a rounding boundary rather than a codebook lookup.

The attack-side margin is left as is so the perturbations still match the
reported attack numbers. The adversarial fine-tuning defense instead uses the
corrected, L2-normalized margin in `src/defenses/vq16/margin.py` and trains
against the real boundary. `scripts/diagnostics/diagnose_vq16_margin_bug.py`
reports how many flipped tokens landed on the true nearest rival rather than some
other code.

## The pure geometric adaptive attack cannot see decoder-only defenses

`scripts/defenses/*/geo_adaptive_*.py` runs two adaptive checks per defense
checkpoint. `geo` maximizes the correct margin against the model's current
boundary. `geo_bpda` blends that margin with BCE through `decode()`, matching the
cached `M3_alpha03` attack but generated live against each checkpoint.

The gradient in `geo` reaches only the encoder, the quant projection, and the
quantizer, never `decode()` or the classifier. The decoder-realism defense
leaves the encoder untouched and trains only the decoder side, so `geo` produces
the same perturbation against `decoder_ft` (and the decoder half of `combined`)
that it would against the undefended tokenizer. A high `geo` AUC for those two
conditions says nothing about robustness; the attack never touched what changed.

This applies to all four tokenizers and all three datasets. For `decoder_ft` and
`combined`, use the `geo_bpda` number, which routes through `decode()`. For
`advft`, which changes the encoder, both numbers are informative. Each script
writes the two as separate grids (`*_geo_adaptive_full_grid.csv` and
`*_geo_bpda_full_grid.csv`).

## Generative-downstream VQA scoring

The LLaVA-Med VQA red-team (`scripts/generative/run_vqa_redteam.py`) decodes
greedily, which is sensitive to small numerical changes in the encoder. Answer
wording varies between runs, and the smallest `eps` that breaks a given image can
move by one step. The pooled rates are stable. The 20 images per dataset are
drawn from `configs/vqa.yaml` (seed 42) rather than a fixed file list, so a rerun
reproduces the pooled numbers but not necessarily every per-image cell.

Two questions are left out of the accuracy tables because the clean model has no
skill on them for the attack to remove: the pneumonia question, where the model
answers "no" to all 20 chest X-rays, and the ISIC lesion-type question, where
clean answers on benign lesions already name melanoma or BCC.
