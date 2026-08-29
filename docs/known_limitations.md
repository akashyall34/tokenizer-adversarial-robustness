# Known limitations

## VQ-16 attack margin uses an unnormalized distance, not the real quantizer's boundary

`src/tokenizers/vq16.py`'s `clean_top1()` and `src/attacks/vq16_attacks.py`'s
`margin_loss()` compute the nearest-codebook distance on the raw,
unnormalized latent and codebook vectors. VQ-16's `quantize` module instead
L2-normalizes both before its own nearest-neighbour lookup.

**Measured impact:** comparing the unnormalized proxy's predicted nearest
code against the quantizer's actual assigned code, on real images, gives only
**51.93% agreement** -- statistically indistinguishable from chance. So the
VQ-16 geometric attacks (`M2-sat` / `M2-unb` / `M3_alpha03`, on all three
datasets, which share this code) are often driving the latent away from the
wrong rival code.

**Scope:** VQ-16 only. The multi-codebook (MedITok / UniTok) margin already
normalizes correctly (`F.normalize` in
`src/attacks/meditok_unitok_attacks.py`), and FSQ's margin is a
distance-to-rounding-boundary, not a codebook nearest-neighbour computation.

**Handling:** the attack-side margin is left unnormalized so the generated
perturbations match the ones behind the reported attack numbers. The
corrected, L2-normalized margin (`src/defenses/vq16/margin.py`) is used for
the adversarial fine-tuning defense, which trains against the real decision
boundary. `scripts/diagnostics/diagnose_vq16_margin_bug.py` measures the
footprint on the cached attack images: of the tokens that flipped, the
fraction that landed on the true nearest rival versus some other code.
