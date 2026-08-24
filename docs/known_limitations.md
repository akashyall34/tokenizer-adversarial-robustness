# Known limitations

## VQ-16 M3 margin attacks use an unnormalized distance, not the real quantizer's boundary

`src/tokenizers/vq16.py`'s `clean_top1()` and `src/attacks/vq16_attacks.py`'s
`margin_loss()` compute nearest-codebook distance on raw, unnormalized latent
and codebook vectors. VQ-16's actual `quantize` module L2-normalizes both
before its own nearest-neighbor lookup internally. This is a faithful port of
`VQ16.ipynb`'s own `vq_clean_top1`/`vq_margin_loss` -- the bug originates in
that notebook, not in the port.

**Measured impact** (from the defense notebook's verification, not assumed):
comparing the unnormalized proxy's predicted nearest code against the real
quantizer's actual assigned code, on real images, gave only **51.93%
agreement** -- statistically indistinguishable from chance. This means the
`M3_alpha03` / `M3_geo_sat` / `M3_geo_unb` attacks for VQ-16 (Kermany, APTOS,
and ISIC alike, since all three share this code) are frequently computing
"push away from the nearest rival code" against the wrong boundary.

**Scope:** VQ-16 only. MediTok/UniTok's margin math already normalizes
correctly (`F.normalize` in `src/attacks/meditok_unitok_attacks.py`), and
FSQ's margin isn't a codebook-nearest-neighbor computation at all.

**Decision:** the attack code stays as-is, faithful to `VQ16.ipynb` --
changing it would mean the ported code no longer matches what was actually
run to produce any already-reported numbers. The corrected, L2-normalized
margin computation is used inside the defense pipeline
(`src/defenses/vq16/margin.py`), which trains against the *real* decision
boundary rather than the buggy proxy. `scripts/diagnostics/diagnose_vq16_margin_bug.py`
quantifies the bug's actual footprint on the cached attack images (of the
tokens that flipped, what fraction landed on the true nearest rival vs. some
other code) -- cite that measurement directly, not an unquantified caveat.
