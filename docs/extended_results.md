# Extended Results and Mathematical Formulations

Extended appendix material for the paper: the full mathematical formulations and the complete per-dataset sweep tables that the space-limited paper summarizes. Every number here is computed from the CSVs in `results/paper_inputs/` (`attack-results/` and `defense-results/`); `scripts/paper/` regenerates the paper's figures and tables from the same data.

---

## Section C: Mathematical Formulations

### Quantization Geometry

Let x be a medical image, δ a perturbation with L∞ norm ≤ ε (all ε in [0,1] pixel units), and T the number of latent positions. The tokenizer pipeline is encoder E, quantizer Q, decoder D; the reconstruction path classifies D(Q(E(x))) and the token path probes the discrete codes Q(E(x)) directly.

**VQ-16.** Given a learned codebook {e_1,...,e_K}, each latent is assigned to its nearest code by squared Euclidean distance. If e_{k1} and e_{k2} are the nearest and second-nearest entries:

```
q_i = argmin_k ||z_i - e_k||_2^2
m_i = ||z_i - e_{k1}||_2^2 - ||z_i - e_{k2}||_2^2     (eq:margin)
```

A negative m_i indicates the current code remains nearest; crossing zero corresponds to a code change. The VQ-16 attack computes this margin using raw vectors, whereas the tokenizer normalizes before lookup (rankings agree only 51.93% of the time).

**MedITok / UniTok.** These tokenizers split each latent into 8 chunks and quantize each against a separate normalized codebook:

```
z_hat = z / ||z||_2
e_hat_k = e_k / ||e_k||_2
q = argmin_k ||z_hat - e_hat_k||_2^2 = argmax_k z_hat^T e_hat_k    (eq:margin-normalized)
```

The margin m = ||z_hat - e_hat_{k1}||_2^2 - ||z_hat - e_hat_{k2}||_2^2 is computed independently for all 8 sub-codebooks.

**FSQ.** FSQ uses no learned codebook. Each of five scalar latent dimensions is bounded and quantized independently. The margin is the minimum distance to a quantization boundary:

```
m = min_{d in {1,...,5}} (0.5 - |u_d - round(u_d)|)    (eq:margin-fsq)
```

where u_d denotes the bounded scalar value. Unlike VQ margins, this quantity is non-negative, so the FSQ attack minimizes m toward zero rather than relying on a sign change.

---

### Attack Objectives

All attacks use projected gradient descent under the same L∞ constraint.

**M0 (direct) and M1 (tokenizer-mediated supervised):**
```
max_{||δ||_∞ ≤ ε}  L_task(C(x+δ), y)                           (M0)
max_{||δ||_∞ ≤ ε}  L_task(C(D(Q(E(x+δ)))), y)                  (M1)
```

**Embedding displacement baseline (Bhagwatkar et al. 2026):**
```
max_{||δ||_∞ ≤ ε}  (1/T) Σ_i ||z_i(x+δ) - z_i(x)||_2          (eq:global)
```

**Boundary-directed attacks M2-sat and M2-unb:**
```
g_sat(m_i) = min(m_i, 0)
g_unb(m_i) = m_i
max_{||δ||_∞ ≤ ε}  (1/T) Σ_i g(m_i(x+δ))                      (eq:geometric)
```

M2-sat stops rewarding movement once a boundary is crossed; M2-unb keeps pushing the latent beyond the boundary. For MedITok and UniTok the objective sums over all 8 sub-codebook margins. For FSQ the non-negative margin is minimized toward zero.

---

### Metrics

```
F(x, x_adv) = (1/T) Σ_i 1[q_i(x_adv) ≠ q_i(x)]               (token flip rate)
D_emb       = (1/T) Σ_i ||z_i(x_adv) - z_i(x)||_2              (embedding displacement)
```

F records whether a discrete assignment changes; D_emb measures continuous latent movement and can continue increasing after a token has already flipped. This distinction is central to the M2-sat versus M2-unb comparison.

---

### Defense Mechanisms

**Stochastic Quantization Smoothing (SQS).** Each latent is quantized K times after adding independent Gaussian noise with scale σ, and the majority token is retained:

```
q^(k) = argmin_j ||z_hat + σ ε^(k) - e_hat_j||_2^2,  ε^(k) ~ N(0,I)
q^SQS = mode(q^(1), ..., q^(K))                                  (eq:sqs-vote)
```

Three variants are evaluated: margin-adaptive noise scale, spatial smoothing, and logit-ensemble. Because margin-adaptive SQS uses a per-token σ, the certified-radius formula below does not apply to it.

**Encoder-FT.** Fine-tunes only the encoder to move adversarial latents back toward their clean quantization regions.

**Decoder-FT.** Freezes the encoder and codebook and fine-tunes the decoder to reconstruct clean images from both clean and adversarial codes, using L1, perceptual, and GAN losses.

**BPDA.** Nearest-neighbor quantization is non-differentiable. BPDA substitutes the identity for the quantizer gradient:

```
q = Q(h)  (forward),   ∂L/∂h := ∂L/∂q  (backward)             (eq:bpda)
```

**Carlini–Wagner L2 attack.** A held-out CW-style L2 attack with κ=0 and a projection step keeping the perturbation within the specified L2 budget.

**Certified radius (SQS with fixed global σ).** Using N noisy predictions with majority-class count n_A:

```
p_A_lower = Beta^{-1}(α; n_A, N-n_A+1)
R = σ · Φ^{-1}(p_A_lower)  if p_A_lower > 0.5                  (eq:certify)
```

using α=0.001 (99.9% confidence). If p_A_lower ≤ 0.5, no radius is certified. β^{-1} is the α-quantile of the Beta distribution; Φ^{-1} is the standard normal inverse CDF.

---

## Section D.3: Additional Qualitative Examples

Per-image qualitative examples referenced by the paper. The figure PDFs are in the paper's supplementary material.

**Figure: qualitative_path_dependence** (`figures/qualitative_path_dependence.pdf`)
Single held-out Kermany image (ground truth: pneumonia), LlamaGen VQ-16 and MedITok, ε=2/255. M1's reconstruction is misdiagnosed despite remaining visually recognizable; M2-unb's reconstruction preserves the correct diagnosis despite visible texture corruption in the adversarial input. Illustrates path-dependence on a single example — the averaged version appears in Figure 2 (Section 5) of the paper.

**Figure: qualitative_appendix_aptos** (`figures/qualitative_appendix_aptos.pdf`)
Single held-out APTOS fundus image (ground truth DR grade: 1), UniTok and FSQ, ε=8/255. Predicted grade shown per panel (argmax over the 5-way softmax). Both attacks change the predicted grade on this image, illustrating per-image variance that Table 1 and Figure 3 average out over the full held-out set. M1's reconstruction shows visible color/texture corruption; M2-unb's reconstruction is visually closer to clean but still changes the predicted grade for this particular image (Table D.4 in this file reports APTOS recon AUC of 0.883–0.939 for UniTok/FSQ under M2-unb at ε=8/255, i.e., most images are unaffected).

**Figure: qualitative_defense** (`figures/qualitative_defense.pdf`)
MedITok, Kermany, ground truth: normal. Undefended reconstruction under M2-unb at ε=8/255 is misread as pneumonia; adding Gaussian pixel noise (σ=0.05) before re-encoding restores a plausible reconstruction and a correct prediction. The same effect appears in aggregate as MedITok/Kermany reconstruction-path AUC rising from 0.632 to 0.893 in Table 3 of the paper.

---

## Section D.3b: Supplementary Figures

Additional figures referenced by the paper; the PDFs are in the paper's supplementary material. The data behind the quantitative ones is in `results/paper_inputs/`, and `scripts/paper/` regenerates the data-driven main-paper figures.

### Defense training curves

**Figure: encft_margin_loss_curves** (`figures/encft_margin_loss_curves.pdf`)
Encoder-FT training: margin loss over training steps, one curve per attack budget (ε ∈ {0.5,1,2,4,8}/255), four tokenizers. Shows that the encoder learns to widen the quantization margin on adversarial inputs.

**Figure: encft_clean_auc_guard_curves** (`figures/encft_clean_auc_guard_curves.pdf`)
Encoder-FT training: clean reconstruction-path AUC over training steps, same layout as above. Monitors the clean-accuracy guard during adversarial fine-tuning.

**Figure: decft_reconstruction_loss_curves** (`figures/decft_reconstruction_loss_curves.pdf`)
Decoder-FT training: total reconstruction loss (L1 + perceptual + GAN) over training steps. Single checkpoint trained at ε=8/255 for each tokenizer.

**Figure: decft_gan_dynamics_curves** (`figures/decft_gan_dynamics_curves.pdf`)
Decoder-FT training: generator and discriminator loss curves. Confirms stable GAN training without mode collapse.

### Alternate renderings of the main-paper figures

Raster renderings of the main-paper figures; the paper renders them as inline TikZ (see scripts/paper/).

**Figure: fig1_spine_dissociation** (`figures/fig1_spine_dissociation.pdf`)
Early export of Figure 1 (path-dependence spine diagram / overview).

**Figure: fig2_flips_vs_damage** (`figures/fig2_flips_vs_damage.pdf`)
Early export of Figure 2 (flip rate vs. probe AUC bar chart).

**Figure: fig3_eps_sweep_paths** (`figures/fig3_eps_sweep_paths.pdf`)
Early export of Figure 3 (budget sweep on both inference paths).

### BRS and boundary-vs-global-displacement figures

BRS (Boundary Risk Score) is an auxiliary per-image analysis; these figures are provided for completeness and are not part of the main paper.

**Figure: rq1_boundary_vs_displacement** (`figures/rq1_boundary_vs_displacement.pdf`)
Boundary-directed vs. global embedding displacement comparison across tokenizers and budgets.

**Figure: boundary_vs_global_eps_sweep** (`figures/boundary_vs_global_eps_sweep.jpg`)
Sweep over ε comparing boundary-directed (M2) and global displacement (Bhagwatkar et al. 2026) attacks on token flip rate and probe AUC.

**Figure: boundary_vs_global_steps_sweep** (`figures/boundary_vs_global_steps_sweep.jpg`)
Same comparison swept over PGD iteration count rather than ε budget.

**Figure: brs_confshift_eps05** (`figures/brs_confshift_eps05.jpg`)
BRS confidence-shift plot at ε=0.5/255. Shows the distribution of per-image BRS scores across clean and adversarial inputs.

**Figure: brs_confshift_eps8** (`figures/brs_confshift_eps8.jpg`)
BRS confidence-shift plot at ε=8/255.

**Figure: brs_min_budget_scatter** (`figures/brs_min_budget_scatter.jpg`)
BRS minimum-budget scatter: per-image BRS vs. minimum ε needed to flip the predicted class.

**Figure: brs_token_quartiles** (`figures/brs_token_quartiles.jpg`)
BRS broken down by token-flip-rate quartile, showing that high-flip images have systematically different BRS distributions.

### Per-tokenizer sweep plots

**Figure: vq16_flip_rate_eps_sweep** (`figures/vq16_flip_rate_eps_sweep.jpg`)
VQ-16 only: token flip rate vs. ε for M2-sat and M2-unb.

**Figure: vq16_m3_probe_eps_sweep** (`figures/vq16_m3_probe_eps_sweep.png`)
VQ-16 only: probe AUC vs. ε under M2 (geometric) attack.

**Figure: vq16_m3_reconstruction_eps_sweep** (`figures/vq16_m3_reconstruction_eps_sweep.png`)
VQ-16 only: reconstruction-path AUC vs. ε under M2 attack.

**Figure: meditok_unitok_geometric_sweep** (`figures/meditok_unitok_geometric_sweep.jpg`)
MedITok and UniTok side-by-side: probe and reconstruction AUC vs. ε under M2-unb.

---

## Section D.4: Full Attack and Defense Sweep Tables

### D.4.1 Attack sweep tables (M1 and M2-sat, all datasets, full ε grid)

**Table: Reconstruction-path AUC under M1, all datasets, full ε grid** (`tab:recon_m1_all`)

n=256 per dataset. M1 is the tokenizer-mediated supervised attack. Lower is a stronger attack.

| Dataset | Tokenizer | clean | 0.5/255 | 1/255 | 2/255 | 4/255 | 8/255 |
|---------|-----------|-------|---------|-------|-------|-------|-------|
| Kermany | LlamaGen VQ-16 | 0.957 | 0.207 | 0.006 | 0.000 | 0.000 | 0.000 |
| Kermany | MedITok | 0.959 | 0.127 | 0.000 | 0.000 | 0.000 | 0.000 |
| Kermany | UniTok | 0.934 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 |
| Kermany | FSQ | 0.810 | 0.099 | 0.002 | 0.000 | 0.000 | 0.000 |
| APTOS | LlamaGen VQ-16 | 0.997 | 0.752 | 0.251 | 0.013 | 0.000 | 0.000 |
| APTOS | MedITok | 0.995 | 0.054 | 0.000 | 0.000 | 0.000 | 0.000 |
| APTOS | UniTok | 0.993 | 0.202 | 0.013 | 0.000 | 0.000 | 0.000 |
| APTOS | FSQ | 0.932 | 0.575 | 0.129 | 0.000 | 0.000 | 0.000 |
| ISIC | LlamaGen VQ-16 | 0.882 | 0.076 | 0.009 | 0.001 | 0.000 | 0.000 |
| ISIC | MedITok | 0.910 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ISIC | UniTok | 0.910 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ISIC | FSQ | 0.797 | 0.021 | 0.000 | 0.000 | 0.000 | 0.000 |

**Table: Token-path AUC under M1, all datasets, full ε grid** (`tab:probe_m1_all`)

n=256 per dataset. Probe is a linear classifier on the frozen discrete codes.

| Dataset | Tokenizer | clean | 0.5/255 | 1/255 | 2/255 | 4/255 | 8/255 |
|---------|-----------|-------|---------|-------|-------|-------|-------|
| Kermany | LlamaGen VQ-16 | 0.877 | 0.854 | 0.805 | 0.676 | 0.548 | 0.442 |
| Kermany | MedITok | 0.938 | 0.906 | 0.865 | 0.719 | 0.489 | 0.274 |
| Kermany | UniTok | 0.921 | 0.862 | 0.757 | 0.640 | 0.480 | 0.339 |
| Kermany | FSQ | 0.869 | 0.869 | 0.871 | 0.857 | 0.818 | 0.718 |
| APTOS | LlamaGen VQ-16 | 0.951 | 0.917 | 0.917 | 0.875 | 0.878 | 0.856 |
| APTOS | MedITok | 0.987 | 0.969 | 0.923 | 0.732 | 0.476 | 0.332 |
| APTOS | UniTok | 0.980 | 0.965 | 0.945 | 0.904 | 0.780 | 0.610 |
| APTOS | FSQ | 0.954 | 0.951 | 0.935 | 0.739 | 0.666 | 0.538 |
| ISIC | LlamaGen VQ-16 | 0.762 | 0.759 | 0.746 | 0.736 | 0.703 | 0.674 |
| ISIC | MedITok | 0.834 | 0.822 | 0.806 | 0.782 | 0.718 | 0.663 |
| ISIC | UniTok | 0.821 | 0.796 | 0.786 | 0.747 | 0.681 | 0.672 |
| ISIC | FSQ | 0.743 | 0.743 | 0.731 | 0.746 | 0.736 | 0.714 |

**Table: Reconstruction-path AUC under M2-sat, all datasets, full ε grid** (`tab:recon_m3sat_all`)

M2-sat is the bounded (saturating) boundary-targeting attack. Barely degrades this path at any budget.

| Dataset | Tokenizer | clean | 0.5/255 | 1/255 | 2/255 | 4/255 | 8/255 |
|---------|-----------|-------|---------|-------|-------|-------|-------|
| Kermany | LlamaGen VQ-16 | 0.957 | 0.933 | 0.938 | 0.940 | 0.933 | 0.912 |
| Kermany | MedITok | 0.959 | 0.963 | 0.958 | 0.964 | 0.954 | 0.926 |
| Kermany | UniTok | 0.934 | 0.939 | 0.939 | 0.935 | 0.938 | 0.906 |
| Kermany | FSQ | 0.810 | 0.806 | 0.798 | 0.822 | 0.835 | 0.825 |
| APTOS | LlamaGen VQ-16 | 0.997 | 0.992 | 0.992 | 0.989 | 0.986 | 0.974 |
| APTOS | MedITok | 0.995 | 0.996 | 0.992 | 0.990 | 0.988 | 0.978 |
| APTOS | UniTok | 0.993 | 0.994 | 0.996 | 0.994 | 0.990 | 0.986 |
| APTOS | FSQ | 0.932 | 0.916 | 0.922 | 0.906 | 0.911 | 0.910 |
| ISIC | LlamaGen VQ-16 | 0.882 | 0.890 | 0.883 | 0.892 | 0.859 | 0.803 |
| ISIC | MedITok | 0.910 | 0.902 | 0.909 | 0.899 | 0.875 | 0.876 |
| ISIC | UniTok | 0.910 | 0.904 | 0.907 | 0.888 | 0.876 | 0.869 |
| ISIC | FSQ | 0.797 | 0.767 | 0.809 | 0.799 | 0.807 | 0.770 |

**Table: Token-path AUC under M2-sat, all datasets, full ε grid** (`tab:probe_m3sat_all`)

MedITok degrades under both M2-sat and M2-unb, but less sharply under M2-sat.

| Dataset | Tokenizer | clean | 0.5/255 | 1/255 | 2/255 | 4/255 | 8/255 |
|---------|-----------|-------|---------|-------|-------|-------|-------|
| Kermany | LlamaGen VQ-16 | 0.877 | 0.880 | 0.892 | 0.880 | 0.872 | 0.852 |
| Kermany | MedITok | 0.938 | 0.912 | 0.846 | 0.803 | 0.747 | 0.690 |
| Kermany | UniTok | 0.921 | 0.916 | 0.920 | 0.900 | 0.908 | 0.865 |
| Kermany | FSQ | 0.869 | 0.873 | 0.860 | 0.852 | 0.844 | 0.849 |
| APTOS | LlamaGen VQ-16 | 0.951 | 0.948 | 0.948 | 0.952 | 0.942 | 0.940 |
| APTOS | MedITok | 0.987 | 0.984 | 0.976 | 0.965 | 0.954 | 0.929 |
| APTOS | UniTok | 0.980 | 0.979 | 0.975 | 0.973 | 0.966 | 0.946 |
| APTOS | FSQ | 0.954 | 0.935 | 0.913 | 0.722 | 0.532 | 0.519 |
| ISIC | LlamaGen VQ-16 | 0.762 | 0.769 | 0.753 | 0.760 | 0.745 | 0.733 |
| ISIC | MedITok | 0.834 | 0.837 | 0.834 | 0.836 | 0.824 | 0.786 |
| ISIC | UniTok | 0.821 | 0.818 | 0.799 | 0.783 | 0.799 | 0.783 |
| ISIC | FSQ | 0.743 | 0.739 | 0.726 | 0.717 | 0.706 | 0.699 |

---

### D.4.2 Defense sweep tables (full ε grid, per-attack breakdown)

**Table: Defense comparison, reconstruction-path AUC under M1, full ε grid** (`tab:defense_recon_m1_full`)

Mean over three datasets, n=128 per dataset.

| ε | Tokenizer | Clean | None | Pixel-noise | SQS | Encoder-FT | Encoder-FT+SQS | Decoder-FT | Encoder-FT+Decoder-FT |
|---|-----------|-------|------|-------------|-----|-----------|----------------|-----------|----------------------|
| 0.5/255 | LlamaGen VQ-16 | 0.942 | 0.349 | 0.538 | 0.386 | 0.723 | 0.700 | 0.751 | 0.794 |
| 0.5/255 | MedITok | 0.947 | 0.057 | 0.894 | 0.098 | 0.473 | 0.484 | 0.704 | 0.777 |
| 0.5/255 | UniTok | 0.937 | 0.072 | 0.919 | 0.108 | 0.621 | 0.620 | 0.723 | 0.824 |
| 0.5/255 | FSQ | 0.831 | 0.256 | 0.306 | 0.489 | 0.714 | 0.697 | 0.910 | 0.839 |
| 1/255 | LlamaGen VQ-16 | 0.942 | 0.084 | 0.357 | 0.275 | 0.555 | 0.538 | 0.570 | 0.636 |
| 1/255 | MedITok | 0.947 | 0.000 | 0.860 | 0.000 | 0.338 | 0.336 | 0.504 | 0.612 |
| 1/255 | UniTok | 0.937 | 0.006 | 0.885 | 0.006 | 0.318 | 0.333 | 0.631 | 0.741 |
| 1/255 | FSQ | 0.831 | 0.036 | 0.044 | 0.183 | 0.620 | 0.619 | 0.891 | 0.839 |
| 2/255 | LlamaGen VQ-16 | 0.942 | 0.006 | 0.174 | 0.249 | 0.033 | 0.268 | 0.440 | 0.475 |
| 2/255 | MedITok | 0.947 | 0.000 | 0.783 | 0.000 | 0.000 | 0.000 | 0.464 | 0.527 |
| 2/255 | UniTok | 0.937 | 0.000 | 0.836 | 0.000 | 0.277 | 0.277 | 0.614 | 0.687 |
| 2/255 | FSQ | 0.831 | 0.000 | 0.000 | 0.006 | 0.567 | 0.569 | 0.850 | 0.801 |
| 4/255 | LlamaGen VQ-16 | 0.942 | 0.000 | 0.000 | 0.193 | 0.233 | 0.272 | 0.525 | 0.480 |
| 4/255 | MedITok | 0.947 | 0.000 | 0.650 | 0.000 | 0.000 | 0.000 | 0.532 | 0.545 |
| 4/255 | UniTok | 0.937 | 0.000 | 0.705 | 0.000 | 0.082 | 0.078 | 0.617 | 0.629 |
| 4/255 | FSQ | 0.831 | 0.000 | 0.000 | 0.000 | 0.276 | 0.269 | 0.791 | 0.730 |
| 8/255 | LlamaGen VQ-16 | 0.942 | 0.000 | 0.000 | 0.121 | 0.000 | 0.125 | 0.662 | 0.553 |
| 8/255 | MedITok | 0.947 | 0.000 | 0.437 | 0.000 | 0.000 | 0.000 | 0.571 | 0.552 |
| 8/255 | UniTok | 0.937 | 0.000 | 0.515 | 0.000 | 0.000 | 0.000 | 0.607 | 0.603 |
| 8/255 | FSQ | 0.831 | 0.000 | 0.000 | 0.000 | 0.046 | 0.046 | 0.605 | 0.602 |

**Table: Defense comparison, token-path AUC under M1, full ε grid** (`tab:defense_probe_m1_full`)

Mean over three datasets, n=128 per dataset.

| ε | Tokenizer | Clean | None | Pixel-noise | SQS | Encoder-FT | Encoder-FT+SQS | Decoder-FT | Encoder-FT+Decoder-FT |
|---|-----------|-------|------|-------------|-----|-----------|----------------|-----------|----------------------|
| 0.5/255 | LlamaGen VQ-16 | 0.871 | 0.850 | 0.859 | 0.839 | 0.868 | 0.859 | 0.850 | 0.852 |
| 0.5/255 | MedITok | 0.913 | 0.892 | 0.888 | 0.893 | 0.885 | 0.888 | 0.892 | 0.895 |
| 0.5/255 | UniTok | 0.903 | 0.867 | 0.850 | 0.873 | 0.852 | 0.854 | 0.867 | 0.871 |
| 0.5/255 | FSQ | 0.786 | 0.786 | 0.787 | 0.670 | 0.602 | 0.604 | 0.786 | 0.615 |
| 1/255 | LlamaGen VQ-16 | 0.871 | 0.832 | 0.834 | 0.805 | 0.855 | 0.839 | 0.832 | 0.836 |
| 1/255 | MedITok | 0.913 | 0.855 | 0.879 | 0.853 | 0.860 | 0.863 | 0.855 | 0.879 |
| 1/255 | UniTok | 0.903 | 0.812 | 0.835 | 0.815 | 0.840 | 0.839 | 0.812 | 0.832 |
| 1/255 | FSQ | 0.786 | 0.777 | 0.776 | 0.680 | 0.601 | 0.601 | 0.777 | 0.614 |
| 2/255 | LlamaGen VQ-16 | 0.871 | 0.772 | 0.784 | 0.747 | 0.812 | 0.787 | 0.772 | 0.795 |
| 2/255 | MedITok | 0.913 | 0.749 | 0.871 | 0.747 | 0.783 | 0.785 | 0.749 | 0.841 |
| 2/255 | UniTok | 0.903 | 0.744 | 0.847 | 0.743 | 0.792 | 0.785 | 0.744 | 0.778 |
| 2/255 | FSQ | 0.786 | 0.705 | 0.712 | 0.679 | 0.606 | 0.605 | 0.705 | 0.613 |
| 4/255 | LlamaGen VQ-16 | 0.871 | 0.725 | 0.729 | 0.676 | 0.758 | 0.735 | 0.725 | 0.750 |
| 4/255 | MedITok | 0.913 | 0.553 | 0.860 | 0.551 | 0.626 | 0.623 | 0.553 | 0.723 |
| 4/255 | UniTok | 0.903 | 0.624 | 0.815 | 0.618 | 0.745 | 0.749 | 0.624 | 0.685 |
| 4/255 | FSQ | 0.786 | 0.692 | 0.688 | 0.622 | 0.609 | 0.609 | 0.692 | 0.608 |
| 8/255 | LlamaGen VQ-16 | 0.871 | 0.655 | 0.669 | 0.598 | 0.688 | 0.643 | 0.655 | 0.688 |
| 8/255 | MedITok | 0.913 | 0.415 | 0.827 | 0.418 | 0.568 | 0.564 | 0.415 | 0.568 |
| 8/255 | UniTok | 0.903 | 0.508 | 0.782 | 0.519 | 0.578 | 0.574 | 0.508 | 0.578 |
| 8/255 | FSQ | 0.786 | 0.583 | 0.584 | 0.585 | 0.573 | 0.575 | 0.583 | 0.573 |

**Table: Geometric-adaptive attack, full ε grid** (`tab:geo_adaptive_full`)

Reconstruction-path AUC, mean over three datasets, n=128 per dataset. Encoder-FT resists the same boundary-directed attack family used for training.

| Tokenizer | Encoder-FT 0.5 | 1 | 2 | 4 | 8 | Decoder-FT 0.5 | 1 | 2 | 4 | 8 | Enc+Dec-FT 0.5 | 1 | 2 | 4 | 8 |
|-----------|---------------|---|---|---|---|----------------|---|---|---|---|----------------|---|---|---|---|
| LlamaGen VQ-16 | 0.925 | 0.919 | 0.910 | 0.912 | 0.898 | 0.938 | 0.936 | 0.931 | 0.934 | 0.938 | 0.952 | 0.936 | 0.927 | 0.921 | 0.925 |
| MedITok | 0.927 | 0.936 | 0.933 | 0.927 | 0.915 | 0.946 | 0.939 | 0.935 | 0.933 | 0.926 | 0.932 | 0.932 | 0.932 | 0.931 | 0.941 |
| UniTok | 0.946 | 0.938 | 0.942 | 0.928 | 0.904 | 0.940 | 0.945 | 0.934 | 0.932 | 0.931 | 0.947 | 0.939 | 0.943 | 0.940 | 0.938 |
| FSQ | 0.748 | 0.751 | 0.685 | 0.764 | 0.800 | 0.920 | 0.920 | 0.910 | 0.908 | 0.915 | 0.842 | 0.842 | 0.856 | 0.851 | 0.866 |

**Table: BPDA attack, full ε grid** (`tab:geo_bpda_full`)

Reconstruction-path AUC, mean over three datasets, n=128 per dataset. Apparent static robustness of fine-tuned defenses collapses to near zero on all learned-codebook tokenizers; FSQ retains partial robustness.

| Tokenizer | Encoder-FT 0.5 | 1 | 2 | 4 | 8 | Decoder-FT 0.5 | 1 | 2 | 4 | 8 | Enc+Dec-FT 0.5 | 1 | 2 | 4 | 8 |
|-----------|---------------|---|---|---|---|----------------|---|---|---|---|----------------|---|---|---|---|
| LlamaGen VQ-16 | 0.489 | 0.217 | 0.000 | 0.000 | 0.000 | 0.501 | 0.214 | 0.024 | 0.003 | 0.000 | 0.529 | 0.251 | 0.017 | 0.001 | 0.000 |
| MedITok | 0.116 | 0.034 | 0.000 | 0.000 | 0.000 | 0.336 | 0.070 | 0.005 | 0.000 | 0.000 | 0.368 | 0.126 | 0.009 | 0.000 | 0.000 |
| UniTok | 0.158 | 0.063 | 0.001 | 0.000 | 0.000 | 0.116 | 0.028 | 0.004 | 0.000 | 0.000 | 0.184 | 0.041 | 0.000 | 0.000 | 0.000 |
| FSQ | 0.691 | 0.614 | 0.434 | 0.355 | 0.246 | 0.839 | 0.755 | 0.578 | 0.347 | 0.159 | 0.796 | 0.701 | 0.532 | 0.390 | 0.284 |

**Table: Carlini-Wagner L2 held-out attack** (`tab:cw_l2_full`)

Reconstruction-path AUC, mean over three datasets, at fixed L2 budget (L2=16). Encoder-FT's five training-budget variants averaged. n=128 per dataset.

| Tokenizer | Encoder-FT | Decoder-FT | Encoder-FT+Decoder-FT |
|-----------|-----------|-----------|----------------------|
| LlamaGen VQ-16 | 0.000 | 0.001 | 0.000 |
| MedITok | 0.000 | 0.000 | 0.000 |
| UniTok | 0.000 | 0.003 | 0.000 |
| FSQ | 0.234 | 0.138 | 0.264 |

**Table: SQS certified radii, all datasets** (`tab:certified_radii`)

Fine-tuned tokenizer with SQS. Certification rate and accuracy from 128-image held-out subset; mean radius over all 128 images (uncertified images contribute 0). Dash indicates the winning SQS variant uses per-token adaptive noise scale, for which the certified-radius formula does not apply.

| Dataset | Tokenizer | Certified rate | Accuracy | Mean radius (L2) |
|---------|-----------|---------------|----------|-----------------|
| Kermany | LlamaGen VQ-16 | 1.000 | 0.586 | 0.028 |
| Kermany | MedITok | 0.961 | 0.672 | — |
| Kermany | UniTok | 0.969 | 0.602 | 0.136 |
| Kermany | FSQ | 1.000 | 0.500 | — |
| APTOS | LlamaGen VQ-16 | 1.000 | 0.758 | 0.030 |
| APTOS | MedITok | 1.000 | 0.961 | — |
| APTOS | UniTok | 0.984 | 0.961 | 0.140 |
| APTOS | FSQ | 0.984 | 0.500 | 0.073 |
| ISIC | LlamaGen VQ-16 | 0.906 | 0.797 | — |
| ISIC | MedITok | 0.898 | 0.836 | 0.110 |
| ISIC | UniTok | 0.898 | 0.844 | 0.106 |
| ISIC | FSQ | 0.867 | 0.633 | 0.072 |
