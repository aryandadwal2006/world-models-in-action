# Chapter 3: Code map

This directory contains the complete experiment runners. The book's central code listings should point to the canonical modules below, not duplicate shortened versions of them.

## Book listing → canonical source

| Concept | Canonical source | What to study |
|---|---|---|
| Image augmentations and positive views | `worldmodels/data/augment.py` | Crop shifts and appearance perturbations |
| Convolutional visual encoder | `worldmodels/models/encoders.py` | CNN layers, latent projection, optional recurrent agent interface |
| NT-Xent / InfoNCE | `worldmodels/losses/ntxent.py` | Normalize, form similarities, identify positive pairs, cross-entropy |
| VICReg | `worldmodels/losses/vicreg.py` | Invariance, variance hinge, covariance regularization |
| Stop-gradient, EMA and redundancy reduction | `worldmodels/losses/noncontrastive.py` | SimSiam, BYOL target update, Barlow Twins |
| Masked reconstruction objective | `worldmodels/losses/reconstruction.py` | Score reconstruction only on masked patches |
| Masked autoencoder | `worldmodels/models/masked_autoencoder.py` | Patchify, mask, encode visible patches, restore tokens, decode |
| Temporal CPC | `worldmodels/models/temporal_cpc.py` | Frame latents, causal GRU context, future latent predictions, InfoNCE |
| Episode-safe temporal windows | `worldmodels/data/sequences.py` | Prevent sequence windows from crossing episode boundaries |
| State representation evaluation | `worldmodels/eval/probes.py` | Frozen linear probes, standardization, angle-aware metrics |

The source files are the implementations used by the experiments. The runners provide complete configurations, optimization, checkpoint metadata, evaluation protocols, figures, and machine-readable outputs.

## Experiment and output map

Run scripts from the repository root.

| Runner | Chapter output / purpose |
|---|---|
| `01_collect_dmc_data.py` | Collect/cache the Control Suite tasks; renders Figure 3.2 |
| `02_views_and_pairs.py` | Shows positive/negative view construction; Figure 3.4 |
| `03_ntxent_from_scratch.py` | Compares loop and vectorized NT-Xent loss and gradients; writes verification JSON |
| `04_train_contrastive.py` | Contrastive training, probe evaluation, Figure 3.6 loss curve and Figure 3.7 latent projection |
| `05_aliasing_diagnostic.py` | Single-frame versus two-frame state-decoding diagnostic; Table 3.3 |
| `06_collapse_demo.py` | Collapse demonstration; Figure 3.9 |
| `07_three_mechanisms.py` | Compares stop-gradient, EMA and redundancy reduction |
| `08_vicreg.py` | VICReg training and its force decomposition; Figure 3.11 |
| `09_masked_autoencoder.py` | Masked reconstruction and frozen-feature evaluation; Figure 3.12 |
| `10_bottleneck_sweep.py` | Latent dimension sweep across tasks; Figure 3.14 |
| `11_temporal_cpc.py` | Temporal representation, next-state and missing-frame evaluations; Figure 3.16 and Table 3.5 |
| `12_assemble_encoder.py` | Loads the learned encoder into the Chapter 1 interface and records the synthesis |
| `table_03_04_comparison.py` | Matched comparison of supervised-pretrained, contrastive, VICReg, and MAE representations; Table 3.4 |

## Important interpretation notes

- Training objectives have different scales; do not rank methods by raw loss values.
- Table 3.4 compares complete configurations, not only loss functions: the MAE arm uses a patch-transformer encoder/decoder while other arms use a convolutional encoder.
- Two-frame stacking did not reliably recover velocity in the tested setup. State this as a bounded empirical result, not a general impossibility.
- The CPC missing-frame result is negative and seed-sensitive. The privileged state-history baseline receives ground-truth physics states, unlike CPC's rendered-image input, so it is not a like-for-like comparison.
- The committed figures/results are experiment artifacts. A new run only reproduces them if the configuration, checkpoint and evaluation contract match the recorded experiment.

Read [the publication-readiness audit](publication_readiness_audit.md) and [the corrected findings](discrepancies_and_findings.md) before using quantitative claims in the manuscript.
