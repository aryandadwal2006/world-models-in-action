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

## Method-fidelity boundary

The implementation follows the central objective/mechanism in each cited paper, but it is not a claim of exact benchmark reproduction.

- [SimSiam (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html) and its [reference projector/predictor implementation](https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py): the demo now includes the BatchNorm structure, predictor, and stop-gradient cosine objective. It uses a small CNN and narrow MLPs, not the paper's large-scale setup.
- [BYOL (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html): the demo keeps an online branch, stop-gradient EMA target branch, predictor, and cosine objective. The target's BatchNorm buffers are updated alongside EMA parameters. Its reduced-width architecture and short DMC training are illustrative.
- [Barlow Twins (PMLR 2021)](https://proceedings.mlr.press/v139/zbontar21a.html): the demo now uses a 3-layer MLP with BatchNorm hidden layers and the cross-correlation objective, at a small output dimension and on DMC frames rather than the original large-scale benchmark.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748): training uses multi-horizon InfoNCE with raw bilinear dot-product scores, without L2-normalizing encoder/prediction vectors or applying a temperature. The one-step rollout diagnostic uses the learned first head w_k[0]. The old committed CPC diagnostic run used unit normalization, temperature scaling, and an additional one-step latent MSE term. Keep it as historical evidence, not as the canonical CPC result. Use a new checkpoint/results directory; checkpoint-only mode intentionally refuses the old objective contract.

A correct objective does not imply a successful experiment. Report the observed outcomes at their actual scope; do not tune hyperparameters until they support the desired narrative.

## Important interpretation notes

- Training objectives have different scales; do not rank methods by raw loss values.
- Table 3.4 compares complete configurations, not only loss functions: the MAE arm uses a patch-transformer encoder/decoder while other arms use a convolutional encoder.
- MAE linear probes use the mean-pooled final encoder tokens (64 features in the current configuration), not a separate pooled projection that the reconstruction loss never trains. The corrected standalone three-seed run reports mean probe R² = 0.170 ± 0.004. The current Table 3.4 MAE results are Cartpole position 0.475 ± 0.012, Cartpole velocity −0.156 ± 0.005, Finger position 0.383 ± 0.010, and Finger velocity −0.034 ± 0.001. These results are configuration-specific; see the committed JSON artifacts.
- Two-frame stacking did not reliably recover velocity in the tested setup. State this as a bounded empirical result, not a general impossibility.
- The CPC missing-frame result is negative and seed-sensitive. The privileged state-history baseline receives ground-truth physics states, unlike CPC's rendered-image input, so it is not a like-for-like comparison.
- The committed figures/results are experiment artifacts. A new run only reproduces them if the configuration, checkpoint and evaluation contract match the recorded experiment.

Read [the publication-readiness audit](publication_readiness_audit.md) and [the corrected findings](discrepancies_and_findings.md) before using quantitative claims in the manuscript.

## Targeted follow-up diagnostics

The branch diagnostics/ch03-experiment-validation adds diagnostics rather than tuning hyperparameters toward expected scores:

- 06_collapse_demo.py records the randomly initialized model at diagnostic epoch 0, then encoder and objective-space spread/rank after each training epoch. Effective rank is scale-invariant, so read it alongside absolute spread.
- 07_three_mechanisms.py uses method-specific BatchNorm heads: a SimSiam-style 3-layer projector plus predictor, online/EMA BYOL heads, and a 3-layer Barlow Twins projector. The backbone, embedding widths, data and training budget are deliberately small; this is a mechanism demonstration, not a reproduction of the original ImageNet-scale results.
- table_03_04_comparison.py remains a bounded comparison of complete model configurations, not a loss-only ablation. Its existing outputs need not be rerun to validate the architecture fixes described here.
- 11_temporal_cpc.py now trains only the canonical multi-horizon bilinear InfoNCE objective over every valid context position, using raw encoder latents and unscaled dot-product scores. The first learned bilinear head w_k[0] supplies the one-step latent rollout diagnostic; there is no auxiliary one-step MSE predictor. Checkpoint-only evaluation validates the experiment version, raw-latent setting, and objective contract before loading any checkpoint.

Recommended sequence from the repository root:

```powershell
python -m unittest discover -s tests -v

python ch03/06_collapse_demo.py --results-dir ch03/results/fidelity_run_20261017/collapse --figures-dir ch03/figures/fidelity_run_20261017/collapse

python ch03/07_three_mechanisms.py --results-dir ch03/results/fidelity_run_20261017/mechanisms

python ch03/11_temporal_cpc.py --gap-sequences 200 --results-dir ch03/results/cpc_canonical_20261017 --checkpoints-dir checkpoints/cpc_canonical_20261017 --figures-dir ch03/figures/cpc_canonical_20261017

python ch03/11_temporal_cpc.py --evaluate-checkpoints --gap-sequences 200 --results-dir ch03/results/cpc_canonical_20261017 --checkpoints-dir checkpoints/cpc_canonical_20261017 --figures-dir ch03/figures/cpc_canonical_20261017 --evaluation-output ch03/results/cpc_canonical_20261017/checkpoint_only_validation.json
```

These are diagnostic runs, not parameter searches. They write to isolated paths so the previously committed outputs remain untouched. The CPC command trains a fresh run with the canonical objective and then reevaluates those exact checkpoints; the final command is a checkpoint-integrity check, not another training run. On reruns, choose new run-directory names to preserve outputs. Do not change settings just to make scores match the narrative.