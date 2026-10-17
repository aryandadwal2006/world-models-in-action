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
| `11_temporal_cpc.py` | Temporal representation and next-state probes plus CPC-native future-latent retrieval; Figure 3.16 and Table 3.5 |
| `12_assemble_encoder.py` | Loads the learned encoder into the Chapter 1 interface and records the synthesis |
| `table_03_04_comparison.py` | Matched comparison of supervised-pretrained, contrastive, VICReg, and MAE representations; Table 3.4 |

## Method-fidelity boundary

The implementation follows the central objective/mechanism in each cited paper, but it is not a claim of exact benchmark reproduction.

- [SimSiam (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html) and its [reference projector/predictor implementation](https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py): the demo now includes the BatchNorm structure, predictor, and stop-gradient cosine objective. It uses a small CNN and narrow MLPs, not the paper's large-scale setup.
- [BYOL (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html): the demo keeps an online branch, stop-gradient EMA target branch, predictor, and cosine objective. The target's BatchNorm buffers are updated alongside EMA parameters. Its reduced-width architecture and short DMC training are illustrative.
- [Barlow Twins (PMLR 2021)](https://proceedings.mlr.press/v139/zbontar21a.html): the demo now uses a 3-layer MLP with BatchNorm hidden layers and the cross-correlation objective, at a small output dimension and on DMC frames rather than the original large-scale benchmark.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748): training uses multi-horizon InfoNCE with raw bilinear dot-product scores over all valid context positions, without L2-normalizing encoder latents or applying a temperature. Evaluation uses the same learned compatibility scores to retrieve the correct held-out future latent from in-batch candidates at each trained horizon, reporting InfoNCE loss, top-1/top-5 retrieval, chance, and the positive-versus-hardest-negative margin. The former recursive "roll out the bilinear output as a latent and decode a missing-frame gap" experiment is removed from the main CPC runner because CPC does not train that output with pointwise latent regression. Treat its committed gap numbers as historical results for that custom diagnostic, not as standard CPC performance.

A correct objective does not imply a successful experiment. Report the observed outcomes at their actual scope; do not tune hyperparameters until they support the desired narrative.

## Important interpretation notes

- Training objectives have different scales; do not rank methods by raw loss values.
- Table 3.4 compares complete configurations, not only loss functions: the MAE arm uses a patch-transformer encoder/decoder while other arms use a convolutional encoder.
- MAE linear probes use the mean-pooled final encoder tokens (64 features in the current configuration), not a separate pooled projection that the reconstruction loss never trains. The corrected standalone three-seed run reports mean probe R² = 0.170 ± 0.004. The current Table 3.4 MAE results are Cartpole position 0.475 ± 0.012, Cartpole velocity −0.156 ± 0.005, Finger position 0.383 ± 0.010, and Finger velocity −0.034 ± 0.001. These results are configuration-specific; see the committed JSON artifacts.
- Two-frame stacking did not reliably recover velocity in the tested setup. State this as a bounded empirical result, not a general impossibility.
- CPC should be judged first by future-latent retrieval under its trained InfoNCE objective. Linear state probes are secondary measures of what the latent/context makes decodable. The retired custom recursive gap rollout is not a canonical CPC metric.
- The committed figures/results are experiment artifacts. A new run only reproduces them if the configuration, checkpoint and evaluation contract match the recorded experiment.

Read [the publication-readiness audit](publication_readiness_audit.md) and [the corrected findings](discrepancies_and_findings.md) before using quantitative claims in the manuscript.

## Targeted follow-up diagnostics

The branch diagnostics/ch03-experiment-validation adds diagnostics rather than tuning hyperparameters toward expected scores:

- 06_collapse_demo.py records the randomly initialized model at diagnostic epoch 0, then encoder and objective-space spread/rank after each training epoch. Effective rank is scale-invariant, so read it alongside absolute spread.
- 07_three_mechanisms.py uses a shared lightweight CNN but method-specific SimSiam, BYOL, and Barlow Twins heads. The rerun budget is 5 seeds, 30 epochs, up to 6,000 training images, a 128-sample batch, and wider heads. It records loss plus encoder/projector spread and effective rank at early, middle, and final checkpoints, alongside held-out per-state linear-probe scores. It remains a small DMC mechanism study, not an exact reproduction of the original ImageNet-scale benchmarks; effective rank must be read with absolute spread.
- table_03_04_comparison.py remains a bounded comparison of complete model configurations, not a loss-only ablation. Its existing outputs need not be rerun to validate the architecture fixes described here.
- 11_temporal_cpc.py trains canonical multi-horizon bilinear InfoNCE and evaluates held-out future-latent retrieval at each trained horizon. The caps are explicitly named `--max-train-sequences` and `--max-val-sequences` because they count sequence windows, not individual frames; older JSON artifacts retain the historical `max_*_frames` config keys. State-probe R² is reported per state variable and seed.

Recommended commands from the repository root. First run tests and short smoke experiments. Then run the larger multi-seed experiments into fresh, isolated output directories; do not tune based on whether results confirm expectations.

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Do not run the experiments." }

# Smoke tests: catch API and data-loader errors before the larger runs.
python ch03/07_three_mechanisms.py --seeds 0 --epochs 1 --batch-size 32 --max-train-samples 256 --latent-dim 16 --hidden-dim 64 --projector-dim 32 --results-dir ch03/results/mechanism_smoke_20261018
if ($LASTEXITCODE -ne 0) { throw "Mechanism smoke test failed." }

python ch03/11_temporal_cpc.py --seeds 0 --epochs 1 --batch-size 8 --seq-len 16 --eval-seq-len 25 --latent-dim 16 --context-dim 32 --k-steps 3 --max-train-sequences 64 --max-val-sequences 32 --retrieval-sequences 20 --results-dir ch03/results/cpc_smoke_20261018 --checkpoints-dir checkpoints/cpc_smoke_20261018 --figures-dir ch03/figures/cpc_smoke_20261018
if ($LASTEXITCODE -ne 0) { throw "CPC smoke test failed." }

# Main diagnostic runs: fresh seeds/checkpoints and no reused outputs.
python ch03/07_three_mechanisms.py --seeds 0 1 2 3 4 --epochs 30 --batch-size 128 --max-train-samples 6000 --results-dir ch03/results/mechanism_diagnostics_20261018
if ($LASTEXITCODE -ne 0) { throw "Mechanism experiment failed." }

python ch03/11_temporal_cpc.py --seeds 0 1 2 3 4 --epochs 20 --batch-size 32 --seq-len 16 --eval-seq-len 25 --latent-dim 16 --context-dim 32 --k-steps 3 --max-train-sequences 4000 --max-val-sequences 1000 --retrieval-sequences 200 --results-dir ch03/results/cpc_retrieval_20261018 --checkpoints-dir checkpoints/cpc_retrieval_20261018 --figures-dir ch03/figures/cpc_retrieval_20261018
if ($LASTEXITCODE -ne 0) { throw "CPC experiment failed." }
```

The mechanism runner reports per-seed validation scores and training trajectories rather than only a final aggregate. CPC reports retrieval accuracy against its actual InfoNCE compatibility objective; it does not use a recursive point-prediction rollout. A successful run is not evidence that a claim is true by itself: interpret retrieval against the random-candidate chance level, check seed-to-seed stability, inspect per-variable probes and collapse diagnostics, and preserve any negative or mixed result.
