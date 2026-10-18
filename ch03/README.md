# Chapter 3: Code map and experiment record

This directory contains the experiment runners behind Chapter 3 of *World Models in Action*. The code listings in the book should point to the canonical implementation modules below; the runners preserve the full protocols used for figures, tables, and machine-readable results.

## Book listing → canonical source

| Concept | Canonical source | What to study |
|---|---|---|
| Image augmentations and positive views | `worldmodels/data/augment.py` | Crop shifts and appearance perturbations |
| Convolutional visual encoder | `worldmodels/models/encoders.py` | CNN layers, latent projection, optional recurrent agent interface |
| NT-Xent / InfoNCE | `worldmodels/losses/ntxent.py` | Similarities, positive-pair identification, cross-entropy |
| VICReg | `worldmodels/losses/vicreg.py` | Invariance, variance hinge, covariance regularization |
| Stop-gradient, EMA, redundancy reduction | `worldmodels/losses/noncontrastive.py` | SimSiam, BYOL target update, Barlow Twins |
| Masked reconstruction | `worldmodels/losses/reconstruction.py` | Reconstruction loss on masked patches |
| Masked autoencoder | `worldmodels/models/masked_autoencoder.py` | Patchify, mask, encode visible patches, restore tokens, decode |
| Temporal CPC | `worldmodels/models/temporal_cpc.py` | Frame latents, causal GRU context, bilinear future-latent scores, InfoNCE |
| Episode-safe temporal windows | `worldmodels/data/sequences.py` | Keep every sequence inside one episode |
| State representation evaluation | `worldmodels/eval/probes.py` | Frozen linear probes, feature standardization, angle-aware metrics |

The source modules are the implementations used by the experiments. The runners add experiment configuration, optimization, checkpoint metadata, evaluation protocols, figures, and machine-readable outputs.

## Experiment and output map

Run scripts from the repository root.

| Runner | Purpose |
|---|---|
| `01_collect_dmc_data.py` | Collect/cache Control Suite tasks and render Figure 3.2 |
| `02_views_and_pairs.py` | Show positive/negative view construction |
| `03_ntxent_from_scratch.py` | Compare loop and vectorized NT-Xent values and gradients |
| `04_train_contrastive.py` | Contrastive training, probes, and latent-space figures |
| `05_aliasing_diagnostic.py` | Compare single-frame with stacked-frame state probes |
| `06_collapse_demo.py` | Demonstrate collapse under attraction-only training and compare a contrastive objective |
| `07_three_mechanisms.py` | Compare SimSiam, BYOL, and Barlow Twins mechanism diagnostics |
| `08_vicreg.py` | Train VICReg and inspect its loss components |
| `09_masked_autoencoder.py` | Masked reconstruction and frozen-feature evaluation |
| `10_bottleneck_sweep.py` | Sweep latent dimensions across selected tasks |
| `11_temporal_cpc.py` | Train temporal CPC; evaluate in-batch future-latent retrieval and state probes |
| `12_assemble_encoder.py` | Check the encoder interface used by the earlier chapter |
| `table_03_04_comparison.py` | Generate the multi-method comparison in Table 3.4 |

## Current primary artifacts

- `ch03/results/table_03_04_objective_comparison.json`: corrected Table 3.4 source metrics.
- `ch03/results/mae_metrics.json`: corrected standalone MAE probe metrics.
- `ch03/results/collapse_demo_metrics.json`: three-seed attraction-only collapse demonstration.
- `ch03/results/three_mechanisms_diagnostics.json`: expanded five-seed SimSiam/BYOL/Barlow Twins diagnostics.
- `ch03/results/table_03_05_cpc_metrics.json`: primary matched-length CPC retrieval result (16-frame training and evaluation windows).
- `ch03/results/cpc_retrieval_25frame_sensitivity.json`: secondary sequence-length sensitivity check; not the primary Table 3.5 result.
- `ch03/figures/fidelity_run_20261017/collapse/`: collapse-demo figures from the recorded run.
- `ch03/figures/cpc_retrieval_matched16_20261018/fig03_16_cpc_future_retrieval.png`: primary CPC retrieval figure.

The diagnostic run summaries are also described in `ch03/publication_readiness_audit.md` and `ch03/discrepancies_and_findings.md`. Timestamped smoke-test outputs are not primary results and are excluded from the release artifact set.

## Result summary and interpretation boundaries

### Three anti-collapse mechanisms

The reduced-scale DMC study uses five seeds, 30 epochs, up to 6,000 training images, a batch size of 128, and a small CNN with method-specific projection/prediction heads.

| Method | Frozen-encoder probe R², mean ± seed SD | Encoder effective rank, mean ± seed SD |
|---|---:|---:|
| SimSiam | 0.266 ± 0.012 | 4.23 ± 0.49 (32-dimensional encoder output) |
| BYOL | 0.268 ± 0.004 | 7.07 ± 0.65 (32-dimensional encoder output) |
| Barlow Twins | 0.285 ± 0.018 | 11.18 ± 0.62 (32-dimensional encoder output) |

Barlow Twins has higher effective rank in this run; its mean probe advantage is small relative to seed variability. Its projector effective rank is 63.42 out of 128 while its encoder feature scale is comparatively small, so rank and absolute spread must be read together. These runs demonstrate differing mechanisms and representation statistics under a small DMC setup. They are not exact reproductions of the original ImageNet-scale results and do not establish a universal ranking.

### Temporal CPC

The CPC checkpoints were trained for 20 epochs with sequence length 16, batch size 32, latent dimension 16, context dimension 32, and three future horizons. The primary evaluation reuses those checkpoints on 200 non-overlapping, episode-safe 16-frame validation windows balanced across five episodes (40 windows per episode).

| Future horizon | InfoNCE loss, mean ± SD | Top-1 retrieval, mean ± SD | Top-5 retrieval |
|---|---:|---:|---:|
| 1 step | 0.115 ± 0.016 | 95.04% ± 0.84 percentage points | 100% |
| 2 steps | 0.133 ± 0.017 | 94.14% ± 0.72 percentage points | 100% |
| 3 steps | 0.171 ± 0.018 | 92.44% ± 0.51 percentage points | 100% |

Retrieval candidates are in-batch, not a single 200-way candidate pool. There are 32 candidates in full batches and eight in the final batch; the query-weighted random top-1 chance is 3.5%. The 200 windows are drawn from five validation episodes, not 200 independent trajectories. Across-seed standard deviations measure model-seed variability, not uncertainty across a large population of environments.

The recurrent context provides only a modest improvement in average current-position and next-state probe R² over the static encoder. Velocity decoding remains weak, including negative R² for pole angular velocity. CPC's successful future-latent identification is not proof of complete physical-state decoding, action-conditioned prediction, stable latent rollout, or planning utility. The training data use random actions, but the model itself receives frames only.

## Method-fidelity boundary

The implementations follow the core objective/mechanism, not the full benchmark scale or every implementation detail of the original papers.

- [SimSiam (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html): stop-gradient cosine loss, predictor, and BatchNorm-based projector in a reduced-scale setup.
- [BYOL (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html): online branch, EMA target, predictor, and cosine objective; reduced architecture and DMC data.
- [Barlow Twins (PMLR 2021)](https://proceedings.mlr.press/v139/zbontar21a.html): cross-correlation redundancy-reduction objective with a reduced-width projector on DMC images.
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748): causal GRU context and multi-horizon raw bilinear InfoNCE scores, without L2-normalizing encoder latents. Evaluation asks whether the correct held-out future latent ranks above other in-batch candidates, using the trained compatibility scores. The earlier recursive gap-rollout diagnostic is not a canonical CPC metric and is not the primary Table 3.5 result.

A correct implementation does not make a scientific claim true by itself. Report the results at their measured scope, preserve negative or mixed outcomes, and do not tune hyperparameters merely to obtain a preferred narrative.

## Reproduction

Install Python 3.11 or newer and the pinned dependencies in `requirements.txt`. From the repository root, first run the regression tests:

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Stop here." }
```

To reproduce the expanded three-method study in a new output location:

```powershell
python ch03/07_three_mechanisms.py --seeds 0 1 2 3 4 --epochs 30 --batch-size 128 --max-train-samples 6000 --results-dir ch03/results/reproduction_mechanisms
if ($LASTEXITCODE -ne 0) { throw "Mechanism experiment failed." }
```

To train CPC from scratch with matched 16-frame training/evaluation windows and isolated outputs:

```powershell
python ch03/11_temporal_cpc.py --seeds 0 1 2 3 4 --epochs 20 --batch-size 32 --seq-len 16 --eval-seq-len 16 --latent-dim 16 --context-dim 32 --k-steps 3 --max-train-sequences 4000 --max-val-sequences 1000 --retrieval-sequences 200 --results-dir ch03/results/reproduction_cpc --checkpoints-dir checkpoints/reproduction_cpc --figures-dir ch03/figures/reproduction_cpc
if ($LASTEXITCODE -ne 0) { throw "CPC experiment failed." }
```

To evaluate the recorded checkpoints without training again, use `--evaluate-checkpoints`, the same training configuration, `--checkpoints-dir checkpoints/cpc_retrieval_20261018`, and a **new** `--evaluation-output` path. Do not point an evaluation run at the primary JSON output path; keep the final record immutable and compare new evaluations separately.

Smoke runs, if needed for API checks, should write to a separate directory and should not be committed as primary results.

See [the publication-readiness audit](publication_readiness_audit.md) and [the corrected findings ledger](discrepancies_and_findings.md) for methods, provenance, and limitations.
