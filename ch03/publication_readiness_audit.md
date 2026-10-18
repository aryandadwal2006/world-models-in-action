# Chapter 3 Publication-Readiness Audit

**Updated:** 18 October 2026  
**Scope:** Chapter 3 experiment code, committed result artifacts, reproduction instructions, and the Chapter 3 design document.  
**Repository:** `aryandadwal2006/world-models-in-action`  
**Release candidate:** `diagnostics/ch03-experiment-validation`

## Decision

**Go-ahead for the cleaned code-companion release, with scientific claims kept within the measured scope.** The five-seed mechanism experiment completed, the five trained CPC checkpoints were evaluated on matched 16-frame training/evaluation windows, and the user-reported local test suite passed 59 tests before the release-document changes. The release result artifacts are now identified by stable paths under `ch03/results/`; smoke outputs and superseded CPC gap-rollout figures are not primary results.

This is **not** certification that the finished book chapter is publication-ready. The repository currently contains the Chapter 3 design document and code companion, not the final authored chapter manuscript or production-layout pages. Book prose, citations, figure rendering, captions, and layout must be checked against these artifacts during manuscript production.

The author-reported 59-test pass was on the predecessor branch state. During this cleanup, the unrelated, unrun Table 3.4 version-7 trained-head diagnostic was reverted to the already-published version-6 source and its three diagnostic-only tests were removed; the CPC test file was renamed without changing test logic. The final cleaned tree has not been rerun locally. This is a remaining verification limitation, although the final source tree restores the tested main-branch Table 3.4 implementation and the experiment code behind the 59-test run is unchanged. No GitHub Actions status checks were returned for the branch commits, and the reviewer cannot access the author's Windows environment to run the suite independently.

## 1. Evidence status

| Area | Current evidence | Release interpretation |
|---|---|---|
| Regression tests | User-reported `python -m unittest discover -s tests -v`: 59 tests passed, before documentation/artifact cleanup | Local execution evidence; no remote CI confirmation |
| Three anti-collapse mechanisms | Five seeds, 30 epochs, up to 6,000 Cartpole frames; metrics stored at `ch03/results/three_mechanisms_diagnostics.json` | Reduced-scale mechanism study, not an ImageNet-scale reproduction |
| CPC objective-aligned retrieval | Five trained checkpoints, re-evaluated on 16-frame windows matching training; canonical metrics at `ch03/results/table_03_05_cpc_metrics.json` | Primary Table 3.5 CPC evidence |
| Corrected Table 3.4 | Experiment version 6, method-facing supervised label clarifies a pretrained encoder plus a separate frozen-feature probe | Method-family comparison; not a loss-only ablation |
| MAE representation path | Corrected model exposes mean-pooled final encoder tokens; standalone and table results are stored in canonical JSON files | Keep configuration and architecture caveats visible |
| Historical CPC gap rollout | Custom recursive latent procedure removed from the CPC runner | Not canonical CPC performance; do not use in the current primary table or figure |

## 2. Three-mechanism experiment

Configuration: five seeds (0–4), 30 epochs, batch size 128, Adam at learning rate 0.001, up to 6,000 training images selected across episodes, 32-dimensional encoder latent, and 128-dimensional projector. The augmentations are ±2-pixel shift with brightness/contrast jitter of 0.05. BYOL uses an EMA coefficient scheduled from 0.99 toward 1.0. This is an intentionally reduced-scale DMC study.

| Method | Frozen-encoder probe R², mean ± seed SD | Encoder effective rank, mean ± seed SD | Encoder mean feature SD |
|---|---:|---:|---:|
| SimSiam | 0.266 ± 0.012 | 4.23 ± 0.49 of 32 | 1.933 ± 0.457 |
| BYOL | 0.268 ± 0.004 | 7.07 ± 0.65 of 32 | 0.306 ± 0.078 |
| Barlow Twins | 0.285 ± 0.018 | 11.18 ± 0.62 of 32 | 0.0616 ± 0.0035 |

The training losses fall and the diagnostics distinguish the learned spaces. Barlow Twins has higher encoder effective rank in this run and projector effective rank about 63.42 of 128. Its encoder feature scale is much smaller than SimSiam's, so effective rank and absolute feature spread must be interpreted together. SimSiam, BYOL, and Barlow Twins have broadly similar aggregate probe R²; five seeds do not establish a universal superiority ranking. Barlow Twins' descriptive probe advantage is small relative to seed variability.

Per-variable probe scores are more informative than the aggregate alone. Cart position and pole angle are substantially more decodable than velocity. Mean cart-velocity R² is near zero for all methods; pole-angular-velocity R² is negative for all three. Negative R² means worse performance than predicting the validation-set mean.

**Claim allowed:** under this small DMC configuration, the three objective mechanisms produce different representation geometry and similarly modest aggregate linear-probe results.  
**Claims not supported:** universal ranking of the methods, exact reproduction of original benchmarks, complete physical-state recovery, or proof that high effective rank alone means no representational failure.

The loss implementation includes method-specific projection/prediction heads. Barlow Twins standardizes batch features inside the loss; the artifact explicitly notes that this is objective-aligned but not code-identical to the reference output-BatchNorm implementation. Do not hide this fidelity boundary.

## 3. CPC: final matched-length result

CPC was trained for 20 epochs with five seeds, batch size 32, sequence length 16, latent dimension 16, context dimension 32, and three future horizons. The existing checkpoints were then evaluated without retraining on 200 non-overlapping, episode-safe 16-frame validation windows. Windows are balanced at 40 from each of five validation episodes. There are 4,925 eligible 16-frame validation windows.

| Future horizon | InfoNCE loss, mean ± seed SD | Top-1 retrieval, mean ± seed SD | Top-5 retrieval | Positive–hardest-negative margin |
|---|---:|---:|---:|---:|
| 1 step | 0.115 ± 0.016 | 95.04% ± 0.84 percentage points | 100% | 12.96 ± 0.65 |
| 2 steps | 0.133 ± 0.017 | 94.14% ± 0.72 percentage points | 100% | 11.52 ± 0.54 |
| 3 steps | 0.171 ± 0.018 | 92.44% ± 0.51 percentage points | 100% | 9.50 ± 0.50 |

The reported retrieval is **in-batch future-latent retrieval**, not a 200-candidate global ranking. Six full batches contain 32 candidates each, and the final batch contains eight. Query-weighted top-1 chance is 3.5%. Top-5 is 100% within those batches. The 200 windows are sampled from five episodes; they are not 200 independent trajectories. Across-seed standard deviations capture trained-model variation, not uncertainty across a large population of environments.

The matched 16-frame result remains close to the earlier 25-frame evaluation: high retrieval at all trained horizons, with gradual deterioration as the prediction horizon increases. Thus, the earlier sequence-length mismatch did not explain away the retrieval performance. Use the matched-length JSON as the primary result. Do not treat the earlier 25-frame output as the canonical Table 3.5 metric.

Secondary frozen linear probes are mixed. Current cart-position R² is 0.805 ± 0.084 from the recurrent context and 0.770 ± 0.033 from the static encoder. Mean next-state R² is 0.259 ± 0.016 from context and 0.238 ± 0.009 from the static encoder. Current cart-velocity R² remains close to zero (context 0.016 ± 0.006), and pole-angular-velocity R² is negative. Therefore:

- The result supports the bounded statement that CPC learns to identify held-out future latents under its trained multi-horizon InfoNCE objective.
- It does **not** establish accurate decoding of all physical state variables, action-conditioned prediction, stable long-horizon latent rollout, or planning utility.
- Data are collected under a random action policy, but the model input contains frames, not actions. The lack of action conditioning is a design limitation, not a controlled demonstration that adding actions alone will improve performance.

### Historical CPC gap-rollout result

Earlier code recursively fed CPC's bilinear compatibility output back as though it were a pointwise predicted latent and decoded that vector using a probe trained on observed latents. CPC's InfoNCE objective does not directly train that output as a latent-regression target. Its negative gap R² values are therefore results for a custom diagnostic, not standard CPC performance. The recursive gap evaluator and its misleading primary-figure framing have been removed from the current CPC runner and the superseded result/figure artifacts are excluded from the release tree. Historical numbers must not be substituted for the retrieval metrics above.

## 4. Other Chapter 3 findings retained

### Table 3.4: method-family comparison

The comparison uses two tasks and three seeds; it compares complete model configurations, not loss functions in isolation. The supervised, contrastive, and VICReg arms use convolutional encoders, whereas the MAE arm uses a patch-transformer encoder/decoder.

Corrected results (mean ± population SD) include:

| Method | Cartpole position | Cartpole velocity | Finger position | Finger velocity |
|---|---:|---:|---:|---:|
| Supervised-pretrained encoder + frozen-feature linear probe | 0.981 ± 0.000 | −0.273 ± 0.012 | 0.995 ± 0.001 | −0.114 ± 0.010 |
| Contrastive (NT-Xent) | 0.672 ± 0.001 | −0.187 ± 0.005 | 0.714 ± 0.029 | −0.007 ± 0.002 |
| VICReg | 0.649 ± 0.022 | −0.183 ± 0.006 | 0.679 ± 0.040 | −0.005 ± 0.004 |
| Masked autoencoder | 0.475 ± 0.012 | −0.156 ± 0.005 | 0.383 ± 0.010 | −0.034 ± 0.001 |

The supervised row trains an encoder/head with privileged state labels and then fits a separate frozen-feature linear probe. The displayed score is not the trained head's direct held-out prediction score. Its row label must remain explicit. Finger Spin's spinner hinge angle is π-periodic in the rendered geometry and is excluded from the aggregate because its full-turn absolute angle is not identifiable from a static image.

### MAE representation-path correction

The earlier MAE feature path included a pooled projection that was not trained by the masked reconstruction loss. The corrected probe uses mean-pooled final encoder tokens (64 features). The corrected standalone result reports mean probe R² = 0.170 ± 0.004 over three seeds. Table 3.4 uses the corrected MAE rows above. Do not silently compare the standalone MAE score with Table 3.4 as if the budgets and evaluation setups were identical.

### Two-frame stacking

The corrected Table 3.3 experiment did not show reliable velocity recovery from stacking two frames. Cart-velocity R² changes by only about +0.002 on average, while pole-angular-velocity R² becomes slightly worse. This is a limited result about the tested image resolution, temporal spacing, augmentation, architecture, and training budget, not proof that velocity is fundamentally unrecoverable from multiple frames.

## 5. Data protocol and leakage checks

The reviewed collection script stores rendered RGB frames, qpos followed by qvel, and episode IDs. Physics states are used as targets for evaluation probes and the explicitly supervised reference arm, not as the training target for the self-supervised objectives. Training and validation data are collected through separate environment calls with different seeds. The temporal sequence dataset only yields windows whose frame episode IDs are constant throughout. The CPC primary evaluation additionally checks its 200 selected windows for non-overlap and stores their start indices and episode counts.

The audit found no direct train/validation label leakage in the reviewed probes: probe fits use training features/labels and validation metrics use the separate validation split. This is a source inspection, not an independent end-to-end reproduction.

## 6. Reproduction and output integrity

- Primary Table 3.5 result: `ch03/results/table_03_05_cpc_metrics.json`.
- Secondary 25-frame sensitivity output: `ch03/results/cpc_retrieval_25frame_sensitivity.json`.
- Collapse demonstration metrics: `ch03/results/collapse_demo_metrics.json`.
- Primary expanded mechanism result: `ch03/results/three_mechanisms_diagnostics.json`.
- Table 3.4 source: `ch03/results/table_03_04_objective_comparison.json`.
- Standalone MAE source: `ch03/results/mae_metrics.json`.
- Primary CPC plot: `ch03/figures/cpc_retrieval_matched16_20261018/fig03_16_cpc_future_retrieval.png`.
- Collapse-demo plots: `ch03/figures/fidelity_run_20261017/collapse/`.

The CPC JSON stores its model seeds, training settings, matched evaluation length, selected window starts, episode balance, candidate-set policy, and prior run configuration. Five CPC checkpoints are retained at `checkpoints/cpc_retrieval_20261018/` to support checkpoint-only re-evaluation. Smoke-test outputs and redundant timestamped backups are not release results.

The 59-test result was reported by the author for the pre-cleanup branch and was not independently reproduced here. The exact cleaned tree has not been rerun after removing three Table 3.4 version-7 diagnostic tests and renaming the CPC retrieval test file. GitHub returned no CI statuses or PR-triggered workflow runs for the latest branch commits. Rerun `python -m unittest discover -s tests -v` after any code changes. Figure 3.16 should receive final visual/layout review in the book-production workflow; code/data inspection cannot replace that visual check.

## Readiness statement

The cleaned code companion and its current experiment artifacts are **ready to merge for book development**. The evidence supports the bounded claims above. The finished Chapter 3 prose is not part of this repository snapshot, so it must still be proofed against these outputs and the Manning production requirements. Avoid claims of universal method superiority, complete state recovery, action-conditioned world modeling, or missing-frame rollout performance from the current CPC experiment.
