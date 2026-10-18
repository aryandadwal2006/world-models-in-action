# Chapter 3 Findings Ledger

**Updated:** 18 October 2026  
**Purpose:** Maintain one current account of empirical findings, fixes, and interpretation limits. This ledger supersedes earlier dated status notes in the commit history.

## Release status

The expanded mechanism experiment and the corrected CPC evaluation have run. The canonical result artifacts are:

- `ch03/results/three_mechanisms_diagnostics.json`
- `ch03/results/table_03_05_cpc_metrics.json`
- `ch03/results/table_03_04_objective_comparison.json`
- `ch03/results/mae_metrics.json`

The user-reported local regression run passed 59 tests on the predecessor branch state. Cleanup reverted the unrun Table 3.4 version-7 trained-head diagnostic and its three tests; the CPC test file was renamed without changing logic. The exact cleaned tree has not been rerun. No GitHub CI status checks were available for this branch, and the experiments were not independently rerun by the reviewer.

## Findings and dispositions

### 1. Capped sampling previously produced prefix bias

The prior first-N selection could leave training or validation examples concentrated in only a few early episodes. The corrected deterministic sampler selects across episode IDs and within-episode time. Temporal sequence sampling preserves contiguous windows in the original arrays and rejects windows that cross episode boundaries.

**Disposition:** corrected sampling is used by the current experiment runners. Keep episode-aware selection and dataset metadata as part of reproduction.

### 2. Two-frame stacking did not reliably recover velocity

Corrected Table 3.3 results are:

| Variable | Single frame | Two-frame stack | Mean change |
|---|---:|---:|---:|
| Cart position | 0.714 ± 0.040 | 0.707 ± 0.010 | −0.007 |
| Pole angle | 0.612 ± 0.069 | 0.586 ± 0.064 | −0.026 |
| Cart velocity | 0.006 ± 0.004 | 0.009 ± 0.007 | +0.002 |
| Pole angular velocity | −0.367 ± 0.001 | −0.375 ± 0.006 | −0.007 |

This is a negative or limited result for the specific resolution, augmentation, architecture, temporal spacing, and training budget. It does not establish a general impossibility of recovering velocity from multiple frames. Present positions as more linearly decodable than velocities in this tested setup; do not claim temporal stacking succeeded.

### 3. Table 3.4 compares complete method configurations

Corrected values (mean ± population SD across three seeds):

| Method | Cartpole position | Cartpole velocity | Finger position | Finger velocity |
|---|---:|---:|---:|---:|
| Supervised-pretrained encoder + frozen-feature linear probe | 0.981 ± 0.000 | −0.273 ± 0.012 | 0.995 ± 0.001 | −0.114 ± 0.010 |
| Contrastive (NT-Xent) | 0.672 ± 0.001 | −0.187 ± 0.005 | 0.714 ± 0.029 | −0.007 ± 0.002 |
| VICReg | 0.649 ± 0.022 | −0.183 ± 0.006 | 0.679 ± 0.040 | −0.005 ± 0.004 |
| Masked autoencoder | 0.475 ± 0.012 | −0.156 ± 0.005 | 0.383 ± 0.010 | −0.034 ± 0.001 |

The supervised row is an encoder pretrained with state labels followed by a separate frozen-feature probe. It is not the trained state-prediction head's own held-out score. The result should not be called an oracle or a direct head-performance metric. The comparison also uses different backbone families (CNN versus patch-transformer MAE), so it is a comparison of method configurations, not a loss-only ablation.

Finger Spin's third qpos coordinate is a spinner hinge angle with π-periodic rendered symmetry. The full-turn coordinate is not identifiable from one static frame and is excluded from the aggregate while remaining documented at the variable level.

### 4. MAE representation-path fix

The earlier probe exposed a pooled projection that the masked reconstruction loss did not train. The corrected representation is the mean-pooled final encoder tokens, 64 features in this configuration. Corrected standalone MAE probe R² is 0.170 ± 0.004 across three seeds. Table 3.4's MAE values were regenerated with the corrected path; the older rows were preserved in the progress artifact's explicitly named legacy archive.

**Disposition:** keep the corrected feature path and its regression test. Do not compare the standalone MAE score with Table 3.4 as though both runs had identical budgets and evaluation protocols.

### 5. Collapse demonstration

The final recorded three-seed collapse demonstration (15 epochs) reports attraction-only encoder feature SD of approximately (7.79\times10^{-5}) and near-zero attraction loss (about (2.55\times10^{-10})). Under the contrastive condition, final encoder feature SD is approximately 2.42. This illustrates that the deliberately underconstrained attraction-only loss can converge to a constant-like representation while the contrastive condition avoids that particular collapse in this setup.

Effective rank is scale-invariant and can remain nonzero for tiny but structured numerical variation. Report feature spread and rank together; do not let rank by itself override the near-zero spread and loss evidence.

### 6. Expanded SimSiam/BYOL/Barlow Twins mechanism study

The final reduced-scale study uses five seeds, 30 epochs, up to 6,000 Cartpole frames, batch size 128, 32-dimensional encoder latents, and 128-dimensional projectors.

| Method | Frozen-encoder probe R² | Encoder effective rank (32-d output) | Mean encoder feature SD |
|---|---:|---:|---:|
| SimSiam | 0.266 ± 0.012 | 4.23 ± 0.49 | 1.933 ± 0.457 |
| BYOL | 0.268 ± 0.004 | 7.07 ± 0.65 | 0.306 ± 0.078 |
| Barlow Twins | 0.285 ± 0.018 | 11.18 ± 0.62 | 0.0616 ± 0.0035 |

Barlow Twins uses more encoder directions by effective rank in this run and has projector effective rank about 63.42 out of 128. Its feature scale is low relative to the other methods; spread and rank must be interpreted together. Aggregate probe scores are close and five seeds do not establish universal superiority. The implementation follows the core mechanisms at small scale and is not an exact reproduction of the original ImageNet-scale papers.

### 7. CPC's primary evidence is future-latent retrieval, not recursive gap rollout

The primary run trains canonical multi-horizon bilinear InfoNCE and then evaluates the matching held-out future latent against other future latents in the same batch. The training sequence length and evaluation sequence length both equal 16. The evaluation uses five trained checkpoints, three future horizons, 200 non-overlapping episode-safe validation windows, and 40 windows from each of five episodes.

| Horizon | InfoNCE loss | Top-1 retrieval | Top-5 retrieval | Positive–hardest-negative margin |
|---|---:|---:|---:|---:|
| 1 | 0.115 ± 0.016 | 95.04% ± 0.84 pp | 100% | 12.96 ± 0.65 |
| 2 | 0.133 ± 0.017 | 94.14% ± 0.72 pp | 100% | 11.52 ± 0.54 |
| 3 | 0.171 ± 0.018 | 92.44% ± 0.51 pp | 100% | 9.50 ± 0.50 |

The top-1 chance level is 3.5% because each query is evaluated against in-batch candidates: 32 candidates for full batches and eight in the last batch. This is not a 200-way retrieval pool. The windows are non-overlapping, but they come from five validation episodes and should not be described as 200 independent trajectories.

State-probe results remain mixed. Recurrent context has current cart-position R² of 0.805 ± 0.084 versus 0.770 ± 0.033 for the static encoder; mean next-state R² is 0.259 ± 0.016 versus 0.238 ± 0.009. Context current cart-velocity R² is 0.016 ± 0.006, and pole-angular-velocity R² is negative. CPC's successful retrieval task therefore does not mean all physical variables are linearly recoverable or that the model is a complete world model.

The former custom recursive gap rollout used a bilinear compatibility output as if it were a pointwise predicted latent, then passed it to a probe trained on observed latents. That is not what the InfoNCE objective directly trains. Its negative gap scores are historical results for that custom procedure and are excluded from the primary CPC runner and Table 3.5 claims. Do not present those scores as standard CPC performance.

The model receives frames, not actions, and the data were collected under random actions. These results do not test action-conditioned prediction, long-horizon latent rollout, or planning. The matched 16-frame results are primary; the earlier 25-frame evaluation is a sensitivity diagnostic.

### 8. Remaining manuscript-production checks

These are not blockers for merging the cleaned code companion, but they remain necessary for the finished book:

1. Insert the current numbers and method caveats from the canonical JSON files into the authored manuscript and table captions.
2. Visually inspect the canonical CPC retrieval figure and the final collapse figures in production layout.
3. Keep the Chapter 2 forklift-occlusion example as conceptual motivation only. The present CPC result measures future-latent retrieval and does not recreate an occlusion or missing-frame experiment.
4. Re-run the test suite after any source-code changes. Changes in this release's documentation/artifact cleanup do not alter the experiment code.

## Canonical paths

- CPC: `ch03/results/table_03_05_cpc_metrics.json`
- Three mechanisms: `ch03/results/three_mechanisms_diagnostics.json`
- Table 3.4: `ch03/results/table_03_04_objective_comparison.json`
- Standalone MAE: `ch03/results/mae_metrics.json`
- CPC plot: `ch03/figures/cpc_retrieval_matched16_20261018/fig03_16_cpc_future_retrieval.png`
- Collapse plots: `ch03/figures/fidelity_run_20261017/collapse/`
