# Chapter 3 Audit Findings — Current Reproducibility and Metric Review

**Audit basis:** repository state at 230b298b9d1ed031e69e6987bb9c2a2e88bc8086 (results saved 2026-10-08), plus the corrective source changes on branch audit/chapter3-methodology-fixes. This document supersedes numerical conclusions in the older draft. It does not claim that the corrective branch has already passed local tests or that its regenerated Table 3.4/CPC outputs have been run.

## 1. Confirmed issues

### 1.1 The Finger Spin state contract was documented incorrectly

worldmodels/data/dmc_data.py stores [qpos, qvel] directly from MuJoCo. The collected Finger Spin target therefore has six values: three generalized positions followed by three generalized velocities. It is not the four-dimensional “pos/vel/touch” target described in the previous version of chapter_03_design.md. The design file has been corrected on the audit branch.

The task's own observation implementation uses proximal/distal angles, spinner-tip position, velocities, and (in the official observation) touch values; it does not expose the raw spinner hinge angle as an observed position. In the collection code, however, raw qpos is deliberately used as privileged evaluation truth. The manuscript must distinguish that privileged target from the pixels and from the suite's observation dictionary.

### 1.2 The absolute spinner hinge angle is not identifiable from the rendered frame

In the official DMC model, the spinner has two identical caps placed at opposite offsets (cap1 and cap2) and a symmetric cylinder decoration. The Finger Spin task also sets the alpha of the tip and target sites to zero. The rendered geometry is therefore invariant to a half-turn of the spinner: its absolute 2π-periodic hinge coordinate is not uniquely recoverable from one static frame. See the upstream [finger.xml](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/finger.xml) and [finger.py](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/finger.py).

The previous Table 3.4 included this coordinate in its Finger Spin position average. That one variable had very large negative scores (roughly −130 to −230 per seed in saved results), overwhelming the other coordinates and yielding position averages around −60 to −80. Those aggregate values are not defensible as a comparison of representation objectives.

**Correction on the audit branch:** the probe now treats this rendered orientation as π-periodic when it is used for angle-aware training/evaluation; Table 3.4 preserves the legacy full-turn score for audit provenance but excludes this unidentifiable variable from aggregate position/overall scores. The three Finger Spin supervised-reference runs are invalidated and will be retrained with the corrected target encoding. The other nine Finger Spin objective runs are retained; their per-variable outputs do not depend on the spinner-angle output column of the independently fitted multi-target Ridge probe.

### 1.3 Temporal CPC gap scores were sampled from only the beginning of validation

The old evaluate_gap took the first 32 eligible windows from a sequential DataLoader. The windows overlap and come from the beginning of the validation trajectory, where target variation can be unrepresentative. The saved gap scores were:

- CPC gap mean R²: seed values approximately −106,939, −231,451, and −2,171.
- Linear state-history gap mean R²: approximately −1,015.93 for every seed.

These results are not suitable for the manuscript. Their magnitude is especially sensitive to evaluation-window selection and local target variance; they do not establish a general failure of CPC.

**Correction on the audit branch:** ch03/11_temporal_cpc.py selects a deterministic, evenly spaced set of validation windows across the eligible sequence dataset, records that sampling policy, and supports --evaluate-checkpoints. This lets the existing trained CPC checkpoints be re-evaluated without retraining the CPC models. The old gap scores must remain labelled superseded until the revised evaluation is run.

### 1.4 Table 3.4 and the chapter notes were stale

The saved Table 3.4 JSON and progress file used probe version 3. Their cartpole results are still usable with the existing state-angle convention; their Finger Spin spinner-angle aggregate is not. The old findings in this file mixed results from multiple earlier runs and must not be quoted. The design file also promised every script would finish in under ten minutes on CPU, which the complete multi-seed comparison, capacity sweep and CPC experiment do not satisfy; that promise has been corrected.

## 2. Results from the saved runs that remain relevant

Unless noted otherwise, these are the current main-branch outputs, not regenerated outputs from the audit branch.

### 2.1 Table 3.3: single frame versus two stacked frames

Means ± population standard deviation over seeds 0, 1 and 2:

| State variable | Single frame | Two-frame stack | Difference |
|---|---:|---:|---:|
| Cart position | 0.582 ± 0.123 | 0.500 ± 0.137 | −0.081 |
| Pole angle | 0.595 ± 0.031 | 0.564 ± 0.050 | −0.031 |
| Cart velocity | −0.485 ± 0.445 | −0.507 ± 0.281 | −0.022 |
| Pole angular velocity | −0.543 ± 0.095 | −0.560 ± 0.121 | −0.017 |

Under this exact setup, two-frame stacking did not improve the velocity probes. State this as the measured result, not as proof that stacking can never recover motion.

### 2.2 Table 3.4: what can be said before the corrected outputs are generated

The existing cartpole position/velocity summaries under the matched Table 3.4 setup (12 epochs, 3,500 training frames, 1,000 validation frames, seeds 0–2) are:

| Objective | Position mean R² | Velocity mean R² |
|---|---:|---:|
| Supervised State-Prediction Reference | 0.967 ± 0.011 | −1.849 ± 0.113 |
| Contrastive (NT-Xent) | −0.065 ± 0.780 | −0.492 ± 0.025 |
| VICReg | 0.616 ± 0.067 | −0.477 ± 0.095 |
| Masked Autoencoder (MAE) | −0.767 ± 0.033 | −0.119 ± 0.013 |

For Finger Spin, the old supervised-reference row must be retrained. The saved non-supervised per-variable outputs, re-aggregated across qpos indices 0–1 while excluding the unidentifiable spinner coordinate at index 2, still produce poor position summaries: contrastive about −9.08 ± 0.45, VICReg about −11.35 ± 1.79, and MAE about −7.64 ± 0.17. These summaries are dominated by the distal-angle score at index 1 (about −15 to −23 per seed). Do **not** present those averages as clean cross-objective rankings until the validation marginal distributions and per-variable predictions are inspected. A small target variance can make R² strongly negative; the metric must be reported with variable-level context.

The Table 3.4 corrective script migrates compatible progress instead of discarding it, archives the legacy results, and queues only the three Finger Spin supervised-reference runs. Do not pass --fresh.

### 2.3 Other experiments

- **VICReg standalone experiment:** saved aggregate probe R² = 0.024 ± 0.111.
- **MAE standalone experiment:** saved aggregate probe R² = −0.552 ± 0.024. Masked reconstruction loss decreased during training, but reconstruction quality is not evidence by itself that the latent is sufficient for state estimation.
- **Anti-collapse mechanisms:** saved mean effective rank was about 1.02 for SimSiam, 1.06 for BYOL, and 5.76 for Barlow Twins. In this run, SimSiam/BYOL had nearly one-dimensional representations under this diagnostic; the result is about these settings, not a universal ranking of the objectives.
- **Latent bottleneck sweep:** cartpole R² means for d = [2, 4, 8, 16, 64, 256] were [−0.252, −0.068, −0.094, −0.012, −0.499, −3.345]. Cheetah means were [−0.090, 0.075, −0.014, 0.208, 0.304, 0.328]. The current evidence does not support a claim that performance monotonically rises and plateaus at the task's state dimension. It shows little positive cartpole decodability in this setup and a steadier rise for cheetah.
- **NT-Xent implementation check:** unrolled and vectorized losses agreed to within 2.4e−7; maximum gradient difference was 7.5e−9.

Do not merge results from the standalone VICReg/MAE runs with Table 3.4: they use different training budgets and data caps.

## 3. Required validation and run order

1. Switch to audit/chapter3-methodology-fixes and run the test suite using the repository's .venv Python. Tests have been added for the angular period and Table 3.4 migration. No passing result is claimed here; verify locally.
2. Run the inexpensive Finger Spin target-distribution diagnostic (standard deviation, quantiles, and episode counts for the exact train/validation subsets used in Table 3.4). This is needed because the saved distal-angle R² values are approximately −15 to −23 even in the supervised reference. Review these values before using an aggregate Finger Spin position score in the manuscript.
3. After the target-distribution check, run ch03/table_03_04_comparison.py. It should migrate version-3 progress, retain compatible results, and retrain only Finger Spin's supervised-reference seeds 0, 1 and 2. Do not pass --fresh.
4. Run ch03/11_temporal_cpc.py --evaluate-checkpoints. This reuses all three trained CPC checkpoints and recomputes metrics using evenly spaced validation windows; it must not train epochs.
5. Run ch03/12_assemble_encoder.py as the final integration check.

Do not recollect data or rerun the completed contrastive, aliasing, collapse, VICReg, MAE or bottleneck experiments merely to obtain the corrected table and CPC gap metrics.
