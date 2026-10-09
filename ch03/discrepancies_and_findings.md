# Chapter 3 Audit Findings — Sampling Bias and Corrective Run Plan

**Audit basis:** committed results at main commit 230b298b9d1ed031e69e6987bb9c2a2e88bc8086 (8 October 2026), the user-provided target-distribution diagnostic, and source changes on branch audit/chapter3-methodology-fixes. The corrective branch has not yet been tested locally. No corrected experiment results are claimed here.

## 1. Confirmed issues

### 1.1 The capped validation sets were prefixes, not representative subsets

The user's diagnostic deliberately loaded the first 3,500 training rows and first 1,000 validation rows from Finger Spin. Those subsets contained 4 training episodes and only 1 validation episode.

For the first 1,000 validation frames, the spinner angle had standard deviation 0.117 and range approximately [-3.585, -2.856]. In the first 3,500 training frames, its standard deviation was 2.020 and range approximately [-3.778, 4.916]. The distal position also had a shifted marginal distribution: training median 0.870 versus validation median -1.507. The validation subset therefore sampled a much narrower, different portion of the dynamics than the training subset.

This is not a theoretical concern: several Chapter 3 scripts use expressions such as frames[:n_val] before constructing the dataset. The same pattern affects Table 3.4, the aliasing diagnostic, standalone VICReg/MAE, the bottleneck sweep, and the original CPC sequence dataset. Their old probe scores and gap scores should be treated as superseded, not interpreted as final generalization performance.

### 1.2 Correct sampling must preserve episode structure

Source fix on the audit branch:
- A shared selector allocates a sample budget across episode IDs and chooses frames evenly within each episode.
- DMCDataset accepts original-array sample indices, so the samples can be distributed across the full dataset without slicing away the underlying temporal context.
- The aliasing experiment selects only valid stack centres whose required history lies in one episode.
- CPC constructs episode-safe sequence windows from the full split and selects sequence starts across episodes, retaining each sequence's contiguous frames.
- Table 3.4 bumps its experiment version, archives the old progress results, and clears the active result slots so no version-3 score can be accidentally reused.
- A reproducible data-only diagnostic was added as ch03/00_audit_dataset_sampling.py.

These changes affect the training samples as well as evaluation samples. As a result, the saved representations from prefix-sampled runs are not a valid substitute for rerunning the experiments under the corrected sampling contract.

### 1.3 Finger Spin's spinner angle is separately unidentifiable from static pixels

The stored state is raw MuJoCo qpos followed by qvel. Finger Spin therefore has six target coordinates, not the old design note's four-dimensional "pos/vel/touch" state. Its spinner geometry has half-turn symmetry, so a single rendered image cannot uniquely identify the absolute 2π-periodic hinge coordinate. See the upstream [finger.xml](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/finger.xml) and [finger.py](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/finger.py).

The audit-branch probe uses a π-periodic representation/score for that coordinate; the Table 3.4 aggregate excludes it and retains the old full-turn score only for provenance. This symmetry correction does not solve the independent prefix-sampling bias.

### 1.4 Barlow Twins normalization was also inconsistent

The old implementation standardized each projection dimension with sample standard deviation (denominator N−1) while dividing the cross-correlation matrix by N. The corrected loss uses population standard deviation, validates the batch shape/size, and has tests for a zero-loss identity cross-correlation matrix. Existing three-mechanism Barlow Twins results are superseded and must be regenerated.

### 1.5 Table 3.4 is a method-family comparison, not an objective-only ablation

The supervised reference, contrastive and VICReg methods use a convolutional encoder, while MAE uses a patch-transformer encoder/decoder. The corrected manuscript notes describe this as a comparison of complete configurations under the stated budget. The code changes do not make it a loss-only ablation.

## 2. What to do with old results

Until the corrected runs finish, do not quote old validation metrics from the following artifacts as final Chapter 3 evidence:

- Table 3.3 aliasing probe values and the associated per-variable conclusion.
- Table 3.4 values for either task.
- The standalone contrastive linear-probe values from ch03/04_train_contrastive.py.
- Standalone VICReg/MAE probe metrics.
- Bottleneck-sweep (R^2) curve.
- CPC current/next-state probe values and gap tracking values.
- The three-mechanism comparison, because both its training subset and Barlow loss implementation changed.
- The collapse-demo latent-standard-deviation curve, because the selected training examples changed.

The cached datasets themselves are still reusable. Do not recollect them.

## 3. Exact validation and rerun order

1. Switch to audit/chapter3-methodology-fixes.
2. Run the unit test suite: .venv Python with -m pytest tests -q.
3. Run ch03/00_audit_dataset_sampling.py. Check that the selected training/validation sample counts cover the full episode sets, and that no one episode takes the whole capped sample budget.
4. Run the corrected, capped-sample experiments once each: ch03/04_train_contrastive.py for seeds 0, 1 and 2; ch03/05_aliasing_diagnostic.py; ch03/06_collapse_demo.py; ch03/07_three_mechanisms.py; ch03/08_vicreg.py; ch03/09_masked_autoencoder.py; ch03/10_bottleneck_sweep.py; and ch03/table_03_04_comparison.py.
5. Run ch03/11_temporal_cpc.py without --evaluate-checkpoints so the trained CPC models use the corrected sequence sample selection as well; the evaluation-only flag is now a diagnostic option for existing checkpoints, not the final corrected training result.
6. Run ch03/12_assemble_encoder.py as the final integration check.

For Table 3.4, do not pass --fresh: the updated migration archives version-3 metrics and queues the full table under the new sampling contract. Old progress values cannot be retained as valid scores because they were evaluated on prefix-only subsets.
