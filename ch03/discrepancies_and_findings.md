# Chapter 3 Audit Findings — Corrected Runs and Interpretation

**Original experiment-result commit:** `7d4c77feb7e7e15bf8cb6b7ca3a1cd5ff411f559`  
**Corrected design/findings on the original audit branch:** most recently inspected at `e659756d7f5a99be91e16be46270fd52cf2d1b2e`.  
**Publication-readiness follow-up:** `audit/chapter3-publication-readiness`; the original 32-window CPC result remains historical, and the expanded checkpoint-only evaluation artifact was committed at `4e105393a27f43aeabfea19edc7d310454b7cbae`.
**Basis:** committed source and JSON/figure/checkpoint artifacts, plus the author-supplied local logs. This report retains dated audit-branch history below but its MAE status was updated after the corrected runs were pushed to `main` on 17 October 2026. The author’s latest log reports 49 passing tests; the reviewer did not independently execute the tests or experiments.

## MAE representation-path correction and rerun (17 October 2026)

The previous MAE probe path pooled encoder tokens through a separate `to_latent` projection, but masked reconstruction loss did not train that projection. The canonical `SimpleMAE` now exposes the mean-pooled final encoder tokens directly as probe features (64 features in the current configuration). The corrected code, tests, figures, and rerun metrics are committed on `main`.

- **Standalone MAE:** mean probe R² = 0.170410 across seeds 0, 1, and 2 (population SD 0.004310); per-seed scores are 0.165494, 0.175988, and 0.169749. The run regenerated Figure 3.12 and `ch03/results/mae_metrics.json`.
- **Table 3.4 MAE row (mean ± population SD across three seeds):** Cartpole position 0.475 ± 0.012; Cartpole velocity −0.156 ± 0.005; Finger position 0.383 ± 0.010; Finger velocity −0.034 ± 0.001. These values are in `ch03/results/table_03_04_objective_comparison.json` (experiment version 6).
- The version-6 table runner migrated the compatible version-5 progress file, archived the prior MAE entries, and reran the six MAE arms (two tasks × three seeds). Existing supervised, NT-Xent, and VICReg results were preserved and resumed.
- The author-supplied local log reports 49 tests passing and successful completion of both MAE runs. No remote CI status was reported for the latest push, and this audit did not independently reproduce the run.

## 1. Execution and sampling status

The earlier corrected rerun completed through `ch03/12_assemble_encoder.py`; that historical Windows log reported 44 tests passed. The subsequent current `main`-branch log reports **49 tests passed** in 2.434 seconds and shows successful completion of the corrected three-seed standalone MAE run and six Table 3.4 MAE runs. These are author-supplied local results, not independently reproduced by this audit; no GitHub Actions status is available for the latest push.

The data-only diagnostic reports 30 training episodes and 5 validation episodes per task. With the 3,500-frame/1,000-frame caps used by the diagnostic, the selected sample contains 116–117 training frames per episode and 200 validation frames per episode. The old prefix-based experiment metrics have been archived or replaced where the scripts version their results.

At the time of the earlier follow-up inspection, `audit/chapter3-methodology-fixes` had advanced to `e659756d7f5a99be91e16be46270fd52cf2d1b2e`, while `main` was still at `230b298b9d1ed031e69e6987bb9c2a2e88bc8086`. Those branch/ref statements are historical, not current status: the relevant CPC evaluation code/artifact, corrected Table 3.4 outputs, and MAE representation fix are now present on `main`. The original 32-window CPC values remain historical; use the checkpoint-only 200-window-per-seed artifact for the expanded evaluation.

## 2. Corrected findings

### 2.1 Episode-prefix sampling was a real source of bias

The previous first-3,500 training-frame subset covered only 4 episodes; the first-1,000 validation-frame subset covered 1 episode. On Finger Spin, the old validation prefix had spinner-angle standard deviation 0.117, versus 2.020 in the training prefix, and the distal-position medians differed substantially. This could distort both training and held-out probe metrics.

The corrected sampler spreads capped samples across episode IDs and within-episode time. For temporal stacks and CPC, it selects valid centres/sequence starts from the full original arrays while keeping the actual windows contiguous and episode-safe. This is the correct fix for the demonstrated prefix bias.

### 2.2 Two-frame stacking did not recover velocity in this setup

Corrected Table 3.3 reports:

| Variable | Single frame | Two-frame stack | Mean change |
|---|---:|---:|---:|
| Cart position | 0.714 ± 0.040 | 0.707 ± 0.010 | −0.007 |
| Pole angle | 0.612 ± 0.069 | 0.586 ± 0.064 | −0.026 |
| Cart velocity | 0.006 ± 0.004 | 0.009 ± 0.007 | +0.002 |
| Pole angular velocity | −0.367 ± 0.001 | −0.375 ± 0.006 | −0.007 |

The tested two-frame, 12-epoch contrastive setup does **not** support a claim that frame stacking recovered velocity. The cart-velocity change is tiny; angular-velocity decoding remains worse than the mean baseline. This is a negative or limited result about this particular representation, temporal spacing, image resolution, augmentation, and training budget. It does not prove that velocity is fundamentally unrecoverable from multiple frames.

A single static image has no explicit time-difference signal, but appearance and dynamics may still be statistically correlated with velocity. Treat velocity identifiability as an empirical question, not an impossibility guaranteed by the renderer.

### 2.3 Contrastive learning and objective comparison

The corrected standalone contrastive runs on Cartpole report per-seed mean probe (R^2) values 0.260, 0.241, and 0.270. Position is substantially more decodable than velocity. These values use the corrected episode-stratified sample and supersede the prefix-sampled results.

Corrected Table 3.4 (mean ± standard deviation across three seeds) reports:

| Method | Cartpole position | Cartpole velocity | Finger position | Finger velocity |
|---|---:|---:|---:|---:|
| Supervised state-prediction reference | 0.981 ± 0.000 | −0.273 ± 0.012 | 0.995 ± 0.001 | −0.114 ± 0.010 |
| Contrastive (NT-Xent) | 0.672 ± 0.001 | −0.187 ± 0.005 | 0.714 ± 0.029 | −0.007 ± 0.002 |
| VICReg | 0.649 ± 0.022 | −0.183 ± 0.006 | 0.679 ± 0.040 | −0.005 ± 0.004 |
| Masked autoencoder | 0.475 ± 0.012 | −0.156 ± 0.005 | 0.383 ± 0.010 | −0.034 ± 0.001 |

Negative (R^2) means worse than predicting the validation-set mean. These values suggest that position is substantially more decodable from one frame than velocity for every method tested. They do not by themselves establish a causal explanation for why velocity is hard to recover.

**Interpretation constraint:** the table compares complete method configurations, not loss functions in isolation. The supervised, contrastive, and VICReg arms use a convolutional encoder, while MAE uses a patch-transformer encoder/decoder. Further, the reported supervised-reference metric is a fresh frozen-feature linear probe after training the encoder with a supervised state head; it is **not** the direct held-out score of that trained head. Do not describe it as an oracle or upper bound without reporting the trained head's own held-out performance.

Finger Spin's stored state has six coordinates (three qpos followed by three qvel). The absolute spinner hinge angle is excluded from the aggregate because the rendered geometry is half-turn symmetric; its identifiable orientation is π-periodic. This correction is encoded in the metric contract.

### 2.4 VICReg, MAE, bottleneck, and collapse observations

- Standalone VICReg reports mean probe R² = 0.248 ± 0.005; the corrected standalone MAE run reports mean probe R² = 0.170 ± 0.004 while masked reconstruction MSE falls to about 0.002. This illustrates that reconstruction quality and linear state decodability measure different things. Do not compare the standalone scores directly to Table 3.4 as if all budgets and configurations were identical.
- In the bottleneck sweep, Cartpole scores rise from about −0.008 at (d=2) to 0.321 at (d=64), then are essentially flat/slightly lower at 0.317 for (d=256). Cheetah rises from 0.060 at (d=2) to 0.422 at (d=256); a plateau is not demonstrated for Cheetah within the tested range.
- The collapse demonstration gives latent standard deviation about (7\times10^{-5}) to (9\times10^{-5}) for attraction-only training versus about 2.30–2.90 for the contrastive condition in these runs.
- The three-mechanism experiment reports effective rank (1.220 pm 0.269) for SimSiam, (1.064 pm 0.029) for BYOL, and (5.986 pm 0.739) for Barlow Twins. Barlow Twins uses more latent directions in this experiment, but its effective rank is still well below the 16-dimensional embedding size. Effective rank is evidence about spectrum concentration, not by itself proof of total collapse or its absence.

These results are descriptive teaching experiments with three seeds and modest training budgets, not a broad benchmark of SSL methods.

### 2.5 CPC is informative but its gap metric needs a stronger evaluation

The corrected CPC run trained all three seeds and saved the checkpoints. Its results are:

- Current Cartpole-position probe: static 0.789 ± 0.007; context 0.828 ± 0.010.
- Current Cartpole-velocity probe: static 0.007 ± 0.004; context 0.143 ± 0.083.
- Mean next-state probe (R^2): static 0.227 ± 0.021; context 0.277 ± 0.049.
- Historical 32-window gap rollout: −3.094, −0.196, and −0.539 per seed; mean −1.276 ± 1.293.
- Expanded checkpoint-only gap rollout (200 non-overlapping windows per seed): −2.853, −0.128, and −0.443 per seed; mean −1.141 ± 1.217 across seeds.
- Expanded per-variable CPC gap mean R² ± population SD across seeds: cart position −3.331 ± 4.267; pole angle −0.382 ± 0.111; cart velocity −0.401 ± 0.504; pole angular velocity −0.451 ± 0.178.
- Privileged state-history extrapolation baseline: mean R² = 0.837, identical across CPC seeds because one shared state-history model is used.

Temporal context improves current-state and next-state probe scores in this setup, while the latent rollout performs poorly on the gap test. The expanded evaluation confirms the negative result but also shows substantial seed sensitivity, driven especially by seed 0 and cart-position prediction. The 200 windows are balanced at 40 per each of five held-out episodes; they are windows, not independent episodes. Episode 3 has the lowest CPC mean gap R² for each of the three seeds, so its trajectory-level errors should be inspected before interpreting the aggregate as a uniform failure mode. The privileged baseline receives true qpos/qvel history, while CPC receives rendered frames; this input asymmetry must be disclosed, not presented as a like-for-like comparison. The author-supplied local log and committed JSON report successful checkpoint-only evaluation; the audit reviewer did not independently execute the evaluation. The current `main`-branch test log includes the CPC window-selection tests and reports 49 total tests passing.

The printed maximum rollout latent norm of approximately 1 is guaranteed by explicit L2 normalization in the model. It is an implementation sanity check, not independent evidence that rollout dynamics are stable.

### 2.6 Encoder assembly

The final script reports successful verification of an observation tensor shaped `(1, 3, 64, 64)` and a persistent state shaped `(1, 32)` over 10 frames. This checks the Chapter 1 wrapper/checkpoint interface; it does not validate long-horizon prediction quality or the CPC rollout.

## 3. Recommended remaining scientific checks

1. **Completed on `main`:** the existing CPC checkpoints were evaluated on 200 episode-balanced, non-overlapping gap windows per seed, with per-seed and per-variable results committed. Do not rerun this evaluation unless debugging; keep the privileged-baseline caveat explicit.
2. Label the Table 3.4 supervised row as a state-label-trained encoder followed by an independent frozen-feature probe. The existing result is not the trained prediction head's direct held-out metric. Only rerun that supervised arm if the chapter specifically needs to claim or tabulate trained-head predictive performance.
3. In the chapter prose, report the negative result for two-frame stacking honestly. Do not claim the current experiment demonstrates velocity recovery.
4. Keep the Table 3.4 method-family comparison caveat and report all negative velocity scores rather than omitting them.
5. Before using any values in the book, ensure figure captions, prose, and result tables reflect the current JSON artifacts committed on `main`.

## 4. Artifact and CI status

The CPC evaluation JSON/figure, corrected Table 3.4 JSON, corrected standalone MAE metrics, regenerated Figure 3.12, and experiment code are committed on `main`. The author-supplied current local log reports 49 tests passing plus successful standalone and Table 3.4 MAE runs. No GitHub Actions status was reported for the latest push, and the reviewer did not independently reproduce the local runs.
