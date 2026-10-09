# Chapter 3 Audit Findings — Corrected Runs and Interpretation

**Audited branch:** `audit/chapter3-methodology-fixes`  
**Experiment commit reviewed:** `7d4c77feb7e7e15bf8cb6b7ca3a1cd5ff411f559`  
**Basis:** source and committed JSON/figure/checkpoint artifacts on the branch, plus the author's local run log supplied on 9 October 2026. The reviewer inspected the committed source and results but did not execute the repository locally.

## 1. Execution and sampling status

The corrected rerun completed through `ch03/12_assemble_encoder.py`. The author's Windows run reports **44 tests passed** (4.23 seconds after the Python-boolean fix), followed by successful completion of the remaining experiment scripts and the Chapter 1 encoder-contract smoke test. GitHub Actions has no run for this commit, so the test evidence is the local log, not remote CI.

The data-only diagnostic reports 30 training episodes and 5 validation episodes per task. With the 3,500-frame/1,000-frame caps used by the diagnostic, the selected sample contains 116–117 training frames per episode and 200 validation frames per episode. The old prefix-based experiment metrics have been archived or replaced where the scripts version their results.

The branch head is `7d4c77f`; `main` remains at `230b298b`. The corrected results are therefore on the audit branch and are not yet merged into the default branch.

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
| Masked autoencoder | 0.164 ± 0.030 | −0.113 ± 0.004 | 0.305 ± 0.012 | −0.010 ± 0.003 |

Negative (R^2) means worse than predicting the validation-set mean. These values suggest that position is substantially more decodable from one frame than velocity for every method tested. They do not by themselves establish a causal explanation for why velocity is hard to recover.

**Interpretation constraint:** the table compares complete method configurations, not loss functions in isolation. The supervised, contrastive, and VICReg arms use a convolutional encoder, while MAE uses a patch-transformer encoder/decoder. Further, the reported supervised-reference metric is a fresh frozen-feature linear probe after training the encoder with a supervised state head; it is **not** the direct held-out score of that trained head. Do not describe it as an oracle or upper bound without reporting the trained head's own held-out performance.

Finger Spin's stored state has six coordinates (three qpos followed by three qvel). The absolute spinner hinge angle is excluded from the aggregate because the rendered geometry is half-turn symmetric; its identifiable orientation is π-periodic. This correction is encoded in the metric contract.

### 2.4 VICReg, MAE, bottleneck, and collapse observations

- Standalone VICReg reports mean probe (R^2=0.248 pm 0.005); standalone MAE reports (0.025 pm 0.018) while its masked reconstruction MSE falls to about 0.002. This is consistent with reconstruction quality and linear state decodability measuring different things. Do not compare these standalone numbers directly to Table 3.4 as if all budgets and configurations were identical.
- In the bottleneck sweep, Cartpole scores rise from about −0.008 at (d=2) to 0.321 at (d=64), then are essentially flat/slightly lower at 0.317 for (d=256). Cheetah rises from 0.060 at (d=2) to 0.422 at (d=256); a plateau is not demonstrated for Cheetah within the tested range.
- The collapse demonstration gives latent standard deviation about (7\times10^{-5}) to (9\times10^{-5}) for attraction-only training versus about 2.30–2.90 for the contrastive condition in these runs.
- The three-mechanism experiment reports effective rank (1.220 pm 0.269) for SimSiam, (1.064 pm 0.029) for BYOL, and (5.986 pm 0.739) for Barlow Twins. Barlow Twins uses more latent directions in this experiment, but its effective rank is still well below the 16-dimensional embedding size. Effective rank is evidence about spectrum concentration, not by itself proof of total collapse or its absence.

These results are descriptive teaching experiments with three seeds and modest training budgets, not a broad benchmark of SSL methods.

### 2.5 CPC is informative but its gap metric needs a stronger evaluation

The corrected CPC run trained all three seeds and saved the checkpoints. Its results are:

- Current Cartpole-position probe: static 0.789 ± 0.007; context 0.828 ± 0.010.
- Current Cartpole-velocity probe: static 0.007 ± 0.004; context 0.143 ± 0.083.
- Mean next-state probe (R^2): static 0.227 ± 0.021; context 0.277 ± 0.049.
- Five-step missing-interval CPC rollout: −3.094, −0.196, and −0.539 per seed; mean −1.276 ± 1.293.
- Privileged state extrapolation baseline: mean (R^2=0.823), identical across seeds because it is one shared state-space baseline, not a seed-varying model.

This supports a tentative statement that temporal context helps decode some state variables and improves mean next-state probe scores in this setup, while the learned latent rollout performs poorly on the current gap test. It does **not** establish a robust failure magnitude: only 32 validation windows are used per seed, and the first seed is a large negative outlier. The privileged baseline receives true qpos/qvel history, while CPC receives rendered frames; the comparison is intentionally asymmetric and must be labelled as such, not presented as a like-for-like model comparison. Re-evaluate the existing CPC checkpoints on a larger gap-window sample before making a strong quantitative claim.

The printed maximum rollout latent norm of approximately 1 is guaranteed by explicit L2 normalization in the model. It is an implementation sanity check, not independent evidence that rollout dynamics are stable.

### 2.6 Encoder assembly

The final script reports successful verification of an observation tensor shaped `(1, 3, 64, 64)` and a persistent state shaped `(1, 32)` over 10 frames. This checks the Chapter 1 wrapper/checkpoint interface; it does not validate long-horizon prediction quality or the CPC rollout.

## 3. Recommended remaining scientific checks

1. Re-evaluate the already trained CPC checkpoints on at least 200 gap windows (no retraining needed) and report per-seed and per-variable metrics. Keep the privileged-baseline caveat explicit.
2. If Table 3.4 describes the supervised arm as a predictive reference or ceiling, add the trained supervised head's direct held-out metrics beside the independent frozen-feature probe results; do not silently conflate them.
3. In the chapter prose, report the negative result for two-frame stacking honestly. Do not claim the current experiment demonstrates velocity recovery.
4. Keep the Table 3.4 method-family comparison caveat and report all negative velocity scores rather than omitting them.
5. Before using any values in the book, ensure figure captions, prose, and result tables all reflect the corrected JSON artifacts on this branch.

## 4. Artifact and CI status

The experiment JSON files, updated figures, and checkpoints are committed on the audit branch. No GitHub Actions runs were found for the reviewed commit; the author's local 44-test pass and terminal log are the available execution record. `main` has not been updated by this branch.
