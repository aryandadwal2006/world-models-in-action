# Chapter 3 Publication-Readiness Audit

**Audit date:** 14 October 2026  
**Baseline inspected:** `audit/chapter3-methodology-fixes` at `e659756d7f5a99be91e16be46270fd52cf2d1b2e`  
**Follow-up branch:** `audit/chapter3-publication-readiness` (contains the checkpoint-only evaluation changes)  
**Default branch:** `main` at `230b298b9d1ed031e69e6987bb9c2a2e88bc8086`; not modified by this audit.

## Executive decision

**The expanded CPC gap evaluation is complete, but Chapter 3 is not ready to freeze for final quantitative claims yet.** The descriptive experiments may be drafted with their measured limitations and negative results preserved. The checkpoint-only gap run now evaluates 200 non-overlapping windows per seed, but the result remains negative and seed-sensitive. The repository contains a design document and experiment code, not the finished book manuscript; the final prose, table, and production captions still need to be checked against the updated artifact.

The author’s local console log and the committed artifact at `ch03/results/table_03_05_cpc_metrics_200_windows.json` show that evaluation-only mode loaded the existing checkpoints without training and evaluated 200 windows per seed. The JSON records 40 windows from each of five episodes, 4,880 eligible validation starts, and no window overlap. The new figure was generated at `ch03/figures/cpc_gap_eval_200/fig03_16_gap_tracking.png`; the original figure path remains unchanged. The Table 3.4 publication-facing label fix is also retained on this branch.

## Prioritized findings

| Priority | Severity | Finding and evidence | Required fix / disposition |
|---|---|---|---|
| P1 | High | The original 32-window result (−3.094, −0.196, −0.539; mean −1.276 ± 1.293) is historical. The expanded checkpoint-only evaluation scores 200 non-overlapping windows per seed, balanced at 40 per episode across five validation episodes. CPC mean gap R² is −2.853, −0.128, and −0.443 by seed (mean −1.141 ± 1.217 across seeds). Per-variable across-seed mean ± population SD: cart position −3.331 ± 4.267, pole angle −0.382 ± 0.111, cart velocity −0.401 ± 0.504, and pole angular velocity −0.451 ± 0.178. The result remains negative and seed-sensitive, especially for cart position; episode 3 has the lowest CPC mean gap R² for every seed. | The requested sample-size follow-up is complete. Use the 200-window artifact for current claims and retain the 32-window metrics only as historical. Present episode-level scores descriptively, label the baseline privileged, and avoid treating windows as independent episodes or generalizing beyond this setup. No retraining is indicated by these results. |
| P1 | High | The state extrapolation baseline is not a like-for-like comparator. It receives true qpos/qvel history, and its history predictor is fitted on raw training-state transitions; CPC receives rendered RGB frames. Its mean R² of 0.837 is a privileged diagnostic, not evidence that a comparable learned model beats CPC. | Call it a **privileged state-history extrapolation baseline** in the prose, table, and figure. State the input asymmetry adjacent to the number. Do not describe the comparison as fair or like-for-like. |
| P1 | High if described as head performance; otherwise medium | `train_supervised` trains an encoder and state-prediction head but returns the result of a new frozen-feature linear probe. `table_03_04_objective_comparison.json` reports that probe, not the trained head's direct held-out prediction score. There is no stored trained-head result in the committed table artifact. | For a representation comparison, relabel the row “Supervised-pretrained encoder + frozen linear probe” and explicitly say the labels train the encoder/head before a separate probe is fitted. If claiming supervised-head predictive accuracy, conduct a targeted supervised-only evaluation that scores the trained head on held-out examples with proper inverse target transformation; do not infer or substitute the existing probe values. |
| P2 | Medium | The earlier Chapter 3 design plan said Table 3.5 would compare a stacked-frame model against single-frame and CPC context. `ch03/11_temporal_cpc.py` has no stacked-frame CPC-table arm. A separate frame-stacking experiment exists in the aliasing diagnostic/Table 3.3. | The plan now specifies only measured CPC current-state and next-state probes plus the separately reported gap test. Keep the stacked-frame comparison in Table 3.3 unless an actual Table 3.5 run is intentionally added. |
| P2 | Medium | Figure 3.16 is generated from the first selected validation window of seed 0, not an aggregate or average. The frame-only re-entry line contains probe reads at observed frames only; it does not predict the missing interval. | A new evaluation figure exists at `ch03/figures/cpc_gap_eval_200/fig03_16_gap_tracking.png`; the canonical `ch03/figures/fig03_16_gap_tracking.png` was not overwritten. Use the new image in the manuscript if it matches production requirements. Its caption must identify one illustrative window, explain that the frame-only trace resumes after the gap, and call the state-history baseline privileged. Report numerical results in the table/JSON, not from the illustrative trace. |
| P2 | Medium | The gap score pools missing-frame rows across windows. This is a valid pooled frame-level descriptive R², but alone it hides episode heterogeneity, and windows within an episode are not independent trajectories. | The new evaluator records episode-balanced window selection, verifies no overlapping 25-frame input windows, saves exact starts and episode counts, and records per-episode metrics. Present per-episode scores descriptively; do not treat 200 windows as 200 independent episodes. |
| P2 | Medium | `discrepancies_and_findings.md` still called `7d4c77f` the current audit-branch head, although the branch has two later commits, ending at `e659756`. | Correct the historical/current head distinction, retain attribution of the local test result, and use this follow-up report as the current disposition. |
| P3 | Low | The printed max latent norm near 1 is guaranteed by L2 normalization in `predict_next_latent` and the rollout. | Report it only as an implementation/numerical sanity check, not evidence of stable rollout dynamics. |
| P3 | Low | Table 3.4 compares full training configurations, not loss functions in isolation: the supervised, NT-Xent, and VICReg arms use a convolutional encoder, while MAE uses a different patch-transformer encoder/decoder. The existing findings disclose this. | Keep the method-family caveat visible wherever Table 3.4 is interpreted. Do not call it a controlled loss-only ablation. |

## Checks that did not reveal a defect

- **No direct train/validation target leakage identified in the reviewed paths.** Table 3.4 uses selected training examples for supervised target normalization and training; the independent probe is fitted on train features/labels and scored on the separate validation split. CPC's static/context probes are fitted on train features and scored on validation features. The privileged state-history model is fitted only from training state transitions. State labels are intentional for the supervised arm and evaluation probes; self-supervised objective losses do not use these labels.
- **Episode boundaries:** the temporal dataset enumerates only contiguous sequence windows whose episode IDs match throughout. The new gap evaluator repeats this constraint for all candidates and adds a non-overlap check. The new test file tests 200 windows on synthetic five-episode data, but those tests have not been executed in this audit environment.
- **Aggregation:** the Table 3.4 seed summary uses arithmetic mean and population standard deviation across seeds, consistently with its report convention. The stored JSON metrics and the corrected findings agree on the headline Table 3.4 position/velocity numbers. The CPC privileged baseline's zero across-seed standard deviation comes from using one shared state-history model, not three separately trained baseline models.
- **Angular metric contract:** the code uses circular scoring for angular variables and handles the rendered Finger Spin hinge angle as pi-periodic. Its absolute 2π coordinate is excluded from the aggregate, rather than allowed to dominate the other aggregate scores.
- **Negative results should remain negative.** The corrected two-frame stacking experiment did not demonstrate velocity recovery: cart-velocity R² improved by only about 0.002 and pole-angular-velocity R² became slightly worse. Cartpole's bottleneck curve is nearly flat from d=64 to d=256, but Cheetah has not demonstrated a plateau over the tested dimensions. Neither result should be tuned or narrated beyond its measured scope.

## CPC follow-up implemented on this branch

Changes on this branch:
- `ch03/table_03_04_comparison.py` defines manuscript-facing method labels without changing existing progress-file keys, so the supervised row is accurately labelled as a supervised-pretrained encoder evaluated with a frozen-feature linear probe. The current result JSON records this mapping under `publication_method_labels`; its metric values are unchanged.
- `tests/test_table_03_04_metrics.py` adds a test for that label.

CPC changes in `ch03/11_temporal_cpc.py`:
1. `--evaluate-checkpoints` remains the no-retraining mode and validates the existing checkpoint metadata.
2. `--gap-sequences 200` requests 200 episode-balanced, non-overlapping 25-frame validation windows from the full eligible validation set, independent of the 1,000-window cap used for state-probe metrics. The script fails loudly if it cannot provide the requested count.
3. Exact window-start indices, episode IDs, counts, no-overlap status, and privileged-baseline scope are included in the result metadata.
4. CPC and privileged-baseline scores are kept per state variable, across seed summary, and per validation episode.
5. `--evaluation-output` writes to a separate path and refuses to overwrite an existing output. The original `table_03_05_cpc_metrics.json` remains the checkpoint configuration reference and historical 32-window result.
6. The figure title now says it is an illustrative single validation-window trace, not an aggregate result.

Tests added at `tests/test_cpc_gap_eval.py` cover 200 windows, five-episode balance, window/episode boundaries, and failure when the requested count is unavailable. **They are not reported as passing until run locally.**

## What is verified, and what is not

**Verified from GitHub source/artifacts:** live refs and branch separation; source logic described above; configuration and aggregate values in committed JSON; checkpoint names and the checkpoint loading/metadata-validation implementation; the Table 3.4 supervised-arm return path; design-plan and figure-generation-code claims.

**Not independently verified by the reviewer:** execution of the test suite on this exact branch and visual inspection of the new PNG. The author’s supplied console log shows the checkpoint-only evaluation completed and the JSON artifact records the reported scores; this reviewer did not rerun the experiment locally. The GitHub Actions endpoint reported no workflow runs for the follow-up branch.

**Scope limitation:** the repository tree contains `chapter_03_design.md` and experiment/results files, but not the completed Chapter 3 book manuscript. The outline has now been updated to the expanded CPC result. The final prose and production caption must still be checked against the new JSON and image; the 200-window evaluation does not by itself make the chapter publication-ready.

## Local commands to finish the minimum work

The 200-window evaluation has already been run. Do not rerun it to the same output path: the script intentionally refuses to overwrite an existing file. To finish local verification on the pushed branch:

```powershell
git fetch origin
git switch audit/chapter3-publication-readiness
git pull --ff-only
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python -c "import json; d=json.load(open('ch03/results/table_03_05_cpc_metrics_200_windows.json', encoding='utf-8')); print('windows:', d['evaluation']['gap_sequence_count']); print('episode counts:', d['evaluation']['gap_window_episode_counts']); print('CPC R2 by seed:', [m['cpc_gap_mean_r2'] for m in d['per_seed_metrics']]); print('CPC mean and population SD:', d['summary']['cpc_gap_mean_r2']); print('CPC per variable:', d['summary']['cpc_gap_r2_per_variable']); print('privileged state-history R2:', d['summary']['linear_gap_mean_r2'])"
```

The existing artifact says `checkpoint_only: true` and `gap_windows_overlap: false`; the original `ch03/results/table_03_05_cpc_metrics.json` remains the historical 32-window output. If a fresh run is necessary for debugging, choose a new `--evaluation-output` and `--figures-dir`; do not retrain or tune solely to make the negative result look better.

## Readiness gate

The sample-size follow-up is complete. The currently supported conclusion is that the latent CPC gap rollout is poor and variable in this DMC setup, while the state-history baseline is privileged and not directly comparable. Before publication lock, run the tests on the updated branch, make sure the manuscript cites the expanded result and the new figure path, and preserve the per-variable/per-episode caveats. Do not interpret the three-seed SD as a confidence interval or the 200 windows as 200 independent trajectories. A final publication-ready decision additionally requires checking the actual authored manuscript against the regenerated artifacts.

Chapter 3 may be drafted now around the already supported objective comparisons, with their limitations and negative results stated plainly. Do **not** freeze the CPC gap result or its figure/caption until the new checkpoint-only evaluation has completed and been reviewed. The minimum required extra experiment is evaluation only, not retraining or data recollection. A final publication-ready decision additionally requires checking the actual authored manuscript against the regenerated artifacts.
