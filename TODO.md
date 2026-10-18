# Chapter 3 Implementation and Release Status

**Status date:** 18 October 2026  
**Current release branch:** `main`  
**Scope:** Code companion and research artifacts for Chapter 3 of *World Models in Action*. This file tracks the current state and the remaining authoring handoff; it is not a claim that the book manuscript is complete.

## 1. Code companion status

- [x] Canonical model/loss/evaluation implementations live under `worldmodels/`.
- [x] Numbered Chapter 3 runners live under `ch03/`; regression tests live under `tests/`.
- [x] The CPC runner uses multi-horizon bilinear InfoNCE and its primary evaluation is held-out, in-batch future-latent retrieval.
- [x] The former custom recursive CPC gap-rollout metric is excluded from the current primary result.
- [x] The five-seed CPC checkpoints and checkpoint-only re-evaluation protocol are retained.
- [x] The expanded five-seed SimSiam/BYOL/Barlow Twins diagnostic and the collapse-demo outputs are retained.
- [x] Table 3.4 uses the corrected version-6 experiment/results; the unrun version-7 trained-head diagnostic is not included.
- [x] Result JSONs include experiment settings and provenance where available, and canonical figures have stable paths.
- [x] Regression workflow is configured at `.github/workflows/tests.yml` for pushes and pull requests to `main`.
- [x] The cleaned regression suite passed 56 tests in GitHub Actions. Use the latest workflow run on the active `main` head for current status.

## 2. Canonical artifacts

- CPC primary result: `ch03/results/table_03_05_cpc_metrics.json`
- CPC 25-frame sensitivity analysis: `ch03/results/cpc_retrieval_25frame_sensitivity.json` (secondary, not Table 3.5 primary)
- Five CPC checkpoints: `checkpoints/cpc_retrieval_20261018/`
- Expanded mechanism diagnostic: `ch03/results/three_mechanisms_diagnostics.json`
- Collapse-demo metrics: `ch03/results/collapse_demo_metrics.json`
- Table 3.4: `ch03/results/table_03_04_objective_comparison.json`
- Standalone MAE metrics: `ch03/results/mae_metrics.json`
- CPC figure: `ch03/figures/fig03_16_cpc_future_retrieval.png`
- Collapse figures: `ch03/figures/fig03_09_collapse_demo.png` and `ch03/figures/fig03_09_objective_space.png`
- Methods, caveats, and reproduction instructions: `ch03/README.md`, `ch03/publication_readiness_audit.md`, and `ch03/discrepancies_and_findings.md`

## 3. Interpretation limits that must survive into the manuscript

- CPC's primary score is **in-batch future-latent retrieval**, not a 200-way global ranking, physical-state prediction, recursive missing-frame rollout, or action-conditioned prediction.
- The 200 retrieval windows are non-overlapping windows sampled from five validation episodes, not 200 independent trajectories. The candidate batch sizes and 3.5% query-weighted chance reference are described in the audit.
- The expanded three-mechanism study is reduced-scale DMC evidence, not an ImageNet-scale reproduction or a universal ranking of methods.
- The current results do not establish complete physical-state recovery, long-horizon rollout stability, or planning utility.
- Preserve negative and mixed findings, especially weak velocity decodability and the limited two-frame aliasing result. Do not tune runs merely to support the planned narrative.

## 4. Manuscript handoff checklist

- [ ] Write the authored Chapter 3 prose from `chapter_03_design.md`, using the canonical code and result JSONs as the source of truth.
- [ ] Correct all numerical claims and caption metadata against the corresponding result artifact, including the differing seed counts between experiments.
- [ ] Keep every CPC claim aligned with future-latent retrieval and the stated candidate protocol. Do not resurrect the historical recursive gap-rollout scores as standard CPC performance.
- [ ] Finish the mathematics, listing excerpts, exercises, and summary in the design's planned teaching order.
- [ ] Add and verify citations/credit lines, including adapted figures and the DeepMind Control Suite source.
- [ ] Review each figure and table in manuscript layout, especially Figure 3.16 and the two collapse-demo plots; code-level inspection is not a layout check.
- [ ] Complete the Manning style checklist in `chapter_03_design.md` against the actual manuscript, not merely this design document.
- [ ] Before production handoff, confirm that the latest GitHub Actions regression run on `main` is green.

## 5. Reproduction rules

Run the regression tests from the repository root:

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Stop here." }
```

Use the commands and settings documented in `ch03/README.md` for deliberate experiment reproduction. Write new outputs to separate directories. Do not overwrite canonical result JSONs, figures, or the retained CPC checkpoints while checking or reproducing the release.

The code companion is the implementation/evidence package. The finished Chapter 3 manuscript and production-layout pages are separate authoring work.
