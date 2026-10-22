# Chapter 3 Implementation and Release Status

**Status date:** 22 October 2026  
**Code-release branch:** `main`  
**Manuscript branch:** `manuscript`  
**Scope:** This file tracks the code companion and experiment artifacts. It is not the live manuscript checklist.

## 1. Code companion status

- [x] Canonical model/loss/evaluation implementations live under `worldmodels/`.
- [x] Numbered Chapter 3 runners live under `ch03/`; regression tests live under `tests/`.
- [x] CPC uses multi-horizon bilinear InfoNCE and its primary evaluation is held-out, in-batch future-latent retrieval.
- [x] The former custom recursive CPC gap-rollout metric is excluded from primary CPC claims.
- [x] Five-seed CPC checkpoints and checkpoint-only re-evaluation protocol are retained.
- [x] Expanded five-seed SimSiam/BYOL/Barlow Twins diagnostics and collapse-demo outputs are retained.
- [x] The method-comparison metrics use the corrected experiment and MAE feature path.
- [x] Result JSONs record experiment settings and provenance where available; experiment outputs have stable paths.
- [x] The regression workflow is configured for pushes and pull requests to `main`.
- [ ] Before relying on code-release status, check the latest regression workflow run on the current `main` head. The 56-test pass recorded in earlier audit notes is historical evidence, not proof that the latest head is green.

## 2. Canonical experiment artifacts

These stable **file paths** include historical names such as `table_03_04` and `table_03_05`. Those filenames are experiment-artifact identifiers, not a guarantee that the current composed chapter prints tables with the same numbers.

- CPC primary matched-length result: `ch03/results/table_03_05_cpc_metrics.json`
- CPC 25-frame sensitivity analysis: `ch03/results/cpc_retrieval_25frame_sensitivity.json` (secondary)
- Five CPC checkpoints: `checkpoints/cpc_retrieval_20261018/`
- Expanded mechanism diagnostic: `ch03/results/three_mechanisms_diagnostics.json`
- Collapse-demo metrics: `ch03/results/collapse_demo_metrics.json`
- Method-comparison metrics: `ch03/results/table_03_04_objective_comparison.json`
- Standalone MAE metrics: `ch03/results/mae_metrics.json`
- Code-generated figures and source panels: `ch03/figures/`

For the current figure/asset and printed-object crosswalk, use `images/README.md` and `chapter3/figure-and-table-production.md`.

## 3. Scientific interpretation boundaries

- CPC's primary score is **in-batch future-latent retrieval**, not a 200-way global ranking, physical-state prediction, recursive missing-frame rollout, or action-conditioned prediction.
- The 200 retrieval windows are non-overlapping windows sampled from five validation episodes, not 200 independent trajectories. Candidate batch sizes and the 3.5% query-weighted chance reference are documented in the code audit.
- The expanded three-mechanism study is reduced-scale DMC evidence, not an ImageNet-scale reproduction or a universal ranking of methods.
- Results do not establish complete physical-state recovery, long-horizon rollout stability, or planning utility.
- Preserve negative and mixed findings, especially weak velocity decodability and the limited two-frame aliasing result. Do not tune runs merely to support a preferred narrative.

## 4. Manuscript status (separate branch)

The current composed source is `manuscript/CH03_Dadwal_World_Models_in_Action.adoc` on the `manuscript` branch. Its build instructions are in `manuscript/README.md`; the generated PDF is an author-side preview rather than final Manning production typesetting.

The old unchecked authoring checklist from the v3 planning stage is superseded. Do not use `chapter_03_design.md`, `agent_memory_mustread/chapter_03_design (1).md`, or the section-draft folder as the current prose or figure-number authority. The design copies are retained as history and marked accordingly. The composed source has 20 numbered figure anchors, two numbered table anchors, and 12 numbered listing anchors; run its verifier and build workflow when reviewing changes.

## 5. Reproduction rules

Run tests from the repository root:

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Stop here." }
```

Use the commands and settings documented in `ch03/README.md` for deliberate experiment reproduction. Write new outputs to separate directories. Do not overwrite canonical result JSONs, figure assets, or retained CPC checkpoints during checking or reproduction.

The code companion is the implementation/evidence package. The composed authored chapter and its PDF preview are maintained separately on `manuscript`.
