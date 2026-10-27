# Chapter 3 Implementation and Release Status

**Status date:** 27 October 2026  
**Code-release branch:** `main`  
**Current manuscript branch:** `manuscript-theory`  
**Scope:** this file tracks code-companion status and points to the current manuscript production status. It does not replace the live figure crosswalk or readiness gate.

## 1. Code companion status

- [x] Canonical model, loss and evaluation implementations live under `worldmodels/`.
- [x] Numbered Chapter 3 experiment runners live under `ch03/`; regression tests live under `tests/`.
- [x] CPC uses multi-horizon bilinear InfoNCE; its primary evaluation is held-out, in-batch future-latent retrieval.
- [x] The former custom recursive CPC gap-rollout metric is excluded from primary CPC claims.
- [x] Five-seed CPC checkpoints and checkpoint-only re-evaluation protocol are retained.
- [x] Expanded five-seed SimSiam/BYOL/Barlow Twins diagnostics and collapse-demo outputs are retained.
- [x] Method-comparison metrics use the corrected experiment and MAE feature path.
- [x] Result JSONs record experiment settings and provenance where available; canonical outputs have stable paths.
- [x] The regression workflow is configured for pushes and pull requests to `main`.
- [ ] Check the latest regression workflow run on the current `main` head before making a new release. Historical passing tests do not prove the latest head is green.

## 2. Canonical experiment artifacts

The following names are stable code/artifact identifiers, not necessarily current printed object numbers in the composed book chapter:

- CPC primary matched-length result: `ch03/results/table_03_05_cpc_metrics.json`
- CPC 25-frame sensitivity analysis: `ch03/results/cpc_retrieval_25frame_sensitivity.json` (secondary)
- Five CPC checkpoints: `checkpoints/cpc_retrieval_20261018/`
- Expanded mechanism diagnostic: `ch03/results/three_mechanisms_diagnostics.json`
- Collapse-demo metrics: `ch03/results/collapse_demo_metrics.json`
- Method-comparison metrics: `ch03/results/table_03_04_objective_comparison.json`
- Standalone MAE metrics: `ch03/results/mae_metrics.json`
- Code-generated plots and source panels: `ch03/figures/`

For the current manuscript's printed figure numbering, use `images/README.md`. For the synchronized figure/table/listing crosswalk, use `chapter3/figure-and-table-production.md`. Do not map printed numbering from result filenames.

## 3. Scientific interpretation boundaries

- CPC's primary score is **in-batch future-latent retrieval**, not a 200-way global ranking, physical-state prediction, recursive missing-frame rollout, or action-conditioned prediction.
- The 200 retrieval windows are sampled from five validation episodes, not 200 independent trajectories. Candidate-batch sizes and the 3.5% query-weighted chance reference are documented in the code audit.
- The expanded three-mechanism study is reduced-scale DMC evidence, not an ImageNet-scale reproduction or a universal ranking of methods.
- Results do not establish complete physical-state recovery, long-horizon rollout stability, or planning utility.
- Preserve negative and mixed findings, especially weak velocity decodability and the limited two-frame aliasing result. Do not tune runs merely to support a preferred narrative.

## 4. Current manuscript status

The current composed source is `manuscript/CH03_Dadwal_World_Models_in_Action.adoc` on **`manuscript-theory`**. The adjacent `manuscript/CH03_Dadwal_World_Models_in_Action.pdf` is a freshly generated author-side preview, not Manning's final typesetting.

The prose and figure-plan revision is complete; verification and a fresh PDF build are the next gate. The current source contains **18 numbered figure placeholders, one numbered synthesis table, and one short PyTorch VICReg listing**. It has a consolidated bibliography, no outbound hyperlinks, no TensorFlow listing, and no standalone JEPA architecture tutorial. The intended Chapter 4 JEPA coverage remains separate.

- Build and verifier instructions: `manuscript/README.md`
- Canonical printed figure register: `images/README.md`
- Synced figures/table/listing crosswalk: `chapter3/figure-and-table-production.md`
- Current prose/PDF production gate: `rough-draft/chapter3/38_dossier_readiness_and_manuscript_gate.md`

Historical design files and section drafts are retained as planning history only. They are not authoritative for the present source, object counts, printed numbering, or remaining work.

## 5. Reproduction rules

Run tests from the repository root:

```powershell
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed. Stop here." }
```

Use the commands and settings documented in `ch03/README.md` for deliberate experiment reproduction. Write new outputs to separate directories. Do not overwrite canonical result JSONs, plot assets, or retained CPC checkpoints during checking or reproduction.

The code companion is the implementation/evidence package. The composed authored chapter and its PDF preview are maintained separately. After figures are approved and inserted, rebuild the PDF and inspect the final integrated layout before calling the chapter production-ready.