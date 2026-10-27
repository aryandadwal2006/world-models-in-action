# Legacy Chapter 3 section-draft workspace

**Status (27 October 2026): archived working material, not the current chapter source.**

This directory contains section-by-section Markdown drafts converted from an earlier Word draft. Those drafts and their embedded figure/production notes remain historical context and are stale relative to the composed chapter. The standalone `figure-and-table-production.md` file is the current synchronized crosswalk; `images/README.md` is the canonical printed-order figure register.

## Current sources of truth

- **Authored chapter source:** [`manuscript/CH03_Dadwal_World_Models_in_Action.adoc`](../manuscript/CH03_Dadwal_World_Models_in_Action.adoc) on `manuscript-theory`.
- **Author-side PDF preview and build instructions:** [`manuscript/README.md`](../manuscript/README.md)
- **Current asset-to-printed-figure map:** [`images/README.md`](../images/README.md)
- **Current figure/table/listing crosswalk:** [`figure-and-table-production.md`](figure-and-table-production.md), synchronized with `images/README.md`.
- **Code and experiment source:** [`ch03/README.md`](../ch03/README.md), `worldmodels/`, `ch03/results/`
- **Historical design rationale:** [`chapter_03_design.md`](../chapter_03_design.md), explicitly marked as superseded planning material.

Do not use the section drafts or the old numbered plans in this folder as the authority for current prose, figure numbering, table numbering, or production readiness. The current composed manuscript has 29 figure placeholders, one numbered synthesis table, and one short VICReg listing. Stable image filename stems do not always match printed figure numbers.

## Retained draft order

These files are retained as reference material, in their former order:

1. [`00-introduction.md`](00-introduction.md)
2. [`3.1-learning-a-state-without-labels.md`](3.1-learning-a-state-without-labels.md)
3. [`3.2-pretext-tasks.md`](3.2-pretext-tasks.md)
4. [`3.3-contrastive-learning.md`](3.3-contrastive-learning.md)
5. [`3.4-collapse-and-anti-collapse.md`](3.4-collapse-and-anti-collapse.md)
6. [`3.5-masked-prediction.md`](3.5-masked-prediction.md)
7. [`3.6-latent-bottlenecks.md`](3.6-latent-bottlenecks.md)
8. [`3.7-temporal-prediction-cpc.md`](3.7-temporal-prediction-cpc.md)
9. [`3.8-assembling-the-encoder.md`](3.8-assembling-the-encoder.md)
10. [`exercises.md`](exercises.md)
11. [`summary.md`](summary.md)
12. [`appendix-a-proof-details.md`](appendix-a-proof-details.md)
13. [`references-and-credits.md`](references-and-credits.md)

## Technical and empirical source-of-truth rules

- Read conceptual implementations from `worldmodels/`; read experiment configuration and runners from `ch03/`.
- Read numeric results from committed JSON files under `ch03/results/`. Do not overwrite canonical result files when reproducing experiments.
- CPC's primary result is held-out, in-batch future-latent retrieval. It is not a physical rollout, action-conditioned prediction, missing-frame reconstruction, or evidence of planning utility.
- Physics-state targets are used for evaluation probes and the explicitly supervised reference arm, not to train the self-supervised objectives.
- A source-level or test-suite pass does not replace a separate visual review of the built PDF.

## Build and validation

For the current chapter preview, follow the instructions in [`manuscript/README.md`](../manuscript/README.md). From a full checkout, the structural verifier can be run with:

```bash
python manuscript/verify-manuscript.py
```

The PDF is an author-side preview, not Manning's final production typesetting.