# Chapter 3 revision and production gate

**Updated:** 27 October 2026  
**Current branch:** `manuscript-theory`  
**Current source:** `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`  
**State:** prose and figure plan revised after the figure-by-figure audit. The source and register now specify 18 figures. Structural verification and a fresh PDF build are still required; no rendered artwork has been approved or inserted.

## Editorial contract

Chapter 3 follows the book's TOC: self-supervised foundations, contrastive objectives, non-contrastive objectives, masked prediction, latent bottlenecks, compact state representations, summary, exercises and references. It builds on Chapter 1's world-model loop and Chapter 2's state contract. It should not reteach the anatomy of a world model; it should explain which relationships each training objective constrains and what that does not establish about downstream state quality.

The reader baseline (MQR) is basic Python/PyTorch, neural networks and transformers, with no assumed self-supervised learning background. Retain first-principles derivations of InfoNCE, BYOL/SimSiam gradient and update paths, teacher distributions, Barlow Twins/VICReg statistics, masked-target objectives and distributional regularizers. Define new symbols and assumptions where they first matter. Do not repeat Chapter 2's full explanations of sufficiency and aliasing; invoke them as the downstream contract and explain their consequences for objective design and evaluation.

## Current inventory

- **18 conceptual figure placeholders**, listed in `images/README.md` in printed order.
- **One numbered synthesis table** and **one short source-attributed PyTorch VICReg listing**.
- One consolidated bibliography and exercises.
- No rendered artwork in the prompt-only `images/` directory.

The reduction removes the artificial family map, repeated method-card overviews and separate diffusion comparison. The two InfoNCE figures are merged into a single loss-and-gradient schematic. The projector path and downstream representation interfaces are combined in a two-lane figure that keeps their roles distinct.

## Validation checklist

| Gate | Required check |
|---|---|
| Source structure | Run `python3 manuscript/verify-manuscript.py`; confirm 18 figure anchors match the canonical register, one table, one listing and unique anchors. |
| Explanations | Confirm the opening builds on Chapters 1–2; retain first-principles derivations rather than repeat earlier state definitions. |
| Mathematics | Check InfoNCE values and gradients; distinguish sampled candidate normalization from a full partition function; qualify the VICReg variance floor relative to `epsilon`; preserve target and gradient paths. |
| Scope | Keep CPC as the temporal-prediction bridge. Keep JEPA architecture in Chapter 4 and diffusion mechanics in Chapter 9. Do not restore the family map or method-card galleries. |
| Figures | Generate only from the 18 approved briefs. Check type at final book-column width. |
| PDF | Run `./build-pdf.sh`; inspect equation formatting, captions, section transitions and bibliography flow. The PDF is an author-side preview, not Manning's final typesetting. |
| Artwork integration | Replace placeholders only after each figure is approved, rebuild the PDF, then inspect the full integrated chapter. |

## Scientific interpretation limits

- Self-supervised loss fit is evidence about the chosen proxy target, not a universal quality score.
- Collapse diagnostics and distribution regularizers do not prove semantic or predictive sufficiency.
- CPC candidate retrieval is not physical-state prediction, an action-conditioned transition model, or recursive rollout.
- A linear probe tests linear accessibility under a protocol; failure does not prove information absence.
- Finite projection/sample diagnostics do not prove exact distribution matching.
- Keep DINO, DINOv2 and DINOv3 components distinct; do not attribute later recipe details to original DINO.

## Sources of truth

- Authored chapter: `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`
- Author-side PDF/build instructions: `manuscript/README.md`
- Canonical printed register: `images/README.md`
- Synchronized crosswalk: `chapter3/figure-and-table-production.md`
- Code and evidence: `ch03/README.md`, `worldmodels/` and canonical JSON under `ch03/results/`

Historical plans remain archived and are not a source for current figure counts or captions.
