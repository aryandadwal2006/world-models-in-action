# Chapter 3 final-draft and production gate

**Updated:** 27 October 2026  
**Current branch:** `manuscript-theory`  
**Current source:** `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`  
**State:** the theory-first prose pass is complete and approved to proceed to figure production. The current source has been rebuilt into the committed author-side PDF. The manuscript itself should remain unchanged during the artwork phase except for approved placeholder replacement and necessary layout fixes.

## Scope and current form

The chapter follows the book TOC: self-supervised foundations, contrastive objectives, non-contrastive objectives, masked prediction, latent bottlenecks, compact state representations, summary, exercises and references. Its organizing idea is that each objective rewards selected relationships in the data, which does not by itself certify a representation as a sufficient state for a world model.

The chapter derives the central mathematical mechanisms, including InfoNCE/NT-Xent and its gradient and sampling assumptions; BYOL and SimSiam update/gradient paths; DINO teacher distributions; Barlow Twins and VICReg statistics; SIGReg/LeJEPA and VISReg distribution constraints; masked-target objectives; and predictive sufficiency. Energy-based learning and diffusion are distinguished from feature-learning objectives. JEPAs are mentioned in the methods survey, while their architecture tutorial remains in Chapter 4. Do not reintroduce a standalone JEPA subsection or a Chapter 4 bridge.

The composed source contains:
- **29 numbered figure placeholders**, each matched to a detailed brief under `images/`;
- **one numbered synthesis table**;
- **one short source-attributed PyTorch excerpt** from the official VICReg implementation;
- one consolidated bibliography and exercises;
- no outbound hyperlinks in the chapter itself.

The source is a theory-first book chapter, not an experiment report or a hands-on implementation tutorial. Empirical findings from the original literature are attributed to their papers; the source excerpt does not claim independent reproduction.

## Current validation status

| Gate | Current position | Remaining action |
|---|---|---|
| Structure and source checks | The manuscript verifier is configured for the current one-table/one-listing structure, 29 figure placeholders, the consolidated bibliography, native AsciiDoc math, unique anchors and no outbound links. The current PDF was generated through the branch's manuscript build workflow. | Rerun verification and build after artwork insertion or any source/layout change. |
| Mathematics and technical claims | The current source distinguishes target construction, optimization mechanism, assumptions and limitations across method families. | During figure production, check every arrow, equation and label against the source. Avoid expanding prose unless a concrete error is found. |
| Prose and scope | The prose pass is complete; redundancy has been reduced and the diffusion discussion is intentionally compact. | No further broad rewrite is planned. Make only evidence-based corrections if a specific issue is found. |
| Figures | All 29 briefs are present. No rendered artwork has been approved or inserted. | Generate original figures from the current briefs; validate each figure and replace the corresponding placeholder only after approval. |
| Numbering and crosswalks | `images/README.md` is the canonical printed-order figure register. `chapter3/figure-and-table-production.md` is synchronized to it and also records the current table/listing. | Keep both crosswalks synchronized when an explicitly approved change affects numbering. |
| Bibliography and attribution | Methods are attributed in prose and entries are gathered in the references section; the chapter intentionally contains no outbound hyperlinks. | Check author/year/title accuracy and the final bibliography layout when reviewing the integrated PDF. |
| PDF and layout | The source has a freshly generated author-side PDF with placeholders. This is not Manning's final production typesetting. | After figures are inserted, inspect page breaks, equation readability, figure sizing/captions, table fit and bibliography pagination. |
| Final production readiness | **Not yet print-ready** because artwork and final integrated layout review remain. | Approve all 29 figures and inspect the rebuilt illustrated chapter before treating the chapter as production-complete. |

## Figure production principles

1. Work from `images/README.md` and the matching `images/CH03_FNN_Dadwal.md` brief; filename stems are not printed figure numbers.
2. Use original teaching schematics, a white background, grayscale-safe dark line work and `#3B6FB6` as the only restrained accent. No gradients, 3D effects, Mermaid or decorative watermarks.
3. Preserve distinctions the prose is careful about: a training score is not semantic sufficiency; anti-collapse constraints do not establish semantic quality; finite distribution diagnostics do not prove exact distribution matching; and latent prediction is not automatically action-conditioned rollout.
4. Do not add invented metrics, paper screenshots, claims of reproduced results, or extra JEPA architecture content.
5. Review each figure at final book-column width and confirm that its caption, labels and arrows remain readable in the rendered PDF.

## Release boundary

Content editing is complete for the current figure-production phase. The next milestone is the **illustrated** chapter preview, not another full rewrite. The chapter becomes production-ready only after all artwork is approved and the final integrated PDF has been inspected.