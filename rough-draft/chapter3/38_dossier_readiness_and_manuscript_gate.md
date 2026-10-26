# Chapter 3 final-draft and publication gate

**Current branch:** `manuscript-theory`  
**State:** theory-first manuscript composed; final rendered review in progress.

## Scope and current form

The chapter now covers objective construction, the candidate-sampling assumptions of contrastive methods, target/gradient mechanics for bootstrap and self-distillation methods, variance/covariance and distribution-shape regularizers, masked prediction targets, the EBM/diffusion boundary, and predictive sufficiency for downstream world models. The JEPA architecture is introduced as a bridge; Chapter 4 remains the full treatment of predictive world models.

The composed source includes **29 conceptual figure placeholders**, each linked to a detailed Markdown production brief in `images/CH03_FNN_Dadwal.md`. This branch deliberately contains no rendered figure files in `images/`; diagrams must be generated and approved later. The chapter includes **two short source-attributed code excerpts** from pinned SimCLR and VICReg source, **one synthesis table**, and a **single consolidated bibliography**. The old end-of-chapter implementation-reference table is removed; method-specific repository links sit beside the relevant explanations.

There are no custom training experiments or complete hand-written reimplementations in the chapter. Upstream code links establish provenance, not an independent reproduction result.

## Final publication checks

| Gate | Current position | Remaining action |
|---|---|---|
| Mathematical correctness | Core equations reviewed against method-specific mechanics; VISReg notation now follows a source-inspected reference implementation's batch reductions | Inspect the latest PDF after all recent equation and link changes; retain source-version caveats where the exact recipe is configuration-dependent |
| Scope and prose | Robotics detours, homegrown experiments and repetitive objective summaries removed | Final read for directness and consistency with Chapter 4; keep frontier JEPA material as a bridge rather than a second full chapter |
| Figures | 29 detailed prompts and matching source placeholders; rendered artwork intentionally absent | Generate artwork from the prompts, validate equations/arrows/captions, and replace each placeholder only after author approval |
| Official sources | Code excerpts are pinned and attributed; method links sit beside the relevant method | Recheck links and source revisions when a final manuscript release is prepared; respect repository-specific licenses |
| AsciiDoc structure | Manning-style anchors, 29 unique figure anchors, one table and two source listings; validator checks prompt-only asset layout and native AsciiDoc links/math | Run `python3 manuscript/verify-manuscript.py` via the build workflow |
| PDF production | Automated AsciiDoc build and PDF style validation are configured on this branch | Inspect the latest artifact page by page for rendered operators, inline math, hyperlinks, equation spacing, table breaks and placeholder pagination |
| Length | The latest checked preview was 42 pages before the final link/math cleanup | Recount the latest successful build; expand theory only where the reader's understanding benefits, not to meet a page target |

## Release boundary

Do not call the chapter print-ready while artwork placeholders remain. A successful source build is necessary but not sufficient: the release gate requires an approved illustration set, a final citation/link pass, page-by-page PDF inspection, and a clear statement of which results are cited from papers versus independently reproduced.
