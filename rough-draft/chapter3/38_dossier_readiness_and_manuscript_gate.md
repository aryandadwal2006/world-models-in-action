# Chapter 3 final-draft and publication gate

**Current branch:** `manuscript-theory`  
**State:** theory-first source revised for a PyTorch-only listing and the specified blue heading accent. The committed PDF has not yet been rebuilt from these latest source/theme changes, so the 41-page preview is stale.

## Scope and current form

The chapter derives contrastive, bootstrap/self-distillation, redundancy-reduction, variance/covariance, distribution-shape and masked-prediction objectives; distinguishes energy-based and diffusion models from feature-learning objectives; and ends by connecting representation design to predictive sufficiency. JEPA is a bridge to Chapter 4, not a second full treatment of world models.

The composed source includes **29 numbered figure placeholders**, each matched to a detailed Markdown production prompt in `images/CH03_FNN_Dadwal.md`. The `images/` directory deliberately contains no rendered artwork. The prompts share a restrained `#3B6FB6` accent, white backgrounds and grayscale-safe flat 2D style. The chapter contains **one short source-attributed PyTorch excerpt** from the pinned VICReg implementation, **one synthesis table**, and **one consolidated bibliography**. The TensorFlow SimCLR listing was removed rather than translated or reimplemented. The end-of-chapter implementation-reference table is removed; official repositories are linked beside relevant methods.

The chapter contains no custom training experiments or full hand-written reimplementations. Official code links establish provenance, not independent reproduction.

## Final publication checks

| Gate | Current position | Remaining action |
|---|---|---|
| Mathematics | Equations for InfoNCE, MoCo variants, BYOL/SimSiam, DINO, Barlow Twins, VICReg, MAE, CPC, diffusion, SIGReg/LeJEPA and VISReg are present. VISReg reductions were checked against a source-inspected reference implementation. | Final equation/source pass after the author approves and inserts actual artwork; preserve configuration-specific caveats. |
| Prose and scope | Repeated explanations trimmed; method families distinguish their targets, gradient paths, update rules and limitations. | Final editorial review against adjacent chapters; avoid expanding JEPA beyond its bridge role. |
| Figures | 29 detailed prompts and matching source placeholders; no rendered art in this branch. | Generate original figures from the prompts, check equations/arrows/captions and page-width readability, then replace each placeholder only after approval. |
| Official sources | Code excerpts are pinned and attributed; method links sit beside the relevant method. | Recheck source links and licenses at release. |
| Structure | The verifier is now configured to require unique Manning-style anchors, 29 placeholders/prompts, one table, one PyTorch listing, no TensorFlow references/listings, one bibliography, native AsciiDoc math/links, and a prompt-only image folder. Source-side checks after the edit confirm the one-listing and no-TensorFlow conditions; the full verifier and PDF build still need to be rerun on this commit. | Rerun the workflow after each substantive source or layout change. |
| Rendered PDF | The stored **41-page PDF predates** the TensorFlow-listing removal and blue-accent theme update; it still contains the old listing and must not be treated as the current source preview. | Rebuild the PDF from the current branch, then inspect page flow, equations, the sole PyTorch listing, table layout and bibliography pagination. |
| Print readiness | The current PDF is a theory-first author preview with visible placeholders, not a final book PDF. | Approve/insert all artwork and perform the final page-by-page review before calling it print-ready. |

## Release boundary

A successful build is necessary but not sufficient. The release gate remains closed until the author approves all 29 figures, checks the final citations/links and confirms the rendered page flow. Any reported experimental results must remain clearly attributed to their papers unless independently reproduced.
