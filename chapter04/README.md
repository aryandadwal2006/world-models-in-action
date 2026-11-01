# Chapter 4: Predicting in representation space

This directory contains the canonical, section-by-section reader-facing source of Chapter 4. Each numbered section is a separate Markdown file. The consolidated AsciiDoc and PDF are generated only after the sources and figures are checked.

## Governing argument
A predictive representation is trained to make a selected target feature predictable from an allowed context. That objective does not by itself specify a sufficient physical state, identify action effects, or supply a planning objective. The chapter derives these distinctions from first principles rather than organizing itself as a survey.

1. Compatibility, energy, probability and task cost.
2. Context, target, query, target distribution and squared-error prediction.
3. I-JEPA spatial context-to-target prediction.
4. V-JEPA spatiotemporal masking and its distinction from causal future prediction.
5. Constant and rank-deficient collapse.
6. Task-relative predictive sufficiency and embedding geometry.
7. Linear downstream risk under ridge and least squares.
8. Nonlinear local-regression risk and Fisher information.
9. Characteristic functions, Cramér–Wold and SIGReg.
10. VISReg's separation of centre, scale and shape.
11. V-JEPA 2's two-stage system and V-JEPA 2.1 dense-feature supervision.
12. Action-conditioned prediction and intervention assumptions.
13. Planning, feedback and model exploitation.
14. Memory, hierarchy, Mode 1/Mode 2 and predictive-agent architecture.
15. A tensor-level walkthrough of one JEPA training step.
16. Synthesis and exercises.

## Source files
The 16 section files are named `01_...` through `16_...`. `references.md` is the reader-facing bibliography source, `figure-register.md` maps all embedded figures to their assets, and `source-audit.md` records the primary-source review and access limitations.

## Figures
The chapter embeds 33 original explanatory SVG diagrams from `../images/chapter04/`. They explain the derivations and information paths, are not copies or tracings of paper artwork, and remain editable vector assets.

## Editorial and technical rules
- Define mathematical objects before using them; derive central equations.
- Distinguish identities, theorems under assumptions, empirical results and architectural proposals.
- Do not infer action-conditioned dynamics from an action-free video objective.
- Do not infer physical sufficiency, causality, calibrated uncertainty or planning competence from low representation loss or a probe score.
- Scope LeJEPA's Gaussian-optimality result to its stated estimators, risk functionals, covariance constraints and regularity assumptions.
- Cramér–Wold concerns all directions; finite stochastic projections are an approximation.
- A derivative with respect to a scale statistic is not a guarantee about every encoder-parameter gradient.
- An energy is not a probability density without a normalization and base measure.
- Do not claim experiments were reproduced when only papers or code were inspected.
- Use professional academic prose, with no author-facing process commentary.

## Source inventory and access
The supplied papers and first-principles JEPA notes remain in `rough-draft/chapter4/`. Their source mapping and review boundary are recorded in `references.md` and `source-audit.md`. The GitHub connector exposed the PDF names, sizes and blob identifiers, but did not expose the stored binary PDF bytes for local page-by-page visual inspection. Matching versioned primary-source texts were reviewed; this is not a claim that all eight exact repository PDF binaries were visually inspected on every page.
