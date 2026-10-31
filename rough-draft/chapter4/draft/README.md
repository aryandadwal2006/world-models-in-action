# Chapter 4 draft — Predicting in representation space

This folder contains the current section-by-section working draft for Chapter 4 of *World Models in Action*. The manuscript is being developed from the author's JEPA notes, the current Chapter 4 design dossier, original source papers and official research repositories.

## Draft order

1. 00-introduction.md — chapter framing and the role of a predictive world model
2. 4.1-why-predict-in-latent-space.md — latent prediction versus observation reconstruction
3. 4.2-jepa-objective.md — generic context encoder, target encoder and predictor
4. 4.2.1-i-jepa.md — image JEPA
5. 4.2.2-v-jepa.md — video JEPA
6. 4.2.3-v-jepa-2-and-2.1.md — action-free pretraining, V-JEPA 2-AC and separate dense-feature V-JEPA 2.1 recipe
7. 4.2.4-regularization-before-distributional-tests.md — collapse, redundancy and distributional shape
8. 4.2.5-lejepa-sigreg.md — risk assumptions, Gaussian target, Cramér–Wold projections and SIGReg
9. 4.2.6-visreg.md — scale, shape, centering and sliced-Wasserstein regularization
10. 4.2.7-adjacent-modal-and-variational-jepas.md — audio, vision-language and probabilistic variants
11. 4.3-action-conditioned-prediction-and-planning.md — action-conditioned world models, planning and energy
12. 4.3.1-path-towards-autonomous-intelligence.md — LeCun's six-module proposal
13. 4.3.2-mode-one-mode-two-and-hierarchical-planning.md — reactive and deliberative modes, distillation, hierarchy and uncertainty
14. 4.4-reading-a-miniature-jepa-system.md — reading official implementations without claiming a reproduction
15. 4.5-summary-and-exercises.md — chapter close and exercises

## Figures

There is no artwork in this draft. Figure markers contain a title and detailed description of the intended visual argument. The AsciiDoc assembly preserves them as explicit figure-description placeholders; no source-paper figure is reused or redrawn at this stage.

## Evidence rules

- Mathematical results are stated with their assumptions.
- Paper-reported numbers are attributed to the paper and evaluation protocol.
- A finite projection sketch is not described as an exact population-level distribution certificate.
- The LeJEPA Gaussian result is not generalized beyond the paper's stated downstream-risk framework.
- VISReg's scalar derivative is distinguished from a lower bound on all encoder-parameter gradients.
- Action-free video representation learning is separated from action-conditioned prediction and planning.
- LeCun's 2022 paper is described as a position paper proposing an architecture.
- No new experiments, reproductions or benchmark claims are introduced by this book draft.
- Official repositories are consulted for implementation structure. Restricted code and model artifacts are not copied.

See the parent Chapter 4 dossier references and claims audit for the complete source map and proof notes.
