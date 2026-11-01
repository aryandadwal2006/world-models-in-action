# Chapter 4: Predicting in representation space

This directory contains the section-by-section working source of Chapter 4. Drafts remain separate Markdown files here; the consolidated AsciiDoc and PDF are produced only after the section sequence, mathematical claims, figures and references have been reviewed.

## Governing argument

A predictive representation is trained to make a selected target feature predictable from an allowed context. That objective does not by itself specify a sufficient physical state, identify action effects, or supply a planning objective. The chapter develops this distinction from first principles.

1. Compatibility and energy: distinguish a compatibility score, a normalized probability and an agent's task cost.
2. The prediction problem: define context, target, query, target distribution and predictor.
3. I-JEPA: derive masked image-feature prediction, target-block sampling and spatial queries.
4. V-JEPA: extend prediction to space-time tokens and distinguish temporal masking from action conditioning.
5. Degenerate solutions: derive constant and rank-deficient representations and separate agreement from information.
6. Predictive sufficiency and geometry: define task-relative state quality and relate embedding geometry to downstream prediction.
7. Linear downstream risk: derive the role of covariance eigenvalues in ridge bias and least-squares variance.
8. Nonlinear risk: derive the density-score term and Fisher-information bound under the assumptions used by LeJEPA.
9. SIGReg: derive projected distribution matching from characteristic functions and Cramér–Wold.
10. VISReg: separate centering, scale and distribution shape.
11. V-JEPA 2 and V-JEPA 2.1: distinguish action-free pretraining, action-conditioned post-training and dense prediction.
12. Action-conditioned prediction: formulate controlled latent dynamics and intervention-aligned data.
13. Planning and model error: connect predicted outcomes to search, task cost and feedback.
14. An agent around the predictor: examine memory, hierarchy, Mode 1/Mode 2 and the proposed cognitive architecture.
15. Reading a JEPA training step: inspect data flow, losses, gradient paths and implementation choices.
16. Synthesis and exercises.

Named methods serve the mathematical argument. The chapter is not organized as a survey of every paper using a JEPA-like architecture. Audio, language and variational extensions are included only where they clarify a general design choice.

## Draft files

| File | Purpose |
|---|---|
| `01_compatibility_and_energy.md` | Energy, compatibility, latent prediction and scoring versus probability |
| `02_the_prediction_problem.md` | Context, target, query, target distribution, predictor and conditional-mean derivation |
| `03_ijepa_spatial_prediction.md` | Spatial masks, target-encoder information flow and I-JEPA |
| `04_vjepa_spatiotemporal_prediction.md` | Video tokens, spatiotemporal masks, L1 prediction and V-JEPA |
| `05_degeneracy_and_collapse.md` | Constant solutions and collapse modes |
| `06_predictive_sufficiency_and_geometry.md` | Task-relative predictive sufficiency and embedding geometry |
| `07_linear_downstream_risk.md` | Ridge bias, least-squares variance and covariance eigenvalues |
| `08_nonlinear_risk_and_fisher_information.md` | Local estimators, score functions and Fisher information |
| `09_sigreg_and_distribution_sketching.md` | Cramér–Wold, characteristic functions and SIGReg |
| `10_visreg_scale_and_shape.md` | Centering, scale, sliced-Wasserstein shape and optimization |
| `11_vjepa2_action_conditioning_and_dense_features.md` | V-JEPA 2, 2-AC and V-JEPA 2.1 |
| `12_action_conditioned_prediction.md` | Interventions and controlled latent dynamics |
| `13_planning_and_model_error.md` | Goal costs, search, receding-horizon control and model exploitation |
| `14_hierarchy_memory_and_agent_architecture.md` | H-JEPA, Mode 1/Mode 2, memory and Configurator |
| `15_reading_a_jepa_training_step.md` | Training graph and implementation audit |
| `16_synthesis_and_exercises.md` | Synthesis, limits and exercises |
| `references.md` | Primary papers and implementation sources |
| `figure-register.md` | Figure plan, primary-source candidates and attribution notes |

## Editorial and technical rules

- Define mathematical objects before using them. Derive central equations in the text rather than replacing derivations with a sequence of paper summaries.
- Distinguish identities, theorems under assumptions, empirical results and architectural proposals.
- Never infer action-conditioned dynamics from an action-free video objective.
- Never infer physical sufficiency, causality, calibrated uncertainty or planning competence from a low representation loss or a probe score.
- Scope LeJEPA's Gaussian-optimality result to its downstream estimators, risk functionals, covariance constraints and regularity assumptions. It is not universal optimality for every task.
- Cramér–Wold concerns all one-dimensional projections. Finite stochastic projections are an approximation.
- A derivative with respect to a scale statistic is not a guarantee about every encoder-parameter gradient.
- An energy is not a probability density without a normalization and base measure. A prediction discrepancy, compatibility energy and task cost are distinct.
- Do not claim experiments were reproduced when only papers or code were inspected.
- Use academic, direct prose. No author-facing process commentary and no em dashes.
- Prefer original explanatory figures when an official figure does not teach the required idea. Check paper version, artwork attribution and reuse terms before reproducing any figure. Do not assume source-code licenses permit commercial book reproduction.

## Source inventory and access

The supplied PDFs and the author's first-principles notes remain in `rough-draft/chapter4/papers_pdfs/` and `rough-draft/chapter4/jepa-notes.md`. The source mapping and access record are in `references.md`. The PDFs are tracked binary files. The GitHub file connector exposes their names, sizes and blob identifiers but not their binary bytes for local extraction in this environment. The review uses the matching versioned full-text papers, including appendices and references where available, plus extracted OpenReview text for LeCun's position paper. This supports a text/equation review but is not a page-by-page visual inspection of all eight supplied PDF binaries; that limitation must remain explicit.
