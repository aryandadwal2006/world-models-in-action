# Chapter 4: Predicting in representation space

This directory is the working source of Chapter 4 of *World Models in Action*. Each numbered Markdown file develops one part of the chapter. No consolidated manuscript or PDF is maintained here. The book-layout source and preview, if required, will be produced only after the chapter text and figure plan have passed review.

## Governing argument

A world model is useful when its internal state allows an agent to distinguish situations that lead to different outcomes under relevant actions. Predictive representation learning addresses part of this problem by training an encoder to make an unobserved target predictable from an available context. The difficult question is not simply how to minimize a feature distance. It is what information the target representation preserves, what distribution of representations the objective permits, and what the resulting representation allows a downstream predictor or planner to infer.

The chapter follows that argument from first principles:

1. **Compatibility and energy.** Express prediction as assigning low incompatibility to a context and a candidate target. Separate a compatibility score from a normalized probability and from an agent's task cost.
2. **The prediction problem.** Define context, target, query, representation, predictor and target distribution. Establish when a predictor is needed and what a feature-space loss actually measures.
3. **Degenerate solutions.** Derive why an unconstrained agreement or feature-prediction objective admits a constant representation. Distinguish complete collapse from loss of task-relevant distinctions.
4. **Predictive sufficiency and geometry.** State the task-relative condition a useful representation should satisfy, then connect the geometry of an embedding distribution to the performance of downstream estimators.
5. **Downstream risk.** Derive the covariance-spectrum results for linear prediction and the density-gradient term that appears in local non-linear prediction. State the assumptions under which isotropic Gaussian embeddings are optimal.
6. **Practical distribution matching.** Derive the projected characteristic-function objective and the separate scale/shape construction. Explain what finite projections can and cannot establish.
7. **Actions and planning.** Separate passive video regularities from predictions conditioned on interventions. Trace the additional model, objective, optimizer and feedback loop required for planning.
8. **An agent around the predictor.** Use the proposed architecture in *A Path Towards Autonomous Machine Intelligence* to connect prediction, costs, memory, action selection and hierarchy.

The named systems are evidence and examples within this argument. They are not the organizing principle of the chapter. I-JEPA and V-JEPA establish concrete forms of feature prediction; V-JEPA 2-AC illustrates the addition of actions; V-JEPA 2.1 illustrates dense prediction targets; LeJEPA and VISReg motivate different ways to constrain the embedding distribution; VL-JEPA shows that the target representation can be linguistic. These methods will be introduced only where they clarify a mathematical or architectural decision.

## Draft files

| File | Purpose |
|---|---|
| `01_compatibility_and_energy.md` | Energy, compatibility, latent prediction and the distinction between scoring and probability |
| `02_the_prediction_problem.md` | Context, target, query, target distribution, predictor and conditional-mean derivation |
| `03_degeneracy_and_collapse.md` | Exact collapse solution and the limits of architectural stabilization |
| `04_predictive_sufficiency_and_geometry.md` | Task-relative predictive sufficiency and the geometry of an embedding distribution |
| `05_linear_downstream_risk.md` | Ridge bias, least-squares variance and covariance eigenvalues |
| `06_nonlinear_risk_and_fisher_information.md` | Local estimators, score functions and Fisher information |
| `07_sigreg_and_distribution_sketching.md` | Cramér–Wold, characteristic functions and SIGReg |
| `08_visreg_scale_and_shape.md` | Centering, scale, sliced-Wasserstein shape and optimization |
| `09_action_conditioned_prediction.md` | Interventions, action-conditioned latent dynamics and V-JEPA 2-AC |
| `10_planning_and_model_error.md` | Goal costs, search, receding-horizon control and model exploitation |
| `11_hierarchy_memory_and_agent_architecture.md` | Multiple futures, H-JEPA, Mode 1/Mode 2, STM and Configurator |
| `12_synthesis_and_exercises.md` | Synthesis, limitations and derivation/design exercises |
| `references.md` | Primary papers and implementation sources |
| `figure-register.md` | Proposed figures, official-source candidates and reuse notes |

## Editorial and technical rules

- Write for a technically curious reader who is learning the subject, not for a specialist who only wants a literature summary. Define every mathematical object before using it.
- Derive the central equations in the text. Do not replace derivations with a sequence of paper-specific objective summaries.
- Separate a mathematical identity, a theorem under assumptions, an empirical result, and an architectural proposal.
- Never infer action-conditioned dynamics from an action-free video objective.
- Never infer physical sufficiency, causality, calibrated uncertainty or planning competence from a low representation loss or a linear-probe score.
- The Gaussian-optimality result must be scoped to the downstream estimators, risk functionals, covariance constraints and regularity assumptions analysed in LeJEPA. It is not universal optimality for every task.
- Cramér–Wold concerns all one-dimensional projections. A finite, stochastic set of projections is an approximation.
- A scalar derivative with respect to a scale statistic is not a guarantee about every encoder-parameter gradient.
- An energy is not a probability density unless a normalization and base measure are specified. A training discrepancy, an energy score and an agent's task cost are distinct objects.
- Do not call code inspection or checkpoint loading a reproduction. No new experiments are claimed in this chapter.
- Use no em dashes. Keep prose academic and direct. Do not address the author or include process commentary in reader-facing text.
- Prefer an original explanation when a paper figure is too specialized or does not teach the point required here. Use an official figure only after checking the paper version, figure identity, attribution and reuse terms. Do not assume a paper's arXiv license grants rights to its artwork or repository code.

## Source inventory

The primary-source PDFs supplied for this chapter remain in `rough-draft/chapter4/papers_pdfs/`. The filenames and exact paper identities are mapped in `references.md`. The author's first-principles research notes remain in `rough-draft/chapter4/jepa-notes.md`. These inputs are source material, not reader-facing chapter sections. They are not modified by this chapter work.

The binary PDF files are present in the repository. The GitHub file connector can report their exact paths, sizes and blob identifiers but does not expose their binary bytes for extraction in this environment. Accordingly, the source review uses the matching versioned full-text papers and PDF page renders where available. Any PDF whose binary could not be retrieved is identified in the source-review record rather than falsely described as having been downloaded.
