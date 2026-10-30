# Chapter 4 research dossier

**Book:** *World Models in Action*  
**Chapter:** 4, “JEPA and Predictive Representation Learning”  
**Target branch:** manuscript  
**Prepared:** 31 October 2026  
**Status:** research dossier and chapter design; not final reader-facing prose.

## Purpose

This folder turns the author's JEPA research notes into a source-aware plan for a substantial Chapter 4. The chapter should make the transition from representation learning (Chapter 3) to predictive world models (Chapters 5–7). Its central question is:

> What does it mean to predict a representation of an unobserved part or future of the world, and under what additional conditions can that prediction become useful for action?

The dossier preserves the mathematical depth of the notes while correcting claims that are stronger than their assumptions or evidence. It distinguishes architectures, training objectives, regularizers, action-conditioned models, planning methods and broader cognitive architectures. These categories are related, but they are not interchangeable.

## Reading order

1. [Chapter design and page budget](chapter4-design.md) — editorial blueprint, chapter spine, page budget, figure/table/listing plan and production gates.
2. [Scope and handoff](00_scope_and_architecture.md) — reader contract, what Chapters 1–3 already establish, what Chapter 4 must add.
3. [The JEPA prediction problem](01_jepa_objective_math.md) — generic architecture, losses, target updates and mathematical limits.
4. [I-JEPA](02_ijepa.md) — image-space context and target-block prediction.
5. [V-JEPA and its descendants](03_vjepa_family.md) — video, V-JEPA 2, action conditioning and V-JEPA 2.1.
6. [Collapse and regularization](04_collapse_regularizers.md) — concise VICReg handoff, then distinctions between moment and distributional constraints.
7. [LeJEPA and SIGReg](05_lejepa_sigreg.md) — risk framework, Gaussian target, Cramér–Wold sketching and characteristic functions.
8. [VISReg](06_visreg.md) — sliced-Wasserstein sketching and separated scale/shape control.
9. [Modalities and variational branches](07_adjacent_modal_variational.md) — A-JEPA, Audio-JEPA, VL-JEPA, Var-JEPA and VJEPA.
10. [Action-conditioned world models](08_action_conditioned_and_planning.md) — V-JEPA 2-AC, EB-JEPA, JEPA-WMs, planning and evaluation.
11. [Path to autonomy](09_path_autonomous_intelligence.md) — LeCun's proposed cognitive architecture and how JEPA fits within it.
12. [Proofs and scientific-claims audit](10_claims_audit_and_proofs.md) — what can be derived, what must be attributed, and what must not be claimed.
13. [Visual and teaching plan](11_figures_tables_exercises.md) — figure briefs, tables, exercises and learning checks.
14. [Official code and reproducibility](12_official_implementations.md) — source-of-truth repositories and the few short code excerpts worth considering.
15. [References](13_references.md) — primary papers, official implementations and supplementary reading.
16. [Book-level assessment](14_book_assessment.md) — provisional editorial assessment and evidence limitations.

## Source hierarchy

Use the following evidence order when writing reader-facing prose:

1. The paper version that introduced the method (and any clearly identified later revision).
2. The paper's official implementation and configuration files for algorithmic details omitted or compressed in the paper.
3. An official project page or author-maintained repository for model availability and practical usage.
4. Later papers for criticism, ablations, extensions or replication findings.
5. The author's JEPA notes for derivations, intuitions, pedagogical examples and questions to investigate.

The personal notes are not themselves evidence for paper-specific results. Where the notes conflict with a primary source or make an unqualified theoretical claim, the claim-audit file takes priority.

## Editorial conventions

- Use the book's current Manning/AsciiDoc conventions. Original schematics only; never paste a paper figure into the manuscript.
- Keep reader-facing prose free of raw URLs. Use author–year citations in prose and one consolidated references section, following Chapter 3.
- Code is PyTorch only, short, and quoted or adapted only from an official implementation after checking its license. The point is to explain one algorithmic operation, not to create a parallel implementation.
- Keep figure captions explanatory but concise (Chapter 3's established budget is roughly four printed lines / 340 characters). Figure numbers, titles and captions belong to the book layout, not inside artwork.
- No em dashes. Use the restrained visual system already used in Chapter 3: white background, grayscale-safe dark text and linework, restrained #3B6FB6 accent.
- State clearly when a result is paper-reported, reproduced by the book's companion experiments, or merely a proposed extension.

## Important scope decision

Chapter 3 already teaches VICReg's invariance, variance and covariance terms, the distinction between decorrelation and task usefulness, latent bottlenecks, and predictive sufficiency. Chapter 4 should not teach those foundations a second time. It should give a short VICReg recap solely to motivate why LeJEPA/SIGReg and VISReg ask a different distributional-regularization question.

Likewise, Chapter 4 should introduce action-conditioned JEPA models and one planning example, but leave general latent dynamics, model-predictive control, Dreamer-style imagination, rollout stability and planning algorithms to Chapters 6 and 7. It should end by making the representation-to-world-model transition concrete, not by exhausting the later chapters.
