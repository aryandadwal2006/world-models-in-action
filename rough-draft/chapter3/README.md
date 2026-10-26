# Chapter 3 theory dossier

## Purpose

This is the unbounded research draft behind the reader-facing Chapter 3, *Representation learning before generation*. It is deliberately much more detailed than the finished chapter should be. Each method note should answer five questions: what problem was visible in the preceding method; what exact training signal is introduced; why the loss has that form; what assumptions or failure modes follow; and what later methods changed.

The dossier is an original synthesis of the cited papers, not a replacement for reading them. Paper PDFs remain at their canonical publisher/arXiv URLs. We do not reproduce complete copyrighted papers inside this repository. Each note reconstructs equations and derivations in new prose, distinguishes paper claims from our deductions, and records implementation details to verify against the original method section.

## Narrative thesis

A self-supervised objective defines a relationship in data and rewards an encoder for making that relationship easy to predict. The relationship may be agreement between transformed views, discrimination among candidates, agreement with a slowly moving teacher, decorrelation of dimensions, assignment to prototypes, or prediction of missing content. These are not cosmetic variations of one objective. They impose different constraints, use different sources of asymmetry, and fail in different ways.

The chapter should progress by the question each method family answers:

1. **What should count as the same example?** Data transformations create positive pairs, but invariance is a design decision.
2. **How can agreement be made nontrivial?** Contrastive objectives compare a positive pair against a set of candidates; InfoNCE is the one loss derived fully in the chapter.
3. **Can a model learn without explicit negatives?** BYOL, SimSiam, and DINO replace instance discrimination with prediction or distribution matching, but must manage collapse.
4. **Can collapse be constrained directly?** Barlow Twins and VICReg use second-order statistics and variance floors rather than a negative dictionary.
5. **Can latent space be organized with prototypes?** DeepCluster and SwAV turn features into pseudo-categories, with different update schedules.
6. **What if the target is missing content rather than another view?** BERT, MAE, BEiT, iBOT, ELECTRA, and data2vec make the target-space choice explicit.
7. **What changes across modalities?** CLIP/SigLIP align image and text; wav2vec 2.0 and HuBERT build speech targets from quantization or clustering.
8. **What is current and what belongs elsewhere?** DINOv3 demonstrates scaled dense visual self-distillation; the JEPA line is introduced as the next architectural step, with its full treatment reserved for Chapter 4.

## Scope rules for the actual chapter

- Deep derivation: InfoNCE/NT-Xent, gradient, negative sampling, and the conditional mutual-information-bound caveat.
- Conceptual but mathematically precise treatments: MoCo, BYOL, SimSiam, DINO, Barlow Twins, VICReg, DeepCluster, SwAV, MAE, BEiT/iBOT, CLIP/SigLIP.
- Compact cross-domain vignettes: BERT versus GPT, ELECTRA, wav2vec 2.0, HuBERT, data2vec.
- Historical note, not a major section: rotation/jigsaw/colorization and CPC. CPC is important to the lineage but does not merit a centerpiece in a contemporary overview.
- Frontier note, not duplicate teaching: I-JEPA, V-JEPA, V-JEPA 2 and DINOv3. Chapter 4 owns JEPA architecture and action-conditioned prediction; the current chapter should only make the handoff intelligible.
- No universal leaderboard. Methods differ in architecture, data, augmentations, compute, target type, training horizon, evaluation and licensing. Comparisons across papers are not controlled head-to-head experiments.
- Only two lightweight demonstrations: derive and inspect a tiny InfoNCE batch; then compare what happens to a constant embedding under invariance-only, Barlow Twins and VICReg-style terms. These are explanatory checks, not claims about ImageNet-scale performance.

## Method inventory

| Family | Core methods | Learning target | Central tension |
|---|---|---|---|
| Hand-designed pretext tasks | rotation, jigsaw, context/colorization | transformation label or withheld image content | The task may reward shortcuts instead of useful invariance |
| Contrastive instance discrimination | CPC, SimCLR, MoCo v1–v3, CLIP | identify a matched view/candidate | Negative-set quality, false negatives, batch/queue design |
| Bootstrap / teacher-student | BYOL, SimSiam, DINO, DINOv2, DINOv3 | another branch's features or distribution | Collapse control without explicit negatives |
| Redundancy reduction | Barlow Twins, VICReg | cross-view invariance plus feature-statistic constraints | Statistical constraints are batch estimates, not semantic guarantees |
| Prototype/clustering | DeepCluster, SwAV; DINO prototypes overlap | cluster labels or assignments | Assignment imbalance, cluster quality, alternating optimization |
| Masked input prediction | BERT, MAE, BEiT, iBOT, ELECTRA, data2vec | missing token, pixel, code or contextual target | Target granularity shapes what must be represented |
| Cross-modal alignment | CLIP, ALIGN, SigLIP | paired text-image embeddings | Pair quality, batch negatives, alignment versus fine-grained grounding |
| Speech SSL | wav2vec 2.0, HuBERT, data2vec | quantized speech units, cluster pseudo-labels, contextual features | Latent sequence resolution and how units are constructed |
| Joint-embedding prediction | I-JEPA, V-JEPA, V-JEPA 2 | target-region or future latent | Predictive feature quality is not automatically state sufficiency or action-conditioned dynamics |

## Review checklist for every note

- State the paper version and modality.
- Define encoder(s), predictor/projector, teacher/target update, and stop-gradient locations.
- Write the exact scalar loss and reduction axes.
- Separate a stationary collapsed solution from a stable training outcome.
- Explain what information the objective is encouraged to retain and what it leaves unconstrained.
- State the paper's own ablations separately from follow-up interpretations.
- Link the official implementation and check its current license.
- Note whether a method is historical, widely used, or currently frontier; do not infer adoption from a paper's benchmark result alone.
