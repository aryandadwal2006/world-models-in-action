# Chapter 3 theory dossier

## Purpose and source policy

This is the unbounded research draft behind the reader-facing Chapter 3, *Representation learning before generation*. It is intentionally much more detailed than the finished chapter and records conceptual lineage, equations, derivations, collapse arguments, method-specific decisions, frontier updates and checks needed for a source-grounded final chapter.

The dossier is an original synthesis, not a replacement for reading canonical papers. It does not reproduce complete copyrighted papers. It provides canonical links, source-by-source reading notes, original derivations, implementation pointers and a record of what each source does and does not support. Exact method claims must be checked against the cited primary paper and official implementation before final typesetting.

## Recommended reading order

1. [Source study notes](13_source_study_notes.md): the three user-supplied sources, the SSL review and the Cookbook, with source hierarchy and limitations.
2. [Landscape and scope](00_landscape_and_scope.md): broad method families and chapter boundaries.
3. [First principles](01_first_principles.md): the representation contract, invariance, target construction, collapse and evaluation.
4. [Contrastive family: full derivations](14_contrastive_family_full_derivations.md): candidate classification, NT-Xent, InfoNCE, MoCo, CPC and neighboring methods.
5. [Bootstrap, statistics and prototypes](15_bootstrap_statistics_and_prototypes_full.md): BYOL, SimSiam, DINO, Barlow Twins, VICReg, W-MSE, DeepCluster and SwAV.
6. [Masked, language and predictive methods](16_masked_language_and_predictive_methods_full.md): MAE, SimMIM, BEiT, MaskFeat, iBOT, BERT, GPT, ELECTRA, speech and JEPAs.
7. [Training design and evaluation](17_training_design_and_evaluation.md): augmentation, projector, batch-size effects, EMA, data balance, collapse diagnostics, transfer tests and world-model-specific evaluation.
8. [Method atlas](18_method_atlas_and_family_relations.md): additional variants, CLIP/SigLIP equations, dense-feature families and modern lineages.
9. [Proof notebook](09_proof_notebook.md): compact derivations and explicit limits of each proof.
10. [Theory notebook](19_theory_proofs_and_counterexamples.md): detailed InfoNCE lower-bound derivation, counterexamples, EMA, moment constraints and representation limits.
11. [LeJEPA/SIGReg/VISReg](20_lejepa_sigreg_visreg.md): first-principles account of Gaussian embedding priors, characteristic-function tests and sliced-Wasserstein regularization.
12. [Loss-gradient derivations](21_loss_gradient_derivations.md): gradients of InfoNCE, CLIP, BYOL/SimSiam, DINO, Barlow Twins, VICReg, SwAV, masked losses, SIGReg and VISReg.
13. [Proposed flow and page budget](10_flow_and_page_budget.md): how to compress the dossier into a coherent reader-facing sequence.
14. [Figure storyboard](11_figure_storyboard.md): conceptual diagrams needed in the final chapter.
15. [Bibliography](12_bibliography.md): canonical citations and source-verification notes.

The original shorter notes remain useful for focused cross-checks: [contrastive InfoNCE](02_contrastive_infoNCE.md), [negative-free bootstrapping](03_negative_free_bootstrap.md), [redundancy reduction](04_redundancy_reduction.md), [clustering and prototypes](05_clustering_and_prototypes.md), [masked prediction](06_masked_prediction.md), [cross-modal and speech](07_cross_modal_speech_and_language.md), [frontier boundaries](08_frontier_and_boundaries.md).

## Narrative thesis

A self-supervised objective constructs a relationship in data and rewards an encoder for making that relationship easy to predict. The relationship may be agreement between transformed views, candidate discrimination, teacher prediction, feature-statistic constraints, prototype assignment, masked-target prediction or latent prediction across space and time. These are not cosmetic variations of one loss: they impose different constraints, use different sources of asymmetry and fail in different ways.

Categories overlap. DINO combines teacher-student distribution matching and prototype outputs; iBOT combines masking and teacher targets; wav2vec 2.0 combines masking, quantization and contrastive discrimination; V-JEPA predicts latent target-region features. The dossier classifies methods along multiple axes rather than forcing them into exclusive categories.

## Scope principle

The dossier must be exhaustive; the printed chapter must be selective. Every detailed method note should answer:
- what problem remained after the preceding family;
- what exact architecture and target were introduced;
- why the loss has its mathematical form;
- what its gradients or update rules do;
- which assumptions are encoded in the data and target;
- which degenerate solutions remain possible;
- what the original paper's evidence supports;
- what later methods changed;
- what evidence matters for a representation used by a world model.

The chapter itself should derive InfoNCE/NT-Xent and its gradient in full; explain SimCLR, MoCo, BYOL, SimSiam, DINO, Barlow Twins, VICReg, DeepCluster, SwAV, MAE and major target-space choices; give compact cross-domain examples from BERT/GPT/ELECTRA, CLIP/SigLIP and speech SSL; keep rotation, jigsaw, context/colorization and CPC as lineage; and hand off full JEPA mechanics to Chapter 4.

Only two small explanatory demonstrations belong in the reader-facing chapter. The long-form derivations and literature map stay here and have no page limit.
