# Chapter 3 theory dossier

## Purpose and source policy

This is the unbounded research draft behind the reader-facing Chapter 3, *Representation learning before generation*. It is intentionally much more detailed than the finished chapter and records conceptual lineage, equations, derivations, collapse arguments, method-specific decisions, frontier updates and checks needed for a source-grounded final chapter.

The dossier is an original synthesis, not a replacement for reading canonical papers. It does not reproduce complete copyrighted papers. It provides canonical links, source-by-source reading notes, original derivations, implementation pointers and a record of what each source does and does not support. Exact method claims must be checked against the cited primary paper and official implementation before final typesetting.

## Recommended reading order

1. [Taxonomy crosswalk](25_taxonomy_source_crosswalk.md): how the beginner guide, Wikipedia, Cookbook and TPAMI review overlap and differ.
2. [Source study notes](13_source_study_notes.md): the supplied sources, SSL review and Cookbook, with source hierarchy and limitations.
3. [Landscape and scope](00_landscape_and_scope.md): broad method families and chapter boundaries.
4. [First principles](01_first_principles.md): representation contracts, invariance, target construction, collapse and evaluation.
5. [Identifiability, invariance and sufficient state](22_identifiability_invariance_and_sufficient_state.md): why a self-supervised proxy loss does not certify a world-model state.
6. [Contrastive family: full derivations](14_contrastive_family_full_derivations.md): candidate classification, NT-Xent, InfoNCE, MoCo, CPC and neighboring methods.
7. [Bootstrap, statistics and prototypes](15_bootstrap_statistics_and_prototypes_full.md): BYOL, SimSiam, DINO, Barlow Twins, VICReg, W-MSE, DeepCluster and SwAV.
8. [Masked, language and predictive methods](16_masked_language_and_predictive_methods_full.md): MAE, SimMIM, BEiT, MaskFeat, iBOT, BERT, GPT, ELECTRA, speech methods and JEPAs.
9. [Cross-modal and speech methods](07_cross_modal_speech_and_language.md): short explanations of image-text and speech SSL.
10. [Training design and evaluation](17_training_design_and_evaluation.md): augmentation, projector, batch-size effects, EMA, data balance, collapse diagnostics, transfer tests and world-model-specific evaluation.
11. [Method atlas](18_method_atlas_and_family_relations.md): additional variants, CLIP/SigLIP equations, dense-feature families and modern lineages.
12. [Mathematical foundations](24_mathematical_foundations.md): probability, cross-entropy/KL, covariance, rank, characteristic functions, Wasserstein distance, EMA and conditional sufficiency.
13. [Theoretical unification](26_theoretical_unification_alignment_uniformity.md): alignment/uniformity, density ratios, CCA/statistics, reconstruction and latent prediction without false equivalence claims.
14. [Proof notebook](09_proof_notebook.md): compact derivations and explicit limits of each proof.
15. [Theory notebook](19_theory_proofs_and_counterexamples.md): detailed InfoNCE lower-bound derivation, counterexamples, EMA, moment constraints and representation limits.
16. [Loss-gradient derivations](21_loss_gradient_derivations.md): gradients of InfoNCE, CLIP, BYOL/SimSiam, DINO, Barlow Twins, VICReg, SwAV, masked losses, SIGReg and VISReg.
17. [LeJEPA/SIGReg/VISReg](20_lejepa_sigreg_visreg.md): first-principles account of Gaussian embedding priors, characteristic-function tests and sliced-Wasserstein regularization.
18. [Method coverage matrix](23_method_coverage_matrix.md): primary-paper map across major families, coverage status and implementation checks.
19. [Primary-paper recipe audit](27_primary_paper_recipe_audit_and_replication_risks.md): method-specific objectives and replication pitfalls, with clear separation between standard equations and implementation details still to verify.
20. [Dense/local, speech and multimodal method cards](28_dense_local_speech_and_multimodal_method_cards.md): DenseCL, PixPro, VICRegL, Leopart, SelfPatch, wav2vec 2.0, HuBERT, WavLM, data2vec, CLIP, ALIGN and SigLIP with equations, target assumptions and official-source pointers.
21. [Historical contrastive lineages](30_historical_contrastive_lineage_and_derivations.md): first-principles derivations and comparisons of NPID, PIRL, CMC and AMDIM.
24. [Upstream code-source audit](29_upstream_code_source_audit.md): exact code paths and limitations for dedicated and shared upstream implementations.
25. [Proposed flow and page budget](10_flow_and_page_budget.md): how to compress the dossier into a coherent reader-facing sequence.
26. [Figure storyboard](11_figure_storyboard.md): conceptual diagrams needed in the final chapter.
27. [Bibliography](12_bibliography.md): canonical citations and source-verification notes.

The original shorter notes remain useful as focused cross-checks: [contrastive InfoNCE](02_contrastive_infoNCE.md), [negative-free bootstrapping](03_negative_free_bootstrap.md), [redundancy reduction](04_redundancy_reduction.md), [clustering and prototypes](05_clustering_and_prototypes.md), [masked prediction](06_masked_prediction.md), and [frontier boundaries](08_frontier_and_boundaries.md).

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

The reader-facing chapter should derive InfoNCE/NT-Xent and its gradient in full; explain SimCLR, MoCo, BYOL, SimSiam, DINO, Barlow Twins, VICReg, DeepCluster, SwAV, MAE and major target-space choices; give compact cross-domain examples from BERT/GPT/ELECTRA, CLIP/SigLIP and speech SSL; keep rotation, jigsaw, context/colorization and CPC as lineage; and hand off full JEPA mechanics to Chapter 4.

Only two small explanatory demonstrations belong in the reader-facing chapter. The long-form derivations and literature map stay here and have no page limit.
