# Dossier readiness and manuscript-start gate

**State at 2026-10-26: research dossier not yet ready to freeze for manuscript synthesis.** The purpose of this file is to make the decision reproducible rather than treating the existence of many notes as evidence that the research phase is complete.

The unbounded dossier should remain exhaustive and navigable. The reader-facing Chapter 3 should be selective. "Ready to move to manuscript" means the source-backed dossier can support a coherent outline and a first composed draft; it does not mean every method in the field has to be derived in the printed chapter.

## 1. Readiness gates

| Gate | Completion evidence | Current state | Remaining work |
|---|---|---|---|
| Foundations | Notation, probability, entropy/KL, similarity, covariance/rank, EMA, stop-gradient and conditional sufficiency | Substantially covered in notes 01, 19, 21, 22 and 24 | Final equation/notation consistency sweep |
| Core visual families | Contrastive, momentum dictionary, negative-free bootstrap, feature statistics, prototypes, masked pixel/token/teacher targets | Broad coverage in notes 14, 15, 16, 19, 21, 27, 31, 32 and 35 | Verify a paper-specific recipe and result caveat for each method selected for prose |
| Temporal/predictive family | CPC, latent prediction, targets/contexts, action conditioning, rollout versus retrieval | CPC and JEPA-family derivations now exist in notes 14, 30, 33 and 37 | Audit CPC exact source/variant; distinguish all retrieval metrics from rollout claims throughout notes |
| Cross-domain coverage | Language, speech, image-text and dense/local vision | Method notes 28, 34 and 36 plus code-source audit | Keep incomplete upstream training loops clearly marked; verify references and objective claims |
| Modern frontier | DINOv2/v3, LeJEPA/SIGReg/VISReg, V-JEPA 2.1 and EB-JEPA | Dedicated notes 20, 32 and 33 | Final paper/version/license/update audit; distinguish each paper's claim from cross-paper synthesis |
| Generative boundary | EBM/NCE, DDPM, score-SDE and flow-matching objective differences | Added in note 37, with pinned source references on main-extra | Decide precise depth in Chapter 3 versus the later generation chapter; avoid duplicate derivations |
| Upstream provenance | Gitlink, .gitmodules entry, immutable source commit and license caveat | main-extra has 45 submodule pointers; every pointer has a URL entry; 5 new CPC/DIM/diffusion pins resolved to non-empty upstream trees | Run final integrity check on all 45, review permissions/licenses for any redistribution, do not claim execution/reproduction |
| Citation integrity | Every equation, paper/version, current claim and arXiv/source link corresponds to the claim made | Partially audited; one V-JEPA citation and one DINO warm-up default discrepancy were corrected | Complete line-by-line citation pass across the dossier and correct every stale/ambiguous author, title, year or identifier |
| Internal consistency | Method names, equation conventions, branch names, links, note numbering and source status agree | Cross-references updated for new notes; known stale CPC implementation claim corrected | Search the full dossier for duplicate/outdated statements; verify all relative links, tables and math notation |
| Editorial handoff | One spine, chapter boundary, page-length intent, original figure plan, one consistent notation and explicit exercise goals | Candidate flow exists in note 10 and figure plan in note 11 | Finalize outline only after source and consistency audits; do not simply paste method notes into composed chapter |

## 2. Current covered terrain

The dossier now separates these broad targets and objective families:

- **Contrastive candidate identification:** instance discrimination, SimCLR/NT-Xent, MoCo queues and momentum keys, CPC future-latent targets, historical NPID/PIRL/CMC/AMDIM, NNCLR and debiased-negative context.
- **Negative-free / teacher targets:** BYOL, SimSiam, DINO, iBOT, DINOv2/v3, KoLeo and selected Gram-anchoring recipes.
- **Moment and distribution constraints:** Barlow Twins, VICReg, W-MSE, LeJEPA/SIGReg and VISReg.
- **Clustering/prototypes:** DeepCluster, SwAV, DINO-family distributions, MeanShift/MSN and local clustering.
- **Masked objectives:** MAE, SimMIM, BEiT-related targets, MaskFeat, iBOT, BERT, GPT, ELECTRA, T5, wav2vec 2.0, HuBERT, WavLM and data2vec.
- **Cross-modal/local features:** CLIP, ALIGN, SigLIP, DenseCL, PixPro, VICRegL, Leopart and SelfPatch.
- **Predictive world-model relation:** I-JEPA, V-JEPA, V-JEPA 2/2-AC/2.1 and EB-JEPA, with action-conditioned prediction and planner evaluation separated from passive video SSL.
- **Generative boundary:** normalized energy models, NCE, Deep InfoMax critic objectives, DDPM noise prediction, score-based SDEs and flow matching.

This is a coverage map, not a claim that all items have identical source-verification depth or belong in the printed chapter.

## 3. Manuscript-start criteria

Start the reader-facing manuscript only when all of the following are true:

1. Each **core** family selected for full derivation has primary paper/version, objective, reduction/normalization, gradient/update path, degenerate case, recipe settings, evidence/limitations, and official-code status.
2. Frontier and supplementary methods have a clearly labelled evidence level: paper-only, source-inspected, or executed in this project. These levels are never substituted for each other.
3. Citation and internal-link audits find no known broken link, wrong paper identifier, stale implementation-availability claim or contradicting default/config statement.
4. The outline makes a decision for each method: core derivation, comparison table, short sidebar, frontier handoff, or later-chapter boundary.
5. The composed chapter can teach the argument in one coherent route without duplicating Chapter 2 or pre-empting Chapter 4's full JEPA treatment.
6. The Manning template and styles are applied to a new manuscript revision, not retrofitted by flattening this unbounded notebook.

## 4. What is not required before the manuscript starts

- Exhaustively deriving every paper in the taxonomy in the printed chapter.
- Running all 45 upstream repositories. They are original references, not one common environment; many require old or large compute/data setups.
- Reproducing each paper's headline benchmark in the book companion. If the chapter explains a published result, accurately attribute it and do not imply the companion reran it.
- Forcing diffusion and flow matching into the SSL section; a clear boundary explanation is enough if the later generation chapter owns their full derivations.

## 5. Immediate next actions

1. Finish source/claim cleanup in method cards and old summaries, especially historical methods whose upstream implementation is now available.
2. Audit every primary paper linked from the method coverage matrix and bibliography for title, author, date, identifier and scope.
3. Audit relative links from README and across the 37-plus note files, plus equations/tables for notation drift.
4. Revisit the storyboard and outline against the Manning TOC and style guide.
5. Only then mark the dossier ready for manuscript synthesis and begin the composed Chapter 3 draft.

No manuscript-ready status is claimed in this commit.
