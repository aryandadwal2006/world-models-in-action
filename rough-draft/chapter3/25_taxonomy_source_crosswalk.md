# Taxonomy crosswalk: synthesizing the overview, beginner guide, Cookbook and review

This note records how the four sources organize SSL and why the dossier chooses an objective-centric taxonomy for the book. It avoids pretending that one source's categories are the only correct classification.

## 1. The sources have different purposes

| Source | Intended purpose | Useful strength | Main limitation for this chapter |
|---|---|---|---|
| Wikipedia: “Self-supervised learning” | Living overview and entry to linked papers | Broad terminology and paper discovery; mentions autoassociative, contrastive, non-contrastive and joint-embedding predictive lines | A concise, editable overview cannot establish derivations or be the final authority for claims |
| Towards AI: “A Beginner's Guide to Self-Supervised Learning” (25 Dec 2025) | Beginner-friendly explanation | Four easy-to-follow buckets: contrastive, non-contrastive, masked prediction and cluster-based SSL | Compresses overlaps and technical distinctions; not a proof or complete current literature survey |
| Balestriero et al., “A Cookbook of Self-Supervised Learning” (2023) | Technical map and training guide | Explains method lineages and training knobs, including augmentations, projectors, batch size, EMA, predictor networks, dimensional collapse, evaluation and compute | Extensive and technically rich, but predates 2025–2026 developments such as DINOv3, V-JEPA 2.1, LeJEPA and VISReg |
| Gui et al., “A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends” (TPAMI 2024) | Broad peer-reviewed taxonomy, applications and open questions | Broad algorithm/application scope and historical synthesis across computer vision and NLP | Earlier preprint than current frontier; its categories should be extended with newer primary papers |

This is why the rough draft contains both a source-by-source reading record and method-first technical notes. The former preserves what the source actually says; the latter develops the equations and architectural logic in original prose.

## 2. Crosswalk of taxonomies

The source labels do not align one-to-one. The mapping below is a teaching crosswalk, not an assertion that the taxonomy terms are formally equivalent.

| Actual training target / mechanism | Beginner guide bucket | Cookbook family/sections | Broad survey / overview connection | Representative methods |
|---|---|---|---|---|
| Synthetic target from a transformation or withheld region | Masked prediction or pretext/task bucket | Early pretext tasks, masked image modeling | Autoassociative / pretext tasks | Rotation, jigsaw, colorization, denoising, inpainting |
| Candidate identification among paired/negative samples | Contrastive | Deep metric learning; contrastive family | Contrastive learning | SimCLR, MoCo, CPC, CLIP, SigLIP |
| Predict a target branch's vector or distribution | Non-contrastive | Self-distillation | Non-contrastive / teacher-student | BYOL, SimSiam, DINO, DINOv2, DINOv3 |
| Match paired features while regularizing batch statistics | Non-contrastive | CCA-related / redundancy-reduction family | Non-contrastive; feature-statistic methods | Barlow Twins, VICReg, W-MSE |
| Predict assignments to pseudo-categories/prototypes | Cluster-based | CCA/prototype family and online clustering | Clustering / pseudo-labeling | DeepCluster, SwAV, MSN; DINO overlaps |
| Predict missing pixels, code IDs, words, replacements or teacher features | Masked prediction | Masked image modeling and NLP-related objectives | Autoassociative / masked objectives | MAE, SimMIM, BEiT, MaskFeat, iBOT, BERT, ELECTRA, data2vec |
| Align naturally paired modalities | Often placed under contrastive | Beyond images/classification; multimodal SSL | Multimodal applications | CLIP, ALIGN, SigLIP / SigLIP 2 |
| Predict target-region or future features in a joint latent space | Not fully covered by a four-bucket beginner taxonomy | Predictive/latent methods and beyond-image sections | Joint-embedding predictive architectures | I-JEPA, V-JEPA, V-JEPA 2 / 2.1, LeJEPA |

The critical distinction is that **training input**, **target type**, **loss geometry** and **collapse mechanism** are separate axes. A masked method can use a teacher target; a contrastive method can be cross-modal; prototype assignments can be balanced online; and a predictive representation can be trained without negative pairs. The categories therefore overlap by design.

## 3. Why the Cookbook must affect the chapter's explanatory spine

The Cookbook is valuable not merely because it lists more methods, but because it shows that recipe variables alter the effective algorithm:
- augmentation determines which transformations the objective rewards the encoder for ignoring;
- projection heads separate the space optimized by the loss from the space transferred downstream;
- candidate count, batch construction and queues alter the sampled discrimination problem;
- teacher EMA, predictors, normalization and stop-gradient alter update dynamics;
- data curation and class imbalance alter the learned representation distribution;
- training budget and evaluation protocol change the meaning of comparisons;
- linear probes, nearest-neighbor probes, fine-tuning, visualizations and dense tasks answer different questions.

These observations lead to a general rule: method descriptions must distinguish **objective**, **architecture**, **training recipe** and **evaluation**. An improvement to the full recipe is not automatically an improvement to the mathematical loss alone.

## 4. What the broad survey adds

Gui et al. provide a cross-domain view of algorithmic motivations and representative applications, and frame open questions. The survey is an important check against making the book entirely ImageNet-centric. However, the final draft must not imply that the review covers the latest methods released after it was written. DINOv3, V-JEPA 2.1, LeJEPA and VISReg are included because their primary papers appeared later and were checked directly.

The survey's broad coverage should lead to questions in the dossier:
- what signals are available in each domain (text order, audio time, image views, video space-time, paired captions);
- how surrogate targets are built in each domain;
- how data collection and target quality differ;
- how performance should be measured beyond recognition;
- which open questions are still open after the cited paper's publication date.

## 5. Reading discipline: source claim versus synthesis

Mark each claim as one of:
1. **Primary-paper statement:** method definition, theorem, experiment or ablation explicitly given by the source.
2. **Derived result:** algebra independently derived from a stated definition.
3. **Cross-paper synthesis:** comparison or interpretation created by the dossier across multiple papers.
4. **Current fact:** release/date/license/status verified from the current official paper or repository.
5. **Editorial recommendation:** how much space the book should give the method or where it belongs in the TOC.

For example, the claim “the standard InfoNCE loss gradient with respect to a candidate logit is softmax probability minus the target indicator” is a mathematical derivation from cross-entropy. The claim “a low InfoNCE loss proves a good world-model state” is not supported and should be rejected. The claim “LeJEPA reports stable training without an EMA target” is a report of that paper's recipe, not a theorem that EMA is always unnecessary. The claim “VISReg improves OOD performance” must be attributed to its tested evaluations and not generalized to every dataset or task.

## 6. Canonical source links

- Wikipedia overview: https://en.wikipedia.org/wiki/Self-supervised_learning
- Towards AI beginner guide (published 25 December 2025): https://pub.towardsai.net/a-beginners-guide-to-self-supervised-learning-27941f2bb4a9
- Balestriero et al., *A Cookbook of Self-Supervised Learning* (2023): https://arxiv.org/abs/2304.12210
- Gui et al., *A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends* (TPAMI 2024): https://doi.org/10.1109/TPAMI.2024.3415112; preprint https://arxiv.org/abs/2301.05712
