# Source study notes: what each source contributes to Chapter 3

**Scope date:** 26 October 2026. This is a reading record and synthesis map, not a reproduction of any source. Method-specific equations and claims should be checked against the original paper before the production manuscript is frozen.

## 1. Source hierarchy

The reading packet contains different kinds of evidence. They should not be treated as interchangeable.

1. **Primary method papers and official implementation repositories** establish a method's actual architecture, objective, notation, training recipe and stated empirical results. A survey's summary is not enough to settle an equation or code detail.
2. **Balestriero et al., *A Cookbook of Self-Supervised Learning* (2023)** is the key architecture-and-training synthesis. It explicitly connects method families, examines mathematical relationships, and devotes substantial space to design choices such as augmentation, projection heads, batch size, target EMA, predictor networks, data distributions, evaluation and computational cost. It is more useful for the rough-draft folder than a list of method names because it explains why recipe details matter.
3. **Gui et al., *A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends* (first arXiv 2023; TPAMI publication 2024)** provides broad taxonomy, application coverage, trend framing and open research questions. It should be used as the broad coverage check and historical survey, while newer 2025–2026 methods must be added from their own papers.
4. **Wikipedia's current SSL article** is a convenient map of headings and linked references. It organizes the topic into pseudo-labels and types including autoassociative, contrastive, non-contrastive, and joint-embedding/predictive architectures. It is not a primary scientific source and should not establish a technical proof or frontier claim.
5. **Towards AI, “A Beginner's Guide to Self-Supervised Learning” (25 December 2025)** is useful as a reader-facing first-pass explanation. It presents four practical buckets: contrastive, non-contrastive, masked prediction and cluster-based SSL. Use it to test whether the narrative is understandable, not as the authority for equations, novelty, contemporary adoption or theoretical guarantees.

The chapter needs to combine these perspectives without copying their taxonomy wholesale. A good teaching taxonomy is allowed to be simpler than the field, but it must say what it omits and allow a method to belong to more than one family.

## 2. What the user's three sources support

### 2.1 Wikipedia: the broad map

URL: https://en.wikipedia.org/wiki/Self-supervised_learning

Useful contributions:
- Defines the broad learning setup as constructing supervisory signals from the input data rather than external labels.
- Links pseudo-labeling, autoassociative/reconstructive objectives, contrastive methods, non-contrastive methods and joint-embedding predictive architectures.
- Offers a readable entry to historical and canonical papers and makes the overlap among current method families visible.
- Can serve as a test that the chapter does not omit the broad joint-embedding/predictive line after covering contrastive and masked methods.

Limitations to preserve:
- A taxonomy page is an orientation, not evidence that its buckets are mutually exclusive or exhaustive.
- Its short claims about collapse control compress a complicated literature. The draft must distinguish a loss admitting a collapsed solution, the local stability of that solution under a specified dynamics, and whether a complete implementation collapses in practice.
- “The model generates its own labels” is a useful approximation, but not every objective literally produces discrete pseudo-labels. Some use continuous targets, cross-view statistics, reconstruction targets or future feature prediction.
- A current article can change. Record the date accessed and verify key method statements against primary references.

### 2.2 Towards AI: a beginner's organization

URL: https://pub.towardsai.net/a-beginners-guide-to-self-supervised-learning-27941f2bb4a9

Useful contributions:
- The four-bucket organization is accessible: contrastive learning, non-contrastive learning, masked prediction and cluster-based SSL.
- It introduces SimCLR and MoCo as accessible contrastive examples and describes positive/negative pairs and batch candidates in plain language.
- Its high-level distinction between creating training targets from unlabeled data and using a mixture of human-labeled and unlabeled data gives a clean first-pass comparison with semi-supervised learning.
- It asks the important practical question of when SSL is useful and when its added complexity may not be justified.

Cautions:
- Four buckets are a pedagogical choice, not a mathematical partition. DINO overlaps teacher-student distillation and prototypes; iBOT combines masked prediction, teacher targets and prototype outputs; wav2vec 2.0 combines masking, quantization and contrastive discrimination.
- The post's “pull positives, push negatives” explanation is intuitive but must be followed by the actual softmax objective and gradient. A negative receives a probability-weighted gradient, not a fixed repulsive force.
- A paired caption is itself a training signal; vision-language methods should not be described as having no supervisory information in every ordinary meaning of the word “supervision.” They avoid conventional human class labels, but exploit naturally occurring pairings or weak text supervision.
- Do not use a beginner blog to validate architecture-specific details, modern benchmark rankings, or claims of universal collapse prevention.

### 2.3 *A Cookbook of Self-Supervised Learning*: the method-and-recipe backbone

URL: https://arxiv.org/html/2304.12210
Canonical citation: Randall Balestriero et al. (2023), arXiv:2304.12210.

This paper is unusually useful because its organization moves from families to the knobs that alter their behavior. Its table of contents explicitly includes:
- the deep metric learning family: SimCLR, NNCLR, MeanSIFT/MeanShift and related contrastive formulations;
- self-distillation: BYOL, SimSiam and DINO;
- the CCA/statistics family: VICReg, Barlow Twins, SwAV and W-MSE;
- masked image modeling;
- attempted theoretical unification, contrastive-loss relationships, hard negatives, projection heads and dimensional collapse;
- pretraining-data quality and balance;
- augmentation, multi-crop, target EMA, predictor networks, batch size, learning-rate schedule, weight decay and ViT-specific recipe details;
- evaluation by k-NN, linear probes, MLP probes, fine-tuning, label-free checks and visualizations;
- training speed, distributed computation, domains beyond natural images, multimodal training and localized/dense features.

The central editorial implication is important: the method is not only its named loss. It is a coupled system of target construction, architecture, view generation, normalization, optimizer, batch and data distribution. The chapter must teach this system logic without becoming a training manual that overwhelms the conceptual progression.

Specific lessons to carry forward:
1. **InfoNCE is not explained completely by a mutual-information slogan.** The paper reviews several explanations, including dependence measures, alignment/uniformity and latent-variable identification under particular assumptions. The chapter should derive the classification objective first and make the assumptions of the information-bound interpretation explicit.
2. **Contrastive objectives weight hard negatives through the softmax.** This does not mean every training setup needs an explicit hard-negative miner; batch composition determines which candidates are available.
3. **Projectors matter.** The vector used by the training objective may be produced by a nonlinear head \(g(f(x))\), while the downstream feature is \(f(x)\). It is a mistake to report only the objective-space statistics and assume the backbone has the same geometry.
4. **Teacher and predictor are separate mechanisms.** EMA smooths target changes; a predictor changes the mapping the online branch must learn. An EMA target is not itself a universal theorem of non-collapse.
5. **Data distribution matters.** Some common objectives or balanced assignment schemes impose implicit assumptions about how features or prototype assignments should be distributed. Those assumptions can be wrong on long-tailed or naturally imbalanced data.
6. **Dimensional collapse is distinct from all-input constant collapse.** A feature can vary across examples while occupying a narrow subspace. Track rank/effective rank, not only per-dimension variance or total loss.
7. **Augmentation is task-dependent.** A transformation useful for ImageNet object classification can remove information relevant to color, texture, geometry, localization or control.
8. **Evaluation is not one number.** Linear probes, k-NN, fine-tuning, visual inspection, label-free criteria, spatial tasks and sequential prediction reveal different properties.

### 2.4 Gui et al. review: breadth and open questions

URLs:
- https://arxiv.org/abs/2301.05712
- https://pubmed.ncbi.nlm.nih.gov/38885108/
- DOI: https://doi.org/10.1109/TPAMI.2024.3415112
- Author resource collection: https://github.com/guijiejie/SSL

Bibliographic detail: the first arXiv version appeared in January 2023; the paper appeared in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, volume 46, issue 12, December 2024, pages 9052–9071, DOI 10.1109/TPAMI.2024.3415112. Use the TPAMI year in the formal bibliography if following final journal publication dates, while noting the earlier preprint only where relevant.

Its abstract says the review addresses algorithmic aspects, representative application domains (including image processing/computer vision and NLP), three research trends and open research questions. Its most useful role in this project is broad coverage and historical framing rather than providing the latest frontier. A 2023 preprint cannot be treated as a review of DINOv3 (2025), V-JEPA 2 (2025), or V-JEPA 2.1 (2026); those require their own current primary sources.

### 2.5 Primary sources required to complete claims

The raw-source reading should branch out into primary papers for at least:
- SimCLR, MoCo v1/v2/v3, NNCLR and CPC for the contrastive lineage.
- BYOL, SimSiam, DINO, DINOv2, DINOv3 and iBOT for bootstrap/distillation.
- Barlow Twins, VICReg, W-MSE, SwAV, DeepCluster, MSN and related prototype/statistical approaches.
- MAE, BEiT/BEiT v2, SimMIM, MaskFeat, iBOT and data2vec for masked target variation.
- BERT, GPT, ELECTRA and T5 for language objectives.
- CLIP, ALIGN and SigLIP/SigLIP 2 for multimodal contrastive objectives.
- wav2vec 2.0, HuBERT, WavLM and data2vec for speech representation learning.
- I-JEPA, V-JEPA, V-JEPA 2 and V-JEPA 2.1 for latent prediction and the current video frontier.

A repository README is implementation evidence, not a replacement for the paper. Record the official repository, revision, license, and whether the implementation is actually runnable under the current dependencies.

## 3. Synthesis rules adopted for this dossier

- Introduce SSL as target construction from data, not as one particular architecture.
- Separate self-supervision from pseudo-labeling, weak supervision and semi-supervised learning instead of asserting that all real-world training signals are interchangeable.
- Organize the visual-method flow by unresolved question: invariance → candidate discrimination → target prediction without explicit negatives → explicit statistical anti-collapse constraints → pseudo-labels/prototypes → masked target construction → multimodal and sequence variants → frontier.
- Treat categories as overlapping “dominant mechanisms.” This is more accurate than forcing each algorithm into one exclusive bin.
- Keep the complete derivations in the dossier and selectively compress them for the reader-facing chapter.
- Do not reproduce a complete copyrighted article or paper inside this repository. Keep bibliographic records, canonical links, implementation links and original explanatory notes. The goal is that a learner can understand the method without needing an unstated intermediary, not that the repository republishes other authors' full texts.
- Mark author claims, independently derived equations, and editorial interpretation separately whenever they could be confused.
- Avoid absolute adoption rankings unless supported by a transparent, dated source. “Foundational,” “widely used,” “modern large-scale example,” and “frontier example” mean different things and should not be treated as synonyms.

## 4. Source ledger

| Source | Type | Use in dossier | Do not use it for |
|---|---|---|---|
| Wikipedia SSL page | Living overview | Entry taxonomy and paper discovery | Final theoretical authority |
| Towards AI beginner article, 2025-12-25 | Beginner overview | Check accessibility and simple family map | Equations, proofs, state-of-the-art adoption |
| Balestriero et al., *A Cookbook of Self-Supervised Learning* (2023) | Technical synthesis/tutorial | Design knobs, method families, evaluation, dimensional collapse, cross-family theory | Treating all claims as current in 2026 |
| Gui et al., TPAMI 2024 review | Broad peer-reviewed survey | Taxonomy, applications, trends, open questions | Coverage of post-2024 frontier methods |
| Primary method papers | Original research | Exact losses, stated assumptions, experiments | Generalizing beyond stated conditions |
| Official code repositories | Source implementations | Details omitted from main text, code path, license, revision | Assuming reproducibility, permissive license or no dependencies |
