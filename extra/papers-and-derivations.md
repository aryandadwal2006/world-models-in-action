# Paper and derivation index

This index distinguishes the original method paper, the implementation URL, and the concept that deserves derivation in the book. It is not a claim that code is executable in a modern environment. Cite the actual paper version used in the manuscript and inspect the original method section before finalizing technical claims.

| Method | Original paper | Key derivation / issue to reconstruct | Later relationship |
|---|---|---|---|
| SimCLR | Chen et al. 2020, https://arxiv.org/abs/2002.05709 | NT-Xent from a 2N-view minibatch, positive index, denominator, temperature, augmentation-induced invariance, batch/negative dependence | Standard baseline for instance discrimination |
| MoCo | He et al. 2020, https://arxiv.org/abs/1911.05722 | Dictionary queue, momentum encoder, key staleness, why a large queue need not equal a large batch | Separates key dictionary size from current minibatch |
| MoCo v2/v3 | Chen et al. 2020/2021, https://arxiv.org/abs/2003.04297 and https://arxiv.org/abs/2104.02057 | Recipe versus architecture changes; transformer training stability | Must not imply v3 is simply v1 with a queue |
| BYOL | Grill et al. 2020, https://arxiv.org/abs/2006.07733 | Online predictor, EMA target, stop-gradient, cosine regression; what prevents trivial collapse in the practical algorithm and what remains theoretical | Negative-free bootstrap family |
| SimSiam | Chen and He 2021, https://arxiv.org/abs/2011.10566 | Symmetric branches, predictor asymmetry, stop-gradient, collapse probes and batch-normalization ablations | Shows EMA teacher is not necessary in this setup |
| DeepCluster | Caron et al. 2018, https://arxiv.org/abs/1807.05520 | Alternation between feature extraction, k-means pseudo-labels, and classifier training; cluster permutation / empty clusters | Offline clustering |
| SwAV | Caron et al. 2020, https://arxiv.org/abs/2006.09882 | Online prototypes, swapped assignments, Sinkhorn-Knopp balanced transport, entropy/imbalance implications | Online clustering without explicit all-pairs negatives |
| Barlow Twins | Zbontar et al. 2021, https://arxiv.org/abs/2103.03230 | Batch-normalized cross-correlation, diagonal alignment, off-diagonal redundancy penalties; invariance versus decorrelation | Redundancy reduction |
| VICReg | Bardes et al. 2022, https://arxiv.org/abs/2105.04906 | Invariance MSE, per-dimension standard-deviation floor, covariance penalty; why invariance alone collapses | Explicit variance/covariance constraints |
| DINO | Caron et al. 2021, https://arxiv.org/abs/2104.14294 | Teacher/student distributions, centering and sharpening, multi-crop, emergent patch structure (empirical claim) | Self-distillation / prototype predictions |
| DINOv2 | Oquab et al. 2023, https://arxiv.org/abs/2304.07193 | Curated large-scale data, teacher distillation, registers and evaluation protocol | Scaling and data quality |
| DINOv3 | Siméoni et al. 2025, https://arxiv.org/abs/2508.10104 | Gram anchoring, dense-feature stability over long training, scaling recipe; current license | Modern large-scale self-distillation |
| MAE | He et al. 2022, https://arxiv.org/abs/2111.06377 | High mask ratio, visible-token-only encoder, decoder reconstruction loss on masked patches, compute asymmetry | Pixel-level masked modeling |
| BEiT | Bao et al. 2022, https://arxiv.org/abs/2106.08254 | Discrete visual tokenizer / masked visual token classification | Semantic discrete target versus pixels |
| iBOT | Zhou et al. 2022, https://arxiv.org/abs/2111.07832 | Self-distillation on class and patch tokens under masking | Distillation and masked prediction meet |
| BERT | Devlin et al. 2019, https://arxiv.org/abs/1810.04805 | Masked language modeling, bidirectional context, corruption/reconstruction and pretrain/fine-tune split | NLP example of self-supervised target construction |
| GPT-2 | Radford et al. 2019, https://openai.com/index/better-language-models/ | Autoregressive factorization by chain rule, causal context, next-token cross-entropy | Generative objective is also self-supervision |
| ELECTRA | Clark et al. 2020, https://arxiv.org/abs/2003.10555 | Replaced-token detection and generator/discriminator; token efficiency trade-off | Discriminative pretraining target |
| CLIP | Radford et al. 2021, https://arxiv.org/abs/2103.00020 | Symmetric image-text contrastive cross-entropy, batch-level negatives, shared embedding space | Cross-modal semantic alignment |
| SigLIP | Zhai et al. 2023, https://arxiv.org/abs/2303.15343 | Pairwise sigmoid objective, logistic interpretation, removing global softmax normalization | Contrastive alternative, not the same loss as CLIP |
| wav2vec 2.0 | Baevski et al. 2020, https://arxiv.org/abs/2006.11477 | Masked latent speech, quantized targets, contrastive discrimination among quantized candidates | Speech-specific SSL |
| HuBERT | Hsu et al. 2021, https://arxiv.org/abs/2106.07447 | Iterative/offline clustering of audio units and masked prediction of pseudo-labels | Cluster-derived speech targets |
| data2vec | Baevski et al. 2022, https://arxiv.org/abs/2202.03555 | Predict contextualized teacher representations from a masked input; EMA target and averaging layers | Modality-general latent targets |
| I-JEPA | Assran et al. 2023, https://arxiv.org/abs/2301.08243 | Predict target-block embeddings from context-block embeddings without pixel reconstruction or negatives | Deeper treatment belongs to Chapter 4 |
| V-JEPA / V-JEPA 2 | Bardes et al. 2024; Assran et al. 2025, https://arxiv.org/abs/2404.08471 and https://arxiv.org/abs/2506.09985 | Video target prediction; what temporal structure and action conditioning change; distinguish V-JEPA 2-AC from passive pretraining | Brief frontier marker in Chapter 3; full JEPA logic belongs in Chapter 4 |

## Other relevant lineages to track

- CPC (van den Oord et al. 2018; https://arxiv.org/abs/1807.03748): historically important bridge from latent future discrimination to later predictive representation learning; not presented as a current dominant image SSL recipe.
- InfoMax and mutual-information bounds: use critically. The InfoNCE lower-bound derivation has distributional assumptions and a finite-negative ceiling; optimizing the bound is not identical to optimizing downstream utility.
- Contrastive objective lineage: N-pair loss, triplet loss, supervised contrastive learning, metric learning. Explain enough to prevent anachronistically treating NT-Xent as the first contrastive objective.
- Classical self-supervision: context prediction, colorization, rotation prediction, jigsaw puzzles. Treat as historically useful pretext-task examples rather than main modern architectures.
- Predictive coding and latent prediction are related but not interchangeable. An objective's name does not establish a learned state is Markovian, physically sufficient, causal, or useful for planning.

| SimMIM | Xie et al. 2022, https://arxiv.org/abs/2111.09886 | Raw-pixel regression with a simple prediction head; masking ratio and patch size provide the learning signal | A useful counterexample to the assumption that masked-image modeling requires a discrete tokenizer or heavy decoder |
| MaskFeat | Wei et al. 2022, https://arxiv.org/abs/2112.09133 | Predict HOG or other handcrafted feature targets from masked visual regions | A target can be a feature descriptor rather than raw pixels or another neural encoder |
| iBOT | Zhou et al. 2022, https://arxiv.org/abs/2111.07832 | Online tokenizer/self-distillation for masked patch-level targets | A clean example where masking, prototypes and teacher-student learning overlap |
