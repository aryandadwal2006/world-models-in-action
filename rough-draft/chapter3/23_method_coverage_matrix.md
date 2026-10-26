# Coverage matrix: primary papers, mechanics and remaining checks

This is the completeness ledger for the rough-draft folder. It is deliberately broader than the final chapter. “Covered” means that an original explanatory note exists in the dossier; it does not mean the manuscript has reproduced the method, benchmark or full implementation. Every method equation, loss reduction, target update and hyperparameter must be rechecked against the primary paper/code before being presented as an exact reproduction.

## 1. Core visual SSL families

| Method | Primary paper | Dominant target/mechanism | Dossier source | Verification checklist |
|---|---|---|---|---|
| SimCLR | Chen et al. (2020), https://arxiv.org/abs/2002.05709 | NT-Xent over two augmented views and in-batch candidates | 02, 14, 19, 21 | Batch convention; denominator candidates; projector and backbone outputs; augmentation details |
| MoCo v1 | He et al. (2020), https://arxiv.org/abs/1911.05722 | Query/key contrastive classification, queue and EMA key encoder | 02, 14, 21 | Queue size; dequeue/enqueue order; key update timing; temperature and normalization |
| MoCo v2 | Chen et al. (2020), https://arxiv.org/abs/2003.04297 | Stronger training recipe over MoCo's queue mechanism | 14, 18 | Which changes are objective changes versus recipe changes |
| MoCo v3 | Chen et al. (2021), https://arxiv.org/abs/2104.02057 | Predictor-to-momentum-key cross-view classification with globally gathered negatives | 14, 18, 29 | Source confirms no v1/v2 queue, (2T) loss scaling and EMA update inside forward; PyTorch source reimplements the paper's TensorFlow/TPU recipe |
| CPC | van den Oord et al. (2018), https://arxiv.org/abs/1807.03748 | Context predicts future latent among candidates | 02, 14 | Horizon-specific scores; negative sampling; no implicit action conditioning |
| BYOL | Grill et al. (2020), https://arxiv.org/abs/2006.07733 | Online predictor matches EMA target representation | 03, 15, 21 | Exact normalized regression loss; target update; predictor; stop-gradient location |
| SimSiam | Chen and He (2021), https://arxiv.org/abs/2011.10566 | Predictor plus stop-gradient Siamese regression | 03, 15, 21 | Shared encoder, batch normalization, gradient path, no EMA target update |
| DINO v1 | Caron et al. (2021), https://arxiv.org/abs/2104.14294 | Student matches centered/sharpened teacher distribution over prototypes | 03, 15, 21 | Teacher/student temperatures, multi-crop, center update, EMA |
| DINOv2 | Oquab et al. (2023), https://arxiv.org/abs/2304.07193 | Scaled self-distilled visual features, curated data and recipe changes | 03, 15, 18 | Separate objective from data-engineering and model-scale contributions |
| DINOv3 | Siméoni et al. (2025), https://arxiv.org/abs/2508.10104 | Large-scale self-distillation with dense-feature stability mechanisms | 03, 08, 15, 18 | Gram anchoring definition, training schedule, license and reported dense-task setup |
| Barlow Twins | Zbontar et al. (2021), https://arxiv.org/abs/2103.03230 | Cross-view cross-correlation matrix toward identity | 04, 15, 19, 21 | Standardization convention, correlation scaling, off-diagonal coefficient |
| VICReg | Bardes et al. (2022), https://arxiv.org/abs/2105.04906 | Invariance MSE + variance floor + covariance penalty | 04, 15, 19, 21 | Variance margin, covariance denominator, loss weights and reductions |
| W-MSE | Ermolov et al. (2021), https://arxiv.org/abs/2007.06346 | Whiten feature batches, then match paired representations | 04, 15, 18 | Exact whitening placement, regularization and covariance-statistic scope |
| DeepCluster | Caron et al. (2018), https://arxiv.org/abs/1807.05520 | Offline k-means pseudo-labels alternate with network training | 05, 15 | Cluster count, class balancing/empty clusters, feature refresh schedule |
| SwAV | Caron et al. (2020), https://arxiv.org/abs/2006.09882 | Predict assignments from another view using balanced prototypes | 05, 15, 21, 29 | Sinkhorn row/column normalization, same-crop exclusion, temperature and optional queue verified in pinned source; run config still matters |
| NNCLR | Dwibedi et al. (2021), https://arxiv.org/abs/2104.14548 | Nearest-neighbor representation used as positive | 14, 18 | Memory bank update, neighbor selection, false-positive relations |
| Mean Shift for SSL | Koohpayegani et al. (2021), https://arxiv.org/abs/2105.07269 | Shift feature toward neighbor mean | 18 | Exact target, neighbor selection, source paper's relation to BYOL |
| MSN | Assran et al. (2022), https://arxiv.org/abs/2204.07141 | Masked Siamese prediction with prototype self-labels | 18 | Assignment, masking and imbalance-handling details |

## 2. Masked visual and token objectives

| Method | Primary paper | Target | Dossier source | Verification checklist |
|---|---|---|---|---|
| MAE | He et al. (2022), https://arxiv.org/abs/2111.06377 | Pixel values in masked image patches; visible-only encoder | 06, 16 | Mask ratio, patch normalization, masked-only loss, encoder/decoder asymmetry |
| SimMIM | Xie et al. (2022), https://arxiv.org/abs/2111.09886 | Masked pixel reconstruction with simple head | 16 | Exact mask input representation and loss scaling |
| BEiT | Bao et al. (2022), https://arxiv.org/abs/2106.08254 | Discrete visual-token IDs from tokenizer | 06, 16 | Tokenizer architecture, target IDs and masked loss |
| BEiT v2 | Peng et al. (2022), https://arxiv.org/abs/2208.06366 | Semantic vector-quantized visual token targets | 16 | Tokenizer pretraining and target vocabulary |
| MaskFeat | Wei et al. (2022), https://arxiv.org/abs/2112.09133 | HOG or feature-descriptor prediction | 16 | Descriptor extraction, masking and normalization |
| iBOT | Zhou et al. (2022), https://arxiv.org/abs/2111.07832 | Masked patch targets through online tokenizer/self-distillation | 06, 16 | Patch/class token losses, teacher and prototype distributions |
| BERT | Devlin et al. (2019), https://arxiv.org/abs/1810.04805 | Predict corrupted token IDs using bidirectional context | 06, 16 | Original corruption strategy, selected-token loss and auxiliary objectives |
| GPT-2 | Radford et al. (2019), OpenAI report | Causal next-token likelihood | 06, 16 | Autoregressive factorization and causal context; no claim of masked-token objective |
| ELECTRA | Clark et al. (2020), https://arxiv.org/abs/2003.10555 | Original/replaced token detection | 06, 16 | Generator sampling, discriminator labels, token-level loss |
| T5 | Raffel et al. (2020), https://arxiv.org/abs/1910.10683 | Text-to-text span corruption | 07, 16 | Sentinel-token construction and decoder target sequence |

## 3. Speech and cross-modal objectives

| Method | Primary paper | Target | Dossier source | Verification checklist |
|---|---|---|---|---|
| wav2vec 2.0 | Baevski et al. (2020), https://arxiv.org/abs/2006.11477 | Quantized latent speech targets at masked steps, contrastive discrimination | 07, 16 | Quantizer, negatives, masking and fine-tuning boundary |
| HuBERT | Hsu et al. (2021), https://arxiv.org/abs/2106.07447 | Cluster-derived labels predicted at masked speech positions | 07, 16 | Clustering layer, target refresh and iteration |
| WavLM | Chen et al. (2022), https://arxiv.org/abs/2110.13900 | Denoising-oriented masked speech prediction | 16 | Corruption mixture and exact pretraining losses |
| data2vec | Baevski et al. (2022), https://arxiv.org/abs/2202.03555 | Contextualized teacher features predicted from masked input | 06, 07, 16 | Teacher-layer aggregation, teacher update and mask pattern |
| CLIP | Radford et al. (2021), https://arxiv.org/abs/2103.00020 | Symmetric image/text candidate classification | 07, 18, 21 | Logit scale, normalized embeddings and both retrieval directions |
| ALIGN | Jia et al. (2021), https://arxiv.org/abs/2102.05918 | Large-scale image-alt-text contrastive pairs | 07, 18 | Pair/data construction and noise; separate from loss novelty |
| SigLIP | Zhai et al. (2023), https://arxiv.org/abs/2303.15343 | Pairwise sigmoid/logistic image-text loss | 07, 18, 21 | Bias, loss normalization and negative weighting |
| SigLIP 2 | Tschannen et al. (2025), https://arxiv.org/abs/2502.14786 | Multilingual/dense vision-language encoders with a combined recipe | 07, 18 | Distinguish recipe contributions from sigmoid loss alone |

## 4. Latent prediction and current regularization frontier

| Method | Primary source | Training target/mechanism | Dossier source | Scope decision |
|---|---|---|---|---|
| I-JEPA | Assran et al. (2023), https://arxiv.org/abs/2301.08243 | Predict target image-region embeddings from context | 08, 16 | Main mechanics reserved for Chapter 4 |
| V-JEPA | Bardes et al. (2024), https://arxiv.org/abs/2404.08471 | Predict target space-time features in video | 08, 16 | Brief handoff in Chapter 3, full JEPA family in Chapter 4 |
| V-JEPA 2 | Assran et al. (2025), https://arxiv.org/abs/2506.09985 | Video latent prediction; separate action-conditioned post-training V-JEPA 2-AC | 08, 16 | Do not conflate passive pretraining with action-conditioned control |
| V-JEPA 2.1 | Mur-Labadia et al. (2026), https://arxiv.org/abs/2603.14482 | Dense predictive loss on visible/masked tokens, deep self-supervision, multimodal tokenizers | 08, 16 | Verify current paper/repo recipe; detailed JEPA regularization belongs in Chapter 4 |
| LeJEPA | Balestriero and LeCun (2025), https://arxiv.org/abs/2511.08544 | Predict view embeddings relative to global-view mean + SIGReg | 20 | Full regularization derivation in dossier; book handoff to Chapter 4 |
| VISReg | Wu et al. (2026), https://arxiv.org/abs/2606.02572 | Prediction loss + explicit scale + sliced-Wasserstein shape + center | 20 | Paper claims are empirical; not a universal no-collapse theorem |

## 5. Dense and local representation learning

| Method | Primary paper | Distinctive target/relation | Dossier source | Main question |
|---|---|---|---|---|
| DenseCL | Wang et al. (2021), https://arxiv.org/abs/2011.09157 | Local/pixel-level correspondences across views | 18 | Does global instance discrimination preserve local correspondence? |
| PixPro | Xie et al. (2021), https://arxiv.org/abs/2011.10043 | Pixel-level consistency and propagation | 18 | How should spatial neighborhoods define positives? |
| VICRegL | Bardes et al. (2022), https://arxiv.org/abs/2210.01571 | Global VICReg plus local feature matching | 18 | How are local correspondence and global constraints combined? |
| Leopart | Ziegler and Asano (2022), https://arxiv.org/abs/2112.10582 | Patch-level clustering for object/part features | 18 | Which unsupervised patch partitions transfer to semantic regions? |
| SelfPatch | Yun et al. (2022), https://arxiv.org/abs/2206.07990 | Patch-level representation learning in ViTs | 18 | Can patch relationships provide useful spatial structure without labels? |

## 6. Important method families not to accidentally omit

- **Classical pretext tasks:** denoising, inpainting, rotation prediction, jigsaw, context prediction, colorization. Their role is to teach surrogate-target construction; do not give them equal page weight to modern large-scale families.
- **Metric-learning precursors:** Siamese/triplet, N-pair, supervised contrastive learning. SupCon uses external class labels and is therefore a comparator, not strictly label-free SSL.
- **Other teacher/prototype families:** MeanShift, MSN, DINO-family and iBOT. State where categories overlap.
- **Other feature-statistic methods:** whitening-based W-MSE and distribution-shape methods SIGReg/VISReg. These need their own equations, not a generic “non-contrastive” explanation.
- **Other multimodal directions:** ALIGN, BLIP/BLIP-2 and related image-text generation/alignment methods. Expand only if the book's later multimodal chapter does not already own them.
- **Generative objectives:** diffusion and flow matching are trained from data but belong to the generation/modeling chapter, not the same taxonomy as feature SSL.

## 7. Completion rule

Before finalizing any method note, ensure the dossier has:
1. source and publication/version;
2. inputs and target construction;
3. exact objective with all normalization/reduction conventions;
4. gradient/update-path explanation;
5. reason the target might be useful;
6. degenerate solutions or sampling assumptions;
7. recipe knobs and compute costs;
8. paper-reported evidence and limitations;
9. official implementation and license/revision;
10. a direct statement of what the objective does **not** establish about world-model sufficiency.


## 8. Historical contrastive lineages added to the dossier

| Method | Primary source | Main contribution | Theory note | Upstream source |
|---|---|---|---|---|
| Non-parametric instance discrimination (NPID) | Wu et al. (2018), https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html | Instance identity as a proxy class; memory bank and NCE | 30 | VISSL's pinned NPID configuration as a maintained reference |
| PIRL / NPID++ | Misra and van der Maaten (2020), https://openaccess.thecvf.com/content_CVPR_2020/html/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.html | Contrastive identity with pretext invariance; NPID is a special case of the formulation | 30 | VISSL's pinned PIRL configuration; verify against paper |
| CMC | Tian et al. (2019 preprint / 2020 publication), https://arxiv.org/abs/1906.05849 | Contrastive pairing between views of a shared source | 30 | Author-published HobbitLong/CMC, pinned in implementations/cmc/ |
| AMDIM | Bachman et al. (2019), https://arxiv.org/abs/1906.00910 | Augmented, multiscale local/global feature discrimination | 30 | Author-published Philip-Bachman/amdim-public, pinned in implementations/amdim/ |

The four rows are complementary historical branches, not a strict sequence of successors. Record whether the method uses NCE or a minibatch InfoNCE softmax and do not equate their estimators without matching their sampling and normalization assumptions.


## 9. Source-backed coverage additions

| Method | Primary source | Direct contribution | Theory note | Source limits to retain |
|---|---|---|---|---|
| iBOT | Zhou et al. (2022), https://arxiv.org/abs/2111.07832 | DINO-style class distillation plus masked patch-token distillation | 31 | Separate class/patch centers, temperatures, masked patch reductions verified in pinned source; run config still required |
| MAE | He et al. (2022), https://arxiv.org/abs/2111.06377 | Visible-only encoder and decoder reconstruction of masked pixels | 16, 27, 31 | Random mask/unshuffle, loss on removed patches and optional pixel normalization verified in pinned source |
| CLIP | Radford et al. (2021), https://arxiv.org/abs/2103.00020 | Symmetric image-text batch classification with learned logit scale | 07, 27, 31 | Normalized embeddings and score matrix verified in official model code; no full public training loop at pinned revision |
| SigLIP | Zhai et al. (2023), https://arxiv.org/abs/2303.15343 | Pairwise sigmoid loss for matched and unmatched image-text pairs | 07, 27, 31 | Pair-sign matrix, scale, bias and log-sigmoid reduction verified in Big Vision; pinned COCO/LiT config is only a minimal example |


## 10. Modern teacher-student objective extensions

| Method/release | Primary source | Core or additional mechanism | Theory note | Configuration caveat |
|---|---|---|---|---|
| DINOv2 | Oquab et al. (2023), https://arxiv.org/abs/2304.07193 | DINO class-token distillation + iBOT patch distillation + KoLeo feature-spreading regularizer | 32 | Centering vs Sinkhorn-Knopp, head sharing, mask and crop choices are config-dependent |
| DINOv3 | Siméoni et al. (2025), https://arxiv.org/abs/2508.10104 | DINO/iBOT targets, with optional Gram-matrix anchoring in selected recipes | 32 | Standard pinned pretraining config disables Gram loss; Gram-anchor config enables it and changes other recipe settings |


## 11. JEPA prediction and action-conditioned variants

| Method | Paper | Direct prediction target | Theory note | Boundary to preserve |
|---|---|---|---|---|
| I-JEPA | Assran et al. (2023), https://arxiv.org/abs/2301.08243 | Target-region image embeddings from context-region features | 33 | Latent prediction is not pixel reconstruction and does not prove state sufficiency |
| V-JEPA | Bardes et al. (2024), https://arxiv.org/abs/2404.08471 | Space-time target features from video context | 33 | Passive clip prediction is not explicitly conditioned on actions |
| V-JEPA 2 / 2-AC | Assran et al. (2025), https://arxiv.org/abs/2506.09985 | Latent video targets; 2-AC adds action-conditioned post-training | 33 | Keep passive pretraining separate from action-conditioned world-model post-training |
| V-JEPA 2.1 | Mur-Labadia et al. (2026), https://arxiv.org/abs/2603.14482 | Dense targets including context/visible and masked tokens, with intermediate-layer supervision | 33 | Exact L1-like loss, context-distance weights and masks are version/config dependent |
| EB-JEPA | https://arxiv.org/abs/2602.03604 | Predicted latent observation sequence, optionally action-conditioned | 33 | Released examples combine prediction with configurable variance/covariance, temporal and inverse-dynamics terms; not one universal recipe |


## Language and speech SSL reference

The new [language and speech objective dossier](34_language_and_speech_ssl_objectives_from_first_principles.md) derives targets and losses for BERT, GPT, ELECTRA, T5, wav2vec 2.0, HuBERT, WavLM and data2vec. It distinguishes verified code paths from paper-level descriptions where a complete upstream trainer is unavailable.


## Visual SSL objective reference

See [Visual SSL objectives](35_visual_ssl_objectives_from_first_principles.md) for derivations and source-pinned comparisons of contrastive, momentum-dictionary, non-contrastive, clustering, self-distillation and masked-reconstruction methods. The note distinguishes the loss target from the method's anti-collapse mechanism and records limits on what upstream source pointers establish.


## Dense/local visual SSL reference

See [Dense and local visual SSL](36_dense_local_visual_ssl_from_first_principles.md) for local correspondence assumptions, patch-level targets, dense-feature evaluation and source/reproduction boundaries for DenseCL, PixPro, VICRegL, iBOT, Leopart, SelfPatch and DINOv2/v3.


## CPC, EBM and diffusion scope note

[The CPC, energy-based and diffusion dossier](37_cpc_energy_based_and_diffusion_objectives.md) provides the first-principles objective boundary: future-latent candidate prediction, joint/product-of-marginal discrimination, normalized energy models, denoising diffusion, score-based SDEs and flow matching.


## Readiness gate

The broad inventory is not equivalent to completed source verification. Use [Dossier readiness and manuscript-start gate](38_dossier_readiness_and_manuscript_gate.md) to distinguish covered families, source-inspected versus executed evidence, unresolved citation work, and criteria for starting the composed manuscript.


## 12. Masked and clustering-family source checks

| Method | Pinned source | Source-verified target | Recipe caveat |
|---|---|---|---|
| DeepCluster | `implementations/deepcluster/` | Offline cluster assignments become pseudo-labels for cross-entropy | Re-clustering interval, cluster count and algorithm matter; repository LICENSE says CC BY-NC 4.0 |
| BEiT | `implementations/unilm/beit/` | Cross-entropy on DALL-E tokenizer code IDs at masked positions | Separate tokenizer prerequisite; original run is resource intensive |
| SimMIM | `implementations/simmim/` | L1 pixel reconstruction over masked regions | Mask patch size, ratio, architecture and normalization vary by config |
| MaskFeat | `implementations/slowfast/projects/maskfeat/` | HOG regression at masked positions in the cited ImageNet config | RGB/HOG and image/video targets are configuration-dependent |

