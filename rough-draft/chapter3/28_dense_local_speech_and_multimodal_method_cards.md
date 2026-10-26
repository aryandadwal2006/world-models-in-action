# Dense/local visual learning, speech SSL and cross-modal objectives: method cards

**Purpose.** This note deepens the method atlas rather than repeating its survey-level descriptions. The main question is where the training signal is attached: to an entire image, a local feature, a masked token, a speech frame, or a paired modality. A representation useful for image-level classification may still have poor spatial correspondence or lose speech properties a downstream task needs.

Equations marked *schematic* express the mechanism being compared, not a claim that every weight, normalizer, queue, sampler and reduction from the original implementation has been reproduced. Use the linked primary paper and upstream repository for exact recipes.

## 1. Global and local features solve different problems

Let an encoder emit a spatial feature map \(F(x)\in\mathbb R^{H\times W\times d}\), and let pooling produce a global vector \(h(x)\in\mathbb R^d\). A global objective typically uses \(h(x)\) to identify, align or predict an entire view. A dense objective additionally defines a signal at spatial locations:

\[
F(x)=\{f_s(x):s\in\mathcal S\}.
\]

The distinction is not merely tensor shape. A global objective may reward invariance to crop position because image identity survives the crop. A local task needs enough correspondence information to match the same region after geometric transformation. Conversely, forcing local invariance across an incorrect match teaches a false relation.

The core design decisions are:
1. **Correspondence:** what counts as the same location after crop, resize, flip or other transforms?
2. **Target geometry:** contrastive classification, feature regression, clustering, or a propagated target?
3. **Spatial prior:** is location itself informative, or should visually similar patches match despite different coordinates?
4. **Evaluation:** are features judged using detection, segmentation or other dense tasks, rather than classification alone?

These assumptions are imposed by the training procedure; none is guaranteed correct across domains.

## 2. DenseCL: contrastive learning at global and local scales

DenseCL starts from a MoCo-v2-style global contrastive recipe and adds a local feature contrastive term. Two augmented views produce spatial feature maps and local embeddings. Dense correspondences are selected, and each matched local pair is trained to be more similar than candidate local negatives.

A conceptual local loss for query \(r^s\) and matched positive key \(t_+^s\) is

\[
\ell_r^s =
-\log
\frac{\exp((r^s)^\top t_+^s/\tau)}
{\exp((r^s)^\top t_+^s/\tau)+
\sum_{t_-}\exp((r^s)^\top t_-/\tau)}.
\]

This is the positive-versus-negatives mechanism; the exact candidate bank, matching, and sampling rules should be taken from the paper/code. The total loss is reported as

\[
\mathcal L_{\mathrm{DenseCL}}
=(1-\lambda)\mathcal L_q+\lambda\mathcal L_r,
\]

where \(\mathcal L_q\) is the global query loss and \(\mathcal L_r\) is the dense local loss. The paper reports \(\lambda=0.5\) for its evaluated setup, not as a universal optimum.

Unlike a global pooled loss, DenseCL directly supervises the spatial feature map. Its primary failure risk is **false local correspondence**: transformed views may not contain the same region, or the matcher may select the wrong region, causing the model to learn the wrong positive relation.

The paper reports improvements on dense prediction tasks relative to its MoCo-v2 baseline. This is empirical evidence under the paper's data, backbone, and downstream training protocol, not a theorem that dense contrast is always better.

**Primary source:** Wang et al. (2021), *Dense Contrastive Learning for Self-Supervised Visual Pre-Training*, https://arxiv.org/abs/2011.09157. Official code: https://github.com/WXinlong/DenseCL. On **main-extra**, the pinned upstream is at **implementations/densecl/**.

## 3. PixPro: pixel-to-propagation consistency

PixPro studies pixel-level pretext tasks and contrasts two ideas:
- **PixContrast:** directly apply contrastive learning to pixel/local features.
- **PixPro:** use pixel-to-propagation consistency.

For the second task, asymmetric branches create the positive pair for a corresponding pixel. One produces an ordinary projected local feature; the other applies a propagation module that filters the feature by propagating features of similar pixels to it. A schematic objective is

\[
\mathcal L_{\mathrm{PixPro}}
=\mathbb E_{(s,s^+)}
d\left(p_s,\operatorname{sg}(\tilde p_{s^+})\right),
\]

where \(p\) is the ordinary local feature, \(\tilde p\) the propagated counterpart, and \(d\) a consistency distance. This omits the paper's exact propagation operator, normalization, transformation mapping and reduction; it is not an implementation formula.

The propagation module aggregates neighboring features according to feature similarity and spatial relationship. The ordinary branch remains spatially sensitive. PixPro is therefore not simply DenseCL with more positives: its defining asymmetry is **ordinary local feature versus propagated local target**, and the consistency task avoids a negative dictionary in the same way that PixContrast does not.

Propagation can smooth noisy local features, but it can also blend distinct objects across a weak boundary. Evaluate small objects and boundaries, not only average segmentation scores. The paper reports dense-transfer gains as empirical results; these are not a universal guarantee for all tasks.

**Primary source:** Xie et al. (2021), *Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning*, https://arxiv.org/abs/2011.10043. Official code: https://github.com/zdaxie/PixPro. Pinned source on **main-extra**: **implementations/pixpro/**.

## 4. VICRegL: global statistics plus matched local features

VICRegL applies VICReg-style variance, invariance and covariance constraints to global feature vectors and to matched local vectors before the final pooling layer. Two differently augmented views are processed by the same architecture.

A high-level objective is

\[
\mathcal L_{\mathrm{VICRegL}}
=\mathcal L_{\mathrm{VICReg}}(h^A,h^B)
+\alpha_{\mathrm{local}}\mathcal L_{\mathrm{VICReg}}(F^A_{\mathcal M},F^B_{\mathcal M}),
\]

where \(\mathcal M\) is a set of local matches. This is the schematic global/local decomposition; exact sampling, weights, statistics and reductions should be copied from the paper configuration.

A distinctive choice is the **local correspondence rule**. A pair of local features is attracted if their feature-space \(L_2\) distance is below a threshold or their relative positions are consistent with the known geometric transformation between the views. Local matching thus combines a content-based condition and a geometric prior.

This differs from DenseCL's local contrastive classification: VICRegL applies variance/invariance/covariance regularization to matched local features. It differs from PixPro's propagated target because propagation is not its defining mechanism.

Global terms preserve image-level information; local terms directly constrain spatial descriptors. The matching prior can still be wrong, and global/local terms compete for capacity. The paper evaluates classification and dense transfer, but does not establish that the resulting feature is physically sufficient or temporally predictive.

**Primary source:** Bardes, Ponce and LeCun (2022), *VICRegL: Self-Supervised Learning of Local Visual Features*, NeurIPS, https://arxiv.org/abs/2210.01571. Official code: https://github.com/facebookresearch/VICRegL. Pinned source: **implementations/vicregl/** on **main-extra**.

## 5. Leopart: spatially dense clustering of object parts

Leopart addresses the gap between image-level SSL and spatially diverse patch features. It starts from a DINO-initialized Vision Transformer and fine-tunes spatial tokens through dense clustering.

The method's pipeline has several coupled parts:
1. Multiple crops pass through student and EMA teacher branches.
2. Patch features produce cluster predictions.
3. Balanced prototype assignments are computed using an optimal-transport/Sinkhorn-style procedure over spatial tokens.
4. Teacher assignments are geometrically aligned across overlapping crops; the ViT attention map helps emphasize foreground patches.
5. A swapped prediction task trains features from one view to predict assignments from another.

A schematic cross-view cluster loss is

\[
\mathcal L_{\mathrm{patch}}
=-\sum_{i,j}m_{ij}\sum_{k=1}^{K}q^{(j)}_{i,k}\log p^{(i)}_{i,k},
\]

where \(q^{(j)}\) is a target cluster assignment from one view, \(p^{(i)}\) is a prediction from another view after alignment, and \(m_{ij}\) marks valid aligned targets. This is an explanatory shorthand, not the complete equation from the original source; exact crop overlap, attention weighting, queue and indexing matter.

Leopart intersects DINO-style teacher updates, prototype clustering, spatial geometry and cross-view prediction. “Clustering the patches” omits the alignment and foreground-selection parts of the target construction.

The goal is to learn reusable parts that can be grouped into objects later, not necessarily to recover one uniquely correct object segmentation. Evaluation should distinguish patch feature quality, unsupervised segmentation postprocessing and supervised segmentation transfer.

**Primary source:** Ziegler and Asano (2022), *Self-Supervised Learning of Object Parts for Semantic Segmentation*, CVPR, https://arxiv.org/abs/2204.13101. Author/team code: https://github.com/MkUuWaUjinga/leopart. Pinned source: **implementations/leopart/** on **main-extra**.

## 6. SelfPatch: make neighboring patch relations explicit

SelfPatch starts from a Vision Transformer property: the encoder produces patch tokens, but many image-level SSL methods supervise mostly a global representation. SelfPatch adds a patch-level pretext task, treating each patch and selected semantically similar neighboring patches as positive samples.

A schematic neighborhood consistency term is

\[
\mathcal L_{\mathrm{patch}}
=\sum_i\sum_{j\in\mathcal N(i)}w_{ij}\,d(z_i,z_j),
\]

where \(\mathcal N(i)\) and \(w_{ij}\) represent the neighbor-selection and weighting rules. This is a conceptual representation of the method's local invariance relation, not its complete paper loss. Exact neighbor selection, the combination with the base image-level SSL loss, and implementation settings must be checked against the source.

Its prior differs from the adjacent methods:
- SelfPatch treats selected neighboring patches as positive relations.
- VICRegL uses global and local VICReg losses plus a correspondence rule.
- DenseCL uses local contrastive classification with negatives.
- PixPro matches ordinary features to propagated features.
- Leopart uses aligned prototype assignments for spatial tokens.

A neighbor can cross an object boundary or come from an adjacent but different object. This makes dense evaluation and boundary-sensitive metrics essential.

**Primary source:** Yun et al. (2022), *Patch-Level Representation Learning for Self-Supervised Vision Transformers*, CVPR, https://arxiv.org/abs/2206.07990. Author/team source: https://github.com/alinlab/SelfPatch. Pinned source: **implementations/selfpatch/** on **main-extra**.

## 7. Speech SSL: shared sequence scaffolding, different targets

Speech methods often map waveform \(x\) through a learned feature encoder to latent frames \(z_1,\ldots,z_T\). A context network processes masked latent spans and produces contextual states \(c_t\). Their main differences are how each frame target is constructed and whether the model predicts a class-like unit or a continuous teacher representation.

### 7.1 wav2vec 2.0: quantized target discrimination

wav2vec 2.0 masks spans in latent space and contrasts a contextual state \(c_t\) with a learned quantized target \(q_t\) among candidate quantized representations. A schematic loss is

\[
\mathcal L_m
=-\sum_{t\in\mathcal M}
\log\frac{\exp(\operatorname{sim}(c_t,q_t)/\kappa)}
{\sum_{\tilde q\in\mathcal Q_t}\exp(\operatorname{sim}(c_t,\tilde q)/\kappa)},
\]

where \(\mathcal Q_t\) contains the positive and sampled distractors, and \(\kappa\) is a temperature. In the original method, negatives are sampled from other masked time steps with detailed sampling rules defined by the paper. A diversity term encourages use of the quantizer codebooks.

The quantizer is jointly learned. Its units are not human transcripts and are not guaranteed to correspond to phonemes. After self-supervised pretraining, labeled ASR fine-tuning is a separate stage (commonly CTC); that downstream supervised objective must not be confused with the SSL loss.

**Primary source:** Baevski et al. (2020), *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS, https://arxiv.org/abs/2006.11477. Official implementation is available in **facebookresearch/fairseq**, pinned at **implementations/fairseq/** on **main-extra**.

### 7.2 HuBERT: masked prediction of offline cluster labels

HuBERT creates frame-aligned pseudo-labels using offline clustering, then trains a BERT-like model to predict them from masked acoustic input. If \(y_t\in\{1,\ldots,K\}\) is a cluster label and \(p_\theta(y_t\mid\tilde x)\) the model distribution, the schematic loss is

\[
\mathcal L_{\mathrm{HuBERT}}
=-\sum_{t\in\mathcal M}\log p_\theta(y_t\mid\tilde x).
\]

The loss is applied on masked regions. The target labels may be regenerated from learned representations over iterations. The paper emphasizes consistency of the unsupervised cluster assignment, not a claim that every cluster is a clean phoneme category.

Compared with wav2vec 2.0, HuBERT uses offline cluster IDs as classification targets; wav2vec 2.0 jointly learns a quantizer and performs candidate contrastive discrimination. Both use masking and temporal prediction, but differ in target construction and loss.

Pseudo-label frame rate must align with the encoded frame rate. The official fairseq README explicitly records cluster-label generation, label rates, pretraining and separate CTC fine-tuning commands.

**Primary source:** Hsu et al. (2021), *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, https://arxiv.org/abs/2106.07447. Upstream method docs: https://github.com/facebookresearch/fairseq/tree/main/examples/hubert.

### 7.3 WavLM: masked prediction plus denoising

WavLM aims at a broader set of speech tasks than ASR alone. It combines masked speech prediction with denoising, uses utterance mixing as an important corruption strategy, and employs gated relative position bias to improve modeling of sequence order.

A schematic objective is

\[
\mathcal L_{\mathrm{WavLM}}
=\mathcal L_{\mathrm{masked}}+\beta\mathcal L_{\mathrm{denoise}},
\]

where \(\beta\) is merely notation for the relative loss weight here; this is not the exact published implementation equation. Reproduce the paper's specific target labels, corruption procedure, weights and reductions from the source configuration.

Masked prediction promotes contextual acoustic modeling; denoising is intended to preserve robustness when speech overlaps or is mixed with other signals. The trade-off is task-dependent: a representation should not erase speaker or event information when the downstream task needs it.

**Primary source:** Chen et al. (2022), *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, https://arxiv.org/abs/2110.13900. Official source is Microsoft UniLM, with method files under WavLM; pinned repository: **implementations/unilm/** on **main-extra**.

### 7.4 data2vec: contextualized teacher features rather than local labels

data2vec predicts contextualized latent representations of the full input from a masked view in a self-distillation setup. Unlike HuBERT's cluster IDs or wav2vec 2.0's quantized targets, the target is a continuous contextual teacher feature. The same broad design applies to speech, vision and language, although the input pipelines and recipes remain modality-specific.

A schematic objective is

\[
\mathcal L_{\mathrm{data2vec}}
=\sum_{t\in\mathcal M}
\rho\left(s_\theta(\tilde x)_t,\operatorname{sg}(y_t^{\mathrm{teacher}})\right),
\]

where the teacher target is contextualized and \(\rho\) stands for the paper's regression loss. The exact target-layer aggregation, normalization, mask, teacher update and loss must be read from the original method/configuration.

Compare the target families directly:
- **quantized latent plus contrastive distractors:** wav2vec 2.0;
- **offline cluster ID plus masked cross-entropy:** HuBERT;
- **contextualized continuous teacher vector:** data2vec;
- **masked target plus denoising and utterance mixing:** WavLM.

A continuous target is not automatically richer or better. Teacher quality, normalization, context and evaluation still determine what information survives.

**Primary source:** Baevski et al. (2022), *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language*, ICML, https://proceedings.mlr.press/v162/baevski22a. Official source: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec.

## 8. Image-text objectives: pair construction is part of the supervision

Image-text models use naturally paired or web-collected text as the positive relation. This is not conventional class-label supervision, but it is still an informative training signal whose quality depends on captions and pair construction.

### 8.1 CLIP: symmetric batch classification

Let normalized image embeddings \(u_i\) and text embeddings \(v_j\) come from \(B\) paired samples. With a logit scale \(s\), define

\[
S_{ij}=s\,u_i^\top v_j.
\]

The image-to-text loss is

\[
\mathcal L_{I\rightarrow T}
=-\frac1B\sum_{i=1}^{B}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ij})}.
\]

The reverse direction applies the same classification to \(S^\top\); CLIP averages the two directions. It learns batch-level pair retrieval, not a complete ontology or guarantee of spatial grounding, reasoning or physical-state sufficiency. Captions can omit entities or describe context instead of the visual property needed downstream.

**Source:** Radford et al. (2021), https://arxiv.org/abs/2103.00020; original code https://github.com/openai/CLIP.

### 8.2 ALIGN: large-scale noisy image-text pairs

ALIGN also scales image-text contrastive training using noisy web-derived pairs. Its central contribution includes data regime and pairing construction, not just a changed objective. No official public ALIGN code release was verified during this dossier update; a third-party implementation should not be described as the author's official repo.

**Source:** Jia et al. (2021), https://arxiv.org/abs/2102.05918.

### 8.3 SigLIP: pairwise sigmoid rather than batch softmax

SigLIP uses pairwise logistic classification rather than CLIP's row/column softmax cross-entropy. For pair label \(y_{ij}\in\{-1,+1\}\) and scaled/bias-adjusted logit \(a_{ij}\), a generic term is

\[
\ell_{ij}=\log\left(1+\exp(-y_{ij}a_{ij})\right).
\]

The exact recipe includes its own scaling, bias, pair sampling and reduction. Conceptually, CLIP normalizes candidate probabilities over rows/columns, while SigLIP classifies each image-text pair through sigmoid terms. It still relies on positive-pair quality, negative composition and data coverage.

SigLIP 2 is a broader recipe intended to improve multilinguality, localization and dense features; it should not be presented as just a loss replacement.

**Sources:** SigLIP (Zhai et al., 2023), https://arxiv.org/abs/2303.15343; SigLIP 2 (Tschannen et al., 2025), https://arxiv.org/abs/2502.14786. Upstream code is in **implementations/big_vision/** on **main-extra**.

## 9. Objective comparison

| Method | Target type | Defining mechanism | Main assumption / risk |
|---|---|---|---|
| DenseCL | Matched local feature | Dense contrastive loss plus global contrastive loss | Spatial correspondence can be wrong |
| PixPro | Propagated local feature | Pixel-to-propagation consistency | Propagation may smooth across object boundaries |
| VICRegL | Global and matched local vectors | VICReg constraints at two spatial scales | Matching rule imposes locality prior |
| Leopart | Patch prototype assignment | Dense balanced clustering and swapped prediction | Balanced clusters/foreground mask need not equal true parts |
| SelfPatch | Neighboring patch relations | Patch-level invariance with image-level SSL | Neighboring patches may be different objects |
| wav2vec 2.0 | Quantized latent | Contrastive discrimination plus quantizer diversity | Codes are not guaranteed to be phonemes |
| HuBERT | Offline cluster IDs | Masked cluster classification | Pseudo-label quality and frame alignment matter |
| WavLM | Masked target plus denoising | Masked prediction and corruption/denoising | May discard nuisance attributes that are task-relevant |
| data2vec | Contextualized teacher feature | Masked-view self-distillation | Teacher target and normalization control what is learned |
| CLIP | Paired image-text label | Symmetric batch-softmax cross-entropy | Caption noise and batch negatives define the relation |
| SigLIP | Pairwise match/non-match label | Sigmoid/logistic pair classification | Pair construction and sampling still matter |

## 10. Evaluate the signal the objective actually trains

A high image-classification linear-probe score establishes that class information is linearly accessible under that protocol. It does not establish strong local geometry, speaker identity, word-boundary timing, localization, spatial correspondence or action-conditioned state.

**For dense visual features**, distinguish global classification, object detection, semantic/instance segmentation, local correspondence under known transformations, and boundary/small-object quality. Match data split, backbone, pretraining data, resolution and downstream schedule before comparing numbers. Some gains arise from the larger whole-recipe package, not one objective component.

**For speech**, distinguish transcript-based ASR fine-tuning, phonetic/content probes, speaker identity, diarization, paralinguistic tasks and robustness to noise/overlap. Masked prediction loss is not ASR performance. Strong word error rate after fine-tuning does not show that frozen features preserve all speaker or environmental information.

**For image-text**, separate text-image retrieval, zero-shot classification, fine-grained localization and robustness to captions omitting entities or describing context. Good retrieval can coexist with poor patch-level grounding.

**For world models**, add latent aliasing and dynamics tests. If two histories map to the same representation but produce different future observations or rewards under the same action, that representation is not sufficient for that prediction target regardless of classification, segmentation or retrieval performance.

## 11. Upstream source map and licensing

The **main-extra** branch collects upstream Git submodules at:
- **implementations/densecl/** — DenseCL: https://github.com/WXinlong/DenseCL
- **implementations/pixpro/** — PixPro: https://github.com/zdaxie/PixPro
- **implementations/vicregl/** — VICRegL: https://github.com/facebookresearch/VICRegL
- **implementations/leopart/** — Leopart: https://github.com/MkUuWaUjinga/leopart
- **implementations/selfpatch/** — SelfPatch: https://github.com/alinlab/SelfPatch
- **implementations/fairseq/** — wav2vec 2.0, HuBERT and data2vec: https://github.com/facebookresearch/fairseq
- **implementations/unilm/** — WavLM and related work: https://github.com/microsoft/unilm
- **implementations/clip/** — CLIP: https://github.com/openai/CLIP
- **implementations/big_vision/** — SigLIP and SigLIP 2: https://github.com/google-research/big_vision

These are pinned submodules, not locally copied source files or one ready-to-run environment. Initialize them with **git submodule update --init --recursive** from the repository root on that branch. Each repository retains its own license and dependencies. DenseCL reports GPL-3.0; VICRegL is archived and its terms are more restrictive than a blanket permissive-license label. Check each pinned repository's license before using or redistributing anything.

No public author/team implementation was verified here for original CPC, NNCLR or ALIGN. Do not silently use a third-party implementation and label it official. Until an upstream source is verified, their papers remain the primary references.

## References

- Baevski, A. et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS. https://arxiv.org/abs/2006.11477.
- Hsu, W.-N. et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*. https://arxiv.org/abs/2106.07447.
- Chen, S. et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*. https://arxiv.org/abs/2110.13900.
- Baevski, A. et al. (2022). *data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language*. ICML. https://proceedings.mlr.press/v162/baevski22a.
- Wang, X. et al. (2021). *Dense Contrastive Learning for Self-Supervised Visual Pre-Training*. CVPR. https://arxiv.org/abs/2011.09157.
- Xie, Z. et al. (2021). *Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning*. CVPR. https://arxiv.org/abs/2011.10043.
- Bardes, A., Ponce, J. and LeCun, Y. (2022). *VICRegL: Self-Supervised Learning of Local Visual Features*. NeurIPS. https://arxiv.org/abs/2210.01571.
- Ziegler, A. and Asano, Y. M. (2022). *Self-Supervised Learning of Object Parts for Semantic Segmentation*. CVPR. https://arxiv.org/abs/2204.13101.
- Yun, S. et al. (2022). *Patch-Level Representation Learning for Self-Supervised Vision Transformers*. CVPR. https://arxiv.org/abs/2206.07990.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. https://arxiv.org/abs/2103.00020.
- Jia, C. et al. (2021). *Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision*. https://arxiv.org/abs/2102.05918.
- Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training*. https://arxiv.org/abs/2303.15343.
- Tschannen, M. et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. https://arxiv.org/abs/2502.14786.
