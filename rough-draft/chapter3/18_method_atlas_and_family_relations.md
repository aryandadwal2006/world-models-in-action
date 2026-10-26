# Method atlas: variants, relationships and missing branches

This atlas is intentionally broader than the reader-facing chapter. It records how important methods relate, what their target construction changes, and which claims need primary-paper verification.

## 1. A multidimensional taxonomy

| Axis | Possible values | Examples |
|---|---|---|
| Input relation | Same-instance views; neighboring frames; paired modalities; masked context/target | SimCLR; CPC; CLIP; MAE |
| Target type | Instance/candidate index; prototype assignment; teacher feature/distribution; pixel/token target; future latent | MoCo; SwAV; DINO; BERT; I-JEPA |
| Loss geometry | Softmax cross-entropy; sigmoid classification; cosine regression; MSE; covariance/correlation; clustering assignment | InfoNCE; SigLIP; BYOL; VICReg; Barlow Twins |
| Target update | None; stop-gradient/shared weights; EMA teacher; offline tokenizer/clustering | SimCLR; SimSiam; BYOL; BEiT; DeepCluster |
| Anti-collapse route | Negatives; predictor/teacher dynamics; variance floor; whitening; assignment balancing | SimCLR; BYOL; VICReg; W-MSE; SwAV |
| Spatial/temporal scope | Global vector; patch features; masked regions; speech tokens; video space-time tokens | DINO; iBOT; MAE; wav2vec 2.0; V-JEPA |
| Supervision source | Raw data; naturally paired language/audio; manual pretraining labels; downstream labels | MAE; CLIP captions; supervised contrastive; linear probe |

This avoids a false dichotomy where every method must be either “contrastive” or “non-contrastive.” CLIP is contrastive and cross-modal. iBOT is masked and teacher-based. SwAV is prototype assignment and cross-view prediction. V-JEPA is masked/target-region latent prediction. Taxonomies may choose a different primary axis; the chapter must say which one it uses.

## 2. Deep metric learning before SimCLR

A Siamese network applies shared parameters to two inputs and learns an embedding where a specified pair relation becomes easy to satisfy. Triplet loss uses an anchor a, positive p and negative n:

\[
L_{\mathrm{triplet}}=\max\{0,d(z_a,z_p)-d(z_a,z_n)+\alpha\}.
\]

If the negative is already farther than the positive by margin alpha, this example contributes zero loss. N-pair formulations extend comparisons to multiple negatives. Modern contrastive SSL adds augmentation-defined positives, minibatch candidates and softmax ranking. InfoNCE is not the first contrastive idea; SimCLR's impact was a scalable recipe coupling the loss, view generation, projection head and training system.

## 3. NNCLR: nearest-neighbor positives

Nearest-Neighbor Contrastive Learning (NNCLR) replaces or supplements the same-instance positive with a nearest-neighbor feature selected from a memory/bank mechanism. This can encourage neighborhoods of semantically related examples rather than exact instance identity. It also imports neighbor errors: a wrong early match can become a self-reinforcing positive relation.

Reference: Dwibedi et al. (2021), *With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations*, ICCV, https://arxiv.org/abs/2104.14548. Before implementing it, check the original repository for the exact memory update, neighbor selection and loss equation.

## 4. MeanShift, MSN and assignments

Mean Shift for Self-Supervised Learning (Koohpayegani et al., 2021) shifts each embedding toward a mean of its neighbors' embeddings instead of using an explicit negative dictionary or adopting a strong prior on the number/shape of clusters. In the paper's setup the closest neighbor can be another augmentation of the same image; with one nearest neighbor the method reduces to a BYOL-like case. Its central target is a neighborhood mean, and the number/quality of neighbors controls the relation. Primary paper: https://arxiv.org/abs/2105.07269; official code: https://github.com/UMBCvision/MSF. Exact loss and implementation details should be checked there.

Masked Siamese Networks (MSN) combine a masked input with prototype-based self-labeling and a Siamese/teacher-style target. It is an important overlapping category: masking defines context, assignment defines the target and teacher/student asymmetry defines target generation. MSN also studies distribution assumptions on unbalanced datasets.

Reference: Assran et al. (2022), *Masked Siamese Networks for Label-Efficient Learning*, ECCV, https://arxiv.org/abs/2204.07141.

Distinguish three prototype-adjacent methods:
- **SwAV:** online balanced assignments by a Sinkhorn-Knopp procedure, followed by cross-view assignment prediction.
- **DINO:** teacher-student cross-entropy over prototype logits, with centering, sharpening and an EMA teacher.
- **MSN:** masked Siamese/prototype self-labeling with its own assignment/regularization choices.

Prototype output does not make these methods the same algorithm.

## 5. W-MSE and whitening

Suppose features have empirical covariance \(\Sigma\). Whitening approximately centers the batch and transforms features by a regularized inverse square root such as \((\Sigma+\epsilon I)^{-1/2}\), making the transformed covariance closer to identity when the estimate is well-conditioned. Paired-view regression can then operate in the whitened geometry without a conventional negative dictionary.

Whitening depends on batch size, feature rank and regularization; a nearly singular covariance makes the inverse square root numerically sensitive. Exact methods differ in where statistics are computed and where whitening is applied. The original paper to use is Ermolov et al. (2021), *Whitening for Self-Supervised Representation Learning*, https://arxiv.org/abs/2007.06346. Do not replace its precise loss and whitening convention with a generic matrix formula.

## 6. DINO lineage and newer regularizers

DINO v1 uses teacher-student distribution matching, centering, sharpening and EMA. DINOv2 combines the lineage with a large curated image dataset, recipe changes and additional regularization, including KoLeo regularization in its published recipe. KoLeo is an entropy-estimation-based regularizer that encourages spread in feature space; it is not the same mechanism as centering teacher logits.

DINOv3 (2025) introduces a scaled recipe with explicit attention to dense spatial feature quality and Gram anchoring. This reflects a key difference: global representation transfer and local spatial feature quality must be diagnosed separately. Do not attribute all gains to a single loss name; data, architecture, resolution, schedule, regularizers and evaluation all matter.

References:
- DINO v1: Caron et al. (2021), https://arxiv.org/abs/2104.14294
- DINOv2: Oquab et al. (2023), https://arxiv.org/abs/2304.07193
- DINOv3: Siméoni et al. (2025), https://arxiv.org/abs/2508.10104

## 7. Dense-feature relatives

Global image-level objectives need not preserve good local correspondences. Dense and spatial methods explicitly change the relation or loss location:

- **DenseCL:** matches local/pixel-level features across augmentations to handle spatially displaced corresponding regions. Wang et al. (2021), https://arxiv.org/abs/2011.09157.
- **PixPro:** uses pixel-to-propagation consistency to learn local representations. Xie et al. (2021), https://arxiv.org/abs/2011.10043.
- **VICRegL:** combines VICReg-style statistical objectives with local feature matching. Bardes et al. (2022), https://arxiv.org/abs/2210.01571.
- **Leopart:** clusters patch-level features. Ziegler and Asano (2022), https://arxiv.org/abs/2112.10582.
- **SelfPatch:** uses relationships between patches within images. Yun et al. (2022), https://arxiv.org/abs/2206.07990.

These methods introduce priors about correspondence and locality. Patch matching is uncertain when crops, occlusion or overlapping objects change which region corresponds to which; a good global embedding does not certify local geometry.

## 8. CLIP, ALIGN, SigLIP and SigLIP 2

### CLIP's symmetric batch classification

Let image encoder \(f_I\) and text encoder \(f_T\) produce normalized embeddings \(u_i,v_i\) for B matched pairs. Define

\[
s_{ij}=\frac{u_i^\top v_j}{\tau}.
\]

The image-to-text loss is

\[
L_{I\rightarrow T}=-\frac1B\sum_{i=1}^{B}\log
\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ij})}.
\]

The text-to-image term applies the same classification in the opposite direction; the symmetric objective averages them. The positive comes from a caption or text paired with the image, not a manually assigned class ontology. The learned relation is consequently shaped by caption quality, missing entities, noisy pairing and the batch's negative examples.

CLIP retrieval alignment does not by itself guarantee spatial grounding or physical-state sufficiency. It establishes a particular cross-modal compatibility property.

### ALIGN

ALIGN also scales image-text contrastive training using noisy web-derived pairs. Its significance lies in the scale and data construction as well as the objective. Naturally paired text is not conventional manual class labeling, but it is still a training signal and is more accurately described as weak/natural language supervision than as “no supervision” in every sense.

### SigLIP's pairwise logistic objective

CLIP normalizes scores over a batch with softmax. SigLIP uses pairwise sigmoid/logistic classification. For pair logit \(a_{ij}\) and \(y_{ij}\in\{+1,-1\}\), the generic logistic term is

\[
\ell_{ij}=\log(1+\exp(-y_{ij}a_{ij})).
\]

This expresses the form of the loss, not every detail of the full SigLIP recipe. The published implementation's scaling, bias and positive/negative weighting must be checked before reproducing the method. Pairwise sigmoid changes normalization and batch behavior; it does not remove the need for reliable positive-pair construction and negative sampling.

SigLIP 2 improves multilinguality, semantic understanding, localization and dense features through a combined recipe that includes data curation and additional training signals. It should not be presented as “just a sigmoid loss.”

References:
- CLIP: Radford et al. (2021), https://arxiv.org/abs/2103.00020
- ALIGN: Jia et al. (2021), https://arxiv.org/abs/2102.05918
- SigLIP: Zhai et al. (2023), https://arxiv.org/abs/2303.15343
- SigLIP 2: Tschannen et al. (2025), https://arxiv.org/abs/2502.14786

## 9. Speech SSL and multimodal target construction

- **wav2vec 2.0:** masks latent audio and identifies quantized target representations with a contrastive objective.
- **HuBERT:** predicts cluster-derived pseudo-labels at masked audio positions.
- **WavLM:** extends masked speech pretraining with denoising-oriented signals to broaden downstream speech representations.
- **data2vec:** predicts contextual teacher features and was framed across speech, vision and language.

Speech makes the target construction especially clear: waveform samples become a latent time sequence, and the objective predicts a quantized unit, cluster ID or contextual vector. The sequence resolution and clustering/quantization recipe are not incidental; they shape which distinctions can be learned.

## 10. Other modalities and methods to cross-reference

- BLIP/BLIP-2 and related vision-language systems combine image-text matching, text generation or query-based representation components. Check the book's later TOC before expanding them in this chapter.
- CPC is a historical bridge from latent future discrimination toward predictive representations, not the current centerpiece of visual SSL.
- I-JEPA/V-JEPA/V-JEPA 2/2.1 are latent prediction systems; Chapter 4 owns the full JEPA treatment.
- Diffusion and flow matching use generative objectives and belong with generation, even when pretrained on unlabeled observations.

## 11. Status vocabulary for the survey

- **Foundational:** establishes a mechanism or vocabulary used to understand later methods.
- **Representative:** clearly illustrates a distinct target-construction choice.
- **Modern large-scale system:** a later system whose results depend on scale, data and engineering as well as the core loss.
- **Frontier example:** a current paper pointing toward an active direction, not necessarily the dominant production method.
- **Historical bridge:** explains a transition but does not need equal space in a current overview.
- **Separate chapter:** belongs elsewhere because full coverage would duplicate or displace that chapter's subject.
