# SSL landscape and chapter boundaries

## 1. The field is a set of target-construction choices

Self-supervised learning is often introduced as learning without labels. The phrase hides a necessary detail: the training procedure still needs a target, comparison, or constraint. The data pipeline constructs that supervision. A transformed image can supply a transformation label; a second view can supply a positive pair; another instance can supply negatives; an exponential-moving-average teacher can supply a target distribution; clustering can produce pseudo-labels; masking can define missing values; text can be paired with an image; speech can be quantized into units.

The major families are best organized by the answer to “what signal is the encoder asked to satisfy?” They are not mutually exclusive. DINO combines self-distillation with prototype probabilities. iBOT combines masked input with teacher-student feature prediction. wav2vec 2.0 combines masking, learned quantization and contrastive discrimination. V-JEPA is masked latent prediction on video, and V-JEPA 2 adds a separate action-conditioned post-training stage for control. A taxonomy should be a map of dominant ingredients, not a partition of the entire literature.

## 2. The historical route to modern visual representations

### 2.1 Hand-designed pretext tasks

Rotation prediction, jigsaw puzzles, context prediction and colorization established the principle that an input can generate its own supervision. They were useful conceptual precursors. Their specific tasks can reward cues that are correlated with the transformation but incidental to the downstream use. For example, predicting image rotation may exploit object-orientation statistics. These tasks are not the best organizing center for current large-scale representation learning, but they make the target-design problem concrete.

### 2.2 Instance discrimination and contrastive learning

The contrastive family makes the target relational. A positive pair should be similar; other items in the denominator compete with the positive. SimCLR made the augmented-view recipe explicit and showed the importance of a large batch and a nonlinear projection head in its setting. MoCo used a momentum key encoder and queue to maintain a large dictionary without requiring every negative to be computed in the current batch. MoCo v2 improved the recipe; MoCo v3 explored the regime of vision transformers and should be described as an empirical architecture/training study, not simply as “MoCo with a newer version number.” CPC moved the candidate discrimination objective to future latent representations. It is conceptually important to predictive representation learning, but its use as a current visual backbone recipe is not comparable to the reach of DINOv2/v3, MAE, or modern masked/teacher-student objectives.

The objective carries two design burdens: selecting invariances and constructing a useful comparison set. A positive pair only means “equivalent” under the data construction. A negative pair only means “not this positive” under the sampling scheme. Neither label is a universal semantic truth.

### 2.3 Removing explicit negatives

BYOL predicts a target encoder's representation from an online branch using an extra predictor and a stop-gradient target; the target parameters are an exponential moving average of the online parameters. SimSiam removes the moving target update and studies a simpler stop-gradient Siamese architecture. This family shifts the central question from “which negative dictionary?” to “which asymmetry and optimization dynamics prevent collapse in practice?” Avoid the oversimplification that stop-gradient itself proves anti-collapse. A constant representation can make a simple view-invariance loss small; actual behavior depends on the complete training system, including predictor, normalization, architecture, target update and optimization.

DINO matches a student distribution to a teacher distribution over learned prototypes. Centering and sharpening alter the teacher distribution and help avoid degenerate assignments. DINOv2 scales the self-distillation recipe with large-scale curated data and engineering. DINOv3 (2025) focuses explicitly on scaling and the preservation of dense features; Gram anchoring is introduced to address degradation of feature maps during long schedules. This is the modern visual foundation-model line to surface in the chapter, while acknowledging that scale, data, distillation and postprocessing are all part of the result.

### 2.4 Statistical anti-collapse constraints

Barlow Twins standardizes outputs and drives the cross-correlation matrix between views toward the identity matrix. The diagonal terms encourage corresponding dimensions to agree; off-diagonal terms penalize cross-dimension redundancy. VICReg separates invariance, variance and covariance into explicit loss terms. Its variance term penalizes dimensions whose batch standard deviation falls below a threshold; the covariance term reduces off-diagonal covariance. The conceptual gain is interpretability: anti-collapse is not an emergent side effect of an elaborate negative dictionary but a stated condition in the objective. The caveat is equally important: finite-batch variances and covariances are estimates, and decorrelated dimensions are not necessarily semantically meaningful or sufficient for control.

### 2.5 Clusters and prototypes

DeepCluster alternates between extracting features, clustering them with k-means and training the network to predict the resulting pseudo-labels. It makes semantic partitioning an explicit intermediate variable, but the pseudo-labels can be noisy and the cluster identities can permute. SwAV predicts the cluster assignment of one view from another view, using online prototypes and Sinkhorn-Knopp to obtain balanced assignments. It avoids constructing explicit instance-level negative pairs and avoids a slow offline clustering loop. DINO also uses prototype distributions but is primarily described as teacher-student self-distillation; the taxonomy should acknowledge that categories overlap.

### 2.6 Masked prediction

BERT masks text tokens and predicts the original tokens from bidirectional context. GPT uses causal autoregressive next-token prediction, so it is self-supervised language modeling but not masked language modeling. These two language-model objectives demonstrate a general lesson: the data's ordering and corruption process define the permissible context. They are useful contrast cases but should not overwhelm a chapter focused on visual representations for world models.

For images, MAE encodes only visible patches and reconstructs masked pixels using an asymmetric decoder, often with a high masking ratio. BEiT predicts discrete visual-token targets generated by a tokenizer. iBOT couples masked patch prediction with teacher-student distillation. The differences are not merely decoder choices: they determine whether the target is raw pixel detail, discrete code identity, or teacher-produced contextual features. Under squared error, a decoder trained on ambiguous targets tends toward a conditional mean, which can be visually smooth and still fail to guarantee task-relevant features.

ELECTRA is another important target-design contrast: corrupt tokens with a generator, then train a discriminator to detect replaced tokens rather than predict the original token only at masked positions. data2vec predicts contextualized teacher representations from masked inputs and was designed as a cross-modality framework. These belong in the survey landscape even if the chapter does not derive all their objectives.

### 2.7 Cross-modal and speech objectives

CLIP aligns image and text embeddings with a contrastive objective, learning a shared space in which paired captions and images rank above mismatched items. ALIGN also scales noisy image-text pairs. SigLIP replaces global softmax contrastive normalization with pairwise sigmoid losses; do not describe it as merely a temperature change to CLIP. These methods supply semantic supervision through naturally paired modalities, but retrieval alignment does not guarantee local grounding, causal understanding or physical-state sufficiency.

wav2vec 2.0 masks latent speech frames, quantizes learned targets, and uses a contrastive objective to identify the correct quantized representation among distractors. HuBERT predicts offline cluster assignments for masked audio segments, iteratively improving the units. data2vec predicts contextualized target representations. The same broad questions recur: what defines the target, which context is visible, and which distinctions can be discarded?

## 3. Where the frontier sits

The frontier is not one universally best loss. It includes scaling strong self-distillation (DINOv3), increasingly spatially faithful features, and joint-embedding predictive architectures for images and video. I-JEPA predicts target-block representations rather than pixel values. V-JEPA extends feature prediction to video. V-JEPA 2 reports strong visual understanding and anticipation; the V-JEPA 2-AC system adds action-conditioned post-training for robotic control. These are not to be collapsed into the same claim: passive video pretraining, prediction of visual latent targets, and action-conditioned control are distinct capabilities and stages.

The main Chapter 3 should give these frontier methods a compact, accurate map and a clear handoff. Chapter 4 will do the detailed JEPA work. Diffusion and flow-matching models generate samples through a different family of generative objectives and belong where the book treats generation; they should not be misclassified as standard discriminative SSL objectives simply because they can also be pretrained on unlabeled data.

## 4. What the chapter must not claim

- No objective guarantees a Markov-sufficient physical state from pixels.
- Low pretraining loss does not establish downstream utility.
- A high linear-probe score is evidence of linearly accessible information, not proof that all relevant information is present.
- A method's name or category does not determine whether it is action-conditioned.
- Better benchmark accuracy does not isolate the effect of a loss if data, scale, backbone, augmentation, compute and recipe all change.
- The current prevalence of a method is not established by the original paper's own evaluation. Use “foundational,” “widely used,” “modern line,” or “frontier example” only when those descriptions are justified by broad literature and released model families.
