# DINOv2, DINOv3 and Gram anchoring: objectives, geometry and verified source details

This note separates the shared teacher-student framework from changes introduced by particular releases and training recipes. DINOv2 combines DINO-style class-token distillation, iBOT-style masked patch distillation and a KoLeo feature-spreading regularizer. DINOv3 retains related distillation mechanisms and provides optional Gram-matrix anchoring. The Gram term is not enabled by every DINOv3 configuration.

## 1. DINOv2 is a combined objective, not one new loss

A useful decomposition is
\[
L=L_{\mathrm{DINO}}+\lambda_{\mathrm{iBOT}}L_{\mathrm{iBOT}}
+\lambda_{\mathrm{KoLeo}}L_{\mathrm{KoLeo}}.
\]
The configuration determines which terms are active and their weights. In the pinned DINOv2 default SSL configuration, the DINO and iBOT loss weights are 1.0 and the KoLeo weight is 0.1. The class-token and patch-token terms train different relations: DINO matches teacher distributions across crops, while iBOT matches the teacher patch distribution at masked student positions.

### 1.1 Class-token distillation and teacher targets

Let the student produce prototype logits \(s_\theta(v)\in\mathbb R^K\) and the teacher produce \(u_\xi(v)\in\mathbb R^K\). A centered and sharpened teacher target is
\[
q_\xi(v)=\operatorname{softmax}\left((u_\xi(v)-c)/T_t\right).
\]
The student minimizes cross entropy \(-\sum_kq_k\log p_k\), with \(p=\operatorname{softmax}(s/T_s)\). For a fixed target, the derivative with respect to student logit \(s_k\) is \((p_k-q_k)/T_s\). The teacher target is detached, while teacher parameters are updated by EMA.

The pinned implementation supports two distinct target-normalization choices:

**Centering.** Subtract an exponential moving average of teacher logits and apply softmax. This controls persistent average output bias but does not directly enforce uniform use of every prototype.

**Sinkhorn-Knopp assignment.** Convert teacher logits to nonnegative weights, then iteratively normalize prototype rows and sample columns. The iterations approximate balanced assignment marginals: each sample's assignment sums to one, while total mass for each prototype is pushed toward an equal share. This discourages dominant prototype usage by construction but also imposes a balancing prior. The two methods are not synonyms; changing this option changes target construction.

### 1.2 Masked patch distillation

For each masked patch \(r\), let \(q_r\) be the detached teacher distribution and \(p_r\) the student prediction. The patch loss is
\[
L_{\mathrm{patch}}=
\frac{1}{B}\sum_{i=1}^B
\frac{\sum_r m_{ir}\left(-\sum_kq_{ir,k}\log p_{ir,k}\right)}
{\max(1,\sum_r m_{ir})}.
\]
Here \(m_{ir}=1\) for masked positions. Per-image division by the number of masked positions prevents images with more masks from automatically receiving proportionally more weight.

DINOv2 supports centered softmax and Sinkhorn-Knopp teacher targets in the patch branch as well. With Sinkhorn-Knopp, the assignment balancing applies to the masked-patch set handled by that implementation; it is not simply subtracting a running center.

### 1.3 KoLeo: regularize neighbourhood geometry

The KoLeo term is inspired by nearest-neighbour estimators of differential entropy. Let \(z_i\) be a normalized student backbone embedding and \(j(i)\) its nearest neighbour under the implementation's distance convention. For \(d_i=\|z_i-z_{j(i)}\|_2\), the regularizer is
\[
L_{\mathrm{KoLeo}}=-\frac{1}{B}\sum_i\log(d_i+\epsilon).
\]
Minimizing the negative log distance encourages nearest-neighbour distances not to become too small. It spreads embeddings on the normalized feature sphere and reduces local crowding.

This is not a semantic clustering objective and does not mathematically guarantee better class accuracy. It acts on backbone representation geometry, not just prototype distributions. Its effect depends on its coefficient, nearest-neighbour definition, normalization and batch sampling.

## 2. DINOv2: source and configuration audit

The official DINOv2 repository is pinned at revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8.

- Class-token loss and Sinkhorn: [dino_clstoken_loss.py](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/dino_clstoken_loss.py)
- Masked patch loss: [ibot_patch_loss.py](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/ibot_patch_loss.py)
- KoLeo regularizer: [koleo_loss.py](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/koleo_loss.py)
- Combined architecture: [ssl_meta_arch.py](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/train/ssl_meta_arch.py)
- Pinned default configuration: [ssl_default_config.yaml](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/configs/ssl_default_config.yaml)
- Paper: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

The pinned default configuration uses 65,536 output prototypes for each DINO and iBOT head, DINO/iBOT weights of 1.0, KoLeo weight 0.1 and centering as the default target normalization, with Sinkhorn-Knopp available as an alternative. The configuration uses mask-sampling probability 0.5, mask-ratio range 0.1–0.5 and eight local crops alongside two global crops. These are defaults for this pinned config, not universal DINOv2 constants or values for every released checkpoint.

The architecture contains a student and a teacher, with teacher parameter gradients disabled. It can share a head for class-token and patch-token outputs or use a separate iBOT head depending on config. A reproduced experiment should record the config, dataset, batch, schedule, centering-vs-Sinkhorn mode and head-sharing decision rather than cite only the method name.

## 3. DINOv3 keeps the core distillation structure and extends selected recipes

The pinned DINOv3 source contains class-token DINO loss, iBOT patch loss and KoLeo. The configuration may change prototype dimensions, masking, normalization, architecture and schedules. A method family can retain the same broad loss families while changing data, architecture, scale or optional regularizers. A benchmark change cannot be attributed to a single loss without controlled comparisons.

### 3.1 Gram anchoring from first principles

A feature vector describes an individual patch; a Gram matrix describes similarities among a set of feature vectors. Let teacher patch features be \(T\in\mathbb R^{N\times d_t}\), and student features be \(S\in\mathbb R^{N\times d_s}\). Their feature dimensions need not match. Construct within-view similarity matrices:
\[
G_T=TT^\top,\qquad G_S=SS^\top.
\]
If features are normalized first, entries are cosine similarities. A Gram loss is
\[
L_{\mathrm{Gram}}=\operatorname{MSE}(G_S,G_T).
\]
This preserves relational geometry: it constrains how patches relate to one another rather than requiring equality between individual teacher and student feature coordinates.

The pinned implementation supports image-level Gram matrices and a mode that flattens features across the batch. It casts features to float before computing similarities. When enabled, normalization is along the feature dimension. Configuration flags can clip negative similarities in both matrices or remove only teacher-side negatives; these options change the relational target. A generic description that Gram loss always matches every signed cosine similarity would therefore be incorrect.

A relational loss has limitations. Similar Gram matrices can occur without preserving correct semantic or physical variables. Gram anchoring does not independently prove that the embedding is a sufficient state for a world model.

### 3.2 Gram anchoring is not enabled in every DINOv3 recipe

The official DINOv3 repository is pinned at revision 6876159a11b4df116f30f667f8c9888617df0751.

- Standard pretraining configuration: [dinov3_vit7b16_pretrain.yaml](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_pretrain.yaml)
- Gram-anchor configuration: [dinov3_vit7b16_gram_anchor.yaml](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml)
- Gram implementation: [gram_loss.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py)
- Class-token loss: [dino_clstoken_loss.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/dino_clstoken_loss.py)
- Patch loss: [ibot_patch_loss.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/ibot_patch_loss.py)
- Paper: [DINOv3: Self-supervised learning for vision at unprecedented scale](https://arxiv.org/abs/2508.10104)

The standard pinned pretraining YAML disables Gram loss. The Gram-anchor YAML enables it, normalizes features, computes Gram matrices per image, uses a larger teacher crop of 512 pixels and applies a scheduled Gram-loss weight. The config also differs in training duration and other recipe settings, so comparing final models without controlled ablations does not isolate the effect of Gram anchoring.

The class-token loss uses teacher/student cross entropy over crop pairs and can exclude diagonal pairs; the source supports Sinkhorn-Knopp target normalization. The iBOT branch provides masked patch prediction with its own loss path. These are still teacher-distribution prediction losses, not pixel reconstruction.

### 3.3 Scale, data and licensing are part of the method report

DINOv2 and DINOv3 are training systems, not only loss functions. Architecture (including register/storage tokens and position encoding), dataset composition, masking and crop distributions, teacher schedules, precision/sharding and training duration influence the learned representations.

The pinned DINOv3 repository refers to a dedicated DINOv3 license agreement rather than automatically inheriting DINOv2's Apache-2.0 terms. Read the exact upstream license and model-card constraints before using or redistributing source or weights.

## 4. What the progression does and does not establish

| Component | Direct target | What it adds | What it does not prove |
|---|---|---|---|
| DINO class loss | Teacher prototype distribution across crops | Global view consistency | Semantic classes or Markov sufficiency |
| iBOT patch loss | Teacher patch distribution at masked sites | Dense local targets | Correct object boundaries or dynamics |
| KoLeo | Nearest-neighbour feature distance | Reduced feature crowding | Correct semantic organization |
| Sinkhorn-Knopp | Balanced prototype assignment | Controls prototype marginal usage | That uniform prototype usage is the true data prior |
| Gram anchoring | Teacher/student patch-similarity matrices | Relational geometry preservation | Correct physical state or planning reliability |

For world-model work, evaluate whether latent states preserve hidden velocity, contact, object identity, reward-relevant context and action-dependent futures. Representation metrics are diagnostics; controlled multi-step prediction and planning under interventions are separate evidence.

## 5. Source and reproduction ledger

The pins in this note verify source identity and code conventions, not executed reproduction. For each claimed result, retain the exact upstream revision, full experiment config, dataset and preprocessing, batch and schedule, checkpoint selection, evaluation protocol and deviations. DINOv3 source and weights have separate license implications; method-level similarity does not erase them.
