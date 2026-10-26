# Teacher distributions, masked targets and cross-modal alignment: first-principles derivations

This note develops DINO, iBOT, MAE, CLIP and SigLIP from the training signals they construct. It separates paper-level mathematics from details confirmed in the pinned official code. These methods use targets constructed without ordinary per-example class labels, but they are not interchangeable: DINO distils global view distributions, iBOT adds masked patch-token distillation, MAE reconstructs masked pixels, CLIP performs batchwise image-text retrieval, and SigLIP classifies image-text pairs with a sigmoid objective.

## 1. Teacher-student learning as distribution matching

Suppose an image \(x\) generates transformed views \(v_1=t_1(x)\) and \(v_2=t_2(x)\). A student network produces logits \(s_\theta(v)\in\mathbb R^K\), and a teacher produces \(u_\xi(v)\in\mathbb R^K\). The \(K\) outputs are coordinates of a learned output space; they need not be human class labels.

The student and teacher probability distributions are
\[
p_\theta(k\mid v)=
\frac{\exp(s_\theta(v)_k/T_s)}
{\sum_{j=1}^{K}\exp(s_\theta(v)_j/T_s)},\qquad
q_\xi(k\mid v)=
\frac{\exp((u_\xi(v)_k-c_k)/T_t)}
{\sum_{j=1}^{K}\exp((u_\xi(v)_j-c_j)/T_t)}.
\]
Here \(T_s,T_t\) are temperatures and \(c\) is an optional teacher center. The distillation cross entropy is
\[
\ell(q,p)=-\sum_{k=1}^{K}q_k\log p_k.
\]
For fixed \(q\), this equals \(D_{\mathrm{KL}}(q\|p)+H(q)\). Since \(H(q)\) does not depend on the student, minimizing cross entropy and minimizing that KL divergence have the same student optimum.

For student logits \(s_j\), the derivative is
\[
\frac{\partial\ell}{\partial s_j}=\frac{p_j-q_j}{T_s}.
\]
The gradient pushes student probability toward the teacher distribution. It does not explain how a useful target distribution arises: DINO and iBOT construct targets from another network branch, and use temperature and centering to control that target distribution.

### 1.1 Teacher asymmetry and exponential moving averages

DINO-style methods stop gradients through teacher outputs and update teacher parameters with an exponential moving average (EMA) of student parameters. If \(\theta_{t+1}\) denotes the student parameters after the gradient update,
\[
\xi_{t+1}=m_t\xi_t+(1-m_t)\theta_{t+1}.
\]
The teacher is a smoothed sequence of student parameter values, not a model trained directly on human labels. Larger \(m_t\) changes the teacher more slowly. EMA alone does not prove collapse is impossible; its effect depends on the target distribution, architecture, views and optimization.

### 1.2 Centering and sharpening are distinct

Centering subtracts a running estimate of average teacher logits:
\[
\bar u=u-c,\qquad
c_{t+1}=\mu c_t+(1-\mu)\operatorname{mean}_{\mathrm{batch,devices}}(u).
\]
It counteracts persistent global preferences for some output coordinates but does not ensure balanced assignments for each image.

Sharpening uses a lower teacher temperature. Smaller \(T_t\) concentrates softmax probability around the largest logits; larger \(T_t\) produces softer targets. An excessively sharp teacher can turn early errors into extreme targets, while overly soft targets can provide weak learning signals. Temperature schedules are optimization choices, not guarantees about representation quality.

## 2. DINO: distil between different image views

### 2.1 Objective

DINO (Caron et al., 2021) applies teacher-student distribution matching across crops of the same image. Common training uses two global crops and optional local crops. The teacher processes the two global crops; the student processes global and local crops. If \(q_1,q_2\) are detached teacher distributions from the global views and \(p_v\) are student distributions for crop \(v\), then
\[
L_{\mathrm{DINO}}=
\frac{1}{|\mathcal P|}
\sum_{(i,v)\in\mathcal P}
-\sum_{k=1}^{K}q_i(k)\log p_v(k),
\]
where \(\mathcal P\) contains teacher/student crop pairs and excludes the same-index comparison in which teacher and student see the same global crop.

That exclusion matters. The student predicts a distribution from a different view rather than simply matching a target computed on the identical input. Output coordinates behave like learned prototypes or clusters, but they are not supplied semantic classes. The objective can discover useful visual groupings without guaranteeing alignment to a human ontology.

### 2.2 Crop consistency is an inductive bias

Global crops preserve broad scene content but change scale and composition. Local crops can omit context. Training the student on local crops to predict teacher distributions from global crops pressures the representation to remain consistent when context changes. This is useful when local evidence identifies shared content; it can fail when a crop removes evidence needed to distinguish objects, interactions or physical states.

DINO is not ordinary class-label classification and not merely cosine-distance matching between two vectors. Its softmax distributions and cross entropy establish competition among output coordinates.

### 2.3 Pinned implementation findings

The official DINO repository is pinned at revision 7c446df5b9f45747937fb0d72314eb9f7b66930a.

- Main training and loss: [main_dino.py](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py)
- The pinned repository uses Apache-2.0 licensing.

The source confirms that the teacher receives only the two global crops while the student receives all crops. Student logits are divided by the student temperature, which defaults to 0.1 in this source; teacher probabilities use centered logits and a scheduled teacher temperature. Same-index teacher/student global pairs are skipped, teacher targets are detached, and the scalar loss is averaged across remaining pairings. The center is updated from distributed mean teacher logits using an EMA coefficient that defaults to 0.9. In the training loop, teacher parameter EMA follows the student optimizer step, and the momentum is scheduled toward one with cosine scheduling. This file defaults to 100 epochs and sets warmup_teacher_temp_epochs to 0, even though its help string says “Default: 30”. Treat the parser value as the code default; a particular run can override it through arguments or configuration.

These findings describe the pinned implementation, not every DINO-family variant or the recipes in later DINOv2/DINOv3 releases.

### 2.4 Failure modes

Centering and temperature schedules help control target dynamics, but do not prove collapse impossible. A noncollapsed prediction space need not preserve velocity, contact, object identity or action effects. A successful linear probe measures linearly accessible information for selected labels; it does not certify that the latent is sufficient for controlled prediction.

## 3. iBOT: global distillation plus masked patch distillation

### 3.1 Two targets and two losses

iBOT (Zhou et al., 2022) combines DINO-like class-token loss with masked image modeling in the teacher's output space. Let the student produce class-token logits \(s^{\mathrm{cls}}\) and patch-token logits \(s^{\mathrm{patch}}_r\), where \(r\) indexes image patches. The teacher produces \(u^{\mathrm{cls}}\) and \(u^{\mathrm{patch}}_r\). Each teacher output is converted to a centered, temperature-scaled distribution:
\[
q^{\mathrm{cls}}=\operatorname{softmax}\left(\frac{u^{\mathrm{cls}}-c^{\mathrm{cls}}}{T_{\mathrm{cls}}}\right),\quad
q^{\mathrm{patch}}_r=\operatorname{softmax}\left(\frac{u^{\mathrm{patch}}_r-c^{\mathrm{patch}}}{T_{\mathrm{patch}}}\right).
\]

The class loss is cross-view cross entropy as in DINO. The patch loss compares teacher and student patch distributions at masked positions:
\[
L_{\mathrm{patch}}=
\frac{1}{B}\sum_{i=1}^{B}
\frac{\sum_{r=1}^{P}m_{ir}
\left[-\sum_{k=1}^{K_p}q^{\mathrm{patch}}_{ir,k}
\log p^{\mathrm{patch}}_{ir,k}\right]}
{\max(1,\sum_{r=1}^{P}m_{ir})},
\]
where \(m_{ir}=1\) indicates a masked patch and \(P\) is the number of patches. Per-example normalization prevents images with more masked patches from receiving proportionally greater weight.

The combined objective is
\[
L_{\mathrm{iBOT}}=\lambda_{\mathrm{cls}}L_{\mathrm{cls}}
+\lambda_{\mathrm{patch}}L_{\mathrm{patch}}.
\]
This is not pixel reconstruction: patch tokens reproduce teacher distributions, not raw RGB values. It is also not just global distillation: patch-level outputs provide a dense signal for masked locations.

### 3.2 Why masking location matters

The student uses visible patches and a mask specification to infer removed positions. The teacher supplies target distributions from the corresponding unmasked image view. A masked location receives direct cross-entropy supervision through student patch logits; teacher targets are detached. Visible tokens provide context for prediction.

### 3.3 Pinned implementation findings

The official iBOT repository is pinned at revision da316d82636a7a7356835ef224b13d5f3ace0489.

- Training and loss: [main_ibot.py](https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/main_ibot.py)
- Output heads: [models/head.py](https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/models/head.py)

The source confirms distinct class and patch centers and teacher-temperature schedules. The class loss matches teacher global crops to student crops except for same-index pairs. The patch loss applies to the corresponding global crop and is weighted by the patch mask; each image's patch loss is divided by the count of masked patches, with a denominator clamp. The two losses are weighted by lambda1 and lambda2. The class center is updated from mean teacher class outputs; the patch center uses teacher patch outputs averaged over patches. They have separate EMA coefficients. This version supports a delayed masked-modeling start and independent temperature schedules. Its default loss weights are 1.0; teacher temperatures default to 0.04 for class outputs and 0.07 for patch outputs, though experiments may override these.

### 3.4 Distinguish DINO, iBOT and MAE

DINO asks whether a teacher distribution from one crop can predict a student distribution on another crop.

iBOT asks that for the global class output and additionally requires masked patch outputs to predict teacher patch distributions.

MAE asks whether visible pixels and spatial context can reconstruct missing pixel patches with a decoder.

The word masked does not determine the target type. A target can be pixels, quantized visual tokens, teacher features or distributions over learned prototypes. The target is a central part of the method.

## 4. MAE: reconstruct pixels only where information was removed

### 4.1 Random masking and encoder/decoder asymmetry

Split an image into \(P\) patches and keep a fraction \(1-\rho\), where \(\rho\) is the mask ratio. MAE samples a random permutation; the encoder processes only visible patch embeddings plus a class token. It therefore does not spend encoder compute on mask tokens. The decoder receives encoded visible tokens, appends learned mask tokens, restores sequence order using the inverse permutation, adds positional embeddings and predicts one pixel-vector per patch.

For patch size \(p\times p\) and RGB targets, each output patch has \(3p^2\) pixel values. A high mask ratio forces prediction from sparse spatial evidence. It can make reconstruction less local, but it cannot guarantee the encoder learns a semantic or dynamical state.

### 4.2 Exact masked-patch loss

Let \(x_r\in\mathbb R^{3p^2}\) be the flattened pixel target for patch \(r\), \(\hat x_r\) the decoder prediction and \(m_r=1\) for removed patches. MAE minimizes
\[
L_{\mathrm{MAE}}=
\frac{\sum_{r=1}^{P}m_r\,\frac{1}{3p^2}
\|\hat x_r-x_r\|_2^2}
{\sum_{r=1}^{P}m_r}.
\]
The average pixel squared error is first calculated per patch; the final loss is averaged over removed patches. Visible patches do not contribute to this objective.

Some configurations normalize pixel targets within each patch:
\[
\tilde x_r=
\frac{x_r-\operatorname{mean}(x_r)}
{\sqrt{\operatorname{var}(x_r)+\epsilon}}.
\]
In the pinned code this operation is optional and defaults to false. When enabled, the implementation adds \(10^{-6}\) before taking the square root. Do not assume a paper discussion of normalized-pixel loss applies to every run.

### 4.3 Pinned implementation findings

The official MAE repository is pinned at revision efb2a8062c206524e35e47d04501ed4f544c0ae8.

- Architecture, mask/unshuffle and loss: [models_mae.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py)
- Pretraining arguments: [main_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py)
- Training iteration: [engine_pretrain.py](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py)

Masking is sampled independently per image by sorting random noise. The encoder sees only visible patch tokens. The decoder appends mask tokens, restores original positions and predicts pixel vectors. Loss is MSE only on removed patches. Mask ratio defaults to 0.75; normalized pixel targets are opt-in and disabled by default. The encoder is deeper/larger than the decoder in the standard configurations, an intentional asymmetry as well as a compute decision.

### 4.4 What reconstruction demonstrates

Low reconstruction loss establishes that the decoder can infer the specified target from visible content under the training distribution. It does not show that the encoder preserves every detail, is minimal, or ensures two histories sharing the same latent have the same action-conditioned future. The decoder can use its own capacity, positional priors and image statistics. Evaluate future-state, reward and action-effect prediction separately for world-model applications.

## 5. CLIP: image-text contrastive classification over a batch

### 5.1 Batchwise candidate classification

Suppose a batch contains \(B\) paired images and texts \((I_i,T_i)\). Let \(f(I_i)\) and \(g(T_i)\) be image and text embeddings, normalized to unit length:
\[
z_i=\frac{f(I_i)}{\|f(I_i)\|_2},\qquad
w_i=\frac{g(T_i)}{\|g(T_i)\|_2}.
\]
CLIP forms
\[
S_{ij}=\alpha z_i^\top w_j,\qquad \alpha>0.
\]
The matching pair is on the diagonal. Each image row is an image-to-text classification task, and each text column is a text-to-image task.

The symmetric contrastive loss is
\[
L_{\mathrm{CLIP}}=
\frac12\left[
\frac1B\sum_i\left(-S_{ii}+\log\sum_j e^{S_{ij}}\right)
+
\frac1B\sum_i\left(-S_{ii}+\log\sum_j e^{S_{ji}}\right)
\right].
\]
The diagonal pair is encouraged to outrank other candidates in both directions. Nonmatching images and texts in the minibatch become negatives, so the objective depends on batch composition; it is not just a distance loss on matched pairs.

The scale \(\alpha\) changes the sharpness of both softmax distributions. CLIP represents the log scale as a learned scalar initialized to \(\log(1/0.07)\), then exponentiates it in the model forward pass. This parameter is distinct from similarity itself.

### 5.2 The text pairing defines the inductive bias

Captions can convey objects, attributes, relations, activities and abstract concepts. The positive relation differs from image augmentation or instance identity. Captions can also omit objects, be ambiguous or be weakly aligned. A diagonal-only target may treat another valid caption as a negative when that relation is absent from the batch labels. The method learns the pairing structure in the data; it does not directly learn a complete ontology of image semantics.

### 5.3 Source boundary: architecture is available, training loop is not

The pinned OpenAI CLIP repository is d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.

- Model normalization, logit scale and pairwise logits: [clip/model.py](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py)
- Official repository: [openai/CLIP](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)
- Paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

At this pinned revision, the public repository provides model architecture, pretrained-model loading and inference/scoring utilities, but not the full pretraining loop for the paper's dataset-scale training. Normalized embeddings and the learned scale are verified in code. The symmetric cross-entropy above is the paper-level objective and must not be labelled a reproduced training function from this repository. Any third-party training loop should be labelled and audited as third-party.

## 6. SigLIP: independent pairwise sigmoid losses

### 6.1 Replace softmax competition with binary pair labels

CLIP uses a row-wise softmax: candidate text probabilities for an image sum to one. SigLIP uses an independent binary objective for every image-text pair in the batch. Let \(z_i,w_j\) be image and text embeddings and define
\[
y_{ij}=\begin{cases}+1,&i=j,\\-1,&i\ne j,\end{cases}
\qquad
a_{ij}=\alpha z_i^\top w_j+b.
\]
The pairwise logistic loss is
\[
L_{\mathrm{SigLIP}}=
-\frac{1}{B}\sum_{i=1}^{B}\sum_{j=1}^{B}
\log \sigma(y_{ij}a_{ij}),
\]
where \(\sigma(a)=1/(1+e^{-a})\). Each pair contributes a binary log-likelihood term. The reduction sums over candidate texts for each image row, then averages rows. There is no row-softmax normalizer.

For one pair,
\[
\frac{\partial \ell}{\partial a}=-y\sigma(-ya).
\]
Gradient descent raises a positive logit and lowers a negative logit. Correctly classified pairs with high confidence receive smaller gradients than ambiguous or misclassified pairs. SigLIP is not contrastive learning without negatives: nonmatching pairs are explicit negative targets. The difference is the pairwise sigmoid objective rather than a softmax over candidates.

### 6.2 Scale and bias have distinct roles

The logit scale \(\alpha\) changes how strongly embedding similarity affects logits. The bias \(b\) shifts all logits and changes the classifier's operating point. In a batch with many negatives relative to positives, a negative initial bias can help prevent high initial probabilities for every pair. The inspected Big Vision configuration sets temperature_init to 10.0 and bias_init to -2.71. These are settings for that configuration, not universal constants of SigLIP.

### 6.3 Pinned implementation findings and limitations

The shared Google Research Big Vision repository is pinned at revision 0127fb6b337ee2a27bf4e54dea79cff176527356.

- SigLIP trainer: [siglip.py](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py)
- Two-tower model: [two_towers.py](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/models/proj/image_text/two_towers.py)
- Example configuration: [siglip_lit_coco.py](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/configs/proj/image_text/siglip_lit_coco.py)
- Paper: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

The trainer computes the image-text score matrix, multiplies it by the model-provided scale, adds the model-provided bias, labels diagonal entries positive and off-diagonal entries negative, and computes log-sigmoid of signed logits. It sums negative log-likelihood across candidates for each image and averages over rows.

Important scope caveat: the pinned siglip_lit_coco configuration is a minimal COCO-caption/LiT-style example, not a standalone reproduction of full web-scale pretraining. It defaults to 5,000 steps, uses COCO captions and has a schedule entry freezing the image tower. Do not present it as evidence that the full-scale SigLIP recipe has been reproduced.

## 7. Comparison by target and objective

| Method | Target constructed from | Loss family | Direct pressure |
|---|---|---|---|
| DINO | Teacher distribution for a different crop | Softmax cross entropy | Cross-view agreement of learned output distributions |
| iBOT | Teacher class distribution plus teacher patch distributions at masked positions | Class and patch cross entropy | Global crop agreement and masked patch distillation |
| MAE | Original patch pixels | Masked mean squared error | Reconstruction of removed patches from visible context |
| CLIP | Diagonal image-text pairs in a paired batch | Symmetric row/column softmax cross entropy | Matched pairs should outrank other batch candidates |
| SigLIP | Diagonal positives and off-diagonal negatives | Pairwise sigmoid log-likelihood | Every image-text pair should receive the correct binary sign |

This describes direct training signal, not guaranteed downstream capability. It does not imply DINO learns only semantics, MAE learns only low-level features, or CLIP understands every caption relation.

## 8. World-model implications

Each loss retains or discards information according to its target relation and training distribution. Teacher-distribution consistency encourages crop invariance; masked patch distillation adds dense local targets; pixel reconstruction pressures the decoder to recover visual detail; language alignment can make caption-related attributes accessible; pairwise sigmoid learning changes how positives and negatives are normalized. None establishes the conditional state contract required for controlled prediction:
\[
P(O_{t+1},R_t\mid H_t,A_t)
=P(O_{t+1},R_t\mid Z_t,A_t).
\]

A world-model evaluation should test visually similar histories that differ in hidden velocity, contact, object identity or action consequences. Measure one-step and multistep prediction, action-conditional alternatives, calibration and planner performance. A linear probe, successful crop matching, pixel reconstruction or image-text retrieval is useful evidence but does not prove Markov sufficiency or reliable planning.

## 9. Source discipline

Keep four records separate: (1) the paper's mathematical objective; (2) implementation behavior at an immutable upstream revision; (3) the exact configuration used for a cited result; and (4) evidence that a specific run was executed and evaluated correctly. The pinned code paths above support source inspection. They do not establish executed reproduction; for CLIP, the official pinned repository does not provide the complete pretraining loop.
