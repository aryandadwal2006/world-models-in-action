# Primary-paper recipe audit: objectives, update paths and replication risks

**Status: working audit, not a claim of exact reproduction.** This file is a method-by-method implementation checklist for turning the conceptual dossier into reliable teaching examples. It distinguishes \(a\) defining objective, \(b\) architecture and update mechanism, \(c\) recipe choices that affect results, and \(d\) claims that require verification against paper appendices or official code. A schematic equation below is not sufficient to reproduce an author's full benchmark.

The main design principle is to never report only a loss name. An SSL algorithm is a coupled specification: data distribution, view construction, target generation, representation heads, gradient paths, update rules, optimizer/schedule and evaluation protocol.

## 1. Contrastive classification: SimCLR and MoCo

### 1.1 SimCLR: a candidate-classification problem

Take a batch of \(B\) source examples and independently sample two views per example. This gives \(2B\) encoder outputs. Let \(h_i=f_	heta(v_i)\) be a backbone representation and \(z_i=g_	heta(h_i)\) be its projection-head output. The usual training similarity is cosine similarity of \(L_2\)-normalized \(z_i\), not necessarily the backbone vector \(h_i\) later used for transfer.

For a paired positive \(j\) corresponding to anchor \(i\), the directional NT-Xent loss is

\[
\ell_{i,j}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k=1}^{2B}\mathbf 1[k\ne i]\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
\]

The denominator contains every other view in the minibatch, including the positive. The self-comparison alone is excluded. With a two-view batch, each source example supplies two directed terms; a common reduction averages over all \(2B\) anchors. The numerator must not silently use a separate denominator that excludes the positive.

**Why the construction matters.** The objective trains a classifier to identify the other view of the same source example among candidates. It does not receive explicit semantic labels for “same object” or “same action.” The augmentation pipeline defines which changes count as irrelevant; the candidate construction defines which examples compete.

**Silent replication mismatches**
- Use \(B\) to describe the number of source examples but \(2B\) for the number of views. Papers and code sometimes use “batch size” differently.
- State whether similarity is computed on \(z\) or \(h\). The original SimCLR design found the nonlinear projection head important; transferring the loss to a backbone feature is a different experimental choice.
- Record augmentation order, crop distributions, color distortion, blur, normalization, temperature, optimizer, learning-rate scaling, total updates and effective global batch size.
- In distributed training, confirm whether negatives span all devices. Per-device negatives and globally gathered negatives create different candidate distributions.
- Report the feature used for transfer separately from the feature used in the loss.

**A useful historical reference configuration from the SimCLR paper.** In the paper's reported default setup for its controlled studies, the base encoder is ResNet-50 and the projection head is a two-layer MLP mapping to 128 dimensions. The stated default pretraining run uses batch size 4096 for 100 epochs, LARS, learning rate 4.8 \((0.3\times 4096/256)\), weight decay \(10^{-6}\), a 10-epoch linear warm-up, and cosine decay without restarts. The stated augmentation recipe combines random resized crop (with random horizontal flip), color distortion and Gaussian blur; the appendix provides exact parameters. The paper also aggregates batch-normalization statistics across devices, noting that local per-device statistics can leak information because positive pairs occur on the same device.

These values document the paper's ablation/reference configuration, not a universally optimal recipe, and should not be copied blindly to a small-compute project or another domain. They demonstrate the level of specificity needed when saying “we used SimCLR.” Source: Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, Algorithm 1 and the “Training with Large Batch Size” / “Evaluation Protocol” discussions, https://arxiv.org/abs/2002.05709.

### 1.2 MoCo: the same broad classification objective with a moving dictionary

A query encoder produces \(q=f_q(x)\). A key encoder produces a positive key \(k^+\) from the paired view, and a queue holds negative keys \(k_1,ldots,k_K\). A schematic query loss is

\[
\ell_q=-\log\frac{\exp(q^\top k^+/\tau)}
{\exp(q^\top k^+/\tau)+\sum_{r=1}^K\exp(q^\top k_r/\tau)}.
\]

The key encoder is updated by an exponential moving average of query-encoder parameters,

\[
\theta_k\leftarrow m\theta_k+(1-m)\theta_q,
\]

and keys are enqueued/dequeued to maintain a dictionary. The queue makes the negative set larger than a conventional minibatch without requiring all keys to be recomputed through the current encoder each step. The price is that queued features were generated by earlier key-encoder states.

The loss above is a structural summary, not the full implementation. A faithful reproduction must establish whether keys are normalized, which tensors are detached, the queue update order, key shuffling or batch-normalization handling, momentum schedule, queue length, distributed gathering and the exact query/key augmentation recipe. Do not assume MoCo v2 and v3 differ only by swapping a CNN for a ViT: the paper-specific architecture and optimization changes need their own audit.

### 1.3 False negatives and sampling assumptions

Contrastive methods treat sampled non-positive candidates as competitors. If a negative is semantically equivalent to the anchor for the intended task, the gradient may push it away despite that equivalence. This is not necessarily a software bug; it follows from the constructed classification task. Larger candidate sets can make discrimination harder and provide more negatives, but do not guarantee better semantic targets. Memory queues, all-gathered minibatches, hard-negative mining and nearest-neighbor positives change the sampling distribution and must be treated as method-defining or method-affecting choices.

For the theory note, distinguish four levels:
1. the finite-batch classification objective;
2. the population-optimal critic under a specified positive/negative sampler;
3. a mutual-information bound under its assumptions;
4. downstream usefulness, which is empirical and task-dependent.

Only the first is directly established by writing down the finite-batch loss.

## 2. Negative-free view prediction: BYOL and SimSiam

### 2.1 BYOL: the target is produced by a slowly moving encoder

The online branch has an encoder \(f_\theta\), projector \(g_\theta\), and predictor \(q_\theta\). The target branch has encoder/projector parameters \(\xi\), updated by EMA and kept out of the gradient path. For two views \(v_1,v_2\), one directional term is schematically

\[
\mathcal L_{1\to2}=2-2\left\langle
\frac{q_\theta(g_\theta(f_\theta(v_1)))}{\|q_\theta(g_\theta(f_\theta(v_1)))\|_2},
\operatorname{sg}\!\left(
\frac{g_\xi(f_\xi(v_2))}{\|g_\xi(f_\xi(v_2))\|_2}
\right)\right\rangle.
\]

The actual BYOL loss averages symmetric view directions and uses the paper's normalization convention. Here \(\operatorname{sg}\) means stop-gradient. The target parameters are updated as

\[
\xi\leftarrow m\xi+(1-m)\theta,
\]

with the exact parameter grouping and momentum schedule checked against the implementation.

**Interpretation.** BYOL does not use a queue of negatives. Its asymmetry comes from the predictor, stop-gradient target, and slow EMA target network. Removing negatives alone does not define BYOL, and adding an arbitrary EMA encoder to another loss does not make it BYOL.

**Audit before coding**
- Check whether the projector and predictor include batch normalization and the exact output dimensions.
- Check target normalization and cosine/MSE reduction; equivalent algebraic forms may have different numerical scales if normalization changes.
- Verify the target branch receives no gradient, and whether buffers (such as normalization running statistics) are updated or copied.
- Verify EMA momentum timing and schedule; the target update may occur before or after optimizer steps in the reference implementation.
- Preserve two-view symmetry and state whether the loss is summed or averaged.
- A training curve with a low loss is not evidence that the representation avoided dimensional collapse; inspect feature variance/rank and transfer.

### 2.2 SimSiam: stop-gradient without a momentum target

SimSiam shares an encoder/projector across both views and adds a predictor on the online side. A normalized negative-cosine term can be written

\[
\mathcal D(p,z)=-\frac{p}{\|p\|_2}\cdot
\operatorname{sg}\!\left(\frac{z}{\|z\|_2}\right),
\qquad
\mathcal L=\tfrac12[\mathcal D(p_1,z_2)+\mathcal D(p_2,z_1)].
\]

The target vector comes from the other view of the same encoder, detached for that loss branch. Unlike BYOL, there is no EMA target encoder. The predictor and stop-gradient are part of the intended optimization asymmetry; remove them and one no longer has the same method.

The SimSiam paper's empirical recipe, particularly normalization layers, matters. A bare equation is not proof of why collapse is avoided for every architecture or optimizer. When explaining the result, separate an empirical finding under the paper's setup from a general theorem for all shared-weight Siamese networks.

## 3. Teacher-student distribution matching: DINO and iBOT

### 3.1 DINO's target is a probability distribution

Let student and teacher produce logits \(u_s,u_t\) over \(K\) output dimensions. A student distribution is formed by a temperature-softmax; the teacher uses its own temperature and a center term. Schematically,

\[
p_s=\operatorname{softmax}(u_s/\tau_s),\qquad
p_t=\operatorname{softmax}((u_t-c)/\tau_t).
\]

One cross-view term is

\[
\ell_{t\to s}=-\sum_{k=1}^K p_{t,k}\log p_{s,k}.
\]

DINO sums selected teacher/student view pairings; original DINO uses teacher global crops and student global/local crops, not necessarily all ordered pairs. The teacher is an EMA of the student, while its center is a moving statistic used to control output imbalance. The temperature and centering/sharpening choices are method-relevant because they change the entropy and balance of the target distribution.

The schematic equation is deliberately not a drop-in reproduction: check the paper's exact view-pair exclusion, teacher-center update, warm-up schedule for teacher temperature, multi-crop construction, prototype output layer, optimizer and architecture. Do not conflate the class token/global feature and patch-token outputs used by different DINO-family variants.

### 3.2 iBOT adds masked patch-level teacher targets

iBOT combines a self-distillation family with masked image modeling. The important conceptual shift is that target construction is not limited to a global image representation: masked patch representations are also trained against teacher-generated targets. In any implementation note, distinguish:
- the class/global-token distillation term;
- the patch-level masked-token term;
- the online tokenizer/prototype output structure;
- teacher/student view and mask construction;
- the target update and centering/sharpening details.

Writing “iBOT is DINO with masking” is a useful memory aid, but insufficient as a method description or reproduction spec. The two target paths and their reductions should be shown explicitly from the paper.

## 4. Redundancy reduction and moment constraints

### 4.1 Barlow Twins

Let \(Z^A,Z^B\in\mathbb R^{B\times d}\) contain projection outputs for two views. Normalize each coordinate across the batch according to the reference method, then compute a cross-correlation matrix

\[
C_{ij}=\frac1B\sum_{b=1}^{B} Z^A_{b,i}Z^B_{b,j}.
\]

The objective is

\[
\mathcal L_{\mathrm{BT}}=
\sum_i(1-C_{ii})^2+
\lambda\sum_{i\ne j}C_{ij}^2.
\]

The diagonal term encourages same-coordinate cross-view agreement; the off-diagonal term penalizes cross-coordinate redundancy. This is a batch-statistic objective and has no explicit negative candidate class. The exact normalized activations and correlation computation matter: using unstandardized activations, a covariance matrix instead of the intended cross-correlation, or a different off-diagonal reduction changes its scale and behavior.

A diagonal cross-correlation condition is a second-order constraint, not a proof that the full embedding distribution is independent or Gaussian. Uncorrelated variables need not be independent without additional assumptions.

### 4.2 VICReg

VICReg uses embeddings \(Z^A,Z^B\) and combines three losses:

\[
\mathcal L=\lambda_{\mathrm{inv}}s(Z^A,Z^B)
+\lambda_{\mathrm{var}}[v(Z^A)+v(Z^B)]
+\lambda_{\mathrm{cov}}[c(Z^A)+c(Z^B)].
\]

A common invariance term is batch-averaged squared error. For a batch covariance matrix \(C\), a variance-floor term penalizes dimensions whose standard deviation falls below a threshold \(\gamma\), while a covariance term penalizes squared off-diagonal entries. One common form is

\[
v(Z)=\frac1d\sum_j\max(0,\gamma-\sqrt{C_{jj}+\epsilon}),
\qquad
c(Z)=\frac1d\sum_{i\ne j} C_{ij}^2.
\]

**Conventions to verify:** the reference paper's exact definitions, denominator \(B-1\) versus \(B\), epsilon placement, standard-deviation convention, whether the off-diagonal covariance sum is divided by feature dimension or another count, and the numeric loss coefficients. The expressions here identify the components and common normalization pattern; the chosen reduction is part of the training recipe.

A variance floor directly makes a constant feature dimension costly, but variance alone permits linearly redundant dimensions. The covariance term addresses second-order redundancy, while the invariance term ties corresponding views. Avoid saying VICReg makes all dimensions statistically independent: its covariance condition is not a general independence condition.

### 4.3 Why these families should not be collapsed into a single category

- Barlow Twins penalizes deviations of the cross-view cross-correlation from identity.
- VICReg explicitly separates invariance, variance floor and within-view covariance regularization.
- Whitening methods transform the feature distribution before a matching loss.
- SIGReg compares one-dimensional projected empirical distributions with a Gaussian target via characteristic-function-based statistics.
- VISReg decouples scale control and distribution-shape matching using a variance term and sliced-Wasserstein-style sketching.

All seek non-degenerate features in different senses. Their assumptions, invariance properties and gradients are not identical.

## 5. Masked reconstruction: MAE versus token/feature targets

### 5.1 MAE's asymmetry is architectural

For an image split into patches, the visible patch tokens are passed through the encoder; masked patches are omitted from this encoder computation. The decoder receives encoded visible tokens plus mask tokens in their original positions and predicts patch content. In the standard MAE objective, loss is evaluated on masked patches rather than uniformly over all input patches:

\[
\mathcal L_{\mathrm{MAE}}=
\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}
\|\hat x_i-x_i\|_2^2,
\]

where \(\mathcal M\) indexes masked patches. Details such as per-patch target normalization are configuration choices to verify in the paper/code. The high mask ratio and encoder/decoder asymmetry are key design choices, not cosmetic implementation details.

An MAE reconstruction loss answers a conditional prediction question about pixels under the patch/mask distribution. It is different from BEiT's discrete token prediction, MaskFeat's descriptor prediction, iBOT's teacher targets, or a JEPA's target-feature prediction. All may mask parts of an input while defining fundamentally different target variables.

### 5.2 The conditional-mean argument and its limit

For squared-error regression, the population minimizer is the conditional expectation:

\[
f^*(x_V)=\mathbb E[X_M\mid X_V=x_V].
\]

This follows by expanding the conditional squared risk around the conditional mean. It explains why a pixel-MSE reconstruction objective estimates a conditional mean, not necessarily a sharp or uniquely correct completion when many completions are possible. This is a statement about the regression target, not a claim that every trained neural network reaches the Bayes optimum.

The target distribution can reward pixel details that are predictable from context, including textures or dataset-specific correlations. Whether those features help the intended task must be measured rather than assumed.

## 6. Image-text matching: CLIP and SigLIP

### 6.1 CLIP: two retrieval directions over a batch

Let normalized image embeddings \(u_i\) and text embeddings \(v_j\) form logits

\[
S_{ij}=\exp(\alpha)u_i^\top v_j,
\]

where \(\alpha\) is a learned logit-scale parameter. For a batch of paired examples, the diagonal entries are matches. The standard loss averages image-to-text and text-to-image cross-entropies:

\[
\mathcal L_{\mathrm{CLIP}}=\tfrac12
\left[\operatorname{CE}(S,\mathrm{diag})+
\operatorname{CE}(S^\top,\mathrm{diag})\right].
\]

The equation is schematic about implementation details but preserves the central symmetry. State whether features are normalized and whether logit scale is clamped or otherwise constrained; verify global negative gathering and how duplicate captions/images are handled. The training target is pair association in a batch, not an ontology of image concepts. Noisy alt-text affects the quality of the positive-pair relation.

### 6.2 SigLIP: pairwise logistic discrimination

SigLIP changes the loss family from a batch-softmax normalization to pairwise sigmoid/logistic classification. A schematic form with \(y_{ij}\in\{-1,+1\}\) is

\[
\mathcal L_{\mathrm{sig}}=
-\sum_{i,j} w_{ij}\log\sigma
\left(y_{ij}(s\,u_i^\top v_j+b)\right),
\]

where diagonal pairs are positive, off-diagonal pairs are negative, and \(w_{ij}\) represents the implementation's reduction/weighting convention. The learned bias and scaling, negative-pair construction and loss normalization must be verified against the paper or official code before claiming exact equivalence.

The conceptual contrast is precise: CLIP's softmax couples all candidates in a row (and its transpose in the reverse direction), whereas pairwise sigmoid treats labels for image-text pairs through logistic terms and does not require the same global softmax denominator. It does not mean that negative sampling, batch composition or loss scaling cease to matter.

## 7. Modern predictive representation methods: don't overgeneralize from the name

### 7.1 JEPAs

A JEPA's defining idea is to predict target representations from context representations instead of necessarily reconstructing target pixels. But “latent prediction” is not a single loss specification. The target may be a region or a future space-time block; methods may differ in target encoder updates, predictor conditioning, target normalization, masks, view selection, aggregation, and whether actions are provided. Use the dedicated JEPA dossier for method-specific equations.

For every I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2.1, LeJEPA or VISReg statement, record:
- exact source version/date and code commit;
- target definition and target-encoder gradient/update mechanism;
- context/target mask or crop construction;
- loss domain (global, patch, token, space-time or intermediate layer);
- regularizer and its placement;
- whether action enters the predictor or training target;
- which paper claim is empirical versus proved under formal assumptions.

Do not conflate passive pretraining with action-conditioned post-training, or a regularizer's theoretical motivation with a guarantee on learned environmental state.

### 7.2 DINOv3 and V-JEPA 2.1 as recipe-level evidence

DINOv3's paper describes Gram anchoring as a way to address degradation of dense feature maps during long training, alongside data/model scaling and post-hoc flexibility. V-JEPA 2.1 describes dense predictive losses on visible and masked tokens, deep self-supervision at intermediate encoder layers, multimodal tokenizers, and scaling. These are useful frontier examples because they show that a strong global representation and strong dense spatial features may require additional recipe mechanisms. They are not evidence that one objective dominates across all modalities, datasets, compute budgets or evaluation targets.

Report the scope of the paper's benchmark suite and compare like-for-like data/model/compute and evaluation. A headline benchmark from a technical report is not itself an independent replication.

## 8. A controlled comparison protocol for the eventual teaching example

A fair experiment should change one major mechanism at a time and hold the data pipeline/evaluation protocol fixed where possible. The experiment log should record at least:

| Category | Minimum record |
|---|---|
| Data | dataset/version, split, deduplication, class/sequence balance, leakage checks |
| Views | crop/temporal sampling, augmentation probabilities and order, normalization |
| Model | encoder, projector/predictor/decoder, dimensions, normalization layers |
| Targets | target source, tokenization/clustering/teacher, stop-gradient location |
| Loss | formula, temperature, coefficients, masking scope, reductions, global/local batch |
| Updates | optimizer, schedule, EMA/momentum, warm-up, gradient clipping, mixed precision |
| Compute | hardware, effective batch, steps, wall-clock, seeds, checkpoint selection |
| Collapse | per-dimension standard deviation, covariance spectrum/effective rank, nearest-neighbor diversity |
| Transfer | frozen linear probe, k-NN, fine-tuning, spatial/dense metrics, data efficiency |
| Temporal use | held-out multi-step prediction, action-conditional tests where relevant, calibration and failure cases |
| Reproducibility | source version, official-code commit, configs, environment, license, known deviations |

Do not use one probe to stand in for the whole representation. A linear probe tests linearly accessible information under a specific label set; k-NN tests local neighborhood structure; fine-tuning tests adaptability with additional supervision; dense evaluation tests spatial precision; multi-step rollout tests predictive stability. For world-model claims, also test whether latent state aliases histories that imply different futures or rewards and whether predictions remain valid under alternative actions.

### Suggested ablations

1. **Objective vs recipe:** keep model, data, and total updates fixed; compare loss mechanisms before introducing paper-specific data-scale differences.
2. **Augmentation prior:** ablate the transformations that assume an attribute is irrelevant; report which downstream attributes are damaged.
3. **Feature location:** compare backbone, projector and predictor outputs separately.
4. **Global vs local features:** evaluate global classification and spatial localization rather than infer one from the other.
5. **Collapse modes:** report constant collapse and dimensional collapse separately.
6. **Temporal sufficiency:** compare one-step and multi-step error, then stratify by action, occlusion, object interaction and visually similar but dynamically distinct histories.
7. **Seed uncertainty:** report mean and spread over seeds where compute permits; distinguish instability from a small average performance difference.

## 9. Verification status ledger

“Standard form” means a conceptual equation consistent with the family, not an exact paper-faithful implementation. Before writing executable listings or claiming exact replication, verify each item directly against the source version to be taught.

| Family | Conceptual objective recorded | Exact-source checks still required |
|---|---|---|
| SimCLR | Core objective verified at pinned official objective.py | Two-view packing, optional L2 normalization, same-view self-mask, cross-view positives, two-direction loss sum and cross-replica gathering verified; exact run config/temperature and experimental recipe remain separate checks |
| MoCo v1/v2/v3 | Core MoCo v1/v2 model verified at last code-bearing upstream revision | Query/key paths, queue logits, key EMA ordering, batch shuffle/unshuffle, temperature scaling and queue update order verified; upstream README and LICENSE conflict, so redistribution status needs legal review; v3 remains to be audited separately |
| BYOL | Core stop-gradient regression and target-parameter EMA verified | Two cross-view regression directions; target parameter EMA follows the online optimizer update; mutable target state is separate from parameter EMA; exact experiment schedule and normalization settings still need a run-level record |
| SimSiam | Core predictor, stop-gradient and loss verified | Official builder shows predictor MLP and detached target outputs; training script averages two negative-cosine directions; BatchNorm placements verified for this pinned implementation |
| DINO/iBOT | Partial/schematic | view-pairing rules, centers, schedules, separate class/patch losses |
| Barlow Twins | Core correlation loss and reductions verified | Non-affine BatchNorm, cross-device cross-correlation aggregation, diagonal squared error plus lambda-weighted off-diagonal squared sum verified; defaults recorded below |
| VICReg | Core loss and reductions verified at pinned official main_vicreg.py | Local paired-view MSE plus globally gathered centered statistics; variance uses sqrt(var + 1e-4); covariance uses batch_size minus one and off-diagonal squared sums divided by feature dimension; defaults recorded below |
| MAE | Yes | target normalization option, masking distribution and loss scaling |
| CLIP | Yes | learned scale behavior, distributed negatives, exact symmetric CE reduction |
| SigLIP | Schematic | exact bias/scale, sample weights, loss normalization and batch scheme |
| JEPA family | Deliberately delegated | method/version-specific target, masking, predictor and loss details |

## 10. Primary references and source discipline

- SimCLR: Chen et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations*, https://arxiv.org/abs/2002.05709.
- MoCo: He et al. (2020), *Momentum Contrast for Unsupervised Visual Representation Learning*, https://arxiv.org/abs/1911.05722; v2, https://arxiv.org/abs/2003.04297; v3, https://arxiv.org/abs/2104.02057.
- BYOL: Grill et al. (2020), *Bootstrap Your Own Latent*, https://arxiv.org/abs/2006.07733.
- SimSiam: Chen and He (2021), *Exploring Simple Siamese Representation Learning*, https://arxiv.org/abs/2011.10566.
- DINO: Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers*, https://arxiv.org/abs/2104.14294.
- iBOT: Zhou et al. (2022), *iBOT: Image BERT Pre-Training with Online Tokenizer*, https://arxiv.org/abs/2111.07832.
- Barlow Twins: Zbontar et al. (2021), https://arxiv.org/abs/2103.03230.
- VICReg: Bardes et al. (2022), https://arxiv.org/abs/2105.04906.
- MAE: He et al. (2022), https://arxiv.org/abs/2111.06377.
- CLIP: Radford et al. (2021), https://arxiv.org/abs/2103.00020.
- SigLIP: Zhai et al. (2023), https://arxiv.org/abs/2303.15343.
- DINOv3: Siméoni et al. (2025), https://arxiv.org/abs/2508.10104.
- V-JEPA 2.1: Mur-Labadia et al. (2026), https://arxiv.org/abs/2603.14482.

The surrounding dossier retains source hierarchy, theoretical derivations, and supplementary-method coverage. This file is an implementation audit ledger and should remain separate from the reader-facing prose until the equations and chosen implementation conventions have been checked against the primary sources.


## 11. Pinned-code findings verified against original implementations

These findings are source inspections, not executed reproductions. Code settles implementation questions such as gradient boundaries, normalization, reduction denominators and update order; it does not prove that a published experiment has been reproduced.

### SimCLR

Pinned repository: google-research/simclr, revision 383d4143fd8cf7879ae10f1046a9baeb753ff438.

- Objective: https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/objective.py
- Call site: https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/model.py
- Projection and schedule utilities: https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/model_util.py

The objective receives 2B feature vectors, optionally L2-normalizes each row and splits them into two views. It calculates within-view and cross-view similarity matrices. Self-comparisons in each within-view matrix are masked with a very large negative value. The positive is located in the cross-view candidate block; one softmax-cross-entropy loss is computed for each direction and the two scalar losses are added. With TPU context, representations are gathered across replicas and positive labels are offset by replica index; otherwise candidates are local. The objective function's default temperature does not identify an actual paper run because the model call site passes a configured flag.

### MoCo v1/v2

Pinned repository: facebookresearch/moco, revision 132aea9da9bb995a411c915c0420be3db62052f3.

- Model, queue and forward pass: https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/moco/builder.py
- Trainer and options: https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/main_moco.py

The builder initializes key parameters from query parameters and disables key gradients. It maintains a normalized feature queue and circular pointer. The pinned builder defaults to feature dimension 128, queue size 65,536, momentum 0.999 and temperature 0.07. During forward, it computes the query; under no-gradient execution it momentum-updates the key encoder, batch-shuffles the key images for BatchNorm, encodes and normalizes keys, and undoes the shuffle. Positive logits are query-key dot products; negatives are query-versus-detached-queue products. The concatenated logits are divided by temperature, and new keys are enqueued after current logits are calculated. This order determines which negatives are available for a batch.

**Provenance and license warning:** upstream commit 8976944da9c6b94cbd9158d7ebe50912aef807ef deleted the repository tree. The pinned revision is its last code-bearing parent. At that snapshot the README states CC-BY-NC 4.0 while LICENSE states MIT; do not infer redistribution rights from one file alone. MoCo v3 is a separate paper/code line and is not verified by this v1/v2 audit.

### BYOL

Pinned repository: google-deepmind/deepmind-research, revision f5de0ede8430809180254ee957abf36ed62579ef, subdirectory byol/.

- Loss and update loop: https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/byol_experiment.py
- Schedules: https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/utils/schedules.py

The loss has two directions: online prediction of view 1 regresses to the stopped-gradient target projection of view 2, and vice versa. Gradients are taken with respect to online parameters; those parameters are updated before the target parameter decay is applied:

\[
\xi_{t+1}=\tau_t\xi_t+(1-\tau_t)\theta_{t+1},
\qquad \tau_t=1-(1-\tau_0)d_t.
\]

Here \(d_t\) is the schedule's decay value, so a single base momentum does not specify a full run. The target model's mutable state is carried and returned separately from its parameter tree; the EMA expression is an operation on parameters, not every state item. The implementation also trains an evaluation classifier using a stopped-gradient backbone embedding, so that classifier loss does not propagate into the representation backbone.

### SimSiam

Pinned repository: facebookresearch/simsiam, revision a7bc1772896d0dad0806c51f0bb6f3b16d290468.

- Projector, predictor and detached outputs: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py
- Training objective and defaults: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py

The projector uses linear layers with BatchNorm/ReLU in hidden stages and affine-free BatchNorm at its output. The two-layer predictor has a BatchNorm/ReLU hidden stage. Forward returns predictor outputs and detached target projections. The training script computes the symmetric negative-cosine objective

\[
L=-\frac12\left[\cos(p_1,\operatorname{sg}(z_2))+
\cos(p_2,\operatorname{sg}(z_1))\right].
\]

Each cosine term is averaged over the batch in code. This explicit stop-gradient is not a momentum target network. The defaults include a 2,048-dimensional projection and a 512-dimensional predictor hidden layer; command-line overrides govern individual runs.

### Barlow Twins

Pinned repository: facebookresearch/barlowtwins, revision 8e8d284ca0bc02f88b92328e53f9b901e86b4a3c.

- Standardization, correlation and loss: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py

Projected outputs are standardized by BatchNorm with affine parameters disabled. The cross-correlation is the standardized-view matrix product; each device divides its local contribution by the configured global batch size before an all-reduce sums device contributions. The loss is a sum of diagonal squared deviations from one plus lambda times the sum of squared off-diagonal entries:

\[
L=\sum_i(C_{ii}-1)^2+\lambda\sum_{i\ne j}C_{ij}^2.
\]

The pinned entry point defaults include batch size 2,048, 1,000 epochs, projector widths 8,192-8,192-8,192 and lambda 0.0051. These defaults must not replace recorded command-line overrides.

### VICReg

Pinned repository: facebookresearch/vicreg, revision 4e12602fd495af83efd1631fbe82523e6db092e0.

- Loss and reductions: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py

The paired-view representation MSE is computed before cross-device gathering. Variance/covariance statistics use globally gathered, separately centered views. Standard deviation uses the framework's sample-variance default with 0.0001 added inside the square root. Each view's variance penalty is the feature-wise mean of max(0, 1 minus standard deviation), and the two view penalties are averaged. Covariance is \(Z^\top Z/(B-1)\); its penalty sums squared off-diagonal entries and divides by the feature dimension, with the two view penalties added. The total objective is

\[
L=\alpha L_{\mathrm{repr}}+\beta L_{\mathrm{std}}+\gamma L_{\mathrm{cov}}.
\]

The pinned defaults are coefficients (25, 25, 1), global batch 2,048 and projector widths 8,192-8,192-8,192. This code's representation MSE is local before gathering, whereas its dispersion and covariance terms use gathered batch statistics.

### Next verification pass

Core loss and update conventions for these methods are now source-checked. Continue with DINO/iBOT and MAE/CLIP/SigLIP, marking exact recipe values separately from defaults and distinguishing source inspection from an executed reproduction.


## 12. Source-backed verification additions: DINO, iBOT, MAE, CLIP and SigLIP

- **DINO:** pinned facebookresearch/dino revision 7c446df5b9f45747937fb0d72314eb9f7b66930a; main_dino.py. Confirmed teacher sees two global crops, student sees all crops, same-index crop pairs are excluded, student logits use student temperature, teacher softmax uses centered logits and scheduled temperature, center uses distributed batch mean with EMA, and teacher parameter EMA follows the optimizer update.
- **iBOT:** pinned bytedance/ibot revision da316d82636a7a7356835ef224b13d5f3ace0489; main_ibot.py and models/head.py. Confirmed distinct class/patch centers and temperature schedules, masked patch loss normalized per image by masked-patch count, class loss excludes same-view pairs, and separate center EMA updates.
- **MAE:** pinned facebookresearch/mae revision efb2a8062c206524e35e47d04501ed4f544c0ae8; models_mae.py and main_pretrain.py. Confirmed random-noise-sort masking, visible-only encoder, sequence restoration, masked-patch pixel MSE, 0.75 default mask ratio, and optional patch normalization disabled by default.
- **CLIP:** pinned openai/CLIP revision d05afc436d78f1c48dc0dbf8e5980a9d471f35f6; clip/model.py. Confirmed normalized image/text embeddings and exponentiated learned logit scale initialized from log(1/0.07). The public pinned tree does not include the full pretraining loop; symmetric cross entropy is recorded as the paper objective rather than a repository training function.
- **SigLIP:** pinned google-research/big_vision revision 0127fb6b337ee2a27bf4e54dea79cff176527356; trainers/proj/image_text/siglip.py, models/proj/image_text/two_towers.py and configs/proj/image_text/siglip_lit_coco.py. Confirmed image-text logits with learned scale/bias, diagonal-positive/off-diagonal-negative signs, log-sigmoid of signed logits, sum over candidates and mean over query rows. The inspected configuration is a minimal 5,000-step COCO/LiT example and freezes the image tower; it is not the full-scale SigLIP recipe.


## 13. DINOv2/DINOv3 source verification

- DINOv2 is pinned at revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8: class-token loss at dinov2/loss/dino_clstoken_loss.py, masked patch loss at dinov2/loss/ibot_patch_loss.py, nearest-neighbour spread regularizer at dinov2/loss/koleo_loss.py, combined student/teacher wiring at dinov2/train/ssl_meta_arch.py, and defaults at dinov2/configs/ssl_default_config.yaml. Verified support for centering and Sinkhorn-Knopp target construction, DINO/iBOT loss branches and KoLeo negative log nearest-neighbour distance. Pinned defaults include DINO/iBOT weights 1.0 and KoLeo weight 0.1.
- DINOv3 is pinned at revision 6876159a11b4df116f30f667f8c9888617df0751: class-token, iBOT patch and Gram losses under dinov3/loss/. Verified GramLoss forms teacher/student within-view Gram matrices and MSE-matches them; optional normalization and negative-similarity filtering change that target. The standard pretraining config disables Gram loss; the Gram-anchor config enables it. The latter changes other recipe settings as well, so its outcome cannot be attributed to Gram loss alone without controlled ablation.
- Licensing: DINOv3 source and weights refer to a dedicated DINOv3 license agreement; do not assume DINOv2's Apache-2.0 terms automatically apply.


## 14. JEPA/V-JEPA/EB-JEPA source audit

- I-JEPA: pinned facebookresearch/ijepa revision 52c1ae95d05f743e000e8f10a1f3a79b10cff048. Verify target/context block sampling, target feature normalization, predictor position encoding and EMA in the exact config.
- V-JEPA / V-JEPA 2 / 2.1: pinned facebookresearch/vjepa2 revision 204698b45b3712590f06245fbfba32d3be539812. The repository has distinct paths for the V-JEPA and V-JEPA 2.1 recipes. In the inspected 2.1 trainer the target encoder executes without gradients, target features are layer-normalized, the predictor receives context and prediction mask indices, the loss exponent of 1.0 yields elementwise absolute error, and target EMA follows the optimizer step. The inspected config uses dense context-token prediction plus distance weighting; it contains placeholder datasets and calls for a large multi-node run.
- EB-JEPA: pinned facebookresearch/eb_jepa revision 966e61e9285b3a876f49b9774e9720d9a99a7925. The trainable class adds prediction cost to a configurable anti-collapse regularizer and supports parallel/autoregressive rollout. The included action-conditioned example uses an observation encoder, action encoder, predictor and goal-latent planning; the simple video example uses Moving MNIST. These toy examples should not be presented as web-scale experiments.


## 15. DINO parser-default discrepancy

The pinned official DINO main script, revision 7c446df5b9f45747937fb0d72314eb9f7b66930a, sets the argument warmup_teacher_temp_epochs to 0. Its adjacent help string says “Default: 30”. These conflict. This dossier records 0 as the executable parser default and flags the help-text inconsistency; do not use the help string alone to reconstruct a training run. Check the actual launch command and configuration for any reported result.
