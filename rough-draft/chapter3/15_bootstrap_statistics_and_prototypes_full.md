# Bootstrap, self-distillation, statistics and prototypes: first principles

This file expands the non-contrastive methods. The most important conceptual distinction is between **what target the loss asks the network to match** and **what mechanism makes nontrivial representations viable during optimization**. Those are separate questions.

## 1. A generic two-view objective

Let \(v_1,v_2\) be views of the same source. The online representation is \(h_\theta(v)=f_\theta(v)\), optionally followed by a projector \(g_\theta\). A target branch may share parameters, use a detached output, or evolve as a separate moving-average network.

A naive agreement objective is

\[
L_{\mathrm{inv}}=\mathbb{E}\left[\lVert h_\theta(v_1)-h_\theta(v_2)\rVert_2^2\right].
\]

For any constant mapping \(h_\theta(v)=c\), the loss is exactly zero. So invariance alone does not identify a meaningful representation. The remaining sections use different strategies to address this issue.

## 2. BYOL: regression to a slow-moving target

BYOL has an online encoder/projector \(z_\theta(v)=g_\theta(f_\theta(v))\) and a predictor \(q_\theta\). A target encoder/projector \(z_\xi(v)\) uses a separate parameter vector \(\xi\). The target is detached from gradient propagation.

One directional loss can be written with normalized features as

\[
L_{\theta,\xi}(v_1,v_2)
=2-2\frac{q_\theta(z_\theta(v_1))^\top z_\xi(v_2)}
{\lVert q_\theta(z_\theta(v_1))\rVert_2
 \lVert z_\xi(v_2)\rVert_2}.
\]

The complete recipe normally adds the reverse direction and averages. The online branch receives gradient descent updates; target parameters follow EMA:

\[
\xi_{t+1}=m_t\xi_t+(1-m_t)\theta_{t+1}.
\]

The target schedule may vary with training. A moving target introduces a second timescale: the online network tracks a slowly changing target rather than chasing a target that changes arbitrarily fast.

### Why the predictor matters

The predictor adds a learnable map between the online projection and the target representation. This changes the optimization geometry. It also allows the online representation space to be mapped into target space without forcing the two intermediate representations to be identical at each step. Removing the predictor often damages performance or leads to collapse in canonical recipes; this empirical fact should not be mistaken for a theorem about every predictor architecture.

### What EMA does not prove

For any fixed mapping \(z_\theta(v)=c\), the normalized target can also be constant, and the cosine loss can be zero when the nonzero constant directions match. Therefore a collapsed solution exists. BYOL's practical success depends on the full architecture, predictor, stop-gradient target, EMA dynamics, normalization, augmentations and optimization. A valid explanation must separate:
1. the set of possible loss minimizers;
2. fixed points of gradient/EMA update equations;
3. the local stability of those fixed points;
4. the empirical behavior under a specified recipe and dataset.

A proof about one linearized or normalized setting cannot silently be extended to arbitrary nonlinear networks.

## 3. SimSiam: stop-gradient with shared parameters

SimSiam shares its encoder and projector across views. One branch passes through a predictor. The other branch's target representation is stop-gradient. The symmetric objective swaps the views:

\[
L=\tfrac12 D(p_1,\operatorname{sg}(z_2))
+\tfrac12 D(p_2,\operatorname{sg}(z_1)),
\]

where \(p_i=q_\theta(z_i)\), and a common distance is negative cosine similarity. The exact sign/convention of D is defined by the implementation; the essential point is that gradients do not flow through the target side of each term.

The stop-gradient operation is not algebraically equivalent to detaching a constant target before both branches are optimized together. It changes which partial derivatives are included in the update. This breaks the symmetry of ordinary two-sided regression.

SimSiam does not use the EMA target-update equation that defines BYOL. The two methods share predictor and target-gradient ideas but should not be described as identical. Canonical SimSiam performance also relies on implementation details such as normalization and batch normalization. An analysis that removes those components is analyzing a different algorithm.

## 4. DINO: distribution matching over prototypes

DINO maps two views through student and teacher networks to vectors of prototype logits. For view v, write student distribution

\[
p_s^{(v)}=\operatorname{softmax}\left(g_s(v)/\tau_s\right)
\]

and teacher distribution

\[
p_t^{(v')}=\operatorname{softmax}\left((g_t(v')-c)/\tau_t\right),
\]

where \(\tau_s,\tau_t\) are temperatures and c is a running center estimated from teacher outputs. The teacher target is stop-gradient. A cross-entropy term is

\[
L(v,v')=-\sum_{k=1}^{K}p_t^{(v')}(k)\log p_s^{(v)}(k).
\]

The full recipe matches selected teacher/student views (including multi-crop configurations). The teacher is updated through EMA; it is not directly optimized by this cross-entropy.

Centering and sharpening act in related but distinct ways. Centering subtracts a slowly moving average of teacher logits, discouraging output concentration in a small set of prototypes. A lower teacher temperature sharpens the target distribution, making its assignment more concentrated. The method can be described as self-distillation because its teacher targets are generated by a learned network on another view, not by an external class label.

The prototype output can be interpreted as an online clustering-like assignment, but DINO's defining mechanism is teacher-student distribution matching. This overlapping description is more accurate than insisting that all methods belong to one exclusive family.

### DINO v1, DINOv2 and DINOv3 are not interchangeable recipes

- **DINO v1 (2021):** introduced self-distillation with multi-crop and found emergent structure in ViT features.
- **DINOv2 (2023):** combines the lineage with a large carefully curated image dataset, engineering and recipe changes; data selection is part of the contribution.
- **DINOv3 (2025):** aims at high-quality visual features at unprecedented scale, with specific attention to dense-feature quality and Gram anchoring. It carries a method-specific license.

The correct comparison asks what changed in data, architecture, objective, regularization, resolution, training horizon and evaluation. It is not sufficient to attribute improvements to a single loss name.

## 5. Barlow Twins: cross-view correlation as the target

Take two batches \(Z^A,Z^B\in\mathbb{R}^{B\times d}\). Standardize every feature dimension across the batch to approximately zero mean and unit variance, obtaining \(\widetilde Z^A,\widetilde Z^B\). Define cross-correlation:

\[
C=\frac1B(\widetilde Z^A)^\top\widetilde Z^B,\qquad
C_{ij}=\frac1B\sum_{b=1}^{B}\tilde z^A_{bi}\tilde z^B_{bj}.
\]

The Barlow Twins loss drives C toward identity:

\[
L_{\mathrm{BT}}=\sum_i(1-C_{ii})^2+
\lambda\sum_{i\ne j}C_{ij}^2.
\]

Diagonal terms encourage corresponding feature dimensions to agree across views. Off-diagonal terms discourage cross-dimension redundancy. The coefficient \(\lambda\) trades the two terms.

This objective expresses a second-order statistical goal. Identity cross-correlation is not a general proof of statistical independence; uncorrelated variables can still be dependent. Nor does it prove that each coordinate has an interpretable semantic meaning. The estimates also depend on batch size and standardization. A small batch produces noisy covariance estimates, and almost constant dimensions raise numerical questions for normalization.

## 6. VICReg: invariance, variance and covariance are different terms

For two feature batches \(Z,Z'\in\mathbb{R}^{B\times d}\), define invariance:

\[
s(Z,Z')=\frac{1}{Bd}\lVert Z-Z'\rVert_F^2.
\]

Let \(\sigma_j(Z)=\sqrt{\operatorname{Var}_b(Z_{bj})+\epsilon}\) be dimension j's empirical standard deviation. With target margin \(\gamma>0\),

\[
v(Z)=\frac1d\sum_{j=1}^{d}\max(0,\gamma-\sigma_j(Z)).
\]

Let \(\operatorname{Cov}(Z)\) be the empirical covariance matrix. The covariance penalty is

\[
c(Z)=\frac1d\sum_{i\ne j}\operatorname{Cov}(Z)_{ij}^{2}.
\]

VICReg combines these with coefficients:

\[
L_{\mathrm{VICReg}}=\lambda s(Z,Z')+
\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')].
\]

For exact constant features, the invariance term can be zero, but the variance term is positive whenever a dimension falls below the positive margin. That term makes exact collapse costly. Above the threshold the variance penalty becomes inactive; it is not an incentive for unbounded feature scale. Covariance penalization reduces linear redundancy, while invariance relates the paired views.

None of these terms tells the encoder which variation is physically meaningful. A batch can have high variance and low cross-covariance while representing nuisance texture instead of the hidden state a controller needs. Explicit anti-collapse is necessary to interpret many non-contrastive objectives, but it is not sufficient for downstream validity.

## 7. W-MSE and whitening

W-MSE (whitening-based mean squared error) is an important adjacent method because it changes the feature geometry through whitening rather than using explicit negatives. At a high level, paired views are mapped to features and transformed using a whitening operation so feature covariance is closer to identity; matching then occurs in the whitened representation. This discourages a representation from concentrating all variation in a small subspace.

Whitening estimates batch covariance \(\Sigma\), regularizes it for numerical stability, and applies a matrix such as \((\Sigma+\epsilon I)^{-1/2}\). This operation rescales and rotates directions based on the empirical second-order statistics. It can be numerically unstable when covariance estimates are noisy or near-singular; batch size, feature dimension, regularization and distributed-statistic aggregation matter.

Do not conflate W-MSE with Barlow Twins or VICReg. They are related by their treatment of cross-view agreement and feature statistics but use different transformations and loss formulations. Use the original W-MSE paper for its exact whitening convention and update placement before implementing a numerical demonstration.

## 8. DeepCluster: clustering alternates with network training

DeepCluster uses an iterative loop:
1. Compute feature vectors for the current dataset using the current encoder.
2. Run a clustering algorithm, conventionally k-means, over the feature set.
3. Treat the cluster IDs as pseudo-labels.
4. Train the network to predict those IDs.
5. Repeat, refreshing clusters as the embedding space changes.

Why can it work? If features contain meaningful similarities, clustering exposes recurring structures; classification on those pseudo-labels trains the encoder to make those structures more stable. Why can it fail? If initial features are poor, the cluster assignments may reflect nuisance variation. Clustering is an extra offline stage. Cluster sizes can be imbalanced, empty clusters may need handling, and IDs are arbitrary up to permutation.

A pseudo-label is not automatically a semantic label. The loop bootstraps from its current representation and may amplify its own mistakes. Inspect assignments and downstream tasks rather than treating cluster purity as ground truth.

## 9. SwAV: online assignment and swapped prediction

SwAV learns prototype vectors \(c_1,\ldots,c_K\). Given a feature z, prototype scores are functions of similarity to these vectors. A normalized score vector defines probabilities over prototypes. SwAV obtains target assignments using a Sinkhorn-Knopp procedure that approximately imposes balance constraints across the batch, then trains a network view to predict the assignment generated from another view. The “swapped” part is the cross-view direction of prediction, not the mere use of clusters.

Balancing discourages an all-to-one-prototype solution. This is an inductive prior rather than a fact about every dataset. In a naturally long-tailed data distribution, equal assignment mass per prototype may not reflect the underlying classes. The number of prototypes, temperature, assignment normalization, multi-crop recipe and batch structure all affect the result.

SwAV therefore sits between contrastive and clustering views: it uses prototype assignments but does not require the same instance-level negative set as SimCLR. Its loss can be read as cross-entropy between an assignment from one view and a predicted distribution from another.

## 10. MeanShift, MSN and prototype-related neighbors

These adjacent methods illustrate variants of the same design space:
- **MeanShift-style SSL:** organizes features through density/neighbor relations and mean-shift-like clustering rather than relying only on independent instance labels.
- **MSN (Masked Siamese Networks):** uses prototype/assignment-based prediction with a teacher/anchor-style setup and masking, with design changes intended to improve learning on unbalanced data.
- **iBOT:** combines the DINO family of teacher distribution targets with masked patch-level prediction. It should be placed at the intersection of teacher-student, prototype, and masked-input methods.
- **NNCLR:** identifies nearest-neighbor feature targets; it can broaden positive relations but inherits neighbor errors.

These method names belong in the landscape register. Before putting an exact equation or implementation recipe in the final manuscript, verify the corresponding primary paper and official code; family resemblance is not enough to infer an objective.

## 11. Comparative equation table

| Family | Target | Positive mechanism | Non-collapse / diversity mechanism | Primary failure mode |
|---|---|---|---|---|
| SimCLR | Candidate index | Two views of same source | Competition against sampled negatives | False negatives, batch dependence |
| MoCo | Candidate index | Query/key views | Negatives from queue, slow key encoder | Stale keys and false negatives |
| BYOL | Target feature | Online prediction of EMA target | Predictor and target/update asymmetry | Collapse remains a possible solution |
| SimSiam | Detached branch feature | Prediction across paired views | Stop-gradient, predictor and normalization | Recipe-sensitive collapse |
| DINO | Teacher probability distribution | Cross-view teacher/student matching | Centering, sharpening, EMA teacher | Assignment concentration or collapse |
| Barlow Twins | Identity cross-correlation | Paired views | Off-diagonal correlation penalty | Moment constraints do not imply semantic sufficiency |
| VICReg | Paired feature vectors | Direct invariance term | Variance floor and covariance penalty | Nuisance variation still satisfies statistics |
| W-MSE | Whitened feature | Paired-view regression | Whitening of feature covariance | Noisy/singular covariance estimates |
| DeepCluster | Cluster ID | Same example receives current cluster assignment | Clustering and pseudo-label prediction | Self-reinforcing bad clusters |
| SwAV | Prototype assignment | Predict the other view's assignment | Balanced Sinkhorn assignment | Balance prior can conflict with long-tail data |
| iBOT | Teacher target at masked patch | Same-image patch/crop relations | Teacher target, prototypes, masking | Target/teacher biases |
| MSN / MeanShift variants | Prototype/neighborhood target | Assignment or neighbor relation | Distribution/assignment constraints | Sensitivity to prototype and data priors |
