# Loss-gradient notebook: what receives a gradient, and why

This is an equation-first companion to the architecture dossiers. The purpose is to make the learning signal explicit for each method family. Formulas are written in a simple convention; implementation-specific reductions, temperature placement, normalization and batch-statistic conventions must be matched to the chosen primary paper/code.

## 1. The generic softmax cross-entropy gradient

Let logits \(a\in\mathbb R^K\) define

\[
p_k=\frac{e^{a_k}}{\sum_j e^{a_j}}.
\]

For one-hot target index y, cross-entropy is \(L=-\log p_y\). Since

\[
L=-a_y+\log\sum_j e^{a_j},
\]

we obtain

\[
\frac{\partial L}{\partial a_k}=p_k-\mathbf 1[k=y].
\]

This identity is the common gradient for candidate classification, masked-token classification, discrete visual-token prediction, teacher-student distillation with a fixed target distribution, and many prototype-prediction objectives. The target distribution differs by method, but the derivative has the same structure.

For a soft target distribution q, such as a teacher's probabilities,

\[
L=-\sum_k q_k\log p_k
\quad\Longrightarrow\quad
\frac{\partial L}{\partial a_k}=p_k-q_k.
\]

The gradient moves student probabilities toward the target distribution. If q is detached, no gradient is propagated through q. This is the main local derivative of DINO-style student cross-entropy, where q is the teacher distribution.

## 2. InfoNCE/NT-Xent: candidate logits and feature gradients

For anchor representation q, candidate vectors \(z_1,\ldots,z_K\), positive index y and temperature \(\tau\), let

\[
a_k=\frac{s(q,z_k)}{\tau}.
\]

Then

\[
\frac{\partial L}{\partial s(q,z_k)}
=\frac{p_k-\mathbf 1[k=y]}{\tau}.
\]

With unnormalized dot product \(s(q,z_k)=q^\top z_k\),

\[
\nabla_q L=\frac1\tau\left(\sum_{k=1}^{K}p_kz_k-z_y\right).
\]

So the query is pulled toward the positive, but its exact direction is corrected by a softmax-weighted average of all candidates. High-probability negatives dominate this average more than easy negatives.

For normalized vectors, this derivative must pass through normalization. Let \(u=q/\|q\|_2\). For nonzero q,

\[
\frac{\partial u}{\partial q}
=\frac1{\|q\|_2}(I-uu^\top).
\]

Thus gradients on the raw feature are projected onto the tangent space orthogonal to u and scaled by feature norm. The exact gradient differs from the unnormalized dot-product expression. Implementations must not conflate the similarity gradient with the raw feature gradient.

In a symmetric two-view SimCLR batch, each view serves both as an anchor and as a candidate for all other anchor losses. The total gradient on one feature vector is the sum of its query role and every candidate role. Looking at only the loss term where it is the anchor omits these additional gradients.

### Temperature's effect

The derivative with respect to the similarity contains \(1/\tau\), but p also depends on \(\tau\). A lower temperature sharpens p, so hard candidates can dominate more strongly, while the explicit scale factor also grows. It is therefore incomplete to say that lower temperature merely “increases the gradient by a constant.”

## 3. CLIP: two coupled candidate classifiers

For normalized image and text embeddings \(u_i,v_j\), define \(a_{ij}=u_i^\top v_j/\tau\). Let row probabilities \(p_{j|i}\) be the image-to-text softmax and column probabilities \(q_{i|j}\) the text-to-image softmax. With a symmetric loss averaged over both directions,

\[
L_{\mathrm{CLIP}}=\frac{1}{2B}
\left[-\sum_i\log p_{i|i}-\sum_j\log q_{j|j}\right].
\]

For a given scaled similarity logit,

\[
\frac{\partial L}{\partial a_{ij}}
=\frac{1}{2B}\left[
p_{j|i}-\mathbf1[i=j]+
q_{i|j}-\mathbf1[i=j]
\right].
\]

A pair contributes to both row and column retrieval. This is why CLIP's loss is not simply a one-direction image-to-text classifier. The gradient updates both modalities through their shared cross-modal similarity matrix.

A pairwise sigmoid loss, such as the family used by SigLIP, has a different derivative. For label \(y\in\{-1,+1\}\) and logit a,

\[
\ell(a,y)=\log(1+e^{-ya})
\quad\Longrightarrow\quad
\frac{\partial\ell}{\partial a}=-y\,\sigma(-ya).
\]

Unlike the row-softmax gradient, this derivative is not normalized over all candidate texts for a single image. Pairwise labels, logit bias, loss weighting and the sampled pair set become particularly important.

## 4. BYOL and SimSiam: a detached target changes the derivative

Consider normalized prediction p and normalized target z, with a common loss

\[
L=2-2\hat p^\top \hat z.
\]

If z is detached, gradient flows through p but not through z for this loss term. For nonzero raw p,

\[
\nabla_p L=-\frac{2}{\|p\|_2}(I-\hat p\hat p^\top)\hat z.
\]

The expression says that the prediction is rotated toward the target direction; the normalization projects away the radial component. The target feature is used to compute the gradient but receives no derivative from this term.

For BYOL, the target parameters are updated by EMA outside this gradient calculation. For SimSiam, the target branch shares encoder parameters but is detached within the term; the encoder still receives gradient when that same branch appears on the prediction side of the symmetric loss. These are different computational graphs, even when their forward features look similar.

Stop-gradient does not directly assert that the target contains the correct semantic information. It changes the update rule. The constant-output counterexample still shows that an agreement target can be satisfied by a collapsed feature; the behavior of the full training dynamics is a separate analysis.

## 5. DINO: cross-entropy to a teacher distribution

Let teacher target \(q\) be a detached probability vector over K prototypes, and student logits a produce probabilities p. The cross-entropy is

\[
L=-\sum_{k=1}^{K}q_k\log p_k,
\]

so

\[
\frac{\partial L}{\partial a_k}=p_k-q_k.
\]

If student logits are divided by student temperature \(\tau_s\), then the gradient with respect to the pre-temperature logit is \((p_k-q_k)/\tau_s\). The teacher target itself is produced by a separate temperature/centering calculation, and teacher parameters are updated through EMA rather than this gradient. Do not backpropagate through the teacher when reproducing the canonical teacher-student target update.

Teacher centering, teacher temperature and EMA affect the distribution q and its timescale; they are not present as separate multipliers in this local cross-entropy derivative. Their effects appear through the target and its evolution.

## 6. Barlow Twins: gradient through the cross-correlation matrix

Let standardized feature batches A and B have shape \(N\times d\) and \(C=A^\top B/N\). Define

\[
L_{\mathrm{BT}}
=\sum_i(1-C_{ii})^2+\lambda\sum_{i\ne j}C_{ij}^2.
\]

The elementwise derivative is

\[
\frac{\partial L_{\mathrm{BT}}}{\partial C_{ij}}=
\begin{cases}
2(C_{ii}-1), & i=j,\\
2\lambda C_{ij}, & i\ne j.
\end{cases}
\]

Writing G for this matrix of derivatives, because \(C=A^\top B/N\),

\[
\nabla_A L_{\mathrm{BT}}=\frac1N BG^\top,
\qquad
\nabla_B L_{\mathrm{BT}}=\frac1N AG.
\]

These equations assume A and B are already the standardized tensors. Gradients must then pass through their batch-wise centering and standardization if the implementation computes those operations in the graph. Near-zero variance makes normalization numerically delicate; the epsilon and normalization convention affect the realized gradient.

The off-diagonal terms directly penalize cross-view covariance entries. They do not directly enforce full independence or semantic meanings for coordinates.

## 7. VICReg: explicit variance and covariance gradients

Use row-oriented centered features \(Z_c=Z-\mathbf1\bar z^\top\), batch size B and empirical covariance

\[
C=\frac1{B-1}Z_c^\top Z_c.
\]

Let M be C with its diagonal set to zero. The covariance penalty is

\[
L_{\mathrm{cov}}=\frac1d\sum_{i\ne j}C_{ij}^{2}
=\frac1d\|M\|_F^2.
\]

Because M and C are symmetric, the gradient with respect to the centered feature matrix is

\[
\nabla_{Z_c}L_{\mathrm{cov}}
=\frac{4}{d(B-1)}Z_cM.
\]

The exact denominator depends on whether a codebase defines covariance using B or B−1 and on the reduction convention; this equation follows the stated sample-covariance definition.

For an active variance-floor coordinate,

\[
L_{\mathrm{var},j}=\max(0,\gamma-\sigma_j),
\quad \sigma_j<\gamma,
\]

and with \(\sigma_j=\sqrt{\operatorname{Var}(Z_j)+\epsilon}\),

\[
\frac{\partial L_{\mathrm{var},j}}{\partial Z_{bj}}
\approx-\frac{Z_{c,bj}}{d(B-1)\sigma_j}.
\]

The approximation notation suppresses edge cases from epsilon and the exact variance estimator. Below the margin, the gradient expands the spread; above it, the hinge term is inactive.

The gradient analysis reinforces the conceptual roles:
- invariance pulls paired views together;
- variance penalizes near-constant dimensions;
- covariance penalizes linearly redundant dimensions.

These terms still cannot tell whether the non-constant feature variation encodes the right state variables.

## 8. SwAV: assignment target is fixed for the prediction step

Let q be a detached assignment vector generated by the balanced Sinkhorn procedure, and p be the student's prototype distribution from softmax logits. The cross-entropy derivative is

\[
\frac{\partial L}{\partial a_k}=p_k-q_k.
\]

The assignment generation itself is a separate operation: exponentiate/rescale prototype scores and alternately normalize rows/columns toward prescribed totals. During the swapped prediction loss, q is treated as the target assignment, and gradients train p to predict it.

The target-building procedure matters because it imposes a balancing prior across prototypes. The derivative of cross-entropy does not express the Sinkhorn balance constraint; that constraint is embedded in q.

## 9. Masked prediction gradients

### Pixel MSE

For prediction \(\hat x_i\) and target x_i on masked positions,

\[
L_{\mathrm{MSE}}=\frac1{|M|d_p}\sum_{i\in M}
\|\hat x_i-x_i\|_2^2
\]

has gradient

\[
\frac{\partial L}{\partial \hat x_i}
=\frac{2}{|M|d_p}(\hat x_i-x_i)
\]

for masked positions, and zero direct loss gradient for positions excluded from the reconstruction loss. The encoder still receives gradient through the decoder's dependence on visible encoded patches.

### Masked token cross-entropy

For target token y and logits a, \(\partial L/\partial a_k=p_k-\mathbf1[k=y]\). BERT and BEiT use this pattern with different token vocabularies and input construction. ELECTRA instead uses binary classification on original/replaced labels. The gradient is determined by the chosen target, not by the fact that a mask was used.

### Autoregressive language modeling

The next-token objective sums cross-entropy over positions t using logits conditioned on prefix \(x_{<t}\). Its gradient at position t is a softmax residual for the observed token. The causal attention mask determines which inputs can influence those logits; it does not change the local softmax derivative.

## 10. SIGReg: structure of the characteristic-function gradient

SIGReg compares the empirical characteristic function of scalar projections with the standard-normal characteristic function. Let \(u_b=z_b^\top w\), \(\hat\varphi(t)=B^{-1}\sum_b e^{itu_b}\), and \(\varphi_0(t)=e^{-t^2/2}\). For a schematic weighted discrepancy

\[
S=\int |\hat\varphi(t)-\varphi_0(t)|^2\,\omega(t)\,dt,
\]

its derivative with respect to projected sample \(u_b\) has the form

\[
\frac{\partial S}{\partial u_b}
=\frac{2}{B}\operatorname{Re}
\int
\overline{\hat\varphi(t)-\varphi_0(t)}
\,(it e^{itu_b})\,\omega(t)\,dt.
\]

This is the chain-rule structure for the schematic integral. The paper's Epps–Pulley implementation uses its exact weighting and numerical quadrature; do not substitute the display above for the canonical code. The scalar gradient then propagates through \(u_b=z_b^\top w\), giving a contribution proportional to w in embedding space.

The statistical insight is that the discrepancy compares an empirical distribution to a specified target. The optimization insight is that the gradient depends on how the characteristic function changes when a projected sample moves. Under the VISReg paper's tested collapse setup, the authors report that this gradient can weaken near collapse; the scale term and sliced-Wasserstein shape loss are introduced partly in response to that diagnostic.

## 11. VISReg: shape gradient through sorting and projection

For projection direction w, write projected samples \(u= \widetilde Zw\), sort them into \(u_{(1)}\leq\cdots\leq u_{(B)}\), and let q contain standard-normal quantiles. A squared-distance shape term for that projection is

\[
L_w=\|\operatorname{sort}(\widetilde Zw)-q\|_2^2.
\]

Away from ties, sorting simply reorders derivatives, so the derivative for the sorted sample is

\[
\frac{\partial L_w}{\partial u_{(i)}}=2(u_{(i)}-q_i).
\]

If an implementation averages over B values or K projections, the corresponding factors \(1/B\) or \(1/K\) multiply this gradient. It backpropagates through the projection \(u=\widetilde Zw\) and then through the normalization. VISReg places stop-gradient on the scale used in normalization so the shape loss does not directly alter that scale through the denominator; a separate variance term controls it.

At ties the sorting map is not classically differentiable, but ties have measure zero under continuous data; automatic-differentiation libraries use an implementation-specific subgradient behavior. The paper's exact reduction and direction sampling should be used for faithful reproduction.

## 12. What gradient derivations can and cannot explain

A local derivative establishes the force produced by a specific loss at a point in parameter space. It does not by itself prove:
- convergence of nonlinear neural network training;
- global stability of non-collapsed solutions;
- semantic quality of the learned features;
- robustness to domain shift;
- Markov sufficiency or causal representation;
- usefulness for planning.

Those are separate questions involving optimization dynamics, data distribution, architecture, evaluation and task structure.
