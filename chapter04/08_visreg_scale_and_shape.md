# 8. Separating scale and shape: VISReg

SIGReg compares projected distributions with a Gaussian target through characteristic functions. VISReg addresses the same broad distributional problem with a different decomposition. It treats the centre, scale and shape of the embedding batch as distinct quantities. The separation is useful because a distribution can have an acceptable covariance while retaining a non-Gaussian shape, and a shape comparison can be made after coordinate scale has been removed.

The aim is not to rank regularizers in isolation. It is to understand why the decomposition is made, what each term constrains, and what the resulting optimization does and does not guarantee.

## 8.1 Three different properties of a batch

Let the embeddings of a batch be \(Z\in\mathbb R^{B\times D}\), with \(B\) samples and \(D\) feature dimensions. Define the batch mean

\[
\mu=\frac{1}{B}\sum_{b=1}^{B}z_b,
\qquad
Z_c=Z-\mu.
\]

Three separate questions can now be asked.

**Centre.** Is the embedding cloud centred at the origin? A nonzero mean shifts every projection, even if the covariance and higher-order shape are otherwise acceptable.

**Scale.** How much variation does each coordinate have? If a representation contracts toward the origin, its standard deviations become small. If some coordinates expand excessively, they can dominate the embedding geometry.

**Shape.** After the mean and scale are accounted for, does the distribution resemble the desired shape? A batch can have unit coordinate-wise variance and identity covariance while remaining non-Gaussian.

These properties should not be treated as one statistic. A variance term does not identify the full distribution. A distributional shape test that normalizes away scale does not, by itself, require the unnormalized embedding cloud to have unit variance.

## 8.2 The scale term

Let \(\sigma_j(Z_c)\) denote the batch standard deviation of coordinate \(j\), computed with the population-style batch estimate used in the reference algorithm. VISReg uses an explicit scale penalty:

\[
\mathcal L_{\mathrm{scale}}
=
\frac{1}{D}
\sum_{j=1}^{D}
\left(1-\sigma_j(Z_c)\right)^2.
\]

The target standard deviation is one in each coordinate. The term penalizes both contraction below unit scale and expansion above it. Its minimum is attained when every coordinate has unit batch standard deviation.

For one coordinate, the derivative with respect to the scalar standard deviation is

\[
\frac{\partial}{\partial\sigma_j}
\left[
\frac{1}{D}(1-\sigma_j)^2
\right]
=
-\frac{2}{D}(1-\sigma_j).
\]

At \(\sigma_j=0\), this scalar derivative equals \(-2/D\). The scale objective therefore has a direct first-order preference for increasing a collapsed standard-deviation statistic. This is the mechanism behind the paper's collapse-gradient analysis.

This statement concerns the derivative with respect to \(\sigma_j\). It is not a universal lower bound on the gradient with respect to every encoder parameter. The derivative of a batch standard deviation with respect to raw embeddings depends on the batch, the variance estimator and the point at which the representation is evaluated. The composition from an embedding statistic back to network parameters depends on the architecture and can be affected by normalization, saturation and finite precision. The paper's gradient comparisons are controlled experiments under the tested setup, not a theorem about every neural network. [Wu, Balestriero, and Levine, 2026]

## 8.3 Removing scale from the shape comparison

Let \(\sigma(Z_c)\in\mathbb R^D\) be the vector of coordinate-wise standard deviations. VISReg constructs a normalized batch

\[
\widetilde Z
=
\frac{Z_c}{\operatorname{sg}(\sigma(Z_c))},
\]

where division is coordinate-wise and \(\operatorname{sg}\) denotes stop-gradient. The reference algorithm detaches the standard-deviation denominator. A numerical safeguard may be added in an implementation for zero or near-zero scale, but it is not part of the displayed scale or shape objective.

The normalization removes coordinate scale before the shape comparison. The stop-gradient means that the shape term does not differentiate through the standard-deviation denominator. The scale objective remains the direct mechanism for adjusting unnormalized coordinate spread, while the shape statistic compares normalized samples.

This gradient boundary is not the same as an EMA target update. An EMA makes a target encoder follow the online encoder slowly. Here, stop-gradient is applied only to the scale statistic used in normalization. It changes which derivatives the shape objective contributes. The project describes VISReg as avoiding the usual EMA teacher-student heuristics; that statement should not be misread as saying the implementation contains no stop-gradient operation anywhere. [Wu, Balestriero, and Levine, 2026]

It is also useful to distinguish what normalization achieves from what it does not. It makes the shape comparison insensitive to the raw coordinate magnitudes that have been divided out. It does not remove every possible dependence between scale and shape in the full optimization problem, nor does it prove that every encoder parameter receives a nonzero gradient.

## 8.4 Sliced Wasserstein distance

The shape term compares projected normalized embeddings with projections of a standard Gaussian. For a unit direction \(w_k\in\mathbb S^{D-1}\), form the scalar samples

\[
u_k=\widetilde Z w_k\in\mathbb R^B.
\]

Let \(u_{k,(1)}\le\cdots\le u_{k,(B)}\) denote the sorted projected values. Let \(q_1,\ldots,q_B\) be the corresponding standard-normal target quantiles, using the reference quantile locations

\[
r_i=\frac{i}{B+1},
\qquad
q_i=\Phi^{-1}(r_i),
\qquad i=1,\ldots,B,
\]

where \(\Phi\) is the standard-normal cumulative distribution function.

For two equally weighted empirical distributions on the real line with the same number of samples, the optimal quadratic transport coupling matches their order statistics. Thus the squared one-dimensional Wasserstein distance can be written as

\[
W_2^2(\widehat P,\widehat Q)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(u_{k,(i)}-q_i\right)^2.
\]

This identity follows from the monotone coupling on the line: crossing assignments cannot reduce the total squared transport cost, so sorted values are paired in order. The formula depends on equal sample weights and equal cardinalities. Weighted samples or unequal sample counts require the appropriate quantile or integral formulation.

VISReg averages the one-dimensional transport discrepancy across \(K_s\) sampled directions:

\[
\mathcal L_{\mathrm{shape}}
=
\frac{1}{K_sB}
\sum_{k=1}^{K_s}
\sum_{i=1}^{B}
\left(
u_{k,(i)}-q_i
\right)^2.
\]

Here \(K_s\) is the number of slices, not the embedding dimension. Each slice examines one direction through the normalized embedding cloud. More directions provide broader coverage of the sphere, but increase computation. As with SIGReg, a finite set of directions is a stochastic sketch rather than an exact all-directions certificate.

The difference between SIGReg and VISReg lies in the one-dimensional statistic. SIGReg compares empirical characteristic functions with the target characteristic function. VISReg sorts projected samples and compares their empirical quantiles with Gaussian quantiles. Both use random projections; they do not compute the same discrepancy.

## 8.5 Centre, scale and shape in one objective

The centering term is

\[
\mathcal L_{\mathrm{center}}
=
\frac{1}{D}\|\mu\|_2^2.
\]

The regularizer combines the three components:

\[
\mathcal L_{\mathrm{reg}}
=
\mathcal L_{\mathrm{scale}}
+
\mathcal L_{\mathrm{shape}}
+
\mathcal L_{\mathrm{center}}.
\]

The method also retains a view-agreement signal. Let \(z_{n,v}\) denote the embedding of sample \(n\) in view \(v\), with \(V_g\) global views among \(V\) total views. Define the centre of the global-view embeddings as

\[
\mu_{g,n}
=
\frac{1}{V_g}
\sum_{v=1}^{V_g}z_{n,v}.
\]

A view-invariance term draws each view embedding toward that global-view mean:

\[
\mathcal L_{\mathrm{pred}}
=
\frac{1}{NV}
\sum_{n=1}^{N}
\sum_{v=1}^{V}
\|\mu_{g,n}-z_{n,v}\|_2^2.
\]

The complete loss is

\[
\mathcal L_{\mathrm{VISReg}}
=
(1-\lambda)\mathcal L_{\mathrm{pred}}
+
\lambda\mathcal L_{\mathrm{reg}},
\]

where \(\lambda\) is the trade-off between view agreement and regularization. The published formulation sums the three regularizer components and uses this single trade-off parameter. [Wu, Balestriero, and Levine, 2026]

This expression makes the responsibilities explicit. The prediction term defines which views should agree. The regularizer constrains centre, scale and normalized shape. The method does not require a separate weight for each of the three components in its stated formulation.

## 8.6 Why sorted one-dimensional transport is practical

The shape comparison is performed after projecting each batch onto one-dimensional directions. The dominant matrix operation projects a \(B\times D\) embedding matrix onto \(K_s\) directions:

\[
\widetilde ZW,
\qquad
W\in\mathbb R^{D\times K_s}.
\]

The projection work is \(O(BDK_s)\). Sorting the \(B\) scalar values in each of the \(K_s\) slices requires an additional \(O(K_sB\log B)\) operation. The paper summarizes the principal scaling of the projection-based method as \(O(NDK)\), linear in batch size \(N\), embedding dimension \(D\), and slice count \(K\), with the slices distributed across GPUs.

This distribution is central to the scaling claim. Each slice can be evaluated independently. If there are \(M\) workers, the slice set can be partitioned across them, and the resulting scalar loss contributions can be aggregated. The authors show that increasing the number of GPUs can compensate for a smaller per-step slice count. The claim is not that a fixed number of slices is mathematically sufficient in every dimension; it is that the sketch can be expanded by distributing independent projections rather than increasing the work on one device alone. [Wu, Balestriero, and Levine, 2026]

## 8.7 What the experiments support

VISReg reports controlled comparisons of regularizer gradients near collapse, tests on low-quality and long-tailed data, low-rank settings, out-of-distribution linear evaluation, transfer learning, dense prediction and generation guidance. The paper reports strong out-of-distribution results, including an ImageNet-22K-pretrained model whose OOD performance is comparable to DINOv2 trained on the substantially larger LVD-142M dataset.

These experiments provide evidence for the proposed decomposition under the paper's architectures, data and optimization settings. They are not a theorem that every parameterization recovers from every collapsed initialization. A controlled gradient comparison does not prove that every implementation of SIGReg has vanishing gradients at every collapse point. A scalar derivative of the scale penalty does not prove that every network can recover from every degenerate state. An OOD score on a particular benchmark set does not establish universal superiority or task sufficiency. [Wu, Balestriero, and Levine, 2026]

## 8.8 The broader design principle

The sequence from covariance penalties to SIGReg and VISReg is not simply a progression of increasingly elaborate anti-collapse tricks.

A variance floor constrains marginal spread. A covariance penalty constrains pairwise linear redundancy. SIGReg compares projected characteristic functions with a Gaussian target. VISReg separates the correction of centre and scale from a sliced-Wasserstein shape comparison. Each objective introduces a different statistical condition.

None of these conditions, in isolation, proves that the representation is sufficient for a physical task. The Gaussian target is motivated by the downstream-risk analysis already developed. The practical objectives approximate distributional constraints on the representation. A separate predictive target is still needed to specify what the representation should preserve.

This distinction becomes decisive when the model is used for interaction. The next section turns from the geometry of passive observations to the information needed to predict the consequences of chosen actions.
