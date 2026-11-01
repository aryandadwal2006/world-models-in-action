# 8. Separating scale and shape: VISReg

SIGReg compares projected distributions with a Gaussian target through characteristic functions. VISReg addresses the same broad distributional problem with a different decomposition. It treats the centre, scale and shape of the embedding batch as distinct quantities. The separation is useful because a distribution can have an acceptable covariance while retaining a non-Gaussian shape, and a shape comparison can be insensitive to the overall scale of the embeddings.

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

**Scale.** How much variation does each coordinate have? If a representation contracts toward the origin, its standard deviations become small.

**Shape.** After the mean and scale are accounted for, does the distribution resemble the desired shape? A batch can have unit coordinate-wise variance and identity covariance while remaining non-Gaussian.

These properties should not be treated as one statistic. A variance term does not identify the full distribution. A distributional shape test that normalizes away scale does not, by itself, require the unnormalized embedding cloud to have unit variance.

## 8.2 The scale term

Let \(\sigma_j(Z_c)\) denote the batch standard deviation of coordinate \(j\). VISReg uses an explicit scale penalty of the form

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

At \(\sigma_j=0\), this scalar derivative equals \(-2/D\). The scale objective therefore has a direct first-order preference for increasing a collapsed standard-deviation statistic.

This statement concerns the derivative with respect to \(\sigma_j\). It is not a universal lower bound on the gradient with respect to every encoder parameter. The derivative of a batch standard deviation with respect to the raw embeddings depends on the variance estimator, the batch, numerical stabilization and the point at which the representation is evaluated. The composition from an embedding statistic back to network parameters depends on the architecture and can be affected by normalization, saturation and finite precision. The paper's gradient comparisons are controlled experiments under the tested setup, not a theorem about every neural network.

## 8.3 Removing scale from the shape comparison

Let \(\sigma(Z_c)\in\mathbb R^D\) be the vector of coordinate-wise standard deviations. VISReg constructs a normalized batch

\[
\widetilde Z
=
\frac{Z_c}{\operatorname{sg}(\sigma(Z_c))+\epsilon},
\]

where the division is coordinate-wise, \(\epsilon>0\) is a numerical stabilizer, and \(\operatorname{sg}\) denotes stop-gradient.

The normalization removes coordinate scale before the shape comparison. The stop-gradient means that the shape term does not differentiate through the standard-deviation denominator. The scale objective remains the direct mechanism for adjusting the unnormalized coordinate spread, while the shape statistic compares normalized samples.

This gradient boundary is not the same as an EMA target update. An EMA makes a target encoder follow the online encoder slowly. Here, stop-gradient is applied to the scale statistic used in normalization. It changes which derivatives the shape objective contributes. The two mechanisms should not be described as the same kind of teacher-student asymmetry.

It is also useful to distinguish what normalization achieves from what it does not. It makes the shape comparison insensitive to the raw coordinate magnitudes that have been divided out. It does not remove every possible dependence between scale and shape in the full optimization problem, nor does it prove that every encoder parameter receives a nonzero gradient.

## 8.4 Sliced Wasserstein distance

The shape term compares projected normalized embeddings with projections of a standard Gaussian. For a unit direction \(w_k\in\mathbb S^{D-1}\), form the scalar samples

\[
u_k=\widetilde Z w_k\in\mathbb R^B.
\]

Let \(u_{k,(1)}\le\cdots\le u_{k,(B)}\) denote the sorted projected values. Let \(q_1,\ldots,q_B\) be the corresponding standard-normal target quantiles. A common midpoint convention is

\[
q_i=\Phi^{-1}\!\left(\frac{i-\frac12}{B}\right),
\]

where \(\Phi\) is the standard-normal cumulative distribution function.

For two equally weighted empirical distributions on the real line with the same number of samples, the optimal quadratic transport coupling matches their order statistics. Thus the squared one-dimensional Wasserstein distance is

\[
W_2^2(\widehat P,\widehat Q)
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(u_{(i)}-q_i\right)^2.
\]

This identity follows from the monotone coupling on the line: crossing assignments cannot reduce the total squared transport cost, so sorted values are paired in order. The formula depends on equal sample weights and equal cardinalities. Weighted samples or unequal sample counts require the appropriate quantile or integral formulation.

VISReg averages the one-dimensional transport discrepancy across \(K_s\) sampled directions:

\[
\mathcal L_{\mathrm{shape}}
=
\frac{1}{K_sB}
\sum_{k=1}^{K_s}
\left\|
\operatorname{sort}(\widetilde Z w_k)-q
\right\|_2^2.
\]

Here \(K_s\) is the number of slices, not the embedding dimension. Each slice examines one direction through the normalized embedding cloud. More directions provide broader coverage of the sphere, but increase computation. As with SIGReg, a finite set of directions is a stochastic sketch rather than an exact all-directions certificate.

The difference between SIGReg and VISReg lies in the one-dimensional statistic. SIGReg compares empirical characteristic functions with the target characteristic function. VISReg sorts projected samples and compares their empirical quantiles with Gaussian quantiles. Both use random projections; they do not compute the same discrepancy.

## 8.5 Centre, scale and shape in one objective

The centering term is

\[
\mathcal L_{\mathrm{center}}=\|\mu\|_2^2.
\]

The regularizer combines the three components:

\[
\mathcal L_{\mathrm{reg}}
=
\lambda_s\mathcal L_{\mathrm{scale}}
+
\lambda_h\mathcal L_{\mathrm{shape}}
+
\lambda_c\mathcal L_{\mathrm{center}}.
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
(1-\alpha)\mathcal L_{\mathrm{pred}}
+
\alpha\mathcal L_{\mathrm{reg}}.
\]

This expression makes the responsibilities explicit. The prediction term defines which views should agree. The regularizer constrains centre, scale and normalized shape. The three regularizer coefficients determine the relative importance of those constraints, while \(\alpha\) balances view agreement against distributional regularization. The exact batch construction, view sampling and loss normalization should be tied to the paper version and reference implementation.

## 8.6 Why sorted one-dimensional transport is practical

The shape comparison is performed after projecting each batch onto one-dimensional directions. The dominant operations are the projection of a \(B\times D\) embedding matrix onto \(K_s\) directions and sorting \(B\) scalars for each direction.

A useful decomposition of the computational work is

\[
O(BDK_s)+O(K_sB\log B).
\]

The first term is projection. The second accounts for sorting each slice. If the number of slices is kept fixed as \(D\) grows, the dependence on embedding dimension differs from a design where \(K_s\) scales with \(D\). In the VISReg paper's practical scaling discussion, the number of slices required for a good approximation is considered in relation to the embedding dimension. If \(K_s\approx C D\) for a factor \(C>1\), the projection cost becomes approximately \(O(CBD^2)\) on a single device, before considering memory layout and parallelization. The paper discusses distributing slices across devices to manage this cost.

It is therefore incomplete to describe the practical computation as simply linear in \(D\) without stating what happens to \(K_s\). Projection count is part of the method's compute budget.

## 8.7 What the experiments support

VISReg reports experiments designed to test the separate roles of scale and shape, including controlled comparisons of gradients near collapsed configurations, low-quality or long-tailed training data, and out-of-distribution transfer. These experiments provide evidence for the proposed decomposition under the paper's architectures, data and optimization settings.

They should not be turned into stronger claims. A controlled gradient comparison does not prove that every implementation of SIGReg has vanishing gradients near every collapse point. A scalar derivative of the scale penalty does not prove that every parameterization can recover from every collapsed initialization. A better out-of-distribution score on a particular benchmark set does not establish universal superiority or task sufficiency.

The most informative conclusion is narrower. A scale objective and a shape objective respond to different properties of the embedding batch. Making that separation explicit gives the optimizer a direct route for correcting coordinate spread while evaluating the distributional shape of normalized features.

## 8.8 The broader design principle

The sequence from covariance penalties to SIGReg and VISReg is not simply a progression of increasingly elaborate anti-collapse tricks.

A variance floor constrains marginal spread. A covariance penalty constrains pairwise linear redundancy. SIGReg compares projected characteristic functions with a Gaussian target. VISReg separates the correction of centre and scale from a sliced-Wasserstein shape comparison. Each objective introduces a different statistical condition.

None of these conditions, in isolation, proves that the representation is sufficient for a physical task. The Gaussian target is motivated by the downstream-risk analysis already developed. The practical objectives approximate distributional constraints on the representation. A separate predictive target is still needed to specify what the representation should preserve.

This distinction becomes decisive when the model is used for interaction. The next section turns from the geometry of passive observations to the information needed to predict the consequences of chosen actions.
