# 10. Separating centre, scale and shape

SIGReg compares projected characteristic functions with a Gaussian target. VISReg approaches distribution matching by separating three properties of a minibatch: its centre, the scale of each coordinate and the shape of the normalized distribution. The purpose is not to add three interchangeable anti-collapse penalties. Each term controls a different statistic, and the shape comparison is constructed so that coordinate scale is handled elsewhere.

## 10.1 The three properties of an embedding batch

Let the embeddings be rows of \(Z\in\mathbb R^{N\times D}\). Define their mean and centred values by

\[
\mu=\frac1N\sum_{n=1}^{N}z_n,
\qquad
\widehat Z=Z-\mu.
\]

The mean determines where the cloud is centred. The coordinate standard deviations determine its scale along each feature axis. The normalized sample distribution determines its shape after those shifts and scales have been removed.

These properties can vary independently. A batch may be centred but have very small variance. It may have unit variance in every coordinate while exhibiting a non-Gaussian shape. It may match a reference shape after normalization but have the wrong unnormalized scale. A single covariance penalty does not impose all three conditions.

## 10.2 Directly constrain coordinate scale

Let \(\sigma_j(\widehat Z)\) denote the batch standard deviation of feature coordinate \(j\). VISReg uses

\[
\mathcal L_{\mathrm{scale}}
=
\frac1D\sum_{j=1}^{D}
\left(1-\sigma_j(\widehat Z)\right)^2.
\]

The target standard deviation is one for each coordinate. The loss penalizes both contraction and expansion. Its derivative with respect to the standard-deviation statistic is

\[
\frac{\partial\mathcal L_{\mathrm{scale}}}
{\partial\sigma_j}
=
-\frac{2}{D}(1-\sigma_j).
\]

At \(\sigma_j=0\), this scalar derivative is finite and nonzero. This is different from a loss whose derivative with respect to variance itself vanishes at zero variance. The paper uses this comparison to motivate a direct scale objective.

The statement concerns the derivative with respect to \(\sigma_j\), not a universal lower bound on the gradient of every encoder parameter. The chain rule still passes through the batch standard deviation and the encoder; its magnitude can depend on the data, representation, numerical stabilizer and network parameterization. The scale term supplies a direct correction at the statistic level, but it is not a theorem that every collapsed network can escape every initialization. (Wu et al., 2026)

## 10.3 Normalize scale before comparing shape

VISReg centres the batch and normalizes each coordinate by its standard deviation:

\[
\widetilde Z
=
\frac{\widehat Z}
{\operatorname{sg}(\sigma(\widehat Z))+\epsilon},
\]

where \(\operatorname{sg}\) is stop-gradient and \(\epsilon>0\) stabilizes division. The normalization removes coordinate scale from the shape comparison. Stop-gradient means that the shape loss does not differentiate through the standard-deviation denominator.

This creates two distinct gradient paths. The scale term acts on the unnormalized centred batch and adjusts coordinate spread. The shape term compares normalized samples, treating the scale statistic as constant for differentiation. It is not an EMA target update: no slowly moving encoder is introduced. The stop-gradient is applied to the normalization statistic, not to a target network.

The centring term is

\[
\mathcal L_{\mathrm{center}}=\|\mu\|_2^2.
\]

It penalizes a shift of the embedding cloud away from the origin. It is separate from scale and shape because a distribution can have the desired coordinate variances and normalized shape while remaining offset.

## 10.4 Sliced Wasserstein matching

After centring and scale normalization, VISReg compares the projected empirical samples with projected samples from a standard Gaussian. For a unit direction \(w_k\),

\[
u_k=\widetilde Z w_k\in\mathbb R^N.
\]

Sort the projected values:

\[
u_{k,(1)}\leq\cdots\leq u_{k,(N)}.
\]

For the standard-normal reference, define equally spaced quantile levels and their Gaussian quantiles:

\[
r_i=\frac{i}{N+1},
\qquad
q_i=\Phi^{-1}(r_i),
\qquad i=1,\ldots,N.
\]

In one dimension, the optimal quadratic transport coupling between two equally weighted empirical distributions matches their order statistics. The corresponding empirical squared Wasserstein distance is

\[
W_2^2
=
\frac1N\sum_{i=1}^{N}(u_{(i)}-q_i)^2.
\]

Averaging this comparison over \(K_s\) random directions gives the sliced shape objective. The paper's displayed formula uses a squared Euclidean norm of the sorted vectors averaged over slices:

\[
\mathcal L_{\mathrm{shape}}
=
\frac1{K_s}
\sum_{k=1}^{K_s}
\left\|
\operatorname{sort}(\widetilde Z w_k)-q
\right\|_2^2.
\]

The paper's accompanying pseudocode computes a mean squared difference over the \(N\) samples and directions, which includes an additional factor \(1/N\) relative to the printed squared-norm formula. This is a normalization discrepancy, not a different transport coupling. Since it changes the scale of the regularizer with batch size, a reproduction must follow one convention consistently and retune weights accordingly. The normalization equation also contains an \(\epsilon\) stabilizer, while the printed algorithm excerpt divides by the detached standard deviation directly; the implementation choice should be recorded. (Wu et al., 2026)

The random projection turns a high-dimensional distribution comparison into several one-dimensional sorting operations. This is computationally attractive because one-dimensional optimal transport has a simple solution. It is still an approximation: a finite set of directions cannot certify exact equality of arbitrary multivariate distributions.

## 10.5 The complete VISReg objective

The regularization component combines the three separate constraints:

\[
\mathcal L_{\mathrm{reg}}
=
\lambda_s\mathcal L_{\mathrm{scale}}
+
\lambda_h\mathcal L_{\mathrm{shape}}
+
\lambda_c\mathcal L_{\mathrm{center}}.
\]

The method also has a view-agreement term. Let \(z_{n,v}\) be the embedding for sample \(n\) under view \(v\), and let the first \(V_g\) views be global crops. Their mean is

\[
\mu_{g,n}
=
\frac1{V_g}
\sum_{v=1}^{V_g}z_{n,v}.
\]

The agreement term draws each view toward this per-sample global-view mean:

\[
\mathcal L_{\mathrm{pred}}
=
\frac1{NV}
\sum_{n=1}^{N}\sum_{v=1}^{V}
\|\mu_{g,n}-z_{n,v}\|_2^2.
\]

The full objective is

\[
\mathcal L_{\mathrm{VISReg}}
=
(1-\lambda)\mathcal L_{\mathrm{pred}}
+
\lambda\mathcal L_{\mathrm{reg}}.
\]

There are four coefficients in this formulation: the outer trade-off \(\lambda\), and the three weights inside the regularizer. The baseline sets the inner weights to one, with ablations that vary their relative contributions. (Wu et al., 2026)

VISReg and SIGReg therefore share a target motivation but use different mechanisms. SIGReg compares characteristic functions along projections. VISReg compares sorted projected quantiles after separately controlling centre and coordinate scale. The two objectives should not be treated as interchangeable or as the same regularizer with different names.

## 10.6 Computational cost and the role of slices

For \(K_s\) random directions, the projection cost is

\[
O(NDK_s),
\]

and sorting the \(N\) projected values for every direction costs

\[
O(K_sN\log N).
\]

The second term may be smaller than the projection cost at the dimensions considered, but both depend on batch size and slice count. More slices improve coverage of the distribution's directions, but increase computation.

The VISReg paper studies distributing slices across devices. Each worker can evaluate a subset of directions and aggregate the resulting loss. The authors report that more devices can compensate for fewer slices per device in their experiments. This is a scaling strategy observed under their configurations, not a theorem that a fixed number of per-device slices is sufficient at arbitrary dimension.

The experiments also compare VISReg on low-rank, long-tailed and degraded data, and evaluate downstream classification, transfer and dense prediction. The reported results support the method's decomposition in the tested settings. They do not prove that any finite projection set recovers the full distribution or that a Gaussian-shaped embedding is sufficient for an arbitrary physical task. (Wu et al., 2026)

## 10.7 Why the distinction matters

The difference between SIGReg and VISReg is instructive. A distributional target alone does not determine the best optimization interface. SIGReg works with characteristic functions. VISReg makes coordinate scale an explicit objective and removes it from the shape comparison with a stop-gradient normalization. Both are attempts to make distributional structure trainable without estimating a full density.

The chapter's central distinction remains unchanged. View agreement specifies which observations should share a representation. Distribution regularization constrains the geometry of the resulting features. Neither alone specifies what a world model must predict under a chosen action. That next requirement changes the conditioning variable, not merely the embedding regularizer.

**Figure 4.16. VISReg assigns separate losses to centre, scale and normalized shape.**

Description: The batch is first centred. A scale loss acts on the unnormalized standard deviations, while a shape loss compares sorted Gaussian projections of the normalized batch. A separate centre loss penalizes a nonzero mean.

**Figure 4.17. Sliced Wasserstein matching reduces each projection to sorted samples.**

Description: Empirical embeddings and Gaussian reference samples are projected along one direction, sorted, and matched by order statistic. Repeating this operation over random directions approximates a multivariate shape comparison.