# 8. Nonlinear readouts, local density and Fisher information

A covariance matrix describes global second-order geometry. It does not say where points are concentrated within that geometry. For nonlinear readouts that estimate a target by averaging nearby examples, this local density matters.

The LeJEPA analysis makes this dependence explicit. Under a specified family of smooth regression functions and local estimators, the leading prediction bias contains the score of the embedding density,

\[
s_p(z)=\nabla\log p(z).
\]

The expected squared magnitude of this score is the Fisher-information functional

\[
J(p)
=
\mathbb E_{Z\sim p}
\left[\|\nabla\log p(Z)\|_2^2\right]
=
\int
\|\nabla\log p(z)\|_2^2p(z)\,dz.
\]

This is a property of the embedding distribution. It is not a property of the encoder architecture alone.

## 8.1 Local regression depends on where examples are concentrated

Let \(Z\in\mathbb R^D\) be a frozen representation with density \(p(z)\), and let the downstream regression function be

\[
\eta(z)=\mathbb E[Y\mid Z=z].
\]

A local estimator predicts \(\eta(q)\) by averaging labels from examples whose embeddings lie near a query \(q\). In a kernel estimator,

\[
\widehat\eta_h(q)
=
\frac{\sum_{i=1}^{N}K_h(q-Z_i)Y_i}
{\sum_{i=1}^{N}K_h(q-Z_i)},
\qquad
K_h(u)=h^{-D}K(u/h).
\]

The kernel weights nearby points more heavily. The bandwidth \(h\) controls the neighbourhood size. The denominator normalizes the weights.

The estimator is affected by two local properties. The target function may curve within the neighbourhood, so averaging nearby labels can differ from the value at its centre. The embedding density may be asymmetric around the query, so one side of the neighbourhood contributes more examples than the other. The second effect creates a bias even when the target function is locally smooth.

## 8.2 Where the density score enters

For large samples, the conditional expectation of the Nadaraya–Watson estimator is approximately

\[
\frac{
\int \eta(q+u)p(q+u)K_h(u)\,du
}{
\int p(q+u)K_h(u)\,du
}.
\]

Assume a normalized, symmetric kernel with zero first moment and isotropic second moment:

\[
\int K(v)\,dv=1,
\qquad
\int vK(v)\,dv=0,
\qquad
\int vv^\top K(v)\,dv=\mu_2(K)I_D.
\]

Expand the regression function and density around \(q\) to second order. Odd terms vanish after integration because the kernel is symmetric. The remaining terms include the Laplacian of the regression function and the cross term between its gradient and the density gradient. After division by the denominator, the leading bias is

\[
\operatorname{Bias}[\widehat\eta_h(q)]
=
\frac{h^2\mu_2(K)}{2}
\left[
\Delta\eta(q)
+
2\nabla\eta(q)^\top\nabla\log p(q)
\right]
+o(h^2).
\]

The first term, \(\Delta\eta(q)\), is target curvature. The second term measures the alignment of target change with the direction in which the embedding density increases. If more training points lie on one side of the query, their labels pull the local average toward that side. If the target function changes along the same direction, the estimator is biased.

The same density-score coupling appears in the leading bias of radius-based \(k\)-nearest-neighbour regression. The coefficient differs because a uniform ball and a smooth kernel have different second moments. The presence of \(\nabla\log p(q)\) is the common feature.

This expansion requires smoothness. The density must be positive and sufficiently differentiable around the query, the kernel must have finite moments, and the regression function must have the required derivatives. The asymptotic regime also assumes a bandwidth small enough to localize the estimator while leaving enough samples in each neighbourhood. The equation should not be applied to arbitrary discrete or highly singular embedding distributions without checking these conditions.

## 8.3 From pointwise bias to the Fisher-information functional

The downstream risk is evaluated over query points, not at one fixed location. Let queries be drawn from the embedding density \(p\). The integrated squared bias is

\[
\operatorname{ISB}
=
\int
\operatorname{Bias}(q)^2p(q)\,dq.
\]

The density-dependent term in the pointwise expansion is
\(\nabla\eta(q)^\top\nabla\log p(q)\). To understand its expected contribution, consider a class of downstream functions whose gradients have isotropic second moment:

\[
\mathbb E_\eta[\nabla\eta(Z)]=0,
\qquad
\mathbb E_\eta[
\nabla\eta(Z)\nabla\eta(Z)^\top
]
=
\tau_g^2 I_D.
\]

Then

\[
\begin{aligned}
&\mathbb E_\eta
\mathbb E_{Z\sim p}
\left[
(\nabla\eta(Z)^\top\nabla\log p(Z))^2
\right]\\
&\quad=
\tau_g^2
\mathbb E_{Z\sim p}
\left[\|\nabla\log p(Z)\|_2^2\right]\\
&\quad=\tau_g^2J(p).
\end{aligned}
\]

The functional \(J(p)\) measures the expected squared score. It is large when the log-density changes sharply over regions carrying substantial probability mass. In a local estimator, this corresponds to density variation that can make one side of a neighbourhood contribute more strongly than the other.

The complete integrated-bias expression also contains a curvature term and a cross term involving the target Hessian and gradient. Those terms do not disappear merely because the score term has been identified. The LeJEPA analysis controls them under its stated regularity and downstream-function assumptions, yielding a density-dependent contribution proportional to \(J(p)\) plus bounded terms. The conclusion is not that every nonlinear estimator has risk exactly equal to \(J(p)\), but that this functional governs a specific part of the risk bound for the estimator classes analysed.

## 8.4 Gaussianity follows from a fixed-covariance inequality

The Fisher-information functional has a useful lower bound. Let \(p\) be a mean-zero density with positive-definite covariance

\[
\Sigma=\mathbb E[ZZ^\top].
\]

Assume sufficient decay and differentiability for integration by parts, and finite Fisher information. Since \(p(z)\nabla\log p(z)=\nabla p(z)\),

\[
\mathbb E[Z\,s_p(Z)^\top]
=
\int z\,\nabla p(z)^\top dz
=
-I_D.
\]

Now consider the expected outer product

\[
\mathbb E
\left[
(s_p(Z)+\Sigma^{-1}Z)
(s_p(Z)+\Sigma^{-1}Z)^\top
\right].
\]

It is positive semidefinite. Expanding it and using the identity above gives

\[
\mathbb E[s_p(Z)s_p(Z)^\top]
-
\Sigma^{-1}
\succeq0.
\]

Taking the trace yields

\[
J(p)\geq\operatorname{tr}(\Sigma^{-1}).
\]

Equality requires the score to be linear in \(z\):

\[
\nabla\log p(z)=-\Sigma^{-1}z.
\]

Integrating this relation gives the Gaussian density with covariance \(\Sigma\). Thus, among densities with a fixed covariance matrix, the Gaussian uniquely minimizes the Fisher-information functional.

This is the first optimization. It identifies the best shape for a given covariance. It does not yet say which covariance is best.

## 8.5 Isotropy follows from a fixed total-variance constraint

Let the eigenvalues of \(\Sigma\) be \(\lambda_1,\ldots,\lambda_D>0\). Then

\[
\operatorname{tr}(\Sigma^{-1})
=
\sum_{j=1}^{D}\frac{1}{\lambda_j}.
\]

Under a fixed trace constraint

\[
\sum_{j=1}^{D}\lambda_j=\tau,
\]

strict convexity of \(u\mapsto1/u\) gives

\[
\sum_{j=1}^{D}\frac{1}{\lambda_j}
\geq
\frac{D^2}{\tau},
\]

with equality only when all eigenvalues equal \(\tau/D\). The covariance is then isotropic:

\[
\Sigma=\frac{\tau}{D}I_D.
\]

Combining the two steps gives the isotropic Gaussian as the unique minimizer of \(J(p)\) under a fixed total-variance constraint:

\[
p^\star
=
\mathcal N\!\left(0,\frac{\tau}{D}I_D\right).
\]

Gaussianity and isotropy arise from different arguments. The first minimizes Fisher information among distributions with a fixed covariance. The second minimizes the inverse-covariance trace among covariances with fixed total variance.

## 8.6 The scope of the result

The argument supports a principled target distribution for the downstream risk framework analysed in LeJEPA. It is not a theorem that Gaussian embeddings are optimal for every task, every estimator or every control problem. The conclusion depends on the local-estimator approximation, smoothness conditions, downstream-function class and covariance constraint.

It is also important to distinguish isotropy from Gaussianity. A non-Gaussian distribution can have zero mean and identity covariance. Such a distribution satisfies the same second-order constraint but can have a different score field and a larger Fisher-information functional. Covariance regularization alone cannot enforce the full distributional property that appears in this risk argument.

The result motivates the next section. If the target is an isotropic Gaussian, a training objective needs to compare a high-dimensional empirical embedding distribution with that target. SIGReg does so through one-dimensional projections and characteristic functions rather than by estimating a full density.

**Figure 4.12. Local prediction bias contains a density-gradient term.**

Description: A query lies in an asymmetric neighbourhood of training embeddings. More samples occur on one side, pulling the local estimate toward the denser region. The score \(\nabla\log p(q)\) describes this local imbalance.

**Figure 4.13. Fisher information and covariance optimality are separate steps.**

Description: The first comparison holds covariance fixed and shows that the Gaussian minimizes the score-based Fisher-information functional. The second holds total variance fixed and shows that isotropic covariance minimizes the inverse-eigenvalue trace.