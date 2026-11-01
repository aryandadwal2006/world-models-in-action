# 8. Nonlinear downstream risk and Fisher information

The covariance spectrum is only a second-order description of an embedding distribution. It does not determine how samples are arranged locally. This matters for nonlinear predictors that estimate a target by examining nearby embeddings. Their predictions depend not only on the target function, but also on where training samples are concentrated.

The key quantity is the score of the embedding density,

\[
s_p(z)=\nabla\log p(z),
\]

and the global statistic that measures its expected squared magnitude,

\[
J(p)
=
\mathbb E_{Z\sim p}
\left[\|\nabla\log p(Z)\|_2^2\right]
=
\int_{\mathbb R^K}
\|\nabla\log p(z)\|_2^2p(z)\,dz.
\]

LeJEPA calls \(J(p)\) the Fisher-information functional of the embedding density. It appears naturally when the bias of local nonparametric estimators is expanded and integrated over query points. The derivation explains why matching only mean and covariance leaves a gap.

## 8.1 A local estimator sees both the target and the sample density

Let \(Z\in\mathbb R^K\) be a frozen embedding with density \(p(z)\). Let the downstream regression function be

\[
\eta(z)=\mathbb E[Y\mid Z=z].
\]

A local estimator predicts \(\eta(q)\) at a query point \(q\) by averaging labels from training embeddings near \(q\). Two examples are radius-based k-nearest-neighbour regression and Nadaraya–Watson kernel regression.

For a fixed radius \(r_0\), the radius-based estimator is

\[
\widehat\eta_{\mathrm{kNN}}(q)
=
\frac{1}{|\mathcal N_{r_0}(q)|}
\sum_{i\in\mathcal N_{r_0}(q)}Y_i,
\qquad
\mathcal N_{r_0}(q)
=
\{i:\|Z_i-q\|_2\le r_0\}.
\]

The number of neighbours is not fixed. It depends on the local density of embeddings. In a dense region, many training points fall inside the ball; in a sparse region, fewer do.

The Nadaraya–Watson estimator uses a smooth kernel:

\[
\widehat\eta_h(q)
=
\frac{
\sum_{i=1}^{N}K_h(q-Z_i)Y_i
}{
\sum_{i=1}^{N}K_h(q-Z_i)
},
\qquad
K_h(u)=h^{-K}K(u/h).
\]

The kernel assigns higher weight to nearby samples. The bandwidth \(h\) controls the size of the neighbourhood. The denominator normalizes the empirical weights.

Both estimators average labels around the query, but they do so with different weighting profiles. Their bias depends on the shape of the target function and on how training points are distributed around the query.

## 8.2 Deriving the density-gradient term

Consider the large-sample conditional expectation of the kernel estimator:

\[
\mathbb E[\widehat\eta_h(q)]
\approx
\frac{
\int \eta(q+u)p(q+u)K_h(u)\,du
}{
\int p(q+u)K_h(u)\,du
}.
\]

The numerator contains the product \(\eta(q+u)p(q+u)\). The denominator contains \(p(q+u)\). To understand the ratio for small bandwidth, expand both functions around \(q\).

For the regression function,

\[
\eta(q+u)
=
\eta(q)
+
\nabla\eta(q)^\top u
+
\frac12u^\top\nabla^2\eta(q)u
+
o(\|u\|^2).
\]

For the embedding density,

\[
p(q+u)
=
p(q)
+
\nabla p(q)^\top u
+
\frac12u^\top\nabla^2p(q)u
+
o(\|u\|^2).
\]

Assume a normalized, symmetric, radially structured kernel with zero first moment and isotropic second moment:

\[
\int K(v)\,dv=1,\qquad
\int vK(v)\,dv=0,\qquad
\int vv^\top K(v)\,dv=\mu_2(K)I_K.
\]

The symmetry makes the first-order terms vanish after integration. The second-order terms contribute traces of the Hessians. Multiplying the expansions in the numerator also produces a cross term between the target gradient and the density gradient. After dividing by the denominator and retaining the leading order in \(h\), the pointwise bias is

\[
\operatorname{Bias}\big[\widehat\eta_h(q)\big]
=
\frac{h^2\mu_2(K)}{2}
\left[
\Delta\eta(q)
+
2\nabla\eta(q)^\top\nabla\log p(q)
\right]
+
o(h^2).
\]

Here \(\Delta\eta(q)=\operatorname{tr}(\nabla^2\eta(q))\) is the Laplacian of the downstream regression function.

The expression contains two distinct terms.

**Target curvature.** The Laplacian \(\Delta\eta(q)\) measures the local curvature of the target function. Even if training points are distributed symmetrically around \(q\), averaging nearby target values can differ from the target value at the centre when the function is curved.

**Density-gradient coupling.** The term \(\nabla\eta(q)^\top\nabla\log p(q)\) measures the alignment between the direction in which the target changes and the direction in which embedding density increases relative to its local value. If nearby points are concentrated more heavily on one side of the query, the local average is pulled toward that side. When the target function changes along the same direction, the resulting average is systematically shifted.

The same density-gradient coupling appears for radius-based k-NN. Its leading bias is

\[
\operatorname{Bias}\big[\widehat\eta_{\mathrm{kNN}}(q)\big]
=
\frac{r_0^2}{2(K+2)}
\left[
\Delta\eta(q)
+
2\nabla\eta(q)^\top\nabla\log p(q)
\right]
+
o(r_0^2).
\]

The coefficients differ because a uniform ball and a smooth kernel have different second moments. The dependence on the local score \\(\\nabla\\log p(q)\\) is common to both estimators. The k-NN expansion assumes a smooth embedding density with bounded derivatives through order three, finite covariance and finite Fisher information, together with a twice continuously differentiable target function. [Balestriero and LeCun, 2025]

## 8.3 From a pointwise bias to Fisher information

A downstream representation is evaluated over a distribution of query points, not at one fixed \(q\). Let queries follow the embedding density \(p\). The integrated squared bias is

\[
\operatorname{ISB}
=
\int
\operatorname{Bias}(q)^2p(q)\,dq.
\]

The density-dependent part of the pointwise bias contains

\[
\nabla\eta(q)^\top\nabla\log p(q).
\]

To expose its contribution, consider the assumptions used in the k-NN analysis of LeJEPA. The density is sufficiently smooth, has bounded derivatives through the required order, finite covariance and finite Fisher information. The regression function is twice continuously differentiable. In addition, the downstream gradient is assumed to have mean zero and an isotropic second-moment matrix:

\[
\mathbb E[\nabla\eta(Z)]=0,
\qquad
\mathbb E[
\nabla\eta(Z)\nabla\eta(Z)^\top
]
=
\tau_g^2 I_K,
\]

with \(\tau_g^2>0\). Queries are drawn from \(p\), and the embedding covariance is subject to a fixed scalar constraint.

Under these conditions, averaging the squared gradient coupling over downstream functions gives

\[
\begin{aligned}
\mathbb E_\eta
\mathbb E_{Z\sim p}
\left[
\big(\nabla\eta(Z)^\top\nabla\log p(Z)\big)^2
\right]
&=
\tau_g^2
\mathbb E_{Z\sim p}
\left[
\|\nabla\log p(Z)\|_2^2
\right]\\
&=
\tau_g^2 J(p).
\end{aligned}
\]

The functional

\[
\boxed{
J(p)=
\int
\|\nabla\log p(z)\|_2^2p(z)\,dz
}
\]

is the expected squared score of the embedding density. It is a property of the density itself, not of one coordinate variance or one covariance entry.

The full k-NN expansion also contains a curvature term and a cross term involving the target Hessian. Under the paper's derivative prior, the curvature term contributes a finite term independent of the embedding density. Unless an additional orthogonality assumption makes the cross term vanish, it also contributes at order \\(O(r_0^4)\\). The density-dependent contribution singled out by the analysis is therefore

\[
\operatorname{ISB}_{\mathrm{kNN}}
=
\frac{r_0^4}{(K+2)^2}
\tau_g^2J(p)
+
O(r_0^4).
\]

For Nadaraya–Watson regression, the paper considers a smoothness class with bounded target gradient and Laplacian. If

\[
\|\nabla\eta(q)\|_2\le L,
\qquad
|\Delta\eta(q)|\le B,
\]

then the integrated squared bias is bounded by

\[
\operatorname{ISB}_{\mathrm{kernel}}
\le
\left(
\frac{h^2\mu_2(K)}{2}
\right)^2
\left[
2B^2+8L^2J(p)
\right]
+
o(h^4).
\]

The kernel result additionally assumes a normalized, symmetric and isotropic kernel with finite second moment and finite roughness, a sufficiently light-tailed embedding density, and the usual small-bandwidth asymptotic regime. In particular, the pointwise expansion is stated where \\(p(q)>0\\), with \\(h\\to0\\) and \\(Nh^K\\to\\infty\\). The theorem bounds worst-case integrated squared bias over a smoothness class of target functions. These conditions delimit the result. [Balestriero and LeCun, 2025]

The pointwise variance also depends on the density:

\[
\operatorname{Var}[\widehat\eta_h(q)]
\approx
\frac{R(K)}{Nh^K}\frac{v(q)}{p(q)},
\qquad
R(K)=\int K(u)^2\,du,
\]

where \(v(q)=\operatorname{Var}(Y\mid Z=q)\). A query in a sparse part of the embedding distribution has fewer nearby training samples and therefore higher pointwise variance. But the paper evaluates the integrated variance under the same density used to sample queries. Multiplying by \(p(q)\) and integrating cancels the density:

\[
\begin{aligned}
\mathcal V(h;p)
&=\int \operatorname{Var}[\widehat\eta_h(q)]p(q)\,dq\\
&\approx
\frac{R(K)}{Nh^K}\int v(q)\,dq.
\end{aligned}
\]

Under this evaluation measure, the leading integrated variance is independent of \\(p\\). The density-dependent optimization therefore enters through the integrated squared bias bound, while the leading integrated variance does not distinguish candidate embedding densities. This is a more precise statement than saying that every pointwise variance term is independent of the density.

The Fisher-information functional appears because the local estimators are biased by nonuniform sample density. The representation controls that density. For the estimator families considered, the density score is therefore part of the downstream prediction error.

## 8.4 Why an isotropic Gaussian minimizes the functional

The next step is a variational statement about probability densities. Let \(p\) be a mean-zero density on \(\mathbb R^K\), with positive-definite covariance matrix

\[
\Sigma=\mathbb E[ZZ^\top].
\]

Under the regularity conditions needed for integration by parts and finite Fisher information, define the score

\[
s_p(z)=\nabla\log p(z).
\]

Because \(s_p(z)p(z)=\nabla p(z)\), integration by parts gives

\[
\mathbb E[Zs_p(Z)^\top]
=
\int z\,\nabla p(z)^\top dz
=
-I_K.
\]

Now consider the positive-semidefinite matrix

\[
\mathbb E\left[
\left(s_p(Z)+\Sigma^{-1}Z\right)
\left(s_p(Z)+\Sigma^{-1}Z\right)^\top
\right].
\]

Expanding the expectation and using
\(\mathbb E[s_p(Z)s_p(Z)^\top]\) for the Fisher-information matrix, together with
\(\mathbb E[ZZ^\top]=\Sigma\), yields

\[
\mathbb E[s_p(Z)s_p(Z)^\top]-\Sigma^{-1}
\succeq 0.
\]

Taking traces gives

\[
J(p)\ge\operatorname{tr}(\Sigma^{-1}).
\]

Equality requires

\[
s_p(z)=-\Sigma^{-1}z
\]

almost everywhere. Integrating this score identity gives a Gaussian density with covariance \(\Sigma\):

\[
p(z)=\mathcal N(0,\Sigma).
\]

Thus, for a fixed covariance matrix, the Gaussian uniquely minimizes the Fisher-information functional.

A second optimization determines the covariance shape. Write the positive eigenvalues of \(\Sigma\) as \(\lambda_1,\ldots,\lambda_K\). Then

\[
\operatorname{tr}(\Sigma^{-1})
=
\sum_{k=1}^{K}\frac{1}{\lambda_k}.
\]

Under a fixed trace constraint

\[
\sum_{k=1}^{K}\lambda_k=c,
\]

strict convexity of \(u\mapsto1/u\) gives

\[
\sum_{k=1}^{K}\frac{1}{\lambda_k}
\ge
\frac{K^2}{c},
\]

with equality only when \(\lambda_k=c/K\) for every \(k\). The covariance must therefore be isotropic:

\[
\Sigma=\frac{c}{K}I_K.
\]

Combining the two steps gives the unique minimizer under fixed total variance:

\[
p^*(z)
=
\mathcal N\!\left(0,\frac{c}{K}I_K\right).
\]

LeJEPA also considers other scalar covariance constraints, including determinant, Frobenius norm and spectral radius. For the specific constraints analysed in the paper, the minimizer remains an isotropic Gaussian with the scale selected to satisfy the constraint.

## 8.5 What the result does and does not say

The argument has a clear chain:

1. Local prediction bias depends on the gradient of the downstream function and the score of the embedding density.
2. Under the stated smoothness and gradient assumptions, integrated bias contains the Fisher-information functional \(J(p)\).
3. For a fixed covariance, the Gaussian uniquely minimizes \(J(p)\).
4. Under a fixed scalar covariance constraint, isotropic covariance minimizes the Gaussian value \(\operatorname{tr}(\Sigma^{-1})\).

This explains the Gaussian target in LeJEPA. It is not a theorem that Gaussian latents are best for every task, every estimator, every control problem or every conditional dynamics model. The result is derived for specified downstream estimator families and risk criteria. Its value is that it gives a principled target under those assumptions, rather than choosing a distribution only because it seems convenient.

It is also important to distinguish isotropy from Gaussianity. A distribution can have covariance proportional to the identity and still be non-Gaussian. Such a distribution matches second-order geometry but may have a different score field and a larger value of \(J(p)\). Covariance regularization alone cannot enforce the density property that appears in the local-estimation risk.

The next section turns this theoretical target into a practical objective. SIGReg does not estimate a full high-dimensional density. It compares one-dimensional projections with the corresponding Gaussian projections. The distinction between the all-directions theorem and the finite stochastic sketch used during training is essential.
