# 7. Why the covariance spectrum affects linear readouts

The covariance spectrum matters because a downstream model must estimate its parameters from a finite sample. If the representation varies strongly in some directions and barely varies in others, the regression problem is poorly conditioned. This is not a statement that isotropic embeddings are best for every task. It is a result about specific estimators under explicit assumptions.

## 7.1 The frozen-feature regression problem

Suppose an encoder is trained and then frozen. A downstream task provides pairs \((z_i,y_i)\), where \(z_i\in\mathbb R^D\) is a representation and \(y_i\) is a scalar target. A linear readout models

\[
y_i=z_i^\top\beta^\star+\varepsilon_i,
\]

where \(\beta^\star\) is the population coefficient and \(\varepsilon_i\) is zero-mean observation noise.

Let the centred design matrix be \(X\in\mathbb R^{N\times D}\), with rows \(z_i^\top\), and let \(\widehat\Sigma=X^\top X/N\) be the sample covariance. We focus on the population covariance \(\Sigma\) that the sample estimates. Assume \(\Sigma\) is positive definite and has eigen-decomposition

\[
\Sigma=U\Lambda U^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_D),
\]

with all \(\lambda_j>0\).

The eigenvalues describe how much variation is available to estimate a coefficient along each principal direction. If an eigenvalue is small, many different coefficients along that direction can produce similar predictions on the finite sample. The readout must estimate that coefficient from weak evidence.

## 7.2 Ridge shrinks weak directions more strongly

For a normalized ridge objective

\[
\widehat\beta_\lambda
=
\arg\min_\beta
\left[
\frac{1}{N}\|y-X\beta\|_2^2
+
\lambda\|\beta\|_2^2
\right],
\]

the solution is

\[
\widehat\beta_\lambda
=
(\widehat\Sigma+\lambda I)^{-1}
\frac{X^\top y}{N}.
\]

Under the population approximation \(\widehat\Sigma\approx\Sigma\), the expected estimate is

\[
\mathbb E[\widehat\beta_\lambda\mid\Sigma]
\approx
(\Sigma+\lambda I)^{-1}\Sigma\beta^\star.
\]

In the eigenbasis, each coefficient is multiplied by

\[
\frac{\lambda_j}{\lambda_j+\lambda}.
\]

The corresponding bias component is

\[
\operatorname{Bias}_j
=
-\frac{\lambda}{\lambda_j+\lambda}\beta^\star_j.
\]

When \(\lambda_j\) is large relative to \(\lambda\), the coefficient is only mildly shrunk. When \(\lambda_j\) is small, the shrinkage is strong. If the downstream task places weight on a weak feature direction, ridge may suppress that task-relevant coefficient.

Now hold the total feature variance fixed:

\[
\operatorname{tr}(\Sigma)
=
\sum_{j=1}^{D}\lambda_j
=
\tau.
\]

The average eigenvalue is \(\bar\lambda=\tau/D\). The largest possible worst-direction ridge bias is controlled by the smallest eigenvalue. Since

\[
\lambda_{\min}\leq \bar\lambda,
\]

an anisotropic covariance has at least one direction no stronger than the isotropic covariance with eigenvalues all equal to \(\bar\lambda\). A task whose coefficient lies in that weak direction suffers at least as much shrinkage as in the isotropic case, and more if \(\lambda_{\min}<\bar\lambda\).

This is a worst-case statement over task directions. It does not say that every downstream task performs worse with anisotropic features. A task aligned with a high-variance direction may be estimated well. The result says that severe anisotropy creates weak directions in which a linear task can become difficult.

## 7.3 Ordinary least squares exposes the variance penalty

Set \(\lambda=0\). Under full column rank, ordinary least squares gives

\[
\widehat\beta_{\mathrm{OLS}}
=
(X^\top X)^{-1}X^\top y.
\]

Assume the classical homoskedastic model

\[
\operatorname{Cov}(\varepsilon\mid X)=\sigma_\varepsilon^2 I.
\]

The conditional coefficient covariance is

\[
\operatorname{Cov}(\widehat\beta_{\mathrm{OLS}}\mid X)
=
\sigma_\varepsilon^2(X^\top X)^{-1}.
\]

With \(X^\top X\approx N\Sigma\), the expected scale of the coefficient variance is governed by

\[
\frac{\sigma_\varepsilon^2}{N}\operatorname{tr}(\Sigma^{-1})
=
\frac{\sigma_\varepsilon^2}{N}
\sum_{j=1}^{D}\frac{1}{\lambda_j}.
\]

Small covariance eigenvalues appear in the denominator. They therefore contribute disproportionately to estimator variance.

At fixed trace \(\sum_j\lambda_j=\tau\), convexity of \(u\mapsto1/u\) and Jensen's inequality give

\[
\frac{1}{D}\sum_{j=1}^{D}\frac{1}{\lambda_j}
\geq
\frac{1}{\frac{1}{D}\sum_{j=1}^{D}\lambda_j}
=
\frac{D}{\tau}.
\]

Hence

\[
\operatorname{tr}(\Sigma^{-1})
\geq
\frac{D^2}{\tau},
\]

with equality only when all eigenvalues equal \(\tau/D\). Under this fixed-trace, full-rank and homoskedastic linear model, isotropic covariance minimizes the trace of the OLS coefficient covariance. This is the linear downstream-risk result developed in LeJEPA; its scope is the specified estimator and assumptions (Balestriero and LeCun, 2025).

## 7.4 What the calculation establishes

The two results isolate complementary effects. Ridge introduces shrinkage bias, which is strongest in weak feature directions. Ordinary least squares avoids ridge shrinkage but becomes statistically unstable when the design contains weak directions. Equalizing the covariance spectrum removes the weakest-direction disadvantage under the assumptions above.

The result concerns coefficient estimation for a linear readout. It does not prove that isotropic covariance minimizes prediction error for every task, that Gaussian embeddings are always preferable, or that isotropy guarantees physical sufficiency. A representation may be isotropic while encoding the wrong variables. A nonlinear downstream function may depend on distributional structure that covariance does not describe.

The motivation for the next step is therefore limited and precise. If downstream prediction uses nonlinear local estimators, the arrangement of samples within the embedding space matters beyond the covariance spectrum. The local density then enters the estimator's bias.

**Figure 4.10. Ridge shrinkage is direction-dependent.**

Description: Coefficients aligned with large covariance eigenvalues are retained more strongly than coefficients aligned with weak eigenvalues. At fixed total variance, anisotropy creates a smallest-eigenvalue direction in which shrinkage is worst.

**Figure 4.11. OLS variance grows with the inverse covariance spectrum.**

Description: Each principal direction contributes in proportion to the reciprocal of its covariance eigenvalue. A narrow direction therefore contributes a large coefficient-variance term, even when the total feature variance is unchanged.