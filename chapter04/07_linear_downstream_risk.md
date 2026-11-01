# 7. Linear downstream risk and the covariance spectrum

A representation is often evaluated by freezing its encoder and fitting a small predictor on top of the resulting embeddings. This procedure isolates one question: how accessible is the information in the representation to a restricted downstream estimator? The simplest case is linear regression. Its mathematics makes the effect of embedding geometry explicit.

The aim of this section is not to prove that linear probes characterize every downstream task. It is to derive one concrete reason why a representation's covariance spectrum matters.

## 7.1 The frozen-feature problem

Suppose a frozen encoder maps observations to \(K\)-dimensional features. For a downstream dataset, arrange \(N\) feature vectors as rows of

\[
X\in\mathbb R^{N\times K},
\]

and collect the scalar targets in \(y\in\mathbb R^N\). A linear probe predicts

\[
\widehat y=X\beta.
\]

With a quadratic penalty on the coefficient vector, ridge regression solves

\[
\widehat\beta_\lambda
=
\arg\min_{\beta\in\mathbb R^K}
\left\{
\|y-X\beta\|_2^2+\lambda\|\beta\|_2^2
\right\},
\qquad \lambda>0.
\]

Differentiating with respect to \(\beta\) and setting the gradient to zero gives

\[
-2X^\top(y-X\widehat\beta_\lambda)
+2\lambda\widehat\beta_\lambda=0.
\]

Therefore

\[
\widehat\beta_\lambda
=
(X^\top X+\lambda I)^{-1}X^\top y.
\]

The learned coefficient vector depends on the geometry of the feature matrix through \(X^\top X\). If the features are centred, this matrix is proportional to their empirical covariance. Its eigenvalues determine how much sample variation is available along each principal direction.

## 7.2 Ridge bias depends on the weak directions

Assume a linear downstream relationship

\[
y=X\beta_{\mathrm{true}}+\varepsilon,
\qquad
\mathbb E[\varepsilon\mid X]=0.
\]

Conditioning on \(X\),

\[
\begin{aligned}
\mathbb E[\widehat\beta_\lambda\mid X]
&=(X^\top X+\lambda I)^{-1}X^\top
\mathbb E[y\mid X]\\
&=(X^\top X+\lambda I)^{-1}X^\top X\beta_{\mathrm{true}}.
\end{aligned}
\]

Subtracting the true coefficient vector gives

\[
\operatorname{Bias}(\widehat\beta_\lambda\mid X)
=
-\lambda(X^\top X+\lambda I)^{-1}\beta_{\mathrm{true}}.
\]

Let the eigendecomposition of \(X^\top X\) be

\[
X^\top X=Q\Lambda Q^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_K),
\]

with positive eigenvalues. In this basis, ridge shrinks the coefficient in direction \(q_k\) by the factor

\[
\frac{\lambda_k}{\lambda_k+\lambda}.
\]

Equivalently, the magnitude of the bias component in that direction is multiplied by

\[
\frac{\lambda}{\lambda_k+\lambda}.
\]

A direction with a large eigenvalue is well represented in the sample: the features vary substantially along it. Ridge therefore shrinks that coefficient less. A direction with a small eigenvalue is weakly supported by the data. Ridge shrinks its coefficient more strongly.

This observation permits a controlled comparison of two feature geometries. Hold the total covariance trace fixed, but allow its eigenvalues to be distributed differently. Let \(\bar\lambda\) denote the mean eigenvalue. In the isotropic case, every eigenvalue equals \(\bar\lambda\). In the anisotropic case, at least one eigenvalue is smaller than \(\bar\lambda\).

Choose a downstream task whose true coefficient vector aligns with the eigenvector associated with the smallest anisotropic eigenvalue, denoted \(\lambda_{\min}\). For this task,

\[
\|\operatorname{Bias}_{\mathrm{aniso}}\|
=
\frac{\lambda}{\lambda_{\min}+\lambda}
\|\beta_{\mathrm{true}}\|,
\]

whereas for the isotropic geometry,

\[
\|\operatorname{Bias}_{\mathrm{iso}}\|
=
\frac{\lambda}{\bar\lambda+\lambda}
\|\beta_{\mathrm{true}}\|.
\]

Because \(\lambda_{\min}<\bar\lambda\),

\[
\|\operatorname{Bias}_{\mathrm{aniso}}\|
>
\|\operatorname{Bias}_{\mathrm{iso}}\|.
\]

The result is a worst-case statement. For any anisotropic covariance with the same total variance, there exists a downstream linear task aligned with its weakest eigen-direction for which ridge incurs greater bias than under the isotropic covariance. It does not say that every task aligned with every direction is worse. [Balestriero and LeCun, 2025]

## 7.3 OLS variance and the inverse spectrum

Ridge introduces bias by shrinking coefficients. To isolate estimator variance, set \(\lambda=0\). The resulting ordinary least-squares estimator is

\[
\widehat\beta_{\mathrm{OLS}}
=
(X^\top X)^{-1}X^\top y,
\]

assuming \(X\) has full column rank. Under the classical homoskedastic linear model with

\[
\operatorname{Cov}(\varepsilon\mid X)=\sigma^2 I,
\]

its conditional covariance is

\[
\operatorname{Cov}(\widehat\beta_{\mathrm{OLS}}\mid X)
=
\sigma^2(X^\top X)^{-1}.
\]

The trace summarizes the total variance across coefficient directions:

\[
\operatorname{tr}\operatorname{Cov}(\widehat\beta_{\mathrm{OLS}}\mid X)
=
\sigma^2\operatorname{tr}\left((X^\top X)^{-1}\right)
=
\sigma^2\sum_{k=1}^{K}\frac{1}{\lambda_k}.
\]

A small eigenvalue becomes a large reciprocal. Consequently, a weak feature direction makes the fitted coefficient sensitive to changes in the sampled data.

Now hold total feature energy fixed:

\[
\sum_{k=1}^{K}\lambda_k=T.
\]

The function \(f(u)=1/u\) is strictly convex for \(u>0\). Jensen's inequality gives

\[
\frac{1}{K}\sum_{k=1}^{K}\frac{1}{\lambda_k}
\ge
\frac{1}{
\frac{1}{K}\sum_{k=1}^{K}\lambda_k
}
=
\frac{K}{T}.
\]

Thus,

\[
\sum_{k=1}^{K}\frac{1}{\lambda_k}
\ge
\frac{K^2}{T},
\]

with equality if and only if all eigenvalues are equal. Under the fixed-trace and full-rank assumptions, the isotropic covariance spectrum uniquely minimizes the trace of the OLS coefficient covariance. [Balestriero and LeCun, 2025]

This derivation explains why anisotropy can increase estimator variance even if the total amount of feature variance is unchanged. Concentrating variance into a few directions forces the remaining directions to become narrower. The inverse spectrum penalizes those narrow directions heavily.

## 7.4 What the linear result establishes

The two calculations isolate complementary effects:

- With positive ridge regularization, a task aligned with a weak covariance direction experiences stronger coefficient shrinkage and larger bias.
- With ordinary least squares and homoskedastic noise, the sum of coefficient variances is proportional to the sum of inverse eigenvalues and is minimized by an isotropic spectrum at fixed total variance.

These results rely on the linear model, the specified regularizer, the design assumptions and the comparison at fixed total variance. They describe the estimator's coefficients, not every possible measure of prediction error for an arbitrary nonlinear learner. In particular, they do not establish that a Gaussian feature distribution is optimal for all downstream tasks.

The linear analysis does establish a concrete design pressure: if a foundation representation must support unknown linear readouts, severe anisotropy can make some readouts poorly conditioned. Equalizing second-order variation removes this weakest-direction disadvantage under the model just analysed.

However, matching covariance does not determine the entire feature distribution. Two distributions can have identical covariance and very different local density structure. A nonlinear predictor that estimates a target by averaging nearby embeddings can be affected by that structure. The next section derives how the density gradient enters local prediction bias, and why the Fisher-information functional appears in the LeJEPA analysis.
