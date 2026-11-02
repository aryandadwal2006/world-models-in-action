# 9. Matching a distribution through its projections

The nonlinear-risk argument gave a reason to prefer an isotropic Gaussian embedding distribution under a stated family of downstream estimators and risk assumptions. It did not yet give a practical training objective. A neural encoder produces a high-dimensional batch of features; estimating and differentiating through its full probability density would be difficult. SIGReg avoids that density-estimation problem by comparing one-dimensional projections of the batch with one-dimensional Gaussian samples.

The construction joins three ideas. Characteristic functions identify probability distributions. The Cramér–Wold theorem says that all one-dimensional linear projections determine a multivariate distribution. The Epps–Pulley statistic provides a differentiable comparison between an empirical one-dimensional distribution and a Gaussian reference.

## 9.1 Why mean and covariance are not enough

Consider two scalar random variables. A standard normal variable has mean zero and variance one. A variable taking values \(-1\) and \(+1\), each with probability one half, has the same mean and variance. Their distributions are nevertheless different. The first is continuous; the second has two point masses.

Matching the first two moments constrains the centre and covariance of a distribution. It does not determine the full shape. The same limitation persists in a high-dimensional embedding. A zero mean and identity covariance do not imply that the batch is Gaussian, unimodal or even approximately Gaussian in every direction.

This matters because the Fisher-information argument did not identify only a covariance matrix. For fixed covariance, it identified the Gaussian as the distribution minimizing the score-based functional. A regularizer that controls only covariance does not enforce that target.

The question is how to compare a high-dimensional empirical distribution with a Gaussian reference without constructing a high-dimensional density estimator.

## 9.2 The characteristic function

For a scalar random variable \(U\), define its characteristic function by

\[
\varphi_U(t)=\mathbb E[e^{itU}],
\qquad t\in\mathbb R.
\]

The complex exponential has unit magnitude, so this expectation exists for every probability distribution. More importantly, the characteristic function uniquely determines the distribution. If two scalar random variables have the same characteristic function at every frequency \(t\), they have the same law.

For a standard normal variable,

\[
\varphi_0(t)=e^{-t^2/2}.
\]

Given scalar samples \(u_1,\ldots,u_N\), the empirical characteristic function is

\[
\widehat\varphi_N(t)
=
\frac1N\sum_{n=1}^{N}e^{itu_n}.
\]

It is a finite average of sine and cosine contributions. Unlike a high-order moment, it does not require raising samples to large powers. The bounded exponential also avoids the polynomial growth that can make high-order moment losses numerically unstable.

A characteristic function comparison still requires a range of frequencies and a weighting rule. Frequencies near zero describe broad distributional structure, while larger frequencies reveal finer differences. A numerical objective must decide how much weight each frequency receives and how the integral is approximated.

## 9.3 Why one-dimensional projections are sufficient in principle

Let \(Z\in\mathbb R^D\) be a random embedding vector and let \(a\in\mathbb S^{D-1}\) be a unit direction. Its scalar projection is

\[
U_a=a^\top Z.
\]

The Cramér–Wold theorem states that two probability distributions on \(\mathbb R^D\) are equal if and only if their projections have the same one-dimensional distribution in every direction:

\[
a^\top X\overset d=a^\top Y
\quad\text{for all }a\in\mathbb S^{D-1}
\quad\Longleftrightarrow\quad
X\overset d=Y.
\]

For a standard isotropic Gaussian \(G\sim\mathcal N(0,I_D)\), every unit projection is standard normal:

\[
a^\top G\sim\mathcal N(0,1).
\]

Consequently, a multivariate Gaussian-matching problem can be characterized through scalar normality in all directions. This is the mathematical reason projection-based distribution matching is possible.

The phrase *all directions* is essential. Equality on a finite collection of projections does not generally identify an arbitrary multivariate distribution. A finite set of directions may miss a discrepancy concentrated in another orientation. Random projections provide a stochastic sketch of the distribution, not a finite-sample proof of exact multivariate equality.

## 9.4 From a goodness-of-fit test to a training objective

For a unit direction \(a\), project a batch of embeddings:

\[
u_n=a^\top z_n,\qquad n=1,\ldots,N.
\]

The Epps–Pulley statistic compares the empirical characteristic function of these projected samples with the standard-normal characteristic function:

\[
T_{\mathrm{EP}}(u_{1:N})
=
N\int_{-\infty}^{\infty}
\left|
\widehat\varphi_N(t)-e^{-t^2/2}
\right|^2w(t)\,dt.
\]

Here \(w(t)\) is a nonnegative frequency window that makes the discrepancy integrable and controls how different frequencies contribute. The target characteristic function and the window serve different purposes: the former describes the null distribution; the latter weights the comparison.

For a set of tested directions \(\mathcal A\), the formal union-intersection testing construction uses the largest directional discrepancy:

\[
T_{\max}
=
\max_{a\in\mathcal A}T_{\mathrm{EP}}(u_{1:N}^{(a)}).
\]

A hypothesis test asks whether any tested direction gives evidence against the Gaussian null. The practical training regularizer uses a different aggregation:

\[
\mathcal L_{\mathrm{SIGReg}}
=
\frac1M\sum_{m=1}^{M}
T_{\mathrm{EP}}
\left(
a_m^\top z_1,\ldots,a_m^\top z_N
\right),
\]

where \(a_1,\ldots,a_M\) are sampled unit directions. Averaging spreads gradient contributions across the sampled projections. The maximum and the average should not be conflated: they correspond to different statistical and optimization purposes. (Balestriero and LeCun, 2025)

The one-dimensional integral is evaluated numerically. In the paper's displayed algorithm, the frequency interval is discretized using 17 points over \([-5,5]\), and a Gaussian weighting is used in the quadrature. The finite grid approximates the integral; it is not the definition of the underlying characteristic-function comparison. The paper also exploits symmetry of the integrand to increase the effective frequency coverage without doubling the computation.

## 9.5 Why resampling directions can work

A unit direction can be sampled by drawing a standard Gaussian vector and normalizing it:

\[
a_m=\frac{g_m}{\|g_m\|_2},
\qquad
g_m\sim\mathcal N(0,I_D).
\]

Each minibatch then evaluates a finite set of random projections. Directions are resampled during training, so the same finite set need not be used throughout optimization. The number of directions seen across many updates can be substantially larger than the number used in one update.

This distinction explains why a small number of slices per minibatch does not mean that the model is constrained only along those directions forever. Across training, different projections expose different aspects of the embedding distribution. The method trades exact all-direction comparison for a stochastic approximation whose coverage accumulates over optimization.

The computation consists primarily of projecting the batch, evaluating scalar characteristic functions and accumulating quadrature values. Its cost depends on batch size \(N\), feature dimension \(D\), number of projections \(M\), and number of frequency knots. The paper's linear-in-\(N\) complexity statement holds when the other quantities are fixed. It should not be read as independence from embedding dimension or number of projections.

For distributed training, each worker can accumulate local complex-exponential sums and combine them through an all-reduce operation. The statistic is therefore compatible with data-parallel training without gathering every projected sample on one device. The communication and projection counts remain part of the practical cost.

## 9.6 SIGReg inside LeJEPA

SIGReg specifies a distributional constraint. It does not say which views of an observation should agree. LeJEPA combines it with a direct multi-view agreement objective.

Let \(z_{n,v}\) denote the embedding of sample \(n\) under view \(v\). Suppose the first \(V_g\) views are global crops. Their per-sample mean is

\[
\mu_n
=
\frac1{V_g}\sum_{v=1}^{V_g}z_{n,v}.
\]

The agreement term draws every view embedding toward this global-view centre:

\[
\mathcal L_{\mathrm{pred}}
=
\frac1{NV}
\sum_{n=1}^{N}\sum_{v=1}^{V}
\|z_{n,v}-\mu_n\|_2^2.
\]

SIGReg is applied to each view's batch distribution separately and then averaged across views:

\[
\mathcal L_{\mathrm{reg}}
=
\frac1V\sum_{v=1}^{V}
\operatorname{SIGReg}
\left(\{z_{n,v}\}_{n=1}^{N}\right).
\]

The combined objective is

\[
\mathcal L_{\mathrm{LeJEPA}}
=
(1-\alpha)\mathcal L_{\mathrm{pred}}
+
\alpha\mathcal L_{\mathrm{reg}}.
\]

This formulation is different from the EMA target-encoder and predictor structure used by I-JEPA and original V-JEPA. The global-view mean provides the agreement target, while SIGReg constrains the distribution of each view. There is no requirement in the base objective for an EMA teacher or a separate predictor network. The regularizer also does not pool all views into one distribution before testing; each view distribution is regularized separately. This matters when global and local views have different crop distributions. (Balestriero and LeCun, 2025)

## 9.7 What SIGReg establishes, and what it approximates

The construction has a clear chain. The downstream-risk analysis motivates a Gaussian target under its assumptions. Cramér–Wold relates multivariate equality to equality of all one-dimensional projections. Characteristic functions identify each projected distribution. Epps–Pulley supplies a scalar goodness-of-fit statistic. Random directions, finite batches and numerical quadrature make the statistic trainable.

Each step has a boundary. A Gaussian target is not justified by Cramér–Wold; it is motivated by the earlier risk result. A finite number of projections is not an exact all-directions certificate. A finite batch gives a noisy empirical characteristic function. A finite frequency grid approximates the integral. A low training regularizer is therefore evidence that the sampled projections approximately match the Gaussian reference under the chosen numerical procedure, not proof that the population embedding distribution is exactly Gaussian.

SIGReg makes distributional shape part of the training objective rather than relying only on variance and covariance. It does so by sketching the distribution through random projections. The next section considers a different decomposition: centre the batch, constrain coordinate scale directly, and compare shape after scale has been removed.

**Figure 4.14. The Cramér–Wold theorem links multivariate distributions to all one-dimensional projections.**

Description: A high-dimensional embedding distribution is projected along many unit directions. Each projection is compared with a one-dimensional Gaussian reference. The theorem concerns all directions; training uses a finite random sample of them.

**Figure 4.15. SIGReg compares empirical and Gaussian characteristic functions.**

Description: The empirical characteristic function is evaluated over a finite frequency grid and compared with the standard-normal characteristic function. A weighted squared discrepancy is integrated numerically, then averaged over sampled directions.