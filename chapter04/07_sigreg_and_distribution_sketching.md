# 7. Distribution matching by projection: SIGReg

The preceding section identified a target distribution through downstream prediction risk. The representation should not merely have nonzero variance or an isotropic covariance matrix. Under the risk assumptions analysed there, the embedding density should be an isotropic Gaussian.

A training objective must now answer a computational question: how can a neural encoder be encouraged to match a high-dimensional distribution without estimating that distribution explicitly?

The key is to replace one high-dimensional comparison with a collection of one-dimensional comparisons. The mathematical justification comes from the Cramér–Wold theorem. The practical statistic used by LeJEPA is based on empirical characteristic functions.

## 7.1 Why moments do not define the distribution

Suppose two embedding distributions have the same mean and covariance. They need not have the same shape. In one dimension, a standard normal variable and a random variable taking values \(-1\) and \(+1\) with equal probability both have mean zero and variance one. Their characteristic functions differ:

\[
\varphi_{\mathrm{normal}}(t)=e^{-t^2/2},
\qquad
\varphi_{\mathrm{two\ point}}(t)=\cos(t).
\]

The first two moments therefore do not identify the distribution. Matching more moments can distinguish more distributions, but matching any fixed finite number of moments still does not determine a general probability distribution. For a particular target family, moment constraints may be useful; they are not a general certificate of distributional equality.

This is the limitation of using only covariance-based regularization when the theoretical target is a Gaussian density. Covariance controls second-order structure. It does not control all higher-order shape.

## 7.2 The characteristic function

For a scalar random variable \(U\), its characteristic function is

\[
\varphi_U(t)=\mathbb E[e^{itU}],
\qquad t\in\mathbb R.
\]

The characteristic function exists for every probability distribution because the complex exponential has unit magnitude. It uniquely determines the distribution of \(U\).

For a standard normal random variable,

\[
\varphi_0(t)=e^{-t^2/2}.
\]

Given scalar samples \(u_1,\ldots,u_N\), the empirical characteristic function is

\[
\widehat\varphi_N(t)
=
\frac{1}{N}\sum_{n=1}^{N}e^{itu_n}.
\]

It is a finite average of complex exponentials. Unlike high-order moments, it does not require raising samples to increasingly large powers. It is differentiable with respect to each sample and can be aggregated over distributed workers using sum reductions.

The characteristic function is therefore a useful object for gradient-based distribution matching. Its bounded integrand also avoids some of the unstable growth that can occur in high-order moment statistics. The weighting of frequencies still matters, and the practical statistic remains an approximation to an integral.

## 7.3 Cramér–Wold: why projections can identify a distribution

Let \(Z\in\mathbb R^K\) be an embedding vector and let \(a\in\mathbb S^{K-1}\) be a unit direction. The one-dimensional projection is

\[
U_a=a^\top Z.
\]

A multivariate distribution is determined by the distributions of all its one-dimensional linear projections. In particular,

\[
a^\top X\overset{d}{=}a^\top Y
\quad\text{for every }a\in\mathbb S^{K-1}
\]

if and only if

\[
X\overset{d}{=}Y.
\]

This is the Cramér–Wold principle. It provides the identification argument for projection-based distribution matching. If every one-dimensional projection of an embedding distribution agrees with the corresponding projection of the target, then the multivariate distributions agree.

For an isotropic Gaussian target,

\[
Z\sim\mathcal N(0,I_K)
\quad\Longrightarrow\quad
a^\top Z\sim\mathcal N(0,1)
\]

for every unit vector \(a\). A high-dimensional Gaussian-matching problem can therefore be examined through scalar normality tests along directions.

The all-directions qualification is essential. Matching a finite set of projections is not, in general, sufficient to identify an arbitrary multivariate distribution exactly. A finite set may miss a structured discrepancy that lies between sampled directions. Increasing the number and diversity of projections improves the sketch, but the finite training objective remains a stochastic approximation to the all-directions condition.

LeJEPA makes a stronger, assumption-dependent argument about how many projections are needed in practice. Its analysis treats the directional discrepancy as a function over the unit sphere and assumes a degree of Sobolev smoothness for the embedding distribution. Under that regularity, the expected discrepancy between the finite-direction test and the all-directions criterion decays at a rate of the form

\[
O\!\left(M^{-2\alpha/(D-1)}\right),
\]

where \(M\) is the number of sampled directions, \(D\) is the embedding dimension and \(\alpha\) describes the smoothness of the directional discrepancy. The bound does not hold independently of regularity, and its constants depend on dimension and smoothness. The paper argues that the Gaussian target and modern encoder architectures produce sufficiently smooth embedding distributions for a number of directions proportional to the dimension to give a useful approximation in the regimes studied.

There is also a distinction between the number of directions in one minibatch and the number observed over training. SIGReg resamples directions as optimization proceeds. The union of directions seen across many steps can cover the sphere much more thoroughly than a fixed set of the same per-step size. The paper reports that a small resampled set can outperform a much larger fixed set in its controlled comparisons. This cumulative-coverage argument is part of the practical method, not a replacement for the all-directions theorem.

These results justify the sketch under stated smoothness and sampling assumptions. They do not make one finite minibatch an exact multivariate distribution test.

## 7.4 A directional statistical test

Let \(P_\theta\) denote the distribution of learned embeddings and \(Q=\mathcal N(0,I_K)\) the target. For a unit direction \(a\), the projected distributions are

\[
P_\theta^{(a)}=(a^\top)_\#P_\theta,
\qquad
Q^{(a)}=(a^\top)_\#Q.
\]

A one-dimensional goodness-of-fit statistic \(T\) can compare samples from \(P_\theta^{(a)}\) with the standard-normal target. Repeating the comparison over a set of directions

\[
\mathcal A=\{a_1,\ldots,a_M\}
\]

produces directional evidence about the multivariate distribution.

The formal union-intersection construction in LeJEPA aggregates directional test statistics through a maximum:

\[
T_{\mathcal A}
=
\max_{a\in\mathcal A}
T\!\left(
\{a^\top z_n\}_{n=1}^{N}
\right).
\]

The maximum is natural for a statistical test: a sufficiently large discrepancy in any tested direction is evidence against the null. The paper relates this construction to the Cramér–Wold principle and gives level and power guarantees under its assumptions.

The training regularizer takes a different form. SIGReg averages the directional statistics:

\[
\mathcal L_{\mathrm{SIGReg}}
=
\frac{1}{M}
\sum_{m=1}^{M}
T\!\left(
\{a_m^\top z_n\}_{n=1}^{N}
\right).
\]

The average distributes gradient contributions across sampled directions. A maximum can concentrate the optimization signal on the current worst direction, leaving other directions with no direct contribution at that step. The maximum and the average therefore serve different roles: the former is a test statistic, the latter a practical training objective.

## 7.5 The Epps–Pulley statistic

LeJEPA uses an Epps–Pulley-style comparison of empirical and target characteristic functions. For a projected sample \(u_1,\ldots,u_N\), define

\[
\widehat\varphi_N(t)
=
\frac{1}{N}\sum_{n=1}^{N}e^{itu_n}.
\]

The Epps–Pulley statistic is

\[
T_{\mathrm{EP}}
=
N\int_{-\infty}^{\infty}
\left|
\widehat\varphi_N(t)-e^{-t^2/2}
\right|^2
w(t)\,dt,
\]

where \(w(t)\) is a frequency weighting function. The paper describes a Gaussian weight, parameterized as \(w(t)=e^{-t^2/\sigma^2}\), with a commonly used scale of one. Its reference implementation uses the target characteristic function \(e^{-t^2/2}\) as the numerical weighting factor in the quadrature. These conventions should be kept distinct: the target characteristic function and the integration weight are separate objects in the abstract statistic, even when a practical implementation chooses related Gaussian forms.

The statistic compares the full empirical characteristic function with the target characteristic function over a range of frequencies. It is not simply a mean-and-variance penalty. Deviations in the shape of the projected distribution can affect the characteristic function even when the first two moments match.

The implementation approximates the integral numerically. The reference LeJEPA implementation uses trapezoidal quadrature over 17 points on the interval \([-5,5]\), with the target characteristic function also used in the weighting factor of the implemented integrand. This is an implementation choice supported by the paper's ablations, not a mathematical requirement of the Epps–Pulley statistic. Changing the frequency range, quadrature grid or weight changes the finite numerical approximation.

## 7.6 From one projection to a minibatch objective

For a batch of embeddings \(z_1,\ldots,z_N\), sample unit directions by drawing Gaussian vectors and normalizing them:

\[
a_m=\frac{g_m}{\|g_m\|_2},
\qquad
g_m\sim\mathcal N(0,I_K).
\]

For each direction, project every embedding:

\[
u_{n,m}=a_m^\top z_n.
\]

Compute the one-dimensional Epps–Pulley statistic for the \(N\) projected samples, then average over directions:

\[
\mathcal L_{\mathrm{SIGReg}}(Z)
=
\frac{1}{M}\sum_{m=1}^{M}
T_{\mathrm{EP}}(u_{1,m},\ldots,u_{N,m}).
\]

The directions are resampled during training. The encoder is therefore exposed to different one-dimensional views of the same high-dimensional embedding distribution over successive optimization steps. This resampling is not equivalent to evaluating all possible directions in one batch.

The computation is organized around matrix multiplication \(ZA\), followed by characteristic-function evaluations and numerical integration. Its cost depends on batch size \(N\), embedding dimension \(K\), number of directions \(M\), and number of integration points. The paper's linear-complexity description concerns the chosen implementation and fixed sketch parameters; it should not be paraphrased as a claim that computation is independent of the number of projections.

For distributed training, each worker can compute local complex-exponential sums and combine them with an all-reduce operation. This makes the empirical characteristic function compatible with data-parallel training without first gathering every projected sample on one device.

## 7.7 Stability and finite-batch effects

The empirical characteristic function is an average of bounded complex values. The Epps–Pulley construction therefore avoids the polynomial growth of high-order moments. LeJEPA proves bounds on the statistic's derivatives with respect to the projected samples under the chosen weighting. By the chain rule, this provides a controlled gradient contribution to the encoder for the statistic under analysis.

This is a statement about the Epps–Pulley statistic and its inputs. It does not imply that the gradient of the complete neural network objective is bounded independently of network architecture, parameterization or all optimization details.

A finite minibatch also introduces a bias in the empirical statistic and its derivative. The paper derives an \(O(1/N)\) term. It reports this bias as small in the tested settings, including minibatches as small as 16, while noting that U-statistic debiasing or sample splitting could provide unbiased alternatives. The training loss is thus a finite-sample estimate, not an exact population test.

There is a related statistical caution. In classical hypothesis testing, failing to reject a null hypothesis does not prove that the null is true. It says only that the observed sample did not supply sufficient evidence against it at the selected test level. Training minimizes a regularizer rather than making a one-shot accept/reject decision, but the logical distinction remains: a small finite-sample projection statistic is evidence of approximate matching under the sampled directions and frequencies, not a certificate of exact population Gaussianity.

## 7.8 Embedding SIGReg inside a predictive objective

SIGReg specifies a distributional constraint. It does not replace the task that tells the encoder which views should be related. LeJEPA combines the two.

For each observation, let \(z_{n,v}\) be the embedding of view \(v\). Let the first \(V_g\) views be global views, and define their mean

\[
\mu_n
=
\frac{1}{V_g}\sum_{v=1}^{V_g}z_{n,v}.
\]

A view-agreement term draws all view embeddings toward this global-view centre:

\[
\mathcal L_{\mathrm{pred}}
=
\frac{1}{NV}
\sum_{n=1}^{N}
\sum_{v=1}^{V}
\|\mu_n-z_{n,v}\|_2^2.
\]

SIGReg is applied to the embedding distribution of the views. The total objective is

\[
\mathcal L_{\mathrm{LeJEPA}}
=
(1-\alpha)\mathcal L_{\mathrm{pred}}
+
\alpha\mathcal L_{\mathrm{SIGReg}},
\]

where \(\alpha\) controls the trade-off between view agreement and distribution matching.

This construction differs from an EMA teacher-predictor system. In the base LeJEPA recipe, the mean of the global-view embeddings provides the centre toward which all global and local views are drawn; the method does not require an EMA target encoder, a stop-gradient target branch or a separate predictor network. The total loss averages SIGReg over views and combines it with this prediction term using one trade-off parameter. This is consistent with the information structure discussed earlier: a predictor is useful when the target contains asymmetric information that must be inferred from the context, but is not mandatory for symmetric view agreement. [Balestriero and LeCun, 2025]

## 7.9 What SIGReg makes possible

The logical contribution of SIGReg is a practical route from a high-dimensional distributional target to differentiable one-dimensional tests:

1. A downstream-risk analysis motivates a target distribution.
2. Cramér–Wold shows that all one-dimensional projections identify the multivariate distribution.
3. Random unit directions produce a stochastic sketch of those projections.
4. The empirical characteristic function represents each projected distribution.
5. Epps–Pulley measures its discrepancy from the Gaussian target.
6. Numerical quadrature and batch aggregation make the statistic usable in gradient-based training.

Each step introduces an approximation or assumption. The Gaussian target is justified by the risk analysis, not by Cramér–Wold. Cramér–Wold is an all-directions result, not a finite-sample guarantee. Random projections reduce computation but do not remove the need to examine the number of directions and batch size. Epps–Pulley provides a stable statistic for the projected distributions, but the numerical integral is still approximated.

SIGReg therefore controls more than the first two moments, but it does so by sampling and testing one-dimensional views of the embedding distribution. The next section considers a different decomposition of the same distributional problem: control the centre and scale explicitly, then compare shape after the scale has been removed.
