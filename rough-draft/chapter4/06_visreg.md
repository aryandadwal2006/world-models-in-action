# 06 — VISReg: separating scale and distribution shape

## 1. Motivation

VISReg (Wu, Balestriero and Levine, 2026), “Variance-Invariance-Sketching Regularization for JEPA training,” proposes a different distributional-regularization design. It is motivated by limitations discussed for VICReg and SIGReg:
- VICReg gives interpretable variance/covariance constraints, but the covariance matrix only captures second-order relationships;
- SIGReg compares projected distributions to a Gaussian through characteristic functions;
- the VISReg paper reports weak gradients for SIGReg close to collapse under the studied setup;
- VISReg combines a scale constraint with a Sliced-Wasserstein shape objective.

These are claims and motivations from the VISReg paper. The book must not elevate them into a universal ranking of objective families.

## 2. Scale and shape are different questions

A representation can shrink in norm while retaining some relative pattern. Conversely, it can have adequate per-coordinate spread while its distribution has an undesirable shape. VISReg's design separates:
- scale: how wide the embedding cloud is;
- shape: how its projected distribution compares with the target;
- center: where the cloud is located.

The exact final loss, weights, normalization, centering, stop-gradient locations and projection sampling must be taken from the current paper and reference implementation. The author's notes contain useful first-principles derivations, but not every displayed formula should be copied without checking the official source.

## 3. One-dimensional optimal transport

For probability measures on the real line, optimal transport has a convenient ordered solution. If two empirical sample sets are sorted, their one-dimensional Wasserstein distance can be written as a distance between corresponding order statistics. In the equally weighted, equal-cardinality case, a squared version has the form:
\[
W_2^2(\hat\mu,\hat\nu)
=\frac1B\sum_{i=1}^{B}
\left(x_{(i)}-y_{(i)}\right)^2,
\]
where \(x_{(i)}\) and \(y_{(i)}\) are sorted samples. The exact formula depends on the weighting and quantile convention.

This is a useful pedagogical step: in one dimension, the optimal coupling can be obtained by sorting rather than solving a full high-dimensional transport problem. The gradients with respect to samples are piecewise smooth, with special handling at ties and quantile conventions.

## 4. Sliced-Wasserstein sketching

To compare a high-dimensional embedding distribution, draw unit directions \(w_1,\ldots,w_K\) and project the samples:
\[
u_{b,k}=w_k^\top z_b.
\]

For each direction, compare the sorted projected samples with the corresponding target quantiles. Average the discrepancy over directions. This gives a stochastic approximation to a high-dimensional distributional comparison.

The method's cost depends on:
- batch size \(B\);
- embedding dimension \(d\);
- number of directions \(K\);
- sorting cost for each direction;
- memory layout, vectorization and distributed implementation.

A schematic cost includes projection work and one-dimensional sorting. If \(K\) itself increases proportionally with \(d\), an expression that is linear in \(d\) for fixed \(K\) can become quadratic in \(d\). Always state the dependence on \(K\), not merely the fixed-\(K\) form.

## 5. Why decouple the gradients?

The VISReg formulation uses separate treatment of scale and shape. The user's notes interpret the stop-gradient boundary as a way to keep shape gradients from cancelling the direct scale correction in the studied formulation. The final manuscript should:
- reproduce the exact normalization and gradient path in the official source;
- derive the scale gradient with respect to standard deviation;
- distinguish a non-vanishing derivative with respect to a statistic from a guaranteed non-vanishing parameter gradient;
- report the paper's collapse experiments and their setup;
- avoid promising recovery for every initialization, network or optimizer.

A non-zero derivative with respect to \(\sigma\) does not by itself prove that the full encoder-parameter gradient is bounded away from zero. The mapping from parameters to batch statistics, network saturation and numerical precision also matter.

## 6. Suggested comparison

| Question | VICReg | SIGReg / LeJEPA | VISReg |
|---|---|---|---|
| Scale control | Variance floor | Distributional target also controls scale under the selected target | Explicit scale term |
| Redundancy/shape | Off-diagonal covariance | Projected characteristic-function discrepancy | Projected sliced-Wasserstein discrepancy |
| Main statistic | Batch covariance | Epps–Pulley-style projected normality test | Sorted 1-D projected samples vs target quantiles |
| Main caveat | Second-order constraints do not determine full distribution | Finite projections approximate all-direction matching | Finite slices and K-dependent compute remain |
| Relationship to JEPA | Generic SSL regularizer used in JEPA-like settings | Explicit LeJEPA predictive objective | Proposed regularizer for JEPA training |

## 7. Figures

**Figure 4.11 — Scale and shape can fail separately.** Show a narrow spherical cloud, an anisotropic cloud and a spread-out but non-Gaussian cloud. Explain which property each regularizer targets. Do not use this drawing to imply that a standard Gaussian is always semantically optimal.

**Figure 4.12 — Sliced-Wasserstein comparison.** Show two high-dimensional point clouds projected onto a unit direction, sorted on a line, then compared with target Gaussian quantiles.

**Figure 4.13 — The role of the number of slices.** Show that more sampled directions improve coverage of the sphere but increase computation. Label this as a conceptual diagram, not a convergence theorem.

## 8. What to verify against the primary sources

Before the manuscript quotes exact formulas, confirm:
- the shape loss uses the intended normalization and centered/normalized feature matrix;
- whether centering is a separate term and how it is weighted;
- the exact target quantile convention;
- the number and sampling distribution of projections;
- which variables are stop-gradient and where;
- whether scale and shape terms are computed over each branch independently;
- the reported hyperparameters, batch sizes and datasets;
- the measured collapse gradients and their units;
- the complexity claims under the paper's chosen K-to-dimension relation.

## 9. Sources

- Wu, Balestriero and Levine (2026), arXiv:2606.02572.
- Project and reference code: https://haiyuwu.github.io/visreg/
- Official repository linked by the project: https://github.com/HaiyuWu/visreg
