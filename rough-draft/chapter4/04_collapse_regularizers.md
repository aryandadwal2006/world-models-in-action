# 04 — Collapse, VICReg and the distributional question

## 1. Start from Chapter 3, do not repeat it

Chapter 3 already explains complete collapse, dimensional collapse, BYOL/SimSiam, Barlow Twins and VICReg. It also distinguishes necessary anti-collapse constraints from task-relevant information. This file should support only a short recap in Chapter 4.

A pure feature-matching objective can admit constant representations. A particular method may avoid collapse through asymmetric branches, target updates, normalization or explicit regularization. These mechanisms are not interchangeable and should not be grouped under a single “anti-collapse trick.”

## 2. VICReg as the bridge

For batches of paired embeddings \(Z,Z'\in\mathbb R^{B\times d}\), VICReg combines:
\[
\mathcal L_{\mathrm{VICReg}}
=\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')].
\]

- The invariance term matches paired views.
- The variance term imposes a per-coordinate standard-deviation floor.
- The covariance term discourages off-diagonal covariance within each view.

The full details, including the variance hinge, covariance estimator and official implementation excerpt, already belong to Chapter 3. In Chapter 4, recap this decomposition and ask what its constraints leave unspecified.

## 3. What moment constraints do not identify

A covariance matrix describes second-order relationships. Zero off-diagonal covariance does not imply statistical independence, and matching mean and covariance does not identify a unique distribution. Higher-order shape, multimodality and tails can differ even when first and second moments agree.

The correct motivation for SIGReg is therefore:
- VICReg gives useful and interpretable scale/redundancy constraints;
- those constraints do not directly compare the full embedding distribution with a chosen target distribution;
- SIGReg seeks a projected distributional comparison to an isotropic Gaussian;
- VISReg changes how distribution shape is measured and separates shape from scale.

Do not say that Gaussianity is universally necessary for learning. It is the target justified by the specific theoretical framework and empirical design of the respective papers.

## 4. The three approaches at a glance

| Method | Invariance/prediction signal | Embedding-space constraint | Main distinction |
|---|---|---|---|
| VICReg | Paired-view agreement | Variance floor + covariance penalty | Interpretable first/second-order constraints |
| LeJEPA | JEPA predictive loss | SIGReg distributional sketch | Uses a Gaussian-target theory and projected characteristic-function test |
| VISReg | JEPA-style predictive agreement | Scale term + sliced-Wasserstein shape term + centering in the reported recipe | Separates scale control from shape matching |

This is a comparison of objective ingredients, not a ranking of universally better methods.

## 5. Questions Chapter 4 must answer

1. What is being regularized: context embeddings, target embeddings, or both?
2. Does the regularizer apply to every branch independently or to a joint/cross-branch statistic?
3. What target distribution is used?
4. How is the distribution discrepancy approximated?
5. Which terms control scale, centering and shape?
6. What is the exact loss normalization and gradient path?
7. What evidence supports stability, scaling or downstream generalization?
8. Which claims are theoretical and which are empirical?

## 6. Claims to avoid

- “VICReg guarantees an informative representation.”
- “The covariance penalty enforces independence.”
- “SIGReg proves that the learned representation is Gaussian.”
- “VISReg guarantees collapse recovery in every architecture.”
- “An isotropic Gaussian embedding is a sufficient world state.”

A regularizer can rule out or penalize selected degenerate geometries under specified conditions. It cannot by itself establish what physical variables the representation encodes.
