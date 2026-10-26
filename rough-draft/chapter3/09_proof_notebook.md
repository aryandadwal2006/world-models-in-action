# Proof and derivation notebook

This note identifies the algebra needed to support the chapter and the limits of each derivation. It is an original notebook, not a transcription of a paper's full proof.

## A. InfoNCE equals candidate classification

Let candidate logits \(a_1,\dots,a_K\) be similarity scores divided by temperature, and let y be the positive index. The softmax probability is \(p_k=e^{a_k}/\sum_j e^{a_j}\). Cross-entropy is

\[
L=-\log p_y=-a_y+\log\sum_j e^{a_j}.
\]

Differentiate:
\[
\frac{\partial L}{\partial a_k}=-\mathbf 1[k=y]+\frac{e^{a_k}}{\sum_j e^{a_j}}
=p_k-\mathbf 1[k=y].
\]

If \(a_k=s_k/\tau\), then
\[
\frac{\partial L}{\partial s_k}=\frac{p_k-\mathbf 1[k=y]}{\tau}.
\]

This proves the gradient sign and softmax weighting. It does not prove that maximizing this objective learns task-relevant features; that claim needs assumptions about data, positive transformations and negative sampling.

## B. The usual InfoNCE mutual-information lower bound

Let positive pairs \((x,y)\) come from a joint distribution \(p(x,y)\) and negatives be sampled independently from the marginal \(p(y)\). The critic's ideal score is proportional to the density ratio \(p(y\mid x)/p(y)=p(x,y)/(p(x)p(y))\). A K-way candidate task contains one positive sample and K-1 negatives. The optimal classifier depends on that ratio; the expected log score relates to mutual information. The standard derivation yields a lower bound often written \(I(X;Y)\ge \log K-L_{\mathrm{NCE}}\), given the candidate-generation convention and sampling assumptions used by that derivation.

Three caveats must accompany the formula:
1. Papers differ in whether K denotes total candidates or number of negatives. Define it locally.
2. Because the bound is at most log K for finite K, increasing the amount of mutual information beyond that ceiling cannot tighten this bound without increasing the candidate count.
3. A larger bound does not guarantee a better representation for a downstream task; it concerns statistical dependence of sampled variables, not the causal sufficiency contract of a world model.

## C. Constant representations and invariance loss

Let \(z_i=c\) for every input. For paired views, an invariance MSE \(L=\mathbb E\|z(x_1)-z(x_2)\|^2\) is exactly zero. Therefore invariance alone admits collapse as a global minimizer. This is an existence proof. It says nothing about whether a particular optimizer reaches the solution from a particular initialization.

For cosine regression, if all vectors are a nonzero constant direction, cosine similarity can be one, yielding zero loss under \(2-2\cos\). The exact constant-zero vector may make cosine undefined in some implementations, but that numerical detail does not rescue the objective: a constant nonzero feature is already a collapsed solution.

## D. VICReg's variance floor rules out exact all-constant solutions below the margin

If all examples have a constant value in dimension j, then the batch standard deviation \(\sigma_j=0\) (ignoring the small numerical epsilon). For margin \(\gamma>0\), its variance penalty is \(\max(0,\gamma-\sigma_j)=\gamma\), which is positive. If training can increase that dimension's variance, the term decreases until the standard deviation reaches the margin. Thus an all-constant representation cannot achieve zero variance penalty. This does not prove the total loss has no bad local minima or that its resulting representation is semantic; it establishes what the stated term penalizes.

## E. Barlow Twins cross-correlation target

For standardized features, if the two views agree dimensionwise and dimension i is uncorrelated with dimension j for i≠j, then the cross-correlation matrix is the identity. The Barlow objective is the squared Frobenius distance to identity with different weights for diagonal and off-diagonal entries. The target matrix expresses cross-view matching plus second-order redundancy reduction. It does not imply higher-order independence without additional distributional assumptions.

## F. Masked MSE and conditional expectation

For a target random vector Y and predictor g(C) conditioned on visible context C, the expected squared-error risk is \(\mathbb E[\|Y-g(C)\|^2]\). For a fixed context c, expanding around conditional mean gives

\[
\mathbb E[\|Y-a\|^2\mid C=c]
=\mathbb E[\|Y-\mathbb E[Y\mid c]\|^2\mid c]
+\|a-\mathbb E[Y\mid c]\|^2.
\]

The cross term vanishes because \(\mathbb E[Y-\mathbb E[Y\mid c]\mid c]=0\). Therefore the conditional mean minimizes expected MSE. When multiple plausible completions are possible, this mean need not be one plausible sample; it can average them. This explains a limitation of pixel-MSE targets without claiming that every MAE prediction is blurry or unuseful.

## G. Autoregressive factorization is not a masked objective

The chain rule gives \(p(x_1,\ldots,x_T)=\prod_t p(x_t\mid x_{<t})\) for any joint distribution. The negative log-likelihood is \(-\sum_t\log p_\theta(x_t\mid x_{<t})\). This derives causal next-token training from observed sequences. It differs from masked language modeling because the context is restricted to preceding tokens, not both sides of a masked token.

## H. EMA as low-pass update

The target update \(\xi_t=m\xi_{t-1}+(1-m)\theta_t\) is an exponentially weighted average of prior online parameters (ignoring initialization transients):
\[
\xi_t=(1-m)\sum_{k=0}^{t-1}m^k\theta_{t-k}+m^t\xi_0.
\]
Thus recent online parameters receive exponentially decreasing weights. This is a smoothing statement, not an anti-collapse theorem.

## I. Sinkhorn-Knopp assignment constraints

SwAV normalizes an exponentiated score matrix by alternately scaling rows and columns toward prescribed sums. Iterative row/column normalization approximates a doubly-constrained assignment, subject to numerical convergence and the exact normalization convention. The balancing constraint reduces the trivial all-to-one-prototype assignment for a batch, but it assumes balanced prototype use. It does not prove learned clusters correspond to true categories.

## J. Claims deliberately not presented as proofs

- A theorem that stop-gradient by itself prevents collapse in every architecture.
- A theorem that any EMA teacher prevents collapse.
- A theorem that decorrelated features are independent or physically sufficient.
- A theorem that better future-latent retrieval implies a better world model.
- A proof that the learned latent is Markov sufficient merely because a linear probe decodes selected simulator coordinates.

These questions require stronger assumptions and separate evidence.
