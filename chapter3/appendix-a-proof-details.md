# Appendix A: InfoNCE bound and gradient details

This appendix contains the algebra moved out of §3.3 to preserve the main chapter’s teaching flow while retaining the full derivation. It uses the same notation as Table 3.2. The claims are conditional on the sampling assumptions described in §3.3; the recorded trajectory batch is not an ideal independent-negative sampler.

## Where the bound comes from

**Sidebar: A proof of the InfoNCE bound**

Suppose one of $K+1$ candidates is drawn from the conditional distribution $p(b\mid a)$, while the others are drawn independently from the marginal distribution $p(b)$. Let $r_j=p(b_j\mid a)/p(b_j)$, and let $S=\sum_{j=0}^{K}r_j$.

**Step 1: The best possible classifier.** Given the anchor and candidates, Bayes' rule assigns candidate $j$ probability $r_j/S$ of being the positive. Any classifier $q_j$ incurs cross-entropy at least as large as this true posterior:

$$
\mathcal{L}(f)\ge
-\mathbb{E}\log\frac{r_m}{S}.
$$

**Step 2: Express that cross-entropy using mutual information.** The expected log likelihood ratio of a positive pair is $I(a;b)$, so the expression on the right is

$$
\mathbb{E}\log S-I(a;b).
$$

**Step 3: Bound the remaining term.** Under the reference distribution where all $K+1$ candidates are independently sampled from the marginal $p(b)$, write $T=S/(K+1)$. The actual candidate-set distribution has density $T$ relative to this reference distribution, so

$$
\mathbb{E}[\log S]
= \log(K+1)+\mathbb{E}_0[T\log T]
\ge \log(K+1),
$$

because $\mathbb{E}_0[T]=1$ and $t\log t$ is convex, so Jensen's inequality makes $\mathbb{E}_0[T\log T]$ nonnegative. Combining the three steps yields

$$
\mathcal{L}_{\mathrm{InfoNCE}}
\ge \log(K+1)-I(a;b),
$$

which rearranges to equation 3.6.

The gradient can also be made precise. The result in equation 3.7 separates attraction toward the positive from repulsion by negatives. For an anchor $i$, define the score $s_{ij}=\operatorname{sim}(z_i,z_j)$. Differentiating the cross-entropy in equation 3.5 with respect to a candidate score gives

$$
\frac{p_i(j)-\mathbf{1}[j=i^+]}{\tau},
$$

where $\mathbf{1}[\cdot]$ equals one if its condition is true and zero otherwise. Applying the chain rule yields the gradient with respect to the anchor representation:

For cosine similarity,

$$
\frac{1}{\lVert u\rVert_2}
\left(
\frac{v}{\lVert v\rVert_2}
-\operatorname{sim}(u,v)
\frac{u}{\lVert u\rVert_2}
\right).
$$

The gradient is perpendicular to $u$, so each score's local update changes the direction of the vector rather than its length to first order.

## Why this bound is not a physical-state guarantee

The lower bound concerns mutual information shared by paired-view representations under the stated sampling assumptions. It does not establish that cart position, angle, velocity, or an action-conditioned predictive state is linearly available. The data-processing inequality also limits shared information to information that survives the view-generation pipeline. These are the precise reasons the chapter follows the training objective with held-out state probes.