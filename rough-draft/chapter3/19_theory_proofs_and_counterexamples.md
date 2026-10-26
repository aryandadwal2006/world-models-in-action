# Theory notebook: derivations, claims and counterexamples

The following proofs are written from the definitions needed by this chapter. They are not a claim that all behavior of nonlinear SSL training can be reduced to these toy calculations. Each proof states exactly what it establishes and what remains outside its scope.

## 1. Mutual information is a dependence measure

For random variables X and Y,

\[
I(X;Y)=D_{\mathrm{KL}}\bigl(p(x,y)\,\|\,p(x)p(y)\bigr)
=\mathbb E_{p(x,y)}\left[\log\frac{p(y\mid x)}{p(y)}\right].
\]

The KL divergence is nonnegative, so mutual information is nonnegative and equals zero exactly when the joint distribution equals the product of its marginals (under standard regularity conditions). The integrand \(\log[p(y\mid x)/p(y)]\) is the pointwise mutual information. A critic that scores a pair by this log density ratio ranks pairs that are more compatible under the joint distribution than they would be under independent sampling.

This identity motivates InfoNCE, but the identity itself does not define the training loss. The finite-candidate classifier and its sampler need their own specification.

## 2. InfoNCE is a classification loss

Draw a positive pair \((x,y_1)\sim p(x,y)\), and independently draw negatives \(y_2,\ldots,y_N\sim p(y)\). A critic \(f_\theta(x,y)\) scores each candidate. Define

\[
\mathcal L_{\mathrm{NCE}}
=-\mathbb E\left[
\log\frac{\exp(f_\theta(x,y_1))}
{\sum_{j=1}^{N}\exp(f_\theta(x,y_j))}
\right].
\]

The denominator normalizes scores into a categorical distribution over candidate indices. The loss is cross-entropy for the correct index 1. With a learned similarity \(s(z_x,z_y)\), a common critic is \(f=s/\tau\), where \(\tau>0\) is temperature.

For logits \(a_k\), probabilities \(p_k=\exp(a_k)/\sum_j\exp(a_j)\), and true index y,

\[
L=-a_y+\log\sum_j\exp(a_j).
\]

Then

\[
\frac{\partial L}{\partial a_k}
=p_k-\mathbf 1[k=y].
\]

This equation is enough to derive the positive and negative gradient signs. For a score \(s_k\) whose logit is \(s_k/\tau\),

\[
\frac{\partial L}{\partial s_k}
=\frac{p_k-\mathbf 1[k=y]}{\tau}.
\]

Hard negatives receive larger gradient because their softmax probabilities are larger; the objective does not repel every negative with the same force.

## 3. Derivation of the standard InfoNCE mutual-information lower bound

### Statement and assumptions

For the sampler above, let N be the **total number of candidates**, including one positive and N−1 independently sampled negatives. Let the critic be any measurable real-valued scoring function for which the expectations exist. The standard bound is

\[
I(X;Y)\geq \log N-\mathcal L_{\mathrm{NCE}}.
\]

This is a bound for that joint-positive/product-of-marginals candidate setup. A loss with arbitrary hard-negative mining or another sampling distribution does not inherit it automatically without a new derivation.

### Step 1: package the candidates into one variable

Let \(V=(Y_1,\ldots,Y_N)\), where \((X,Y_1)\) follows the joint distribution and the remaining candidates are sampled independently from \(p(y)\). Because \(Y_2,\ldots,Y_N\) are independent of X and of Y1,

\[
I(X;V)=I(X;Y_1,Y_2,\ldots,Y_N)=I(X;Y_1)=I(X;Y).
\]

The extra negatives provide a candidate set for the classification task, but they do not increase the mutual information between X and the bag beyond that already carried by the positive.

### Step 2: construct a normalized variational conditional density

Define

\[
g(x,v)=
\frac{\exp(f_\theta(x,y_1))}
{\frac1N\sum_{j=1}^{N}\exp(f_\theta(x,y_j))}.
\]

Now consider \(V\) under the product distribution \(p(v)=\prod_{j=1}^{N}p(y_j)\), where all candidates are independent marginal samples. Under this distribution the candidate positions are exchangeable. For every fixed x, each expectation

\[
\mathbb E_{p(v)}
\left[
\frac{\exp(f_\theta(x,y_i))}
{\frac1N\sum_j\exp(f_\theta(x,y_j))}
\right]
\]

has the same value for i=1,...,N. Sum these N equal expectations:

\[
\begin{aligned}
N\,\mathbb E_{p(v)}[\text{one term}]
&=\mathbb E_{p(v)}
\left[
\frac{\sum_i\exp(f_\theta(x,y_i))}
{\frac1N\sum_j\exp(f_\theta(x,y_j))}
\right]\\
&=N.
\end{aligned}
\]

Therefore \(\mathbb E_{p(v)}[g(x,V)]=1\). Hence

\[
q(v\mid x)=p(v)\,g(x,v)
\]

is a normalized conditional distribution over candidate bags.

### Step 3: apply the Barber–Agakov variational inequality

For any normalized conditional density q,

\[
I(X;V)\geq
\mathbb E_{p(x,v)}\left[\log\frac{q(v\mid x)}{p(v)}\right].
\]

To see why, subtract the right-hand side from I(X;V):

\[
\begin{aligned}
I(X;V)-\mathbb E_{p(x,v)}\log\frac{q(v\mid x)}{p(v)}
&=\mathbb E_{p(x,v)}
\log\frac{p(v\mid x)}{q(v\mid x)}\\
&=\mathbb E_{p(x)}
D_{\mathrm{KL}}(p(v\mid x)\|q(v\mid x))\\
&\geq 0.
\end{aligned}
\]

The last inequality is Gibbs' inequality for KL divergence. Substitute \(q(v\mid x)=p(v)g(x,v)\):

\[
\begin{aligned}
I(X;V)&\geq\mathbb E_{p(x,v)}[\log g(X,V)]\\
&=\mathbb E\left[
f_\theta(X,Y_1)-
\log\left(\frac1N\sum_{j=1}^{N}e^{f_\theta(X,Y_j)}\right)
\right]\\
&=\log N-\mathcal L_{\mathrm{NCE}}.
\end{aligned}
\]

Since \(I(X;V)=I(X;Y)\), the desired result follows:

\[
I(X;Y)\geq \log N-\mathcal L_{\mathrm{NCE}}.
\]

### Step 4: why the estimate cannot exceed log N

The classifier's probability for the correct candidate is at most 1, so its negative log probability is nonnegative:

\[
\mathcal L_{\mathrm{NCE}}\geq 0.
\]

Thus

\[
\log N-\mathcal L_{\mathrm{NCE}}\leq\log N.
\]

This is an exact finite-candidate ceiling for the InfoNCE expression. It is distinct from a broader result that distribution-free high-confidence lower bounds on mutual information grow only on a logarithmic scale with sample size.

### What the theorem means and does not mean

It means that under the stated sampler, a small InfoNCE loss certifies a lower bound on dependence between X and Y. It does not mean that:
- the critic is an unbiased numerical estimator of I(X;Y);
- the bound is tight at finite N;
- maximizing this term alone guarantees disentangled or interpretable features;
- the information preserved is task-relevant;
- an encoder is Markov sufficient, causal, action-conditioned or useful for planning.

The critical primary references are van den Oord et al. (2018), Poole et al. (2019), and Tschannen et al. (2020). Tschannen et al. specifically caution that the empirical success of MI-inspired objectives cannot be explained by MI alone: architectural inductive bias and estimator parameterization also matter.

## 4. The ideal contrastive critic is a density ratio

For a positive pair, the ideal score is proportional (up to an x-only additive term) to

\[
f^*(x,y)=\log\frac{p(y\mid x)}{p(y)}+c(x).
\]

The additive function \(c(x)\) does not change the softmax distribution among candidate y values for fixed x because it adds the same offset to every candidate logit. This establishes the density-ratio interpretation of an ideal critic. A finite neural network trained with finite data and finite negatives may not attain this ideal.

## 5. Pure invariance has a constant global minimizer

Let the objective be

\[
L_{\mathrm{inv}}=\mathbb E\|z(T_1(x))-z(T_2(x))\|_2^2.
\]

Choose \(z(x)=c\) for every x. Then each paired difference equals zero, so \(L_{\mathrm{inv}}=0\). Since squared error is nonnegative, this is a global minimizer. Therefore view agreement alone does not identify a non-collapsed representation.

For a cosine loss \(2-2\cos(z_1,z_2)\), a shared nonzero constant direction also yields zero loss. The exact zero vector can make cosine undefined, but the nonzero constant solution is enough to establish the issue.

This is an existence result. It does not prove that a specific optimizer initialized randomly will converge to collapse. Conversely, the fact that an implementation avoids collapse does not prove no collapsed solution exists.

## 6. VICReg explicitly penalizes exact constant features

For a feature dimension j, VICReg's standard-deviation penalty has form

\[
v_j=\max(0,\gamma-\sigma_j),
\]

for margin \(\gamma>0\). If every example has the same coordinate value, \(\sigma_j=0\) (ignoring a small numerical epsilon), and \(v_j=\gamma>0\). The constant representation therefore cannot make the variance term zero. If \(\sigma_j\geq\gamma\), that dimension's term becomes inactive. The penalty is a variance floor, not a reward for arbitrarily large variance.

This establishes what the term penalizes. It does not prove that all total-loss local minima are good, that features encode semantics, or that covariance regularization removes every form of dimensional collapse.

## 7. Barlow Twins' target expresses two second-order conditions

If standardized views have corresponding dimensions aligned and distinct dimensions uncorrelated across views, then the cross-correlation matrix C approaches identity: \(C_{ii}=1\) and \(C_{ij}=0\) for i≠j. Barlow Twins minimizes a weighted squared deviation from that matrix. This is a statement about second-order cross-view statistics. It does not establish mutual independence for arbitrary distributions: uncorrelated coordinates can remain dependent through higher-order relationships.

In practice, standardization and C are estimated from a finite batch. The estimate's variance, treatment of nearly constant dimensions and synchronization across devices are part of the implementation.

## 8. The conditional mean minimizes squared prediction risk

Let Y be a vector target and C the observed context. For fixed c, the predictor \(a\) minimizes

\[
R(a\mid c)=\mathbb E[\|Y-a\|_2^2\mid C=c].
\]

Write \(m=\mathbb E[Y\mid C=c]\). Expanding \(Y-a=(Y-m)+(m-a)\) gives

\[
R(a\mid c)=
\mathbb E[\|Y-m\|^2\mid c]+\|m-a\|^2,
\]

because the cross term vanishes: \(\mathbb E[Y-m\mid C=c]=0\). The minimum is attained at \(a=m\). When many targets are plausible, that mean can lie between plausible samples. This is the mathematical basis of the conditional-mean caution for pixel MSE, not proof that every reconstruction must be blurry.

## 9. EMA is an exponential smoother, not an anti-collapse theorem

Given \(\xi_t=m\xi_{t-1}+(1-m)\theta_t\) with \(0\leq m<1\), iterating gives

\[
\xi_t=(1-m)\sum_{r=0}^{t-1}m^r\theta_{t-r}+m^t\xi_0.
\]

Recent online parameters receive larger weights, while old ones decay geometrically. This smooths target evolution. No feature variance or rank constraint appears in the equation, so EMA alone cannot guarantee non-collapse.

## 10. Augmentation can destroy state information: a constructive example

Let the underlying state be \(s=(p,c)\), with position p and a small color code c that determines which action is safe. The observation is \(x=R(p,c)\). Suppose training augmentations randomly recolor the code so that views of the same scene always disagree in c, while a consistency objective rewards their embeddings for agreement. If no other target provides pressure to preserve c, a representation that discards c can lower the objective. Yet a controller that needs c to select a safe action cannot use that representation.

This is not a claim that all color jitter destroys color information in every model. It is a counterexample to the universal claim that augmentation invariance is automatically task-aligned. Whether a feature is nuisance or signal depends on the downstream contract.

## 11. A static objective does not establish Markov sufficiency

A representation z=f(o) trained with a static view objective depends on current observation o and its augmentation relation. The objective contains no constraint that two observations sharing z must have the same conditional distribution over next observations, rewards or transitions under every action. Therefore Markov sufficiency is not implied by the static objective alone.

To support that stronger claim, an evaluation must test future-conditioned distinctions. For example, if two histories map to the same latent but have different velocities, the distributions \(p(o_{t+1}\mid z_t,a_t)\) may differ. A good reconstruction or linear-probe score for selected variables cannot rule out every such aliasing case.

## 12. Claims that require more than the proof notebooks

The following are intentionally not presented as general theorems here:
- stop-gradient alone prevents collapse for any architecture;
- any EMA target prevents collapse;
- decorrelated features are statistically independent or physically meaningful;
- better future-latent retrieval proves a better world model;
- global classification features imply good local geometry;
- a high linear-probe score proves causal sufficiency;
- the best objective is independent of data distribution and augmentation.

Each requires additional assumptions or direct evidence.
