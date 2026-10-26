# LeJEPA, SIGReg and VISReg: distributional regularization for predictive representations

**Status:** frontier dossier. The reader-facing Chapter 3 should give this lineage a short handoff only; the full derivation belongs with the detailed JEPA treatment in Chapter 4. This document keeps the mechanics complete enough to understand the claims without confusing paper-reported results with general theorems.

Primary sources:
- Balestriero and LeCun (2025), *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544, current HTML version v3: https://arxiv.org/html/2511.08544.
- Wu, Balestriero and Levine (2026), *VISReg: Variance-Invariance-Sketching Regularization for JEPA training*, arXiv:2606.02572: https://arxiv.org/abs/2606.02572; project page https://haiyuwu.github.io/visreg/.
- The conceptual predecessor is VICReg (Bardes et al., 2022), which decomposes learning into invariance, variance and covariance.

## 1. Why this lineage matters

VICReg made the non-contrastive design principle explicit: matching views is not enough, so add constraints on feature variance and covariance. SIGReg changes the statistical target from second-order moment constraints toward matching the embedding distribution to an isotropic Gaussian. LeJEPA combines a joint-embedding predictive loss with SIGReg. VISReg keeps VICReg's separation of scale and shape, but replaces the covariance term with a distribution-shape regularizer built from sliced Wasserstein distances.

The progression is not simply “a newer loss beats an older loss.” Each method chooses a different proxy for a non-collapsed, well-shaped representation:

| Method | Invariance/prediction term | Distribution regularizer | What it directly controls |
|---|---|---|---|
| VICReg | Direct paired-view MSE | Variance floor plus off-diagonal covariance | Per-dimension spread and second-order redundancy |
| SIGReg | Combined with a prediction loss in LeJEPA | Random one-dimensional projections compared to a Gaussian using a characteristic-function test | Projected distributional shape rather than only covariance |
| LeJEPA | Predict embeddings of views relative to a mean of global-view embeddings | SIGReg | Predictive agreement and target embedding distribution without the usual EMA teacher or stop-gradient heuristics |
| VISReg | Mean-global-view embedding prediction | Variance scale + sliced-Wasserstein shape + centering | Separates scale from full projected distribution shape and feature center |

The category overlaps with JEPA rather than forming an independent new chapter family. This dossier is detailed because the full logic is central to current non-contrastive representation learning, but it should not displace Chapter 4's main treatment of predictive world models.

## 2. VICReg as the baseline to improve

For paired batches \(Z,Z'\in\mathbb{R}^{B\times d}\), VICReg uses three components:

\[
L_{\mathrm{VICReg}}
=\lambda s(Z,Z')+\mu[v(Z)+v(Z')]+\nu[c(Z)+c(Z')].
\]

The invariance component is

\[
s(Z,Z')=\frac{1}{Bd}\|Z-Z'\|_F^2.
\]

For feature dimension j, the variance term penalizes \(\sigma_j\) below a margin \(\gamma\):

\[
v(Z)=\frac1d\sum_{j=1}^{d}\max(0,\gamma-\sigma_j(Z)).
\]

The covariance term penalizes off-diagonal entries of the empirical covariance matrix:

\[
c(Z)=\frac1d\sum_{i\ne j}\operatorname{Cov}(Z)_{ij}^{2}.
\]

The attraction of this decomposition is interpretability. The three terms answer separate questions: should views match, should each coordinate vary, and should different coordinates be linearly redundant?

The limitation identified by later distributional methods is that first and second moments do not determine a multivariate distribution in general. Two distributions can share the same mean and covariance yet have very different shapes, multimodality or tails. Decorrelated dimensions need not be independent, and the feature space may still differ sharply from the target distribution even when its covariance is identity.

## 3. Why an isotropic Gaussian target appears in LeJEPA

LeJEPA's central theory claims that, under the paper's specified JEPA downstream-prediction-risk analysis, the isotropic Gaussian is the optimal distribution for the embeddings. This statement must remain attributed to that theory and its assumptions. It is not a universal theorem that every representation-learning problem has a Gaussian-optimal embedding: a particular task may benefit from structure, discreteness, multimodality or geometry that does not match a standard normal distribution.

Let \(Z\in\mathbb{R}^{d}\). A standard isotropic Gaussian has

\[
Z\sim\mathcal N(0,I_d).
\]

The target imposes zero mean, unit variance along every direction, no preferred orientation (rotational isotropy), and Gaussian distributional shape rather than merely uncorrelated dimensions.

If \(w\) is any unit vector, then \(w^\top Z\sim\mathcal N(0,1)\). This property is crucial: instead of comparing two high-dimensional empirical distributions directly, sketch the embedding distribution onto many one-dimensional directions and test whether those projections look standard normal.

A finite set of random directions provides only a stochastic approximation to the all-directions condition. It does not establish exact distributional equality from a finite batch. The theory supplies the reason for the target; the finite sketch makes the regularizer computationally practical.

## 4. SIGReg from characteristic functions

### 4.1 The characteristic function

For a scalar random variable X, the characteristic function is

\[
\varphi_X(t)=\mathbb E[e^{itX}],\qquad t\in\mathbb R.
\]

It represents the distribution through its Fourier transform. A standard normal random variable has characteristic function

\[
\varphi_{\mathcal N(0,1)}(t)=e^{-t^2/2}.
\]

Given scalar projected samples \(u_1,\ldots,u_B\), the empirical characteristic function is

\[
\widehat\varphi_B(t)=\frac1B\sum_{b=1}^{B}e^{itu_b}.
\]

SIGReg uses a weighted discrepancy between the empirical characteristic function and the target standard-normal characteristic function, evaluated over a finite set of integration points and averaged over random one-dimensional projections. This is based on an Epps–Pulley normality-test construction. In schematic form,

\[
S(u_{1:B})\approx
\int_{\mathbb R}
\left|\widehat\varphi_B(t)-e^{-t^2/2}\right|^2 w(t)\,dt,
\]

where \(w(t)\) is the test's prescribed weighting function. The canonical paper gives its exact normalization, quadrature and weights; those should be used for implementation rather than treating this schematic integral as the code-level formula.

### 4.2 The sketching argument

The Cramér–Wold theorem says two probability distributions over \(\mathbb R^d\) agree if their one-dimensional projections agree along every direction. Testing all directions is impractical. SIGReg samples a manageable set of directions per training step and computes a normality discrepancy in one dimension for each projection.

The practical procedure is:
1. Sample embeddings \(z_b\in\mathbb R^d\).
2. Draw unit directions \(w_k\), commonly from a rotationally symmetric distribution.
3. Project \(u_{bk}=z_b^\top w_k\).
4. Evaluate the empirical characteristic-function discrepancy against \(\mathcal N(0,1)\) for each k.
5. Average projected discrepancies and add the result to the predictive loss.

Resampling directions across minibatches helps approximate the all-direction condition over training. The random-direction count is a computational/statistical knob. A fixed small set can miss a problematic direction; more directions cost more time. The paper analyses both the number of directions and numerical integration settings.

### 4.3 Why moments are not the same as distribution matching

Mean and covariance leave higher-order aspects unspecified. A non-Gaussian distribution can have zero mean, unit covariance and still be multimodal or heavy-tailed. A characteristic function identifies a scalar distribution, and the Cramér–Wold argument connects all scalar projections to the multivariate distribution. The regularizer therefore targets more than decorrelation.

### 4.4 Computational logic and caveats

SIGReg's argument is that stochastic one-dimensional tests avoid computing a full high-dimensional distribution distance or dense covariance inverse. Its operation count depends on batch size, embedding dimension, number of projections and integration points. Those parameters should be reported.

The VISReg paper reports that SIGReg's Epps–Pulley gradients weaken near collapse in its tested setup. This is a result discussed by the VISReg authors, not a blanket theorem about all statistical-test regularizers or all collapse trajectories. Treat “SIGReg matches Gaussian shape” as its design target, and “SIGReg's gradient can be weak near collapse under the studied configuration” as a paper-reported diagnostic.

## 5. LeJEPA objective

LeJEPA combines the predictive loss of a joint-embedding architecture with SIGReg. In the paper's implementation, global views define a mean embedding for each source instance, and embeddings from all views are predicted toward that mean. A high-level notation is:

\[
\mu_g(x)=\frac{1}{V_g}\sum_{v\in\mathcal V_g} z(v),
\]

where \(\mathcal V_g\) denotes global views of the same source. If \(\mathcal V\) includes global and local views, the prediction term is schematically

\[
L_{\mathrm{pred}}=
\frac1{|\mathcal V|}\sum_{v\in\mathcal V}
\|\mu_g(x)-z(v)\|_2^2.
\]

The implementation reshapes batched global/all-view embeddings so the mean is computed over views belonging to the same image. Do not flatten the batch before the mean: averaging across different source images would define a different objective.

Then LeJEPA adds SIGReg to the embedding distribution:

\[
L_{\mathrm{LeJEPA}}=(1-\lambda)L_{\mathrm{pred}}+\lambda L_{\mathrm{SIGReg}}.
\]

Here \(\lambda\) balances predictive agreement with distribution regularization. The paper reports that the recipe can be trained without the EMA teacher, stop-gradient and specialized target schedules often associated with earlier methods. This is one of its claims and design goals; it does not mean all SSL systems can omit those mechanisms.

### What changed in the optimization problem?

In BYOL, online features predict an EMA teacher's target and stop-gradient defines which branch receives gradients. In SimSiam, the predictor and stop-gradient change gradient flow without EMA. In LeJEPA, the global-view mean forms the target and SIGReg constrains the distribution; it does not rely on the same teacher-student update. Its simplicity follows from this different objective design, not from the claim that teacher-based methods were mathematically pointless.

### What the paper reports

The authors evaluate LeJEPA over many datasets and architectures and report stable training with a single objective trade-off parameter, linear time/memory complexity in their formulation, and good frozen-backbone results. These are empirical paper claims under its protocol. Comparisons must preserve training data, architecture, compute, view augmentation and evaluation protocol; they should not be paraphrased as a universal proof that LeJEPA always outperforms teacher-based SSL.

## 6. VISReg: separate scale, shape and center

VISReg is motivated by two limits that the authors identify:
1. VICReg's covariance objective regulates only second-order structure.
2. SIGReg's Epps–Pulley discrepancy can provide weak gradients near collapse in the reported analysis.

VISReg retains explicit scale regularization but replaces covariance with a sliced-Wasserstein distribution-shape loss, and adds a centering term.

### 6.1 Scale term

Center the embedding batch:

\[
\widehat Z=Z-\mu,\qquad
\mu=\frac1B\sum_{b=1}^{B}z_b.
\]

For dimension j, let \(\sigma_j(\widehat Z)\) be its empirical standard deviation. The paper's scale term is

\[
L_{\mathrm{scale}}=
\frac1d\sum_{j=1}^{d}(1-\sigma_j(\widehat Z))^2.
\]

Unlike a hinge-style variance floor, this term pulls each dimension's standard deviation toward one, subject to the paper's centering and numerical conventions.

### 6.2 Separate scale from shape

VISReg normalizes centered embeddings using the standard deviation with a stop-gradient:

\[
\widetilde Z=
\frac{\widehat Z}{\operatorname{sg}(\sigma)+\epsilon}.
\]

The purpose of stop-gradient here is to prevent shape gradients from being entangled with changes to scale. This has a different declared role from stop-gradient as a branch-asymmetry mechanism in SimSiam. The operation is not merely another anti-collapse trick; it decomposes two terms that would otherwise interfere.

### 6.3 Sliced-Wasserstein shape loss

Let \(w_k\) denote a sampled projection direction. For each direction, project normalized embeddings into one dimension and sort the batch projections. Compare the ordered values with the sorted quantiles of a standard normal sample/reference:

\[
L_{\mathrm{shape}}=
\frac1K\sum_{k=1}^{K}
\left\|
\operatorname{sort}(\widetilde Z w_k)-q_{\mathcal N}
\right\|_2^2.
\]

The vector \(q_{\mathcal N}\) contains the fixed standard-normal quantiles used by the implementation. In one dimension, squared 2-Wasserstein distance between empirical distributions can be computed from sorted samples/quantiles. Repeating this comparison across random directions forms a sliced Wasserstein distance. Its target is projected distribution shape, not only zero off-diagonal covariance.

The Cramér–Wold theorem motivates why projections can characterize distributions when all directions are considered. A finite random set of slices is an approximation; K is a practical regularization parameter trading compute for coverage of directional structure.

### 6.4 Centering term and full regularizer

The paper adds

\[
L_{\mathrm{center}}=\|\mu\|_2^2.
\]

The regularizer is

\[
L_{\mathrm{Reg}}=
\lambda_{\mathrm{scale}}L_{\mathrm{scale}}+
\lambda_{\mathrm{shape}}L_{\mathrm{shape}}+
\lambda_{\mathrm{center}}L_{\mathrm{center}}.
\]

The paper describes the full training loss as a weighted combination of prediction and regularization, with the prediction term matching each view embedding to the mean embedding of global views. The exact weighting and parameter grouping should follow the paper version and code selected for reproduction.

### 6.5 How VISReg differs from SIGReg

SIGReg uses a frequency-domain characteristic-function discrepancy based on an Epps–Pulley normality test. VISReg uses a transport-based shape distance: sort scalar projections and match their empirical quantiles to standard-normal quantiles. The two share the goal of matching projected distributions to an isotropic Gaussian but use different statistics and gradient behavior.

The VISReg authors report stronger corrective gradients near collapse in their experiments and favorable OOD/low-rank/long-tailed results. These are paper-reported results, not a formal guarantee that every VISReg model avoids every possible failure. Report the tested architecture, initialization, dataset, optimizer and objective weights when discussing the result.

## 7. VICReg → SIGReg/LeJEPA → VISReg

| Property | VICReg | LeJEPA / SIGReg | VISReg |
|---|---|---|---|
| Prediction target | Paired view embedding | Mean global-view embedding / JEPA prediction | Mean global-view embedding / JEPA prediction |
| Scale | Standard-deviation floor | Gaussian target sets scale as part of the full target | Explicit standard deviation toward one |
| Shape | Off-diagonal covariance | Epps–Pulley characteristic-function test over random projections | Sliced-Wasserstein distance of random projections to Gaussian quantiles |
| Center | Depends on feature normalization | Captured by distribution test | Explicit squared batch mean |
| Target mechanism | Paired views, no teacher required | No EMA teacher or stop-gradient required by the paper's recipe | Stop-gradient appears inside normalization to decouple scale and shape, not as a BYOL-style target branch |
| Main claim | Interpretable variance/invariance/covariance decomposition | Gaussian embedding distribution is optimal under the authors' downstream-risk theory; SIGReg approximates it efficiently | Separate scale and shape while using sliced-Wasserstein distribution matching |
| Main limitation | Second-order constraints do not determine distribution shape | Finite random projections/test discretization and assumptions of Gaussian-optimality theory | More regularizer coefficients and random-slice budget; empirical claims need matched protocols |

## 8. What the reader should not infer

- Isotropic Gaussian embeddings are not a universally optimal target for every task. LeJEPA's optimality result is theorem-bound to the paper's predictive-risk setup and assumptions.
- Matching projected samples to Gaussian references over a finite number of directions does not prove the entire empirical high-dimensional distribution is Gaussian.
- Non-collapse is not semantic sufficiency. A representation can have a well-shaped distribution and still discard velocity, contact, object identity or another task-relevant variable.
- A strong gradient under a toy collapse diagnostic is evidence about that parameterization and regularizer, not a theorem that no local minima, dimensional collapse or pathological representations exist.
- OOD transfer or linear-probe results do not prove that a representation supports action-conditioned dynamics or planning. Those need distinct evaluations.
- “Heuristic-free” in the paper title is shorthand for avoiding particular commonly used teacher/EMA/stop-gradient/scheduler heuristics in the proposed recipe; VISReg itself uses stop-gradient for scale/shape decoupling. The role and location of an operation matter more than its name.

## 9. Place in the book

Chapter 3 should close this lineage with a short statement: statistical anti-collapse evolved from second-order constraints toward explicit distribution-shape targets, and predictive architectures provide a place to use those targets. Chapter 4 should derive the JEPA prediction objective and then compare VICReg/LeJEPA/VISReg regularizers as choices for the embedding distribution. That flow avoids teaching JEPA twice while showing a meaningful current progression.

## References

- Bardes, A., Ponce, J. and LeCun, Y. (2022). VICReg. https://arxiv.org/abs/2105.04906.
- Balestriero, R. and LeCun, Y. (2025, v3 dated 14 November 2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. https://arxiv.org/abs/2511.08544.
- Wu, H., Balestriero, R. and Levine, M. (2026). VISReg: Variance-Invariance-Sketching Regularization for JEPA training. https://arxiv.org/abs/2606.02572.
- Epps, T. W. and Pulley, L. B. (1983). A test for normality based on the empirical characteristic function. Biometrika.
- Cramér, H. and Wold, H. (1936). Some theorems on distribution functions.
- Bonneel, N. et al. (2015). Sliced and Radon Wasserstein barycenters.
