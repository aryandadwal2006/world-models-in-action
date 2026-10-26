# A theoretical unification of SSL objectives: what is common, what is not

This note expands the theoretical-unification thread in the Cookbook. Its purpose is not to claim that every SSL algorithm optimizes mutual information or that every loss is mathematically equivalent. It provides a shared vocabulary for comparing target construction, feature geometry and the assumptions needed for each interpretation.

## 1. A shared abstraction: samples, views, representation and target

Let X be an observation, T a stochastic data transformation, \(V=T(X)\) a view, and \(Z=f_\theta(V)\) the representation. An SSL recipe constructs one or more target signals from the data and minimizes an empirical risk

\[
L(\theta)=\mathbb E_{(V,Y)\sim \mathcal D_{\mathrm{SSL}}}
\left[\ell\bigl(g_\theta(V),Y\bigr)\right],
\]

where \(\mathcal D_{\mathrm{SSL}}\) is induced by the data, view process, sampling and any teacher/cluster/tokenizer. This notation is broad by design; it does not imply all methods have a separate target Y or one single encoder pass.

The first question for any method is: what is the distribution \(\mathcal D_{\mathrm{SSL}}\)? In contrastive learning, it includes positive pairs and sampled negatives. In masked modeling, it includes visible context and the withheld target. In teacher-student learning, target features or distributions come from the teacher's view and update rule. In clustering, pseudo-labels depend on the current feature space and assignment algorithm. In predictive latent learning, the target is an embedding of another region or time.

Only after the target-generating distribution is specified does it make sense to compare losses.

## 2. Alignment and uniformity for normalized representations

Wang and Isola (2020) analyze contrastive representation learning using two geometric properties of L2-normalized embeddings on the unit hypersphere. For paired views \((x,x^+)\) from a positive-pair distribution, define alignment as

\[
L_{\mathrm{align}}=
\mathbb E_{(x,x^+)}
\|f(x)-f(x^+)\|_2^\alpha,
\]

where \(\alpha>0\) controls the distance power. This measures how close matched views become.

For independent samples x,y from the data distribution, define uniformity

\[
L_{\mathrm{uniform}}=
\log\mathbb E_{x,y}
\exp(-t\|f(x)-f(y)\|_2^2),
\]

for a positive scale parameter t. A low value corresponds to embeddings spread out over the unit sphere rather than concentrated in a small region. The exponential weighs nearby pairs strongly, so this metric penalizes crowding.

The key interpretation is that contrastive learning tends to trade off:
- **alignment:** positive pairs are made close;
- **uniformity:** unrelated examples remain distributed throughout the normalized feature space.

This is a geometric lens on contrastive objectives, not an exact rewriting of every SSL loss as two independent terms under arbitrary training protocols. Its stated assumptions include normalized representations and the sampling setup used in the paper.

## 3. Why alignment alone collapses

If \(f(x)=c\) for every input x, then every positive pair agrees perfectly, and \(L_{\mathrm{align}}=0\). But all embeddings are identical, so the representation is useless for distinguishing examples. Alignment alone cannot prevent collapse.

Uniformity pushes in the opposite direction by penalizing feature crowding, but “uniformity” is not a synonym for every possible anti-collapse constraint. It describes a particular geometry for normalized embeddings. VICReg's coordinate-wise variance floor, Barlow Twins' cross-correlation penalty, SIGReg's Gaussian target and VISReg's projected-quantile matching use different spaces and statistics.

The abstract commonality is that a nontrivial representation needs more than cross-view agreement. The actual mathematical condition is family-specific.

## 4. InfoNCE, density ratios and dependence measures

Given anchor x and candidates \(y_1,\ldots,y_K\), a contrastive classifier minimizes

\[
L_{\mathrm{NCE}}=
-\mathbb E\log\frac{\exp f_\theta(x,y^+)}
{\sum_{j=1}^{K}\exp f_\theta(x,y_j)}.
\]

Under the appropriate positive-joint/negative-product-of-marginals sampler, the ideal critic is related to the log density ratio

\[
f^*(x,y)=\log\frac{p(y\mid x)}{p(y)}+c(x),
\]

where \(c(x)\) does not change the normalized candidate distribution for fixed x. This makes candidate classification a form of density-ratio discrimination.

Under the sampler and candidate-count convention used in the standard derivation, InfoNCE also yields

\[
I(X;Y)\geq \log K-L_{\mathrm{NCE}}.
\]

But several different statements must not be conflated:
1. **Classification identity:** InfoNCE is cross-entropy over candidates. This is exact.
2. **Density-ratio optimum:** the ideal critic is related to the joint/marginal ratio under the specified sampler. This is a population statement with assumptions.
3. **Mutual-information lower bound:** this follows under the bound's sampling and normalization conditions.
4. **Downstream representation quality:** this is an empirical and task-dependent question, not a corollary of the bound.

Tschannen et al. (2020) argue that MI maximization alone does not explain the success of representation learning; architecture, estimator parameterization and inductive bias also matter. Therefore the book should derive InfoNCE operationally first, explain the bound second, then state why it is not a proof of task-relevant state.

## 5. CCA and redundancy reduction

Classical canonical correlation analysis identifies linear projections of paired random vectors that maximize cross-view correlation. SSL methods such as Barlow Twins and VICReg are related in spirit because they use paired views and constraints on cross-view feature statistics, but neither should be called identical to classical CCA.

Barlow Twins matches the cross-correlation matrix to identity. VICReg combines direct view invariance with variance and off-diagonal covariance penalties. These objectives encode a different constraint than uniformity on a hypersphere:
- cross-correlation identity is a second-order condition;
- a variance floor prevents coordinates from becoming constant;
- uniformity describes global spread in a normalized space;
- Gaussian distribution matching imposes more than a covariance matrix.

A distribution can have identity covariance but non-Gaussian higher-order structure. Conversely, a Gaussian target fixes higher-order distributional shape as well as first/second moments. These are related ways to encourage diversity, not interchangeable objectives.

## 6. Clustering as latent-variable supervision

DeepCluster produces assignments from current embeddings, then uses assignment IDs as pseudo-labels. SwAV produces online assignments to prototypes and predicts the assignments across views. DINO outputs probability distributions over prototypes and matches a teacher distribution.

All three exploit a discrete latent assignment, but differ in how it is created and updated:
- offline clustering recomputes dataset-level assignments;
- SwAV creates balanced online assignments with an iterative normalization procedure;
- DINO uses teacher-student distribution matching, with centering and sharpening;
- MSN combines masking with prototype self-labeling.

A probabilistic interpretation can treat the cluster ID as a latent variable, but the learned labels remain algorithm-generated targets. They do not prove that each prototype corresponds to a ground-truth semantic class. Assignment balancing itself is a prior over how much probability mass each prototype should receive.

## 7. Reconstruction objectives as conditional prediction

Masked reconstruction often minimizes

\[
L_{\mathrm{rec}}=\mathbb E[\ell(\hat Y(X_V),Y_M)].
\]

The loss implies an observation model. For example, squared error is the negative log-likelihood of a Gaussian target with fixed variance, up to constants; categorical cross-entropy is the negative log-likelihood of a categorical target. The interpretation depends on the assumed target distribution and reduction.

MAE predicts pixels, BEiT predicts discrete visual tokens, MaskFeat predicts handcrafted descriptors, iBOT predicts teacher/prototype targets, and data2vec predicts contextualized teacher features. Their shared masked-input skeleton hides different conditional prediction problems. The target's information content and noise model shape the learned representation.

Reconstruction can be useful, but its loss only certifies the ability to predict the chosen target from visible context under the training distribution. It does not directly certify state sufficiency or causal dynamics.

## 8. Teacher-student objectives as moving-target regression

A generic teacher/student objective may be written

\[
L_{\mathrm{distill}}=
-\mathbb E\sum_k q_{\mathrm{teacher},k}\log p_{\mathrm{student},k},
\]

or as feature regression between student prediction and target feature. These two target spaces produce different gradients. A teacher distribution is matched through softmax cross-entropy; a vector target is matched through cosine/MSE or another regression loss.

EMA alters how the target evolves, while stop-gradient alters which pathways receive gradients. A predictor changes the function the online branch must learn. Centering/sharpening alter the teacher distribution; a variance/covariance penalty instead constrains batch statistics. There is no single mathematically universal “teacher mechanism.”

The target may evolve slowly enough to stabilize learning but still permit collapse in principle. Loss geometry and update dynamics must be analyzed together.

## 9. Joint-embedding prediction as a distinct predictive relation

A JEPA-style objective predicts a target representation from a context representation:

\[
L_{\mathrm{pred}}=
\mathbb E\left[\|\hat z_{\mathrm{target}}-
z_{\mathrm{target}}\|^2\right]
\]

as a schematic form; actual methods may predict multiple regions, use a view mean, normalize features or aggregate losses differently.

The key choice is to predict a representation, rather than necessarily reconstructing the raw target input. This can ignore unpredictable pixel detail while retaining structure that the target encoder encodes. Whether it does retain the variables required by the environment depends on target construction, data, model architecture, prediction horizon and evaluation.

Passive frame/video feature prediction does not imply action conditioning. If action A does not enter the predictor or its target-generation mechanism, the objective alone does not define how alternative actions change the future.

## 10. A unified comparison checklist

For any SSL method, ask these questions in order:

1. **Distribution:** what joint, conditional, marginal or target distribution is sampled?
2. **Equivalence:** which views/examples are declared related?
3. **Target:** candidate index, target vector, teacher distribution, pseudo-label, masked symbol or pixel?
4. **Geometry:** cosine similarity, dot product, MSE, cross-entropy, covariance, prototype assignment, Gaussian test or transport distance?
5. **Gradient path:** which branches receive gradients, and which targets are detached?
6. **Update dynamics:** EMA, clustering refresh, tokenizer training, memory queue or ordinary gradient descent?
7. **Degenerate solutions:** constant collapse, dimensional collapse, assignment collapse, false negatives, target shortcuts or over-reconstruction?
8. **Assumptions:** balanced data, well-curated positives, correct augmentation invariance, Gaussian embedding prior, spatial correspondence?
9. **Evidence:** which probe or downstream task was tested, and under what compute/data recipe?
10. **World-model gap:** what additional evidence is needed for sufficient state, action-conditioned prediction and planning?

This is the unification the chapter can safely use: every method makes assumptions about the target relation and imposes a geometry or statistical condition on embeddings. It is not justified to claim that every method maximizes the same mutual information, approximates the same likelihood or guarantees a universally useful representation.

## References

- Wang, T. and Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. https://arxiv.org/abs/2005.10242.
- Poole, B. et al. (2019). On Variational Bounds of Mutual Information. https://proceedings.mlr.press/v97/poole19a.html.
- Tschannen, M. et al. (2020). On Mutual Information Maximization for Representation Learning. https://arxiv.org/abs/1907.13625.
- Balestriero, R. et al. (2023). A Cookbook of Self-Supervised Learning. https://arxiv.org/abs/2304.12210.
