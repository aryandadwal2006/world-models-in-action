# Redundancy reduction: Barlow Twins and VICReg

## 1. The shared problem

Given two views of the same source example, we want their representations to agree on common information. If that is the only constraint, the constant map solves the objective: every view receives the same vector. Contrastive methods counter this with candidate discrimination. Redundancy-reduction methods instead impose constraints on the distribution of features across a minibatch.

The constraints should be separated into three mathematical ideas:

- **Invariance:** representations of two views of the same source should agree.
- **Non-collapse / variance:** each feature dimension should vary across examples rather than take one constant value.
- **Redundancy:** different dimensions should not carry linearly redundant variation, measured through covariance or cross-correlation.

These conditions are not equivalent to semantic structure. A random high-variance orthogonal code can satisfy moment constraints without encoding the information a downstream task needs.

## 2. Barlow Twins from the cross-correlation matrix

Take a batch of N paired examples. The two branches map them to representations \(z^A,z^B\in\mathbb R^{N\times d}\). Normalize each feature dimension across the batch to obtain \(\tilde z^A,\tilde z^B\), with approximately zero mean and unit variance. Define the cross-correlation matrix

\[
C_{ij}=\frac{1}{N}\sum_{b=1}^{N}\tilde z^A_{b,i}\tilde z^B_{b,j}.
\]

If corresponding dimensions agree and distinct dimensions are decorrelated across views, then C is close to the identity matrix. Barlow Twins minimizes

\[
L_{\mathrm{BT}}=\sum_i (1-C_{ii})^2+
\lambda\sum_i\sum_{j\ne i}C_{ij}^2.
\]

The diagonal term aligns corresponding features across views. The off-diagonal term discourages a feature in one view from being correlated with a different feature in the other view. The coefficient λ controls the trade-off; the standard formulation does not make the diagonal and off-diagonal penalties identical.

### What this objective does

When \(z^A=z^B\) is constant, batch standardization becomes degenerate and the desired cross-correlation identity cannot be obtained in the ordinary well-defined way. The objective therefore creates pressure away from collapse through the normalized cross-view statistics. However, finite batches and normalization choices matter, and implementation details determine behavior for nearly constant dimensions.

An identity cross-correlation matrix is a second-order statistical condition. It is not proof of statistical independence unless stronger distributional assumptions hold, and it is not proof that the coordinates correspond to physical variables. The method reduces redundancy in a specific measured sense.

## 3. VICReg separates the constraints

VICReg makes the three terms explicit. Let \(z,z'\in\mathbb R^{N\times d}\) be the two feature batches.

The invariance term is a mean-squared error:

\[
s(Z,Z')=\frac{1}{Nd}\lVert Z-Z'\rVert_F^2.
\]

For dimension j, calculate its empirical standard deviation over the batch:

\[
\sigma_j(Z)=\sqrt{\operatorname{Var}_{b}(Z_{b,j})+\epsilon}.
\]

The variance penalty is

\[
v(Z)=\frac{1}{d}\sum_{j=1}^{d}\max(0,\gamma-\sigma_j(Z)),
\]

with an analogous term for Z'. When a feature dimension's standard deviation is at least the margin γ, the term contributes zero; below the margin, it pushes the dimension to vary more. The covariance penalty uses the off-diagonal entries of the empirical covariance matrix:

\[
c(Z)=\frac{1}{d}\sum_{i\ne j}\operatorname{Cov}(Z)_{ij}^{2}.
\]

The full objective is a weighted combination:

\[
L_{\mathrm{VICReg}}=\lambda\,s(Z,Z')+
\mu\,[v(Z)+v(Z')]+\nu\,[c(Z)+c(Z')].
\]

The published implementation uses a particular set of coefficients and numerical conventions. The equation is a structural description, not a universal parameter setting.

### Why the terms are not redundant

- If two views differ arbitrarily, invariance penalizes the mismatch.
- If every example maps to the same vector, invariance can be zero but variance is below the margin.
- If many dimensions co-vary linearly, the covariance term penalizes off-diagonal covariance.
- If features vary but encode irrelevant noise, variance and covariance terms alone do not solve the representation problem.

The variance penalty can be understood coordinate by coordinate. For \(\sigma_j<\gamma\), increasing the spread of that coordinate lowers the penalty; once the margin is met, this term is inactive. It is therefore a floor, not a reward for unbounded variance. Other objective terms and the rest of the architecture determine the eventual scale.

## 4. Relation between the methods

Both methods match views and reduce redundancy. Their formulations differ:

| Question | Barlow Twins | VICReg |
|---|---|---|
| Invariance | Diagonal cross-correlation should approach one | Direct MSE between paired representations |
| Anti-collapse | Cross-correlation identity under batch normalization | Explicit per-dimension standard-deviation floor |
| Redundancy | Off-diagonal cross-correlation | Off-diagonal covariance |
| Scale/normalization | Standardized dimensions are central | Feature scale and covariance enter separate terms |
| Main conceptual benefit | One cross-correlation objective | Separates invariance, variance and covariance so each contribution is inspectable |

Do not claim one universally prevents collapse better. That depends on data, architecture, batch size, normalization, coefficients, optimizer and evaluation. In the chapter, their purpose is conceptual: demonstrate that negatives are one anti-collapse strategy, not the only one.

## References

- Zbontar, J. et al. (2021), Barlow Twins: Self-Supervised Learning via Redundancy Reduction, arXiv:2103.03230.
- Bardes, A., Ponce, J. and LeCun, Y. (2022), VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning, arXiv:2105.04906.
