# 05 — LeJEPA and SIGReg: distributional regularization

## 1. The research question

VICReg makes non-collapse constraints explicit through variance and covariance penalties. LeJEPA asks whether the geometry of the embedding distribution can be justified from a downstream prediction-risk perspective, and SIGReg provides a tractable way to regularize toward the proposed target.

The primary source is Balestriero and LeCun (2025), “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics,” arXiv:2511.08544. Its central claims must be tied to the paper's mathematical setup and assumptions. Do not convert the paper's optimum into a universal theorem about representations for every possible task.

## 2. Why an isotropic Gaussian appears

The paper analyses downstream prediction risk under a particular statistical framework and argues that the embedding distribution should be isotropic Gaussian under that analysis. The desired target is:
\[
Z\sim\mathcal N(0,I_d).
\]

This target has:
- zero mean;
- unit variance in every direction;
- isotropic covariance;
- Gaussian projected distributions along every unit direction.

A crucial boundary: an isotropic Gaussian is not automatically the best representation for every downstream problem. Discrete structure, manifolds, multimodality, task-specific invariances and control variables may call for a different geometry. Attribute the Gaussian-optimality claim to LeJEPA's risk assumptions and identify those assumptions when summarizing the proof.

The full risk derivation in the user's notes should be treated as a proof study, not copied into the main text in one uninterrupted chain. The printed chapter needs the problem setup, the theorem's assumptions, the central geometric idea, and a concise proof sketch. The longer regression and Fisher-information derivations can live in an extended proof notebook.

## 3. Characteristic functions

For a scalar random variable \(U\), its characteristic function is:
\[
\varphi_U(t)=\mathbb E[e^{itU}].
\]
For a standard normal variable:
\[
\varphi_{\mathcal N(0,1)}(t)=e^{-t^2/2}.
\]

Given projected samples \(u_1,\ldots,u_B\), the empirical characteristic function is:
\[
\widehat\varphi_B(t)=\frac1B\sum_{b=1}^{B}e^{itu_b}.
\]

A distributional test compares the empirical characteristic function with the theoretical target over a range of frequencies, using a weight function and numerical integration or quadrature. SIGReg uses an Epps–Pulley-style normality statistic for this purpose. The chapter should give the schematic discrepancy first, then quote the exact statistic, normalization and quadrature from the primary paper and official implementation.

Do not invent a convenient simplified SIGReg expression and present it as the exact paper formula. The exact weights, integration points, scaling and direction-sampling details are algorithmic choices.

## 4. Cramér–Wold and random projections

The Cramér–Wold theorem states that a multivariate probability distribution is determined by the distributions of all its one-dimensional projections. If two distributions have identical projections in every direction, they are identical.

For a high-dimensional embedding vector \(Z\in\mathbb R^d\), choose unit directions \(w_k\) and form:
\[
u_{b,k}=w_k^\top z_b.
\]

SIGReg evaluates a one-dimensional distributional discrepancy for each sampled direction and averages over the finite set. Sampling directions makes the test computationally useful, but it is an approximation. A finite number of sampled projections and finite minibatch do not certify exact distribution equality.

The chapter should explicitly distinguish:
- the exact Cramér–Wold identifiability result;
- the stochastic finite-direction estimator used during training;
- the empirical performance of the method under a particular number of projections and batch size.

## 5. The LeJEPA objective

At a high level, LeJEPA combines a JEPA predictive term with a SIGReg term:
\[
\mathcal L_{\mathrm{LeJEPA}}
=\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}}.
\]

The exact predictive target, averaging over views, target construction and implementation details must be checked against the paper version being cited. This formula is a conceptual decomposition, not a substitute for the method's precise mathematical definition.

The paper emphasizes a leaner formulation that avoids several common heuristics such as EMA teacher updates and stop-gradient asymmetry in its specified setup. Do not generalize this into a statement that all JEPA methods can omit target stabilization or that every architecture will train reliably with this objective without tuning.

## 6. SIGReg's computational argument

The point of the projection approach is to avoid constructing a full high-dimensional distribution distance. The cost depends on:
- batch size;
- embedding dimension;
- number of sampled directions;
- frequency/integration points per one-dimensional test;
- how projections and statistics are implemented.

State complexity from the primary method and code, not by repeating an unverified asymptotic phrase from the notes. A theoretically linear form in one parameter does not imply that the full recipe is independent of the number of random projections or batch size.

## 7. Strengths and limitations

**What the method contributes**
- an explicit distributional target;
- a mathematically motivated Gaussian geometry within its stated risk framework;
- a projected test built from characteristic functions;
- a recipe designed to reduce dependence on several architectural anti-collapse heuristics.

**What it does not establish**
- exact Gaussianity from a finite stochastic estimate;
- causal or physical completeness of the representation;
- sufficient state for every downstream target;
- action-conditioned transition accuracy;
- robust long-horizon rollouts;
- universally superior OOD performance on arbitrary data.

The VISReg paper discusses gradient behaviour of SIGReg near collapse in its tested setup. If that comparison is included, attribute it to the VISReg authors and state the tested conditions. Avoid describing it as a theorem about all SIGReg implementations.

## 8. Figures and worked examples

**Figure 4.9 — From a batch of embeddings to projected normality tests.** Draw an embedding cloud, several sampled directions, the resulting scalar samples and the characteristic-function comparison against a standard-normal target.

**Figure 4.10 — Exact theorem versus finite sketch.** Depict the all-directions condition as an idealized statement and the finite set of random directions as a stochastic training approximation.

A short worked example can compare two one-dimensional projected sample sets that share mean and variance but differ in shape. This visually motivates why a moment penalty does not fully specify a distribution.

## 9. The LeJEPA proof notebook

Include a concise theorem statement only after checking the original assumptions. A longer appendix should organize the argument as:
1. define the downstream prediction risk and admissible predictor family;
2. specify fixed constraints such as covariance/scale;
3. derive the risk dependence on embedding geometry;
4. identify the Gaussian minimizer under the stated functional;
5. add the isotropy condition under the specified scalar constraint;
6. state what changes if the assumptions or downstream estimator family change.

Do not compress this into “Gaussian is optimal for all downstream tasks.” The paper's result is valuable precisely because it specifies a mathematical setting.

## 10. Sources

- Balestriero and LeCun (2025), arXiv:2511.08544.
- Project/code pointer recorded by the primary paper: https://github.com/rbalestr-lab/lejepa
- The user's extended derivations in the JEPA notes, cross-checked against the paper before quotation.
