# 10 — Proof notebook and scientific-claims audit

This file is the guardrail for turning the author's research notes into a technically reliable textbook chapter. Each note-derived claim should be classified as: (A) a definition or direct identity, (B) a theorem under assumptions, (C) an empirical result reported by a paper, (D) a useful intuition, or (E) an unsupported or overgeneralized statement.

## 1. Claim audit

| Note-derived claim | Status | Required treatment |
|---|---|---|
| A constant embedding can minimize a pure view-agreement/prediction loss | Mathematically true for the stated unregularized objective | Derive with a simple counterexample; do not claim every training run collapses |
| Branchwise variance/covariance penalties guarantee information content is preserved | Too strong | They constrain selected batch statistics, not semantic/task information |
| VICReg decorrelation guarantees independent dimensions | False in general | Say it reduces linear covariance, not arbitrary dependence |
| Covariance has a cost involving batch size and dimension squared | Correct for explicitly forming a dense covariance matrix | State the dimensions, implementation and hardware; do not label it categorically fatal |
| Covariance regularization is a dead end for frontier models | Unsupported generalization | Compare actual bottlenecks and algorithms; it remains a design trade-off |
| First and second moments do not determine a general distribution | Correct | Use a simple counterexample or distribution sketch |
| An isotropic Gaussian is universally optimal for any unknown downstream task | False as stated | Attribute the result to LeJEPA's precise risk model, estimator class and constraints |
| Every finite random-projection sketch matches the full distribution | False | Distinguish Cramér–Wold's all-directions theorem from the finite stochastic approximation |
| SIGReg's gradient vanishes exponentially near every collapse point | Too broad | Report VISReg's analysis/experiments under the configurations studied; verify exact assumptions |
| VISReg's scale term has a nonzero derivative at zero standard deviation | Can be true for its stated scalar scale loss | Distinguish a derivative with respect to the standard deviation from a bound on the full network-parameter gradient |
| A decoupled shape gradient guarantees the model cannot collapse | Too strong without a theorem covering optimization and parameterization | Describe the intended gradient separation and paper-reported behaviour |
| VISReg complexity is linear | Incomplete without projection count | State dependence on batch B, dimension d and number K of directions; if K scales with d, the dependence changes |
| One-dimensional empirical Wasserstein comparison can use sorted samples | Correct under the specified equal-weight/sample convention | State quantile and weighting assumptions; note piecewise differentiability/ties |
| Hierarchical JEPA is the inevitable architecture for long horizons | Not established | Present as a research proposal unless supported by a specific method/paper |
| Latent planning with JEPAs is guaranteed to outperform input-space planning | False | Treat as an empirical research question; report planner, data and environment |
| A JEPA latent is automatically causal, controllable or Markov | False | Require task-relative predictive tests and action-conditioned evidence |
| A Path to Autonomous Machine Intelligence is a validated end-to-end system | False | Identify it as a position paper and proposed architecture |

## 2. Proof 1: conditional mean for squared-error prediction

For context \(C=c\), define \(\mu=\mathbb E[H\mid C=c]\). Then:
\[
\mathbb E[\|H-a\|^2\mid c]
=\mathbb E[\|H-\mu\|^2\mid c]+\|a-\mu\|^2.
\]
The cross term is zero by the definition of the conditional expectation. Thus the unique minimizer (when the conditional second moment exists) is \(a=\mu\).

**Interpretation:** a deterministic MSE predictor averages all target outcomes compatible with its context. Under a multimodal future, the average may lie between modes. A probabilistic target/predictor can represent uncertainty more explicitly, but only if its likelihood/objective and calibration are appropriate.

## 3. Proof 2: EMA is a weighted parameter history

For constant \(m\in[0,1)\):
\[
\xi_{t+1}=m\xi_t+(1-m)\theta_{t+1}.
\]
Iterating:
\[
\xi_t=m^t\xi_0+(1-m)\sum_{j=0}^{t-1}m^j\theta_{t-j}.
\]
This establishes smoothing of parameter history. It does not establish stability or prevent all collapsed solutions.

## 4. Proof 3: moment matching does not determine a distribution

Use a one-dimensional example. A standard normal and a symmetric two-point distribution at \(\{-1,+1\}\) both have mean zero and variance one, but their characteristic functions differ:
\[
\varphi_{\mathrm{two-point}}(t)=\cos(t),
\qquad
\varphi_{\mathcal N(0,1)}(t)=e^{-t^2/2}.
\]
This is enough to demonstrate that mean and variance (and hence covariance in a one-dimensional case) do not determine the full distribution.

For multivariate examples, a non-Gaussian distribution can have identity covariance. Do not overcomplicate the main text with an unnecessary catalogue of shapes.

## 5. Proof 4: Cramér–Wold and its approximation boundary

The theorem compares distributions of projections in every direction. Training samples a finite set of directions. Therefore:
- all-direction equality is an identifiability result;
- finite-direction matching is an approximation/stochastic estimator;
- minibatch estimates add sampling error;
- projection count and distribution affect estimator variance.

The manuscript must never equate the theorem with exact finite-sample Gaussianity.

## 6. Proof 5: one-dimensional empirical transport

For equal-mass empirical distributions on the real line, the monotone coupling matches order statistics. The squared 2-Wasserstein distance becomes the average squared difference between corresponding sorted samples. The identity depends on equal weights/cardinalities; weighted or unequal-size distributions require the appropriate quantile/integration formulation.

## 7. Action-conditioned sufficiency

Let \(X\) be available history, \(Z=f(X)\) the latent representation, \(A\) an action, and \(Y\) a future target. A task-relative sufficient state for this prediction problem would satisfy:
\[
P(Y\mid X,A)=P(Y\mid Z,A).
\]
The action is conditioned on both sides. If \(f(x_1)=f(x_2)\) but the future distributions under the same action differ, the representation aliases histories that a predictor cannot distinguish.

This is a direct extension of the Chapter 2/3 state contract, not a new universal theorem about every possible world model.

## 8. Complexity audit checklist

For every complexity statement, specify:
- batch size \(B\);
- embedding dimension \(d\);
- projection count \(K\);
- token count \(N\), if the operation acts on tokens;
- sequence length \(T\), if temporal attention or prediction is involved;
- whether memory or time is being described;
- whether \(K\) is fixed or scales with \(d\);
- whether the quoted cost covers only a regularizer or the entire training step.

Do not describe a regularizer's asymptotic cost as the cost of the whole JEPA.

## 9. Empirical-claim audit

Any benchmark table must record:
- the paper/version/date;
- model size and training data;
- frozen vs fine-tuned features;
- probe type and training data;
- split and metric;
- whether the result is an encoder, predictor or agent result;
- whether planning is simulated or physical;
- environment familiarity and task-specific adaptation.

Avoid a leaderboard-style table mixing classification accuracy, feature retrieval, prediction loss and robot success percentage.

## 10. Mandatory attribution

The Gaussian-target theorem belongs to LeJEPA and its assumptions. The finite projection statistic belongs to SIGReg. The scale/shape treatment belongs to VISReg. The action-conditioned robot-planning evidence belongs to V-JEPA 2-AC and JEPA-WMs papers. The broader cognitive architecture belongs to LeCun's 2022 position paper. None of these claims should be attributed to “JEPA” generically.
