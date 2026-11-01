# 4. Predictive sufficiency and the geometry of a representation

Avoiding a constant representation does not establish that the model has learned a useful state. A representation can vary across observations and still merge two situations that have different consequences. The question is therefore not whether the embedding contains variation. It is whether it preserves the distinctions needed for the predictions and decisions under consideration.

## 4.1 A task-relative definition of a useful state

Let \(X\) denote the available history of observations, \(A\) an action or action sequence, \(Y\) a future quantity relevant to a task, and

\[
Z=f_\theta(X)
\]

the learned representation. The representation is sufficient for the chosen prediction problem when conditioning on \(Z\) and \(A\) preserves the conditional distribution of \(Y\) that was available from \(X\) and \(A\):

\[
P(Y\mid X,A)=P(Y\mid Z,A).
\]

This equality is task-relative. The target \(Y\), action domain, time horizon and data distribution must be specified. A representation sufficient for classifying an image need not be sufficient for predicting the motion of an object. A representation sufficient for short-horizon prediction may fail at a longer horizon. A representation that predicts passive video may fail to predict the effect of a control input.

The action appears on both sides of the equality for a reason. A state representation used for control should retain enough information that the future distribution can be predicted when the action is known. If two histories map to the same \(Z\) but respond differently to the same action, the representation has merged states that the controller must distinguish.

## 4.2 Predictive equivalence between histories

The condition can be understood through an equivalence relation on histories. Fix a family of targets and a set of admissible actions. Define two histories \(x_1\) and \(x_2\) to be predictively equivalent when

\[
x_1\sim x_2
\quad\Longleftrightarrow\quad
P(Y\mid X=x_1,A=a)
=
P(Y\mid X=x_2,A=a)
\]

for every relevant action \(a\) and target \(Y\) in the task specification.

A minimal sufficient predictive state would assign the same representation to histories in the same equivalence class and different representations when the relevant conditional future distributions differ. In practice, a neural representation may preserve more than the minimum needed, and the equivalence relation is only defined relative to the selected tasks and data.

This perspective clarifies why a representation should not be called a physical state simply because it is temporally predictive. A learned feature may support some future predictions without preserving every hidden variable, and the relevant variables can change with the action space and objective. (LeCun, 2022)

## 4.3 The target encoder defines which distinctions count

Suppose the target encoder maps two possible target observations to the same feature:

\[
f_{\bar\theta}(y_1)=f_{\bar\theta}(y_2).
\]

A predictor matching this target representation has no direct loss-based reason to distinguish \(y_1\) from \(y_2\). The two targets are equivalent under the feature space used for training. This can be an intentional invariance, for example when a change in illumination should not affect the representation. It can also be a failure if the difference corresponds to object position, contact, velocity or another variable that matters downstream.

The target representation is thus not a neutral measurement instrument. It defines the geometry in which prediction errors are measured. The learned invariances of that target become the invariances of the predictive objective.

The same applies to dense and pooled representations. A globally pooled vector can be adequate for a scene-level classification target while discarding precise spatial correspondence. A dense token representation can preserve spatial structure, but its predictive objective may still fail to encode the variable required for a particular control task. Dense prediction changes the information being supervised; it does not certify control sufficiency.

## 4.4 Probes measure accessibility under a chosen decoder

A frozen representation is often evaluated by training a downstream predictor on top of it. The choice of predictor determines which notion of accessibility is measured.

A linear probe asks whether the target can be estimated by a linear function of the representation. It is useful because it isolates the geometry exposed by a simple readout. A nonlinear probe asks whether a more expressive function can extract the information. A nearest-neighbour or kernel probe asks whether local geometry supports useful predictions. A planning evaluation asks whether a model and a planner can use predicted states to select successful actions in an environment.

These tests answer different questions. A strong linear probe indicates that certain information is readily accessible to a linear readout. A weak linear probe does not prove that information is absent; a nonlinear readout may recover it. Strong representation benchmarks do not establish that the latent state is Markov, causal or sufficient for control. Planning success is evidence about the complete model-planner-environment system, not an encoder in isolation.

A useful evaluation therefore matches the readout to the claim. Representation quality, target-feature prediction, action-conditioned prediction and closed-loop control should not be collapsed into one leaderboard or one metric. (Balestriero and LeCun, 2025; Assran et al., 2025)

## 4.5 Covariance as a first view of embedding geometry

Let a batch of \(B\) embeddings in \(D\) dimensions be arranged as rows of a matrix

\[
Z=
\begin{bmatrix}
z_1^\top\\
\vdots\\
z_B^\top
\end{bmatrix}
\in\mathbb R^{B\times D}.
\]

Let \(\bar z\) be the batch mean and \(Z_c\) the centred matrix. The empirical covariance is

\[
\widehat\Sigma
=
\frac{1}{B-1}Z_c^\top Z_c.
\]

The covariance matrix describes coordinate-wise variance and pairwise linear dependence. Its eigenvectors give principal directions of variation; its eigenvalues quantify the variance along those directions.

If the eigenvalues are all equal, the covariance is isotropic:

\[
\widehat\Sigma=\sigma^2 I_D.
\]

If a small number of eigenvalues dominate while others are close to zero, the embedding cloud is anisotropic or approximately low rank. The model may use only a small set of directions, even when every coordinate has some nonzero marginal variance.

Covariance is useful because it provides a tractable description of second-order geometry. It is not a description of the full distribution. Two distributions can share the same mean and covariance while differing in multimodality, tail behaviour or support. Zero covariance does not imply statistical independence, and a full-rank covariance matrix does not imply that the representation preserves the right semantic variables.

This limitation motivates the distinction between moment regularization and distributional regularization. Before introducing the latter, however, it is useful to understand why covariance geometry matters even for a simple downstream estimator.

## 4.6 Why the spectrum matters

Consider two embedding distributions with the same total variance,

\[
\operatorname{tr}(\Sigma)
=
\sum_{k=1}^{D}\lambda_k.
\]

An isotropic distribution spreads this variance equally across directions. An anisotropic distribution concentrates it unevenly. The total amount of variation can be identical while the weakest direction is much narrower in the anisotropic case.

A downstream estimator that relies on a narrow direction may be poorly conditioned. Regularization can shrink its coefficient strongly because the data provide little variation along that direction. Without regularization, small eigenvalues can amplify estimator variance. These are not merely visual differences between point clouds. They affect the statistical stability of downstream prediction.

The next section derives this dependence for ridge regression and ordinary least squares. The derivation is deliberately limited to those estimators and their assumptions. It is not a proof that isotropic covariance is optimal for every nonlinear task or every controller.
