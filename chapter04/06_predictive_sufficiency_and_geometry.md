# 6. What should a predictive representation preserve?

A non-collapsed representation can still be a poor state. The relevant question is not whether the embedding varies, but whether it distinguishes histories that lead to different predictions or decisions. This makes representation quality a task-relative property.

## 6.1 Predictive sufficiency is defined relative to a target

Let \(X\) denote the observation history available to an agent, \(A\) an action or action sequence, \(Y\) a future quantity of interest, and

\[
Z=f_\theta(X)
\]

the learned representation. For the chosen prediction problem, \(Z\) is sufficient when

\[
p(Y\mid X,A)=p(Y\mid Z,A)
\]

almost surely under the data distribution.

The equality says that once the representation and action are known, the original history provides no additional information about the target distribution. This is a statistical definition, not a claim that \(Z\) equals the simulator's physical state.

The target \(Y\), action set, prediction horizon and data distribution must be specified. A representation can be sufficient for object classification but insufficient for contact prediction. It can be sufficient for a short visual forecast but insufficient for a long physical rollout. It can be sufficient for passive future prediction yet insufficient for predicting the effect of a candidate action.

The action appears on both sides because a controller needs to know whether two histories respond differently to the same command. If \(x_1\) and \(x_2\) map to the same \(Z\), but the same action produces different future distributions, then the representation has merged histories the controller must distinguish.

## 6.2 Predictive equivalence between histories

The condition can be understood as an equivalence relation. Fix the family of targets and admissible actions relevant to the task. Two histories \(x_1\) and \(x_2\) are predictively equivalent when

\[
x_1\sim x_2
\quad\Longleftrightarrow\quad
p(Y\mid X=x_1,A=a)
=
p(Y\mid X=x_2,A=a)
\]

for every relevant target \(Y\) and action \(a\).

A minimal sufficient representation would assign the same code to histories within one equivalence class and different codes to histories whose relevant conditional futures differ. A learned encoder is not guaranteed to recover these classes. It is trained through a proxy target, and the target encoder may already identify some observations as equivalent.

Consider two images of the same cup under different lighting. If the lighting change leaves every downstream prediction unchanged, treating the images as equivalent is useful. Now consider two images with the cup at different distances from a gripper. If the gripper will move under the same command, those positions may lead to different collision or contact outcomes. Treating them as equivalent could be destructive.

The correct invariances are therefore determined by the task, not by visual similarity alone.

## 6.3 The target encoder decides which differences count

Suppose two target observations satisfy

\[
f_{\bar\theta}(y_1)=f_{\bar\theta}(y_2).
\]

A predictor matching this target feature has no direct reason to distinguish \(y_1\) from \(y_2\). The target encoder has made them equivalent for the loss.

This can be exactly what is wanted. The target may remove small lighting differences, sensor noise or background texture. But if it also removes object position, relative orientation or a contact configuration required downstream, the prediction loss cannot recover that distinction by itself.

The target encoder is therefore part of the task specification. Its architecture, training data, augmentations and previous objectives define the geometry in which prediction errors are measured. A dense feature map may preserve spatial correspondence better than a pooled vector, but density alone does not guarantee that the right spatial information is present.

This also explains why a representation benchmark is not a universal measure of state quality. A feature can perform well on an image classification benchmark while discarding information needed for manipulation. A probe only measures the information that its target and readout ask for.

## 6.4 What a probe actually measures

A frozen representation is often evaluated by fitting a predictor on top of it. The predictor class determines the claim supported by the result.

A linear probe asks whether the target is accessible through a linear map. A nonlinear probe asks whether a more expressive function can extract it. A nearest-neighbour or kernel estimator tests whether local geometry supports prediction. A closed-loop control evaluation tests whether the complete model and controller can use the representation to select actions in an environment.

These tests are not interchangeable. A weak linear probe does not prove that the information is absent; it may be encoded in a form that a linear map cannot extract. A strong nonlinear probe does not show that the information is easy to use for planning. Successful planning is evidence about the complete system, including the transition predictor, cost, search procedure and feedback loop.

A useful evaluation should therefore match the readout to the claim. Representation quality, feature prediction, action-conditioned prediction and closed-loop control should be reported separately.

## 6.5 Covariance describes only second-order geometry

Arrange \(N\) centred embeddings in a matrix \(Z\in\mathbb R^{N\times D}\). The empirical covariance is

\[
\widehat\Sigma
=
\frac{1}{N-1}Z^\top Z.
\]

Its diagonal entries are coordinate variances. Its off-diagonal entries measure pairwise linear co-variation. Its eigenvectors give principal directions of variation, while its eigenvalues quantify variance along those directions.

If the eigenvalues are equal, the covariance is isotropic:

\[
\widehat\Sigma=\sigma^2 I_D.
\]

If a few eigenvalues dominate while others are small, the embedding cloud is anisotropic. The representation varies more in some directions than others. This can matter for downstream estimation: a small eigenvalue means the dataset contains little variation along that feature direction.

Covariance is not the full distribution. Two distributions can have identical means and covariance while differing in skewness, multimodality, tails or support. Zero covariance does not imply independence, and a full-rank covariance matrix does not imply semantic sufficiency.

The covariance spectrum is nevertheless a useful starting point because it determines the conditioning of simple linear readouts. The next section derives that dependence for ridge regression and ordinary least squares. This gives a concrete reason to care about weak feature directions before considering more complicated nonlinear estimators.

**Figure 4.8. Predictive equivalence is defined by downstream futures, not visual similarity.**

Description: Two histories that differ only in nuisance appearance are grouped into one predictive-equivalence class when their target distributions agree. Histories that look similar but respond differently to the same action must remain distinguishable.

**Figure 4.9. Equal total variance can hide very different covariance spectra.**

Description: An isotropic embedding distributes variance equally across directions. An anisotropic embedding concentrates variance in a few directions and leaves weak directions with small eigenvalues. Both clouds can have the same covariance trace, but their downstream estimation problems are differently conditioned.