# 2. What exactly is being predicted?

The notation of a JEPA is compact, but the statistical problem is determined by the variables supplied to it. A method must say what the context contains, what the target contains, how the target location is specified and how target examples are sampled. If these choices are left implicit, two objectives that use the same encoder and loss can be mistaken for the same task even when they condition on different information.

## 2.1 The joint distribution defines the task

Let \(X\) denote the available observation, \(C\) the context-selection variable, \(T\) the target-selection variable and \(Q\) the query supplied to the predictor. Let \(Y_T\) denote the target-side observation at the selected positions. The data pipeline induces a joint distribution

\[
p_{\mathcal D}(X,C,T,Q,Y_T).
\]

The predictor does not learn an abstract relation between context and target in isolation. It learns under this distribution. A different context sampler changes what can be observed. A different target sampler changes what must be inferred. A different temporal stride changes the time interval separating context and target. A different query changes which part of the target representation is requested.

For example, predicting a small block adjacent to a visible block can be solved using local continuity. Predicting a distant object region may require information distributed across the scene. Predicting a feature from a later video frame requires temporal evidence. Predicting the consequence of a robot command requires the command itself to be part of the condition.

It is therefore more accurate to describe a JEPA as learning a conditional prediction problem than as simply learning to “predict missing patches”. The patches are one way to instantiate the variables.

## 2.2 The mask is part of the conditional distribution

Let \(M\) denote the sampled mask. It determines the visible input \(X_C\) and the target positions \(T\). The predictor is trained against a distribution of the form

\[
p(Z_T\mid X_C,Q_T,M).
\]

A mask is not a neutral deletion operation. It determines which evidence remains available and which distinctions the model must infer. If a target can be recovered from a tiny adjacent region, the model may rely on local texture. If the target is far from the visible region, it must combine more global context. If the target is hidden across several frames, the model must use temporal information that survives elsewhere in the clip.

There is no universally correct mask independent of the task. A mask that makes targets almost trivial can reward shortcuts. A mask that removes nearly all useful evidence can make the conditional target highly uncertain. The objective is shaped by the range of conditional problems sampled during training, not merely by the choice of neural architecture.

The query matters for the same reason. A target feature is indexed by a location. If the predictor is not told which location it should estimate, it must either produce all target positions or infer the requested position from another convention. Supplying explicit position information makes the prediction interface well-defined.

## 2.3 What does squared-error prediction return?

Suppose the target feature is a random vector \(H\), and the predictor must output one deterministic vector \(a\) after observing context \(X_C=x\) and query \(Q=q\). Under squared error, its ideal output solves

\[
a^*(x,q)
=
\arg\min_a
\mathbb E\!\left[
\|H-a\|_2^2\mid X_C=x,Q=q
\right].
\]

Let

\[
\mu(x,q)
=
\mathbb E[H\mid X_C=x,Q=q].
\]

Write \(H-a=(H-\mu)+(\mu-a)\). Expanding the squared norm gives

\[
\begin{aligned}
\mathbb E[\|H-a\|_2^2\mid x,q]
&=
\mathbb E[\|H-\mu\|_2^2\mid x,q]\\
&\quad+\|\mu-a\|_2^2\\
&\quad+2\,\mathbb E[(H-\mu)^\top(\mu-a)\mid x,q].
\end{aligned}
\]

The final term is zero because \(\mathbb E[H-\mu\mid x,q]=0\). The first term does not depend on \(a\). The minimum is therefore attained at

\[
a^*(x,q)=\mathbb E[H\mid X_C=x,Q=q].
\]

A deterministic squared-error predictor estimates the conditional mean of the target features. It does not necessarily select one of the possible target outcomes.

This matters when the conditional target is multimodal. Imagine a scene in which a ball may pass to the left or right of an obstacle. If the two outcomes are represented by distant feature vectors and occur with similar probability, their mean may lie between them. The mean can minimize squared error without corresponding to a plausible physical outcome. The predictor has averaged uncertainty rather than represented the alternatives.

For absolute error, the scalar minimizer is a conditional median. Applied coordinate-wise to a vector, an \(L_1\) loss produces coordinate-wise conditional medians, which may also combine components from different modes. A point-estimation loss therefore does not, by itself, represent a full distribution over possible futures.

## 2.4 When one prediction is not enough

If the task requires multiple alternatives, the model needs an output or inference mechanism capable of representing them. One option is to introduce a latent variable \(U\) that selects a hypothesis:

\[
\widehat Z_T=g_\phi(Z_C,Q_T,U).
\]

Another is to predict parameters of a conditional distribution. These extensions change the prediction problem: training must specify how latent hypotheses or distributional outputs are supervised and how uncertainty is evaluated.

A latent variable also introduces a possible shortcut. If it can carry the target feature directly, the predictor may ignore the context. The model must constrain how the latent is sampled or inferred, or otherwise prevent it from becoming an unrestricted channel for the answer. A latent variable should represent uncertainty that remains after conditioning on context, not leak the target into the predictor.

Not every JEPA requires a multimodal output. For a target that is nearly determined by context, a deterministic feature estimate can be adequate. The point is to match the output parameterization to the conditional uncertainty of the task. A low pointwise error on an averaged feature is not evidence that alternative outcomes have been represented.

## 2.5 The target branch is not an independent copy of the target crop

In I-JEPA and V-JEPA, the target encoder processes the complete image or video before target features are selected. The context encoder receives only visible tokens. This asymmetry is central: the target feature is contextualized by the target encoder's view of the full observation, while the predictor must estimate it from a restricted context.

The target is therefore not generally the output of encoding an isolated crop. Encoding a patch in isolation could discard its relation to the surrounding scene, while the full-image target feature can include global context. The target pathway defines what the predictor is being asked to approximate.

A second design choice is how the target parameters change. In the original I-JEPA and V-JEPA recipes, the target encoder is updated as an exponential moving average (EMA) of the online context encoder. The target tensor is detached during the prediction loss; the target parameters change through the EMA update rather than through backpropagation. These are two distinct operations. Stop-gradient specifies the derivative path. EMA specifies how the target network evolves between training steps.

Neither choice is a universal requirement for every method called JEPA. LeJEPA, for example, uses direct view agreement with an explicit distributional regularizer rather than the same teacher-predictor arrangement. What matters is that the optimization has a well-defined target and a reason the target cannot simply follow the predictor's current output.

## 2.6 A predictor is useful when the target contains withheld information

If two branches receive equivalent views and the objective only asks their embeddings to agree, a direct consistency loss may be sufficient. A predictor becomes important when the target contains information intentionally absent from the context, or when the target representation is defined by a distinct branch. The predictor must then map from what is available to what is requested.

This distinction prevents a common overgeneralization: a predictor is not a decorative network inserted into every self-supervised system. Its role follows from the information flow. If the target can be obtained from the context by a simple agreement relation, a separate predictor may be unnecessary. If the target depends on masked locations, later observations or an action that is not present in the context, the prediction interface must account for that missing condition.

**Figure 4.2. The mask and query define the conditional prediction problem.**

Description: A single observation is divided into visible context and withheld target positions. The query identifies the target locations, while the mask sampler determines which evidence remains available. Nearby, distant, temporal and action-conditioned targets correspond to different conditional distributions.

**Figure 4.3. A conditional mean can fall between plausible target modes.**

Description: Two separated clusters represent alternative target features compatible with the same context. A squared-error predictor returns their probability-weighted mean, which may lie between the clusters rather than on either plausible outcome.