# 2. The prediction problem in representation space

The expression “predict a representation” is compact, but it hides several modelling decisions. A method must specify what information is available, what is withheld, what the target encoder observes, and which variable tells the predictor what it is expected to estimate. These decisions define the actual statistical problem. The loss function alone does not.

## 2.1 Context, target and query

Let \(X\) denote the information available to the model and \(Y\) the target information it is asked to predict. The variables need not be individual frames. \(X\) may be a crop, a set of visible image patches, a video prefix or a history of observations. \(Y\) may be a hidden image block, selected video tokens, a later observation, or a representation from another modality.

A query \(Q\) specifies which part of the target is requested. For an image, it may contain spatial positions. For video, it may contain spatial and temporal positions. For an action-conditioned model, the conditioning information must also include the action or action sequence. The query is part of the problem definition, not just an implementation convenience.

A general representation-prediction system can be written as

\[
S_X=f_\theta(X),\qquad
S_Y=f_{\bar\theta}(Y),\qquad
\widehat S_Y=g_\phi(S_X,Q).
\]

The training objective is an expectation over the joint distribution of context, target and query:

\[
\mathcal L_{\mathrm{pred}}(\theta,\bar\theta,\phi)
=
\mathbb E_{(X,Y,Q)\sim p_{\mathrm{data}}}
\left[
d\!\left(
g_\phi(f_\theta(X),Q),
f_{\bar\theta}(Y)
\right)
\right].
\]

The distribution \(p_{\mathrm{data}}(X,Y,Q)\) is as important as the function \(d\). It determines which contexts occur, which targets are sampled, and which information the model must use to make a prediction. Changing the mask sampler changes the conditional distribution the predictor faces. Changing temporal stride changes the prediction horizon. Changing the action distribution changes the interventions represented in the training data.

## 2.2 A feature-space target is not the same as a cropped input

The context and target branches do not necessarily receive symmetric inputs. In I-JEPA, for example, the context encoder processes the selected context tokens, while the target encoder processes the full image before the target positions are selected from its feature map. The model therefore has access to a target feature at a location that is hidden from the context branch. It does not simply encode an isolated target crop through an otherwise identical pathway.

This detail matters because a target encoder defines the features being predicted. The representation may contain spatial context from outside the target block. The target is not equivalent to an independently encoded patch unless the architecture explicitly makes it so. (Assran et al., 2023)

The same caution applies to video. A visible-context encoder and an unmasked target encoder create different information paths. The predictor is asked to infer target features from the context pathway, not to copy the target encoder's input. Positional information tells it where the target lies in the image or clip. The masking pattern determines whether the prediction requires local continuity, global scene context, temporal context or some combination.

A figure of a JEPA should therefore show the information flow, not just three boxes labelled encoder, predictor and target. In particular, it should make clear which pixels or tokens each branch receives and where target positions are selected.

## 2.3 The mask defines a conditional distribution

The target sampler determines what is unknown. Suppose \(M\) denotes a random mask and \(Y_M\) the selected target under that mask. The predictor is trained against a family of conditional distributions

\[
p(Y_M\mid X_{\neg M},M),
\]

where \(X_{\neg M}\) denotes the visible context. A mask selecting a small region near visible content may make the target predictable from local texture or continuity. A large, distant target may require broader context. A target spanning later times may require temporal information. If the context removes a variable essential to the target, uncertainty remains even for an ideal predictor.

There is no mask geometry that is optimal independently of the intended prediction problem. A method that makes targets too easy can reward shortcuts. A method that makes targets too ambiguous can produce weak or noisy learning signals. The target distribution is part of the inductive bias.

This is why image, video and dense-feature objectives should not be described as though they were the same prediction task at different resolutions. I-JEPA's spatial blocks test relations between visible and withheld image regions. The original V-JEPA uses masked spatiotemporal features, which can be spatially or temporally withheld. V-JEPA 2.1 changes the target supervision again by including dense features from visible as well as masked tokens and by applying supervision at intermediate depths. That choice asks the representation to preserve spatially structured information across the token grid. It is not the same as adding an action input. (Bardes et al., 2024; Assran et al., 2025; Mur-Labadia et al., 2026)

## 2.4 The deterministic squared-error predictor

A simple predictor minimizes squared distance in the target representation space. For a fixed context \(X=x\) and query \(Q=q\), let the random target feature be \(H=f_{\bar\theta}(Y)\). The optimal deterministic prediction is

\[
a^*(x,q)
=
\arg\min_a
\mathbb E\!\left[
\|H-a\|_2^2\mid X=x,Q=q
\right].
\]

Assume the conditional second moment exists and define

\[
\mu(x,q)=\mathbb E[H\mid X=x,Q=q].
\]

Add and subtract \(\mu(x,q)\):

\[
\begin{aligned}
\mathbb E[\|H-a\|_2^2\mid x,q]
&=
\mathbb E[\|H-\mu+\mu-a\|_2^2\mid x,q]\\
&=
\mathbb E[\|H-\mu\|_2^2\mid x,q]
+\|\mu-a\|_2^2\\
&\quad+
2\,\mathbb E[(H-\mu)^\top(\mu-a)\mid x,q].
\end{aligned}
\]

The cross term is zero because \(\mathbb E[H-\mu\mid x,q]=0\). Hence

\[
\mathbb E[\|H-a\|_2^2\mid x,q]
=
\mathbb E[\|H-\mu\|_2^2\mid x,q]
+\|a-\mu\|_2^2.
\]

The first term does not depend on \(a\), so the unique minimizer is

\[
a^*(x,q)=\mathbb E[H\mid X=x,Q=q].
\]

This is a property of squared-error regression. It is not a pathology unique to JEPA.

## 2.5 What happens when the target is multimodal?

A conditional mean can fail to correspond to any single plausible outcome. Consider a one-dimensional target with two equally likely values, \(-1\) and \(+1\), given the same context. The conditional mean is zero, even though zero is never observed. A deterministic squared-error predictor returns the point between the two modes.

In an image or video representation, the same phenomenon can arise when one context is compatible with distinct futures. A single predicted feature may average across alternatives. Whether this is harmful depends on the geometry of the target space and the downstream task. For a representation-learning objective, a mean feature may still be useful. For a planner that must distinguish mutually exclusive outcomes, a single averaged future can be misleading.

A system that needs multiple alternatives must represent them explicitly, for example through a latent variable, a distribution over future features, an ensemble or multiple hypotheses. Merely naming the model a world model does not supply uncertainty. The conditional distribution and the prediction objective must support it.

This observation will matter when introducing the latent variable \(z\) in energy-based prediction. It can index alternative compatible futures, but it must not become an unconstrained channel through which the target is copied.

## 2.6 When is a predictor necessary?

A predictor is useful when the target contains information that is not directly available in the context, or when the model must transform context information into a target-specific representation. A future feature conditioned on a history and an action is an obvious example. A masked region whose location is specified by a query is another.

However, a separate predictor network is not a universal requirement of joint-embedding learning. For symmetric views that are intended to represent the same content, a loss can directly bring the view embeddings toward a shared centre. LeJEPA uses a multi-view construction in which a mean of global-view embeddings serves as the centre and local and global view embeddings are drawn toward it. Its basic recipe does not require the EMA teacher and predictor arrangement used by I-JEPA or V-JEPA. (Balestriero and LeCun, 2025)

The choice depends on information asymmetry:

- If the task is to make two views of the same underlying sample agree, direct view agreement may be appropriate.
- If the target is a hidden part specified by a location or query, a predictor can map context features to that target.
- If the target is a future that depends on a selected action, the predictor must receive the action and the model must be trained on aligned action-observation transitions.
- If the target is linguistic, the predictor maps visual context to a text representation; that is a cross-modal prediction problem, not a physical dynamics model.

The predictor should be justified by the conditional problem, not included merely because a previous JEPA implementation has one.

### A linguistic target: VL-JEPA

The target representation need not be visual. In VL-JEPA, a visual context and a text query are used to predict an embedding of a target text response. The primary predictive objective operates in the continuous text-embedding space rather than requiring autoregressive token generation at every training step. A language decoder can be applied when an explicit text answer is required.

This is another instance of the same conditional structure: the visual context and query define what is available, and the text embedding defines the target. It also makes the limit of the abstraction clear. Predicting a linguistic representation from an image is cross-modal prediction; it is not, by itself, an action-conditioned model of physical dynamics.

## 2.7 Target updates are a separate design choice

The notation \(f_{\bar\theta}\) does not imply a particular update rule. Several possibilities exist:

1. An EMA target encoder, updated from the trainable encoder after each optimization step.
2. A target branch that receives no gradient but is updated by another rule.
3. A frozen encoder used as a fixed feature target.
4. Shared or symmetric branches for multi-view objectives.
5. A predictor-free objective in which the view embeddings are compared directly.

For an EMA target,

\[
\bar\theta_{t+1}
=
m_t\bar\theta_t+(1-m_t)\theta_{t+1}.
\]

This update makes the target a smoothed history of the online parameters. It does not define the meaning of JEPA and is not a general proof that every collapse solution is impossible. The target-update rule, the loss, the mask sampler and the architecture form a method-specific training system.

It is important not to conflate the EMA used to update a target encoder with the stop-gradient applied to a tensor in a regularizer. Both affect gradient flow, but they serve different purposes. VISReg, for example, detaches the scale statistic in the shape normalization to separate scale and shape gradients. That is not the same mechanism as using an EMA teacher.

## 2.8 What a low prediction loss establishes

A low value of \(\mathcal L_{\mathrm{pred}}\) establishes that the predictor matches the selected target representation under the training distribution, the chosen mask or query sampler, the normalization and the optimization procedure. It does not establish, without additional evidence, that:

- the target representation preserves every variable relevant to a downstream task;
- the representation is a sufficient state for physical prediction;
- action effects are identified from passive observations;
- predictions remain accurate under recursive rollout;
- the model captures multiple futures or calibrated uncertainty;
- a planner will find desirable or safe actions;
- the result generalizes to unobserved action-state combinations.

The next section examines the simplest failure permitted by an unconstrained agreement objective: every input can receive the same representation and achieve zero prediction error.
