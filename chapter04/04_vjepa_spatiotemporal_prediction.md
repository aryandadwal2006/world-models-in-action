# 4. V-JEPA: predicting spatiotemporal features from video

An image contains spatial relationships, but it does not directly reveal how those relationships change. A single frame may show an object at a particular position without specifying whether it is moving, accelerating, being pushed or resting. Video adds temporal evidence. The challenge is to define a prediction problem that makes this evidence useful without requiring the model to reconstruct every pixel of every frame.

V-JEPA extends feature prediction from a spatial grid to a spatiotemporal grid. Its target positions have both spatial and temporal coordinates. The model observes part of a video clip and predicts target features at selected positions. This still does not make the objective action-conditioned. The video supplies evidence about what tends to happen; the model is not necessarily told which intervention caused the observed transition. (Bardes et al., 2024)

## 4.1 A video is a volume of tokens

Let a clip contain \(T\) frames:

\[
X=(x_1,x_2,\ldots,x_T).
\]

A video tokenizer converts the clip into spatiotemporal tokens. A token location can be indexed by

\[
r=(t,h,w),
\]

where \(t\) denotes temporal position and \(h,w\) spatial position. In the original V-JEPA configuration, the video tokenizer uses tubelets spanning two consecutive frames and a \(16\times16\) spatial patch. A token is therefore a small spatiotemporal volume, not simply a patch from one independently processed frame.

The context encoder receives a selected subset of tokens. The target encoder computes target features for the clip, and the predictor estimates target features at selected spatiotemporal positions:

\[
Z_C=f_\theta(X_C,C),\qquad
H_Y=f_{\bar\theta}(X)_Y,
\]

\[
\widehat H_Y=g_\phi(Z_C,Q_Y),
\]

where \(C\) is the visible set and \(Q_Y\) describes the requested target positions. The query includes time as well as space. A model can be asked to predict a feature at a later time, at a different spatial location, or both.

As in I-JEPA, the target features come from the target encoder's output at selected positions. The target is not an RGB crop. The objective is to predict a learned feature that summarizes the target pathway's representation of the clip.

## 4.2 Why temporal masking is different from image masking

Adjacent video frames are highly redundant. An object may move only a few pixels between consecutive frames. If the model sees nearly all of a frame or nearby frames, it may infer the target from trivial local continuity. A task that is too easy can reward a shortcut rather than a representation that captures useful temporal relations.

V-JEPA samples spatially contiguous regions and extends them across the temporal dimension of the clip. Its target masks therefore select spatiotemporal regions rather than independent random holes in individual frames. The paper uses two families of target masks. A short-range mask is the union of eight randomly sampled target blocks, each covering 15% of a frame; a long-range mask is the union of two blocks, each covering 70% of a frame. The aspect ratio of each sampled block lies between 0.75 and 1.5. The unions produce an average masking ratio of approximately 90%. The values 15% and 70% describe the spatial scale of the sampled blocks, not the fraction of the complete video removed by the final union. Both mask families extend each selected spatial block through the clip's full temporal dimension. (Bardes et al., 2024)

The encoder paths also matter. The visible complement is applied at the input of the context encoder by dropping masked tokens. The target encoder processes the complete clip; target positions are selected from its contextualized output. Thus, the model does not create a target by encoding an isolated RGB crop.

The geometry of a target tube matters. If the same spatial region is hidden across time, the model cannot simply inspect the corresponding patch in another nearby frame. It must use the rest of the clip to predict the target features. If a target is withheld at only one time but visible at neighbouring times, temporal interpolation may be sufficient. Both are valid prediction problems, but they test different conditional information.

V-JEPA keeps the core I-JEPA insight and changes the coordinate system of prediction. A target is no longer described only by its location in an image. Its time coordinate is also part of the query. The training distribution can emphasize spatial context, temporal context or their interaction.

## 4.3 The feature target and the loss

Let \(H_Y\) be target features at selected spatiotemporal positions and \(\widehat H_Y\) the corresponding predictions. The original V-JEPA paper uses an \(L_1\) feature-prediction loss:

\[
\mathcal L_{\mathrm{V\text{-}JEPA}}
=
\frac{1}{|Y|d}
\sum_{r\in Y}
\left\|
\widehat h_r-h_r
\right\|_1.
\]

This differs from the squared-error form often used to introduce generic feature prediction. The discrepancy function changes the statistical estimator approached by an ideal deterministic predictor.

For a scalar random target \(H\), the constant \(a\) minimizing expected absolute deviation,

\[
a^*=\arg\min_a\mathbb E[|H-a|\mid C=c],
\]

is a conditional median of \(H\) given the context. The constant minimizing expected squared error is the conditional mean. In multiple dimensions, an \(L_1\) vector loss is typically applied coordinate-wise; its minimizers are coordinate-wise conditional medians, subject to possible non-uniqueness. This property should not be used to claim that V-JEPA explicitly models a full multimodal distribution. It describes the pointwise estimator favoured by the chosen loss under its assumptions.

The paper motivates its \(L_1\) choice in the context of feature prediction and target-update dynamics. The predictor estimates target features from the context representation, while the target encoder is updated as a moving average. The EMA target provides a slowly changing feature target while the online encoder and predictor are trained. This arrangement is specific to the original V-JEPA recipe.

## 4.4 Target updates and gradient paths

The original V-JEPA retains an online context encoder and an EMA target encoder. A stop-gradient is applied to target features in the prediction loss, so gradients from feature regression update the context encoder and predictor rather than directly updating the target branch. The target parameters change through the momentum update.

The training step can be described as follows:

1. Sample a clip and a spatiotemporal target mask.
2. Construct visible context tokens and target positions.
3. Encode visible tokens with the online context encoder.
4. Encode the clip with the target encoder and select target features.
5. Supply target positions as queries to the predictor.
6. Compute the \(L_1\) discrepancy between predicted and target features.
7. Update the online encoder and predictor by gradient descent.
8. Update the target encoder using EMA.

This sequence makes the information paths explicit. The target branch supplies a feature target, not a gradient path through the prediction loss. The EMA update is a separate parameter update. The phrase “stop-gradient teacher” can obscure the distinction between detaching the target tensor in the loss and changing target parameters through a moving average.

Neither mechanism by itself proves that collapse is impossible for every objective and architecture. The result depends on the complete training system, including mask construction, data, optimizer, normalization, target update and predictor.

## 4.5 What time contributes to the representation

Video feature prediction can make temporal relations useful to the training objective. If a target feature depends on motion, persistence or an interaction visible in context, a predictor that ignores those relations may incur larger error. The model can therefore learn features that support motion understanding and temporal anticipation.

But temporal evidence is not the same as physical state. A feature that predicts the next visible frame need not encode velocity explicitly. A feature that supports action anticipation need not encode force, contact or mass. A feature that predicts what usually follows an observed action may reflect correlations in the dataset rather than the effect of an intervention selected by the learner.

Consider a clip of a hand approaching a cup. The next observed frame may show the cup being grasped. A video predictor can learn that this visual pattern is often followed by a particular feature configuration. Without an explicit action variable and suitable intervention data, however, it does not necessarily learn what would happen if the hand moved away, approached from another direction or pushed the cup instead. Observed sequences mix environment dynamics with the policy that generated them.

The claim that video prediction can learn useful temporal structure follows from the objective. The stronger claim that video prediction alone teaches a causal model of how the world responds to actions does not.

## 4.6 Representation evaluation is not target prediction

A video encoder may produce a grid of feature vectors, but that does not mean every useful property is immediately available to a linear classifier. The original V-JEPA work evaluates frozen representations through downstream protocols, including video classification. One practical finding is that attentive pooling can improve substantially over simple average pooling for the unnormalized token features. A learnable query uses cross-attention to extract a task-specific representation from the feature grid.

This has a broader implication. The pretraining objective determines which relationships the model is optimized to preserve, but a downstream readout determines how those relationships are accessed. A linear probe tests whether a distinction is linearly accessible from frozen features. An attentive probe tests whether a small trainable attention mechanism can combine token features to expose the distinction. They are different tests.

The evaluation protocol must be stated alongside reported scores. A result obtained using an attentive head is not a pure measure of linear separability. A fine-tuned score is not a frozen-feature score. A feature-prediction loss is not a classification metric. These evidence types should not be mixed.

The original V-JEPA paper reports strong frozen-feature performance on video understanding benchmarks using large-scale video pretraining. These results establish that spatiotemporal feature prediction can yield useful visual representations under the tested protocols. They do not show that the model can plan under candidate actions or remain accurate over recursive rollouts.

## 4.7 What V-JEPA does not condition on

The phrase *future prediction* is sometimes used loosely in discussions of video models. A masked target can occur later than visible context, but temporal ordering is not enough to establish a controlled transition model.

The original paper explicitly compares three masking regimes: random tubes, causal multi-block masks, and multi-block masks sampled across the entire clip. In the causal variant, the context is restricted to the first six or twelve frames of a sixteen-frame clip before spatiotemporal blocks are removed. The standard multi-block setting, by contrast, can use context from the whole clip, including frames later than a target position. In the reported frozen evaluation, the full-clip multi-block setting outperforms the causal-prefix alternatives on Kinetics-400, Something-Something-v2 and ImageNet-1K. The result is important for interpretation: the principal V-JEPA recipe is a strong masked-video representation objective, not simply a causal next-frame predictor. Temporal ordering exists in the data, but the mask does not impose a past-only information boundary. (Bardes et al., 2024)

To be action-conditioned, the predictor must receive a variable representing the candidate action or intervention. A controlled latent transition might be written as

\[
\widehat s_{t+1}=g_\phi(s_{\leq t},a_t),
\]

where \(a_t\) identifies the action whose consequence is being predicted. If the action is absent, the model estimates what tends to occur under the trajectory distribution in its data. It may infer likely actions from visual motion, but that is not the same as conditioning on a chosen action.

This distinction matters when a planner compares alternatives. To evaluate actions \(a\) and \(a'\), the model must be able to produce different predictions when the action changes while the observed context is held fixed. An action-free model has no explicit input with which to represent this counterfactual difference.

The original V-JEPA is therefore a video representation-learning method, not the complete action-conditioned system used for physical planning. The next extension adds that interface as a separate post-training stage rather than simply renaming the original video encoder.

## 4.8 From video features to controlled dynamics

I-JEPA asks which features at hidden image locations can be predicted from visible spatial context. V-JEPA asks which features at hidden space-time locations can be predicted from visible video context. The model gains temporal evidence, and its target grid can express motion and interactions over a clip.

Neither method is required to distinguish outcomes under a specified action. That requires the model to receive actions, train on trajectories with aligned observations and actions, and be evaluated on action-conditioned predictions. The distinction is not terminological. It changes the conditional distribution the model must estimate.

V-JEPA 2 makes this separation explicit. It first scales action-free video pretraining, then adds a distinct action-conditioned predictor trained on robot trajectory data. V-JEPA 2.1 later strengthens dense feature prediction, but its robot-planning experiments again use a separately trained action-conditioned predictor. These stages will be treated separately rather than combined into one undifferentiated “V-JEPA family”.

**Figure 4.1. A video mask is a spatiotemporal selection.** Draw a short clip as a volume of frame patches. Highlight visible context tokens and a target tube spanning multiple frames. Label token coordinates with time and space, and show target positions entering the predictor as queries. Distinguish temporal masking from action conditioning.

**Figure 4.2. A masked feature target is not an action-conditioned future.** Use two panels with the same observed video context. The first predicts a withheld feature region without an action input. The second adds two candidate actions and shows that the conditional prediction problem now differs by action. Do not imply that the original V-JEPA includes the second pathway.

The next section examines what the target representation is allowed to forget, and how the geometry of the resulting features affects downstream estimation.
