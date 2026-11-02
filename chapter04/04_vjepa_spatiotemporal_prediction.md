# 4. V-JEPA: extending prediction through time

An image describes a scene at one instant. It does not directly specify how that scene changes. A sequence adds temporal evidence: an object may move, a hand may make contact, or a tool may alter another object's state. V-JEPA extends feature prediction from a spatial grid to a spatiotemporal grid. It asks the model to predict selected video features from the remainder of the clip.

This is a more demanding representation-learning problem than image masking, but it is not automatically a causal or action-conditioned dynamics model. That distinction depends on which frames are visible and whether the predictor receives an explicit action.

## 4.1 Video tokens have spatial and temporal coordinates

Let a video clip contain frames \(X=(x_1,\ldots,x_T)\). A video encoder converts short spatiotemporal volumes, or tubelets, into tokens. Each token can be indexed by

\[
r=(t,h,w),
\]

where \(t\) is temporal position and \(h,w\) are spatial coordinates.

A context mask selects visible tokens \(C\), and a target mask selects positions \(T\). The online encoder processes the visible subset, while the target encoder processes the complete clip:

\[
Z_C=f_\theta(X_C),
\qquad
H_T=[f_{\bar\theta}(X)]_T.
\]

The predictor uses the context representation and the spatiotemporal target query to estimate \(\widehat H_T\). The loss compares this estimate with \(H_T\).

The additional time coordinate changes what can be predicted. A target at the same spatial position in a later frame may depend on motion. A target at a different position may depend on object displacement or interaction. A target extended over several frames may require the predictor to estimate a temporally coherent feature sequence.

## 4.2 Why the target is masked across time

Video contains strong redundancy. Nearby frames often show nearly the same background and similar object appearance. If a target patch is hidden at one frame but visible at adjacent frames, the model may infer it from a near-duplicate rather than learn a useful relation across the clip.

The original V-JEPA uses 3D multi-block masking. It samples spatially contiguous blocks and extends each selected spatial region through the temporal dimension. Its reported short-range configuration takes the union of eight target blocks with spatial scale \(0.15\); its long-range configuration takes the union of two blocks with spatial scale \(0.7\). Block aspect ratios are sampled between \(0.75\) and \(1.5\). The union of blocks produces an average masking ratio near \(90\%\) in the reported setup. These settings define the experiments in the paper; they are not universal constants of video JEPA.

The spatial mask is repeated across the clip's temporal dimension. The context encoder therefore cannot inspect the same spatial region in neighbouring frames. It must use other spatial regions and the remaining temporal evidence to predict the target tube.

Mask scale changes the inference problem. Smaller blocks can test whether local and moderately distributed context predicts withheld content. Larger blocks remove more of the scene and require broader evidence. Combining the two exposes the predictor to more than one spatial scale. It still does not force a strictly past-to-future prediction, because the visible context may include frames both before and after a target time.

## 4.3 The original objective predicts target features

The original V-JEPA predicts the target encoder's features, not RGB pixels. Its paper formulates the feature regression with an \(L_1\) loss:

\[
\mathcal L_{\mathrm{V\text{-}JEPA}}
=
\frac{1}{|T|D}
\sum_{r\in T}
\left\|
\widehat h_r-h_r
\right\|_1,
\]

where \(D\) is feature dimension and \(h_r\) is the target feature at spatiotemporal position \(r\). The target feature is supplied by the slowly updated target encoder and detached from the prediction-loss gradient.

This choice has a statistical interpretation. For a scalar target \(H\), the constant \(a\) minimizing \(\mathbb E[|H-a|\mid C]\) is a conditional median. Squared error instead selects the conditional mean. A coordinate-wise vector \(L_1\) loss therefore favours coordinate-wise conditional medians, with possible non-uniqueness. This does not mean V-JEPA explicitly models a multimodal probability distribution. It means the chosen pointwise loss has a different optimum from squared-error regression.

The target encoder is updated through an EMA of the online encoder, as in I-JEPA. The target branch is not updated by backpropagation through the feature loss. This creates a moving target that changes more slowly than the online context encoder. It is an optimization design, not a universal guarantee against collapse.

## 4.4 Temporal evidence is not the same as an action

A video model can exploit temporal regularities. If a hand approaches a cup, the next observed frames may often show a grasp. The model can learn that this visual pattern predicts a particular feature configuration. Yet it has not necessarily learned the effect of a chosen intervention.

The distinction is between two conditional distributions:

\[
p(Z_{t+1}\mid X_{1:t})
\qquad\text{and}\qquad
p(Z_{t+1}\mid X_{1:t},A_t).
\]

The first describes future features under the action distribution present in the observed trajectories. The second conditions on a specified action \(A_t\). If the action is omitted, the predictor cannot be asked to compare the consequences of two alternative commands while holding the history fixed.

Temporal order alone also does not guarantee that context is past-only. The original V-JEPA paper studies both full-clip multi-block masking and more restrictive causal-prefix variants. The standard multi-block objective can expose the context encoder to frames later than a selected target position. It is therefore not a causal next-frame objective. The paper's ablations show that the full-clip context setting can be effective for learning video representations; they do not turn it into an intervention model.

This distinction should be maintained even when an evaluation is called action anticipation. Predicting a likely human action label from a video is a semantic forecast. Predicting the physical consequences of a robot command requires the command itself to be an input and requires training data that connects actions to subsequent observations.

## 4.5 What V-JEPA demonstrates

The original V-JEPA results show that masked spatiotemporal feature prediction can produce useful frozen video representations. Evaluation on action recognition and video understanding tasks indicates that features learned from temporal context transfer to downstream tasks. The paper also studies mask design and the contribution of temporal information.

Those results support a limited but important conclusion: it is possible to learn video features by predicting withheld spatiotemporal representations without reconstructing every pixel. They do not demonstrate a complete dynamics model, a calibrated distribution over futures or successful planning under candidate actions.

A representation can be useful for recognizing motion while omitting variables needed for physical prediction. It can support classification without being Markov. It can predict features of observed clips without remaining accurate when its own predictions are fed back as inputs. Each stronger claim needs a different evaluation.

The progression from I-JEPA to V-JEPA therefore adds temporal evidence but not control. To predict what follows a chosen action, the action must enter the conditioning information. V-JEPA 2 makes that transition explicit by retaining a pretrained visual encoder and training a separate action-conditioned predictor on robot trajectories.

**Figure 4.5. A spatiotemporal mask with full-clip context is not inherently causal.**

Description: A spatial target region is withheld through several frames, while the context contains the remaining tokens from the clip. Some context may lie after a target time. A separate causal-prefix design would restrict context to earlier frames only.

**Figure 4.6. Masked video prediction and action-conditioned prediction answer different questions.**

Description: The masked-video path estimates withheld features from visible video context. The controlled path receives an explicit action alongside the current state and predicts a resulting feature or state. Changing the action while holding context fixed must be able to change the controlled prediction.