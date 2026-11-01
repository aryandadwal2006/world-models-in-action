# 12. Reading a JEPA training step

The equations of a predictive objective become easier to reason about when each variable is connected to an operation on tensors. This section traces one training step through an image-based JEPA. The purpose is not to prescribe a complete training recipe for every domain. It is to make the data paths, gradient paths and mask semantics explicit.

I-JEPA provides a useful example because its architecture separates a context encoder, a target encoder and a predictor, and because it predicts several target blocks from one context block. The same structure can then be compared with video feature prediction and with predictor-free multi-view matching.

## 12.1 From an image to patch representations

Let an image batch have shape

\[
X\in\mathbb R^{B\times C\times H\times W}.
\]

A patch embedding layer divides each image into \(N\) patches and maps every patch to a \(D\)-dimensional token:

\[
T=\operatorname{PatchEmbed}(X)
\in\mathbb R^{B\times N\times D}.
\]

The transformer encoder operates on these tokens together with positional information. Before any masking, the image can therefore be viewed as a grid of feature locations. A mask does not simply delete arbitrary values from an already-computed feature vector. It defines which image patches are supplied to the context pathway and which positions are requested as targets.

Let \(M_c\) be the context mask and \(M_1,\ldots,M_M\) the target masks. For an image-based task, these masks are spatial. In a video model, they become spatiotemporal and index locations across both position and time.

A basic shape check is useful. If an image is divided into a grid of \(h_p\times w_p\) patches, then \(N=h_pw_p\). A target block containing \(n_i\) patch locations should produce a target tensor with \(n_i\) tokens per image. The predictor must return exactly the same target locations and feature dimension. A model that predicts the right number of tokens in the wrong spatial order has not solved the intended task.

## 12.2 Constructing context and target blocks

I-JEPA samples one relatively large context block and several target blocks. The target blocks are sampled independently of the context. They can overlap the context before masking is resolved, so any overlapping target locations are removed from the context block. The remaining context is spatially distributed and informative, but does not reveal the target tokens directly.

The paper uses target blocks of a sufficiently large spatial scale to encourage semantic prediction rather than trivial local interpolation. It also uses a context block that covers most of the image while retaining some spatially distributed missing regions. These choices define the difficulty of the task. A very small target may be predictable from local texture. A target that is too large or too distant may be ambiguous even when the context contains all useful information.

The masks are therefore part of the model's inductive bias. They determine which conditional relationship is repeatedly presented during training.

For one image, the context encoder receives the visible context positions:

\[
T_c=T[M_c].
\]

The notation \(T[M]\) denotes selection of the tokens indexed by mask \(M\), not multiplication by a binary matrix. The context encoder processes only these visible tokens:

\[
S_c=f_\theta(T_c).
\]

If the context contains \(N_c\) patches, then

\[
S_c\in\mathbb R^{B\times N_c\times D}.
\]

The context pathway does not need to run attention over the missing patches. This is one reason representation prediction can be less computationally expensive than a full-resolution reconstruction objective.

## 12.3 The target encoder and target selection

The target branch defines the feature vectors that the predictor must match. It is not enough to say that the target branch “encodes the target crop.” In the official I-JEPA implementation, the target encoder processes the full image and produces a spatial feature map. The target positions are then selected from this feature map. The target encoder can therefore use image context outside an individual target block when constructing its reference features. This is a specific information path, not an implementation detail that can be omitted from an architecture diagram. (Assran et al., 2023; official I-JEPA implementation)

Write the target encoder as

\[
S_t=f_{\bar\theta}(T)
\in\mathbb R^{B\times N\times D}.
\]

For target block \(i\), select its target locations:

\[
S_t^{(i)}=S_t[M_i]
\in\mathbb R^{B\times N_i\times D}.
\]

The target encoder parameters \(\bar\theta\) do not receive the prediction-loss gradient. They are updated by an exponential moving average of the online/context encoder:

\[
\bar\theta_{t+1}
=
m_t\bar\theta_t+(1-m_t)\theta_{t+1}.
\]

The target branch is thus a slowly moving reference. It is neither a fixed pretrained network nor an independent network optimized by the same gradient as the context encoder.

The implementation may apply a normalization layer to the target features before or after selecting the requested positions, depending on the exact method and code version. The order matters because normalization statistics can couple tokens or dimensions. When describing a particular implementation, state the actual order rather than drawing a generic target crop through the encoder.

## 12.4 A target position is a query

The predictor cannot infer which output locations are requested from the context features alone. It receives a query describing the target positions. I-JEPA constructs this query with learned mask tokens and positional embeddings.

For target block \(i\), let

\[
Q_i=\{q_j:j\in M_i\}
\]

be the sequence of query tokens. Each token contains information about the position it represents. The predictor receives both the context features and the query:

\[
\widehat S_t^{(i)}
=
g_\phi(S_c,Q_i).
\]

Its output has the same shape as the selected target:

\[
\widehat S_t^{(i)}
\in\mathbb R^{B\times N_i\times D}.
\]

This is not an ordinary decoder that reconstructs pixels. The query asks for features at specified positions, and the predictor returns vectors in the target encoder's feature space.

Positional information is essential. Without it, the predictor would be asked to produce a set of target features without knowing which location each feature belongs to. A permutation-invariant set of outputs would not preserve the intended spatial correspondence. The target mask is therefore both a sampling device and a query specification.

## 12.5 The loss and its gradient paths

For \(M\) target blocks, I-JEPA averages a squared feature discrepancy over the requested target positions:

\[
\mathcal L_{\mathrm{I\text{-}JEPA}}
=
\frac{1}{M}
\sum_{i=1}^{M}
D\!\left(
\widehat S_t^{(i)},S_t^{(i)}
\right),
\]

where \(D\) sums or averages the squared Euclidean differences over the selected patch features according to the implementation's normalization convention.

The gradient flows through the predictor and the context encoder. It does not flow through the target encoder. The target encoder follows the online encoder through the EMA update after the optimizer step.

The training step can be represented schematically as the following pseudocode:

    target_map = target_encoder(patchify(image_batch))
    context = context_encoder(select(patchify(image_batch), context_mask))

    loss = 0
    for mask in target_masks:
        query = mask_tokens_with_positions(mask)
        prediction = predictor(context, query)
        target = select(target_map, mask)
        loss = loss + feature_distance(prediction, target)

    loss = loss / number_of_target_blocks
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_target_encoder_ema()

This pseudocode omits batching details and the implementation's exact normalization, but it makes the computational graph explicit. In particular, the target map is computed from the image rather than from a crop that contains only the requested target block. The predictor receives positional queries, and the EMA update is separate from ordinary gradient descent.

A figure showing only an encoder, a predictor and a target encoder hides the most important part of the method. The reader should be able to trace which pixels reach each branch, which positions are selected, where positional queries enter, which tensors are compared and which parameters receive gradients.

## 12.6 What changes for video

V-JEPA retains the feature-prediction principle but replaces spatial image patches with spatiotemporal tubelets or tokens. A video is patchified across time, height and width. Its context mask can remove tubes or spatiotemporal blocks, and its target can include positions distributed across the clip.

The original V-JEPA objective uses an \(\ell_1\) feature regression loss and an EMA target encoder with stop-gradient. This differs from the squared \(\ell_2\) discrepancy used in the I-JEPA formulation. The predictor is conditioned on the requested target locations, and the masking strategy determines whether the target is a spatially hidden region, a temporally withheld region or a combination. (Bardes et al., 2024)

The original V-JEPA paper studies several masking regimes, including random tubes, causal multi-block masks that restrict context to a video prefix, and multi-block masks that can draw context and targets across the clip. (Bardes et al., 2024) The causal-prefix setup asks a different question from a non-causal mask over the entire video. The latter can use visible frames both before and after a target time, so it is not equivalent to predicting a future from the past. A reader should never infer temporal causality from the fact that a model was trained on video.

V-JEPA 2 retains masked feature prediction for visual pretraining and then introduces a separate action-conditioned post-training stage. That stage changes the input information and the transition problem, as discussed in the following sections. (Assran et al., 2025)

## 12.7 What changes for predictor-free view matching

LeJEPA illustrates a different case. It constructs global and local views of an observation and moves their embeddings toward the mean of the global-view embeddings. The mean acts as a shared centre. SIGReg regularizes the distribution of the embeddings.

There is no separate predictor that receives a spatial or temporal query and produces a target block. The training views are intended to represent related content, and the method uses direct view agreement. The distributional regularizer constrains the embedding batch while the prediction term defines which views should be close. (Balestriero and LeCun, 2025)

This is not an exception to the JEPA principle. It demonstrates why the predictor should be tied to the information structure of the task. When a target requires a location-specific or action-conditioned prediction, a predictor is useful. When the task is symmetric agreement across related views, a direct loss may suffice.

## 12.8 An implementation audit

Before trusting a feature-prediction training run, check the data and gradient paths rather than looking only at the scalar loss.

**Mask leakage.** Verify that target positions are not passed to the context encoder as visible inputs. If target blocks overlap the sampled context, resolve the overlap before constructing the context sequence.

**Target correspondence.** Check that predicted token \(j\) is compared with the target feature at the same requested spatial or spatiotemporal position. A mismatch in flattening order can create a numerically valid but semantically incorrect objective.

**Query information.** Confirm that the predictor receives a position-bearing query for every requested target location. A mask token without its location cannot specify which feature is to be predicted.

**Gradient routing.** Confirm that the target branch is detached from the prediction loss when the method calls for an EMA target, and that the EMA update is performed after the online optimizer step. An accidental gradient through the target branch changes the objective.

**Target updates.** Log the online and target parameter distance. A target that is not updated, or is copied directly from the online encoder, does not implement the intended moving-reference dynamics.

**Feature normalization.** Record where normalization is applied and whether it is per token, per feature dimension or over the batch. Changing the location of a normalization layer can change both the target geometry and the meaning of the loss.

**Prediction scale.** Report whether the feature discrepancy is summed or averaged over patches, dimensions, blocks and batch items. The numerical value of the loss changes with these conventions, even when the underlying tensor differences are identical.

**Temporal semantics.** For video, record whether context is causal, whether the target lies in the future or anywhere in the clip, and how the temporal stride affects the prediction horizon. A random spatiotemporal mask is not automatically a future-prediction task.

**Action alignment.** For a controlled predictor, verify that every action is paired with the state transition it caused or was intended to cause. A one-step shift in action indexing changes the conditional prediction problem.

A low loss is meaningful only after these checks. Otherwise, the model may be solving a different task from the one the architecture diagram suggests.

The worked example provides the operational interpretation of the central equations: a JEPA predicts features at requested locations in a target representation space. The mask distribution defines what is withheld, the target encoder defines what counts as a correct feature, the predictor maps context to the query, and the gradient path determines which parameters learn from the discrepancy. The next section returns to the complete argument and the limits of what such a training objective establishes.
