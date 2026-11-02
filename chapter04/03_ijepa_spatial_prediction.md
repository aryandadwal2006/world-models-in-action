# 3. I-JEPA: spatial prediction from an image

I-JEPA is a useful first concrete example because its prediction problem can be drawn on a single image. It asks whether features at withheld image locations can be estimated from a spatially distributed context. The method does not need an image decoder to reconstruct the missing pixels. It uses a target encoder to define the feature representation and a predictor to estimate selected target features from visible context.

The important part is not simply that some patches are hidden. The context and target masks are chosen to make the predictor use information that extends beyond the immediate neighbourhood of a target.

## 3.1 From an image to tokens

Let an image be divided into non-overlapping patches. A Vision Transformer maps these patches, together with positional embeddings, into a sequence of tokens. We can index the token at spatial location \(r\) by \(x_r\).

The context mask selects a set \(C\) of visible positions. The target mask selects one or more disjoint or partially overlapping regions \(T_1,\ldots,T_M\). The context encoder receives the visible tokens:

\[
Z_C=f_\theta(X_C).
\]

The target encoder receives the full image and produces a feature map:

\[
H=f_{\bar\theta}(X).
\]

The target features for block \(T_m\) are then selected:

\[
H_{T_m}=[H]_{T_m}.
\]

This order matters. The target encoder has processed the full image before the target features are read out. Its representation at a target location can therefore depend on surrounding content. The predictor is not asked to reconstruct an isolated patch representation; it is asked to estimate the feature produced by the full-image target pathway.

## 3.2 A context block and several target blocks

I-JEPA uses a large context region and multiple target blocks. The context is spatially distributed rather than being a small crop adjacent to every target. The target blocks are selected at locations that are not supplied as visible input to the context encoder.

The released ImageNet training configuration uses one context block and four target blocks. Its context scale is approximately \(0.85\) to \(1.0\) of the image area, while each target block is sampled at a smaller scale, approximately \(0.15\) to \(0.2\), with aspect ratios sampled between \(0.75\) and \(1.5\). These values describe that recipe, not a definition of I-JEPA. The method is the broader context-to-target feature-prediction construction; mask count and geometry are hyperparameters that alter the sampled prediction problem.

The multiple targets matter. A context may contain enough information to predict several spatially separated regions. Training against multiple blocks discourages the predictor from treating one adjacent boundary as the only useful source of evidence. It does not guarantee that the model learns object-level semantics, but it changes what evidence is useful for reducing the loss.

The context and target regions also need a clear overlap rule. In the standard disjoint setting, target positions are removed from the context input. If overlap is allowed, a target feature may be partly visible to the context encoder, making the prediction easier for reasons unrelated to the intended inference. The mask construction is therefore part of the experimental method, not an incidental preprocessing detail.

## 3.3 Target positions are queries

The predictor must know which target positions it is expected to estimate. I-JEPA supplies positional information for the target blocks. Conceptually, the predictor receives the context features and a set of target-position queries:

\[
\widehat H_{T_m}
=
g_\phi(Z_C,Q_{T_m}).
\]

The predictor is not given the target pixels or the target encoder's feature values. It receives the spatial coordinates of the requested features and must infer their values from context.

This separates two roles that can otherwise be confused. The mask says which input tokens are withheld. The query says where the target features are located. The predictor uses the query to place its estimates in the target feature map. If the positions were omitted, the model would not know which spatial features the loss compares.

The predictor in the released architecture is narrower than the main encoder. This is a computational choice, not the conceptual reason prediction works. The predictor provides a mapping from context features to target features; its width and depth determine capacity and cost.

## 3.4 The loss and the target update

At the level of the paper's objective, I-JEPA averages a feature-distance loss over the selected target positions and target blocks. The target branch is updated by an EMA of the online encoder. The exact training implementation includes feature normalization and uses smooth-\(L_1\) regression in the released training script; this is worth distinguishing from the paper's compact equation, which presents an \(L_2\)-style feature distance.

A simplified expression for the implementation is

\[
\mathcal L_{\mathrm{I\text{-}JEPA}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\operatorname{SmoothL1}
\left(
\widehat H_{T_m},
\operatorname{sg}(\operatorname{LN}(H_{T_m}))
\right),
\]

where \(\operatorname{LN}\) is feature-wise layer normalization and \(\operatorname{sg}\) denotes stop-gradient. The notation suppresses the average over batch and spatial positions.

The online context encoder and predictor are updated by backpropagation. The target encoder receives no gradient from this loss. Its parameters follow the online encoder through

\[
\bar\theta\leftarrow
m\bar\theta+(1-m)\theta,
\]

with a momentum coefficient \(m\) scheduled during training. This update makes the target evolve more slowly than the online model. It reduces the tendency for the target to move immediately in response to the predictor's current error, but it is not, by itself, a theorem that rules out every degenerate solution.

The distinction between paper equation and released implementation is not cosmetic. A mathematical description should state the objective being analysed; an implementation description should state the numerical operations actually used. When those differ, report both rather than silently treating one as the other.

## 3.5 Why spatial prediction can produce useful features

A target block is easier to predict when the context retains information related to its content. That information can include object identity, scene layout, shape, texture and spatial relationships. A large distributed context allows the predictor to combine evidence from more than the target's immediate neighbourhood.

The target encoder also changes what counts as a successful prediction. Because it processes the full image, the feature at one target location can summarize both local content and relationships to the rest of the scene. The predictor learns to estimate this contextualized feature from the visible context.

The method's evidence comes from downstream evaluations, not just from a decreasing pretraining loss. The I-JEPA paper evaluates frozen or lightly adapted representations on tasks that include image classification, low-shot classification, object counting and depth estimation. These tests show that spatial feature prediction can yield representations useful for several visual tasks. They do not show that the learned features form a physically sufficient state for control.

The target mask ablations are also informative. Large target regions and a sufficiently informative context performed better than alternatives in the reported evaluations. This supports the idea that mask geometry controls the information the model must use. It does not establish one universally optimal mask across datasets or downstream tasks.

## 3.6 What I-JEPA does not learn

I-JEPA has no temporal index in its prediction target. It does not have to distinguish two histories that lead to different motion, because the target and context belong to one image. It has no action variable, so it cannot answer what would happen under a selected robot command. It is not trained through recursive prediction of its own estimates, so its loss does not test long-horizon consistency.

These are boundaries of the chosen problem, not defects in the method. I-JEPA demonstrates that useful image representations can be learned by predicting target features from spatial context without reconstructing the image. To ask about motion, the target must extend through time. To ask about controlled consequences, the action must become part of the condition.

**Figure 4.4. I-JEPA predicts contextualized features at hidden image locations.**

Description: A large context region supplies visible patch tokens to the online encoder. The target encoder processes the full image, after which several target blocks are selected. Positional queries tell the predictor which target features to estimate; no target pixels or target feature values enter the predictor.