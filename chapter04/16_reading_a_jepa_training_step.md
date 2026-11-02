# 16. Following one JEPA training step

The architecture becomes easier to understand when followed through one batch. Consider an I-JEPA training step with a batch of images. Each image is divided into patches, a context mask selects visible positions, and target masks select regions whose features must be predicted. The context encoder, target encoder and predictor do not receive identical inputs. The order of operations determines what information can flow through the loss.

## 16.1 From pixels to patch tokens

Let the image batch be

\[
X\in\mathbb R^{B\times 3\times H\times W},
\]

where \(B\) is batch size. A patch embedding layer maps each image to \(L\) tokens of dimension \(D_e\):

\[
X\longrightarrow P\in\mathbb R^{B\times L\times D_e}.
\]

Positional embeddings are added so the encoder can distinguish patch content from patch location. The resulting sequence is the input to the transformer encoder.

The exact number of tokens depends on image size and patch size. For a square image divided into \(p\times p\) patches, \(L=(H/p)(W/p)\). This notation describes the conceptual tensor; implementations may flatten, transpose or group dimensions differently.

## 16.2 Construct context and target positions

The mask sampler produces context positions \(C\) and one or more target blocks \(T_1,\ldots,T_M\). Conceptually, the context indices have shape

\[
C\in\mathbb N^{B\times L_C},
\]

and a target block has indices

\[
T_m\in\mathbb N^{B\times L_T}.
\]

The indices select tokens from the patch sequence. They do not themselves contain image information. Their job is to specify which tokens are visible and which target locations will be queried.

The online context encoder receives only the visible tokens:

\[
Z_C=f_\theta(P_C)
\in\mathbb R^{B\times L_C\times D_e}.
\]

The target encoder receives the full patch sequence:

\[
H=f_{\bar\theta}(P)
\in\mathbb R^{B\times L\times D_e}.
\]

The selected target features are

\[
H_{T_m}=[H]_{T_m}
\in\mathbb R^{B\times L_T\times D_e}.
\]

This difference is important. The context encoder cannot read target tokens, while the target encoder processes the full image before features are selected. Target features can therefore incorporate surrounding image context from the target branch.

## 16.3 The target branch and gradient path

The target encoder is evaluated without gradient tracking in the prediction step. Its output is detached before entering the loss. The prediction loss therefore updates the online context encoder and predictor, but not the target encoder.

The target parameters change after the online optimizer step through an exponential moving average:

\[
\bar\theta_{k+1}
=
m_k\bar\theta_k+(1-m_k)\theta_{k+1}.
\]

The target update is parameter-level. It is not a gradient from the target feature tensor. Keeping these operations separate avoids the misleading statement that the target encoder is trained by the same loss but with a smaller learning rate. In this recipe, the target branch follows the online encoder through its EMA rule.

## 16.4 The target position is a query

The predictor needs both the context features and the requested target locations. The target mask is converted into positional query embeddings. The predictor combines these queries with context features and produces one predicted feature per target position:

\[
\widehat H_{T_m}
=
g_\phi(Z_C,Q_{T_m})
\in\mathbb R^{B\times L_T\times D_e}.
\]

The query carries the location, not the answer. Supplying the target feature itself would defeat the prediction task.

The predictor may project context tokens and query embeddings into a different internal dimension. Its output is then projected back into the target feature dimension so it can be compared with \(H_{T_m}\). The tensor shape is a statement about the interface; internal attention blocks, normalization and projection layers determine how the mapping is computed.

## 16.5 Loss, optimizer and EMA order

The implementation computes a feature loss for each target block and averages across target positions and the batch. In the released I-JEPA training code, the target features are normalized and detached, and the prediction loss is implemented with smooth-\(L_1\) regression. The paper's concise mathematical formulation is written as an \(L_2\)-style feature distance. A reproduction should retain this distinction rather than treating equation and implementation as exactly identical.

A conceptual training step is:

1. Sample an image batch and context/target masks.
2. Embed patches and add positional information.
3. Run the context encoder on visible tokens.
4. Run the target encoder on the full image without gradients.
5. Select target features and build position queries.
6. Predict target features from context and queries.
7. Compute the feature discrepancy against detached targets.
8. Backpropagate through predictor and context encoder.
9. Update their parameters with the optimizer.
10. Update target parameters by EMA.

The order matters. The target features used by the current loss are produced before the EMA update. The target update then prepares a slowly changing target for a later step.

## 16.6 What changes for video

In V-JEPA, the image token index is replaced by a spatiotemporal token index \((t,h,w)\). Patch embedding becomes tubelet embedding. The mask sampler selects spatiotemporal blocks, and target queries include temporal positions as well as spatial positions. The target encoder processes the full clip; the context encoder processes the visible tokens. The same conceptual loss path remains: predict target features from context and queries.

The temporal target sampler does not automatically impose a causal boundary. Whether future frames are visible to the context encoder depends on the mask design. For an action-conditioned predictor, the inputs change more substantially: actions and relevant robot state enter the transition predictor, which is trained on aligned trajectories. That is not simply V-JEPA with a different mask.

## 16.7 What changes in predictor-free view agreement

LeJEPA illustrates a different use of the representation objective. Rather than predicting masked target tokens through an EMA target branch and a separate predictor, its base objective draws embeddings from multiple views toward the per-sample mean of global-view embeddings and regularizes the distribution of each view with SIGReg.

The computational graph and statistical objective are therefore different. It is reasonable to discuss both under the wider family of joint-embedding self-supervised methods, but the term predictor should be reserved for a network that actually estimates a target representation from a specified context and query. A direct view-agreement term does not have the same information flow as masked feature prediction.

## 16.8 What to inspect in an implementation

When reading code for a JEPA training step, trace the tensors rather than relying on module names. Identify which image or clip is passed to each encoder, where masks are applied, when target features are selected, how positional queries are built, which tensors are detached, and exactly which parameters the optimizer updates. Then locate the EMA update and verify when it occurs relative to the optimizer step.

This procedure can expose errors that a high-level block diagram hides. Applying target masks before the target encoder when the method requires full-image contextualized targets changes the task. Passing target features to the predictor leaks the answer. Updating target parameters through the loss instead of the intended EMA rule changes the gradient path. Treating a full-clip mask as causal changes the interpretation of the video objective.

A training implementation is therefore part of the scientific definition of the method. The paper explains the objective; the code specifies operational details that may be necessary for an exact reproduction. Neither should be substituted silently for the other.

**Figure 4.28. One I-JEPA step from patches to target update.**

Description: A batch is converted into patch tokens. The context encoder receives visible tokens; the target encoder receives the full image under no-gradient evaluation. Target positions become predictor queries. The loss updates the predictor and online encoder, then an EMA update advances the target encoder.