# JEPA prediction, dense targets and action-conditioned world models: from first principles

This note connects image and video representation learning with controlled prediction. It separates four questions: how an encoder represents a view, how a predictor forecasts another latent, how targets are trained without pixel reconstruction, and whether the learned state predicts outcomes under selected actions. It covers I-JEPA, V-JEPA, V-JEPA 2, the dense predictive extension V-JEPA 2.1, and action-conditioned examples in EB-JEPA. These are related architectures and research lines, not one identical loss with interchangeable details.

## 1. The basic joint-embedding predictive problem

Let an observation be \(x\), and let a masking procedure split it into context positions \(C\) and target positions \(Y\). For video, positions index space and time; for images, they index spatial patches. A context encoder maps only visible/context input to embeddings:
\[
z_C=f_\theta(x_C,C).
\]
A target encoder computes target representations for the original observation:
\[
h_Y=f_{\bar\theta}(x)_Y.
\]
A predictor receives context embeddings, their positions, and the locations to predict:
\[
\hat h_Y=g_\phi(z_C,C,Y).
\]
A generic squared-error prediction objective is
\[
L_{\mathrm{pred}}(\theta,\phi;\bar\theta)
=\frac{1}{|Y|D}\sum_{r\in Y}
\left\|\hat h_r-\operatorname{sg}(h_r)\right\|_2^2,
\]
where \(D\) is target feature dimension and \(\operatorname{sg}\) denotes stop-gradient. A particular paper or implementation can use a different norm, target normalization, reduction, layer weighting, masking protocol or extra regularizer. Those details are part of the method.

The word **joint-embedding** indicates that context and target are compared in a learned feature space. The word **predictive** indicates that the model predicts target embeddings from context rather than reproducing target pixels. Neither word implies that predicted features constitute the full physical state.

### 1.1 Why distinguish context encoder, target encoder and predictor?

- The **context encoder** learns features from the visible input. Its input is restricted by the context mask; it cannot directly read the target tokens from that same encoder pass.
- The **target encoder** supplies target embeddings for the original observation or clip. In standard JEPA recipes its gradients are stopped and its parameters are updated by an EMA of the context encoder.
- The **predictor** uses context representations and target-position information to infer the target embedding. It is not a pixel decoder; it can be a transformer or another spatial/sequence predictor.

The target branch ordinarily processes the full or less-corrupted observation, so it can calculate features at target positions. The target positions are selected afterwards. The context branch sees the restricted input and predicts the selected target. This asymmetry creates the learning task.

### 1.2 Target encoder and exponential moving average

Let online encoder parameters after a gradient update be \(\theta_{t+1}\) and target encoder parameters be \(\bar\theta_t\). The target update is
\[
\bar\theta_{t+1}=m_t\bar\theta_t+(1-m_t)\theta_{t+1},
\qquad 0\leq m_t<1.
\]
For constant \(m\), target parameters are a weighted history of online parameters, with exponentially decreasing weight on older values. A schedule increasing \(m_t\) toward one makes the target evolve more slowly later in training. The target is not updated by backpropagation through the target prediction loss.

EMA adds temporal asymmetry but is not a proof of anti-collapse. A constant representation can make predictions easy; whether the system escapes that solution depends on architecture, target normalization, data variation, masking and any explicit regularizers. A JEPA description that mentions only EMA while ignoring collapse diagnostics and the full recipe is incomplete.

### 1.3 What does regression optimize?

For a fixed context \(c\) and stochastic target \(H\), squared-error prediction minimizes
\[
\mathbb E[\|H-a\|_2^2\mid C=c].
\]
Its minimizer is the conditional mean \(a^*(c)=\mathbb E[H\mid C=c]\). If distinct target states are plausible from the same context, a deterministic squared-error predictor can average them in latent space. The average need not match any one target exactly.

Elementwise absolute-error loss yields a conditional median in a scalar coordinate under regularity conditions. These facts explain why the target and loss matter; they do not imply L1 or L2 is always better. For representation learning, target embeddings may already encode invariances and semantic structure. For dynamics, an ambiguous future can require a probabilistic or action-conditioned predictor.

A low training loss establishes prediction of the chosen representation under the training mask and data distribution. It does not establish good pixel generation, accurate out-of-distribution transitions or Markov sufficiency.

## 2. I-JEPA: predict spatial target blocks in an image

I-JEPA (Assran et al., 2023) uses a context block from an image to predict representations of other target blocks. Let an image \(x\) be decomposed into patch tokens. The context mask selects a visible region \(C\); multiple target masks select blocks \(Y_1,\ldots,Y_M\). The context encoder processes the context input, while the target encoder computes features at target positions on the original image. A predictor combines the context output with positional information for each target block.

A generic objective is
\[
L_{\mathrm{I\mbox{-}JEPA}}
=\frac1M\sum_{j=1}^{M}
\frac1{|Y_j|D}\sum_{r\in Y_j}
\left\|g_\phi(z_C,C,Y_j)_r
-\operatorname{sg}(f_{\bar\theta}(x)_r)\right\|_2^2.
\]
The exact target normalization and loss reduction should follow the paper version and code. The defining structure is prediction of multiple target regions' embeddings from a context region, not RGB reconstruction.

### 2.1 Why multiple target blocks?

One context may be compatible with several target regions. Predicting multiple blocks asks the model to retain content predictable from the context while avoiding the shortcut of reading target tokens from an unrestricted context path. Multi-block prediction also makes target coverage and the distribution of context-target distances explicit design choices.

Not every target block is equally predictable. A target may be occluded or ambiguous from the context. JEPA trains on the sampled target distribution, so performance depends on target construction and the invariances encoded in the data.

### 2.2 Why not reconstruct pixels?

A pixel loss assigns a target to every reconstructed RGB value. A JEPA loss assigns a target to the encoder's representation. If the target encoder maps together changes deemed irrelevant, a predictor can succeed without reproducing those changes. That can reduce pressure to model pixel-level texture. It can also discard information that is irrelevant to the pretraining target but essential downstream.

The safe interpretation is: JEPA learns a representation that supports the specified latent prediction task. It does not follow that the latent is semantic, minimal, physically complete or sufficient for planning.

**Primary sources:** [I-JEPA paper](https://arxiv.org/abs/2301.08243); [official I-JEPA repository at pinned revision](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048).

## 3. V-JEPA: predict space-time target features from video context

V-JEPA applies the same broad mechanism to video. A clip has temporal as well as spatial patch positions, so context and target masks occupy a three-dimensional space-time token grid. A context encoder processes visible/context tokens. A target encoder processes the corresponding clip to supply feature targets at selected space-time positions. A predictor uses context features and positions to forecast those targets.

Writing \(X=(x_1,\ldots,x_T)\) for a clip and \(r=(t,h,w)\) for a patch's temporal and spatial coordinates, a generic loss is
\[
L_{\mathrm{V\mbox{-}JEPA}}
=\frac1{|Y|D}\sum_{r\in Y}
\left\|\hat h_r-
\operatorname{sg}\big(f_{\bar\theta}(X)_r\big)\right\|_2^2.
\]
Temporal masks change the target from an arbitrary image region to a region of a trajectory. The model can learn features predictive of motion, interaction and temporal continuity to the extent those variables affect the sampled target embeddings and are inferable from visible context.

### 3.1 Space-time masking is part of the learning problem

The mask generator determines whether the target is spatially local, temporally extended, spatially extended or a combination. The mask changes which target features are predictable from context. A model trained to predict nearby patches may exploit spatial redundancy; a model trained to predict future time blocks must rely more on temporal regularities. Neither regime contains the same information as the other.

Training protocol also determines whether the target encoder sees a full clip, a transformed crop or a temporally sampled sequence. These choices should be recorded alongside the loss. The name V-JEPA alone does not identify an exact mask distribution, target normalization, architecture or schedule.

### 3.2 Passive video prediction is not action-conditioned dynamics

If an encoder/predictor sees video but no control input, it learns relationships within the distribution of observed trajectories. It does not explicitly learn which future follows a chosen action. Correlations between states and actions can be learned only when the objective includes appropriate action information; even then, counterfactual validity under actions rare or absent from training needs separate testing.

A representation may encode motion cues and still fail to predict outcomes after an intervention. Therefore distinguish passive latent prediction, action-conditioned transition prediction, and planning with those transitions.

**Primary sources:** [V-JEPA paper, Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471); [pinned official JEPA repository](https://github.com/facebookresearch/jepa/tree/51c59d518fc63c08464af6de585f78ac0c7ed4d5).

## 4. V-JEPA 2: representation pretraining and action-conditioned post-training

V-JEPA 2 is a video representation/prediction line trained on large video collections. Its standard pretraining objective predicts target latent video features from context rather than reconstructing the full video at pixel level. The official repository also contains V-JEPA 2-AC, an action-conditioned post-training stage using robot interaction trajectories. These should not be collapsed into one claim that ordinary passive video pretraining already gives an action-conditioned world model.

A controlled latent transition can be written as
\[
z_t=f_\theta(o_{\leq t}),\qquad
\hat z_{t+1}=g_\phi(z_t,a_t),
\]
with training target
\[
z_{t+1}^{\mathrm{target}}
=\operatorname{sg}(f_{\bar\theta}(o_{\leq t+1}))
\]
or the target representation at the exact positions/times used by the source recipe. The action \(a_t\) enters the predictor, allowing predictions to be conditioned on control. Training still constrains this mapping only over the observed state-action distribution and chosen representation target.

For goal-conditioned planning, a separate decision rule can search over action sequences:
\[
a_{0:H}^{*}=\arg\min_{a_{0:H}}
\sum_{t=0}^{H} c(\hat z_t,z_{\mathrm{goal}}),
\qquad \hat z_{t+1}=g_\phi(\hat z_t,a_t).
\]
This explains the interface between a latent world model and a planner; it is not a universal V-JEPA 2 planning cost or optimization algorithm. A planning result must be evaluated for model exploitation and real-environment robustness, not inferred from representation pretraining loss.

**Sources:** [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985); [official V-JEPA 2 repository, including V-JEPA 2-AC and 2.1](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812).

## 5. V-JEPA 2.1: dense prediction and deep self-supervision

The pinned V-JEPA 2 repository includes V-JEPA 2.1 code. The release description states that the recipe adds dense predictive loss, deep self-supervision at intermediate encoder representations, multimodal image/video tokenizers, and model/data scaling. The paper is [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482).

### 5.1 Dense predictive loss includes context tokens

The code's dense loss allows both masked/predicted target tokens and context/visible tokens to contribute. With the pinned training configuration's **predict_all: true**, the loss includes normal predicted target outputs as well as a context loss. The latter compares predictor outputs for context positions against corresponding target-encoder features and uses the configured context-loss weight.

The relevant code path is [app/vjepa_2_1/train.py](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py). The implementation normalizes target features with LayerNorm before computing losses. When multi-level prediction is enabled, it normalizes separate intermediate feature groups before concatenating them into the predictor target.

For a generic loss exponent \(p>0\), define
\[
\ell_p(\hat h,h)=\frac1p\,\operatorname{mean}(|\hat h-h|^p).
\]
In the pinned V-JEPA 2.1 training config, **loss_exp: 1.0**, so the code's primary term is an elementwise absolute-error loss (L1-type), not squared error. Changing the exponent changes the loss family. Exact reductions and which tokens enter each term are part of the version-specific recipe.

### 5.2 Context loss and distance weights

The inspected V-JEPA 2.1 config enables **predict_all** and **weight_distance_loss**. The trainer computes distances between the coordinates of context-mask tokens and target-mask positions; when distance weighting is enabled, a term is weighted inversely by the computed context-to-prediction distance, with a source-specific softened distance transformation. It then applies the same latent prediction family to context outputs. This is a recipe-specific dense-loss construction, not a generic requirement of every JEPA method.

The target encoder parameters are updated by EMA after the online optimizer step in the pinned training loop. The target branch runs under no-gradient mode. The configuration shown in the repository is a large-scale setup with 16 nodes and 8 tasks per node, 16-frame clips at 256-pixel crop size, a giant video transformer, batch 24 and 1000 epochs; dataset paths are placeholders. This source config is evidence about intended scale and code, not evidence that this project executed the run.

### 5.3 Deep self-supervision changes the target set

If target features from multiple intermediate encoder representations are included, the loss constrains more than the final embedding. The predictor must match several levels of representation structure. This can improve dense feature consistency, but it changes the optimization objective and compute. A result from this multi-level recipe cannot be ascribed to one final-layer prediction loss alone.

**Pinned implementation and config:** [V-JEPA 2.1 trainer](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py); [predictor](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/predictor.py); [example pretraining config](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitg16/pretrain-256px-16f.yaml).

## 6. EB-JEPA: prediction plus explicit anti-collapse and dynamics regularizers

The EB-JEPA repository is a library of examples and utilities for joint-embedding predictive architectures. Its simple video example, action-conditioned video example, and base JEPA class are related but not identical training systems. The name does not imply every example uses one universal energy function. The pinned code implements a latent prediction cost together with configurable regularizers.

Let an observation encoder produce \(z_t=E(o_t)\), and an action encoder produce \(u_t=A(a_t)\). A predictor produces
\[
\hat z_{t+1}=P(\hat z_{\leq t},u_{\leq t})
\]
or predicts from a sliding context window, depending on its recurrent/parallel mode. In the video example without actions, the action path can be unused. In the action-conditioned example, action encoding is part of the prediction path.

The training objective takes the form
\[
L=L_{\mathrm{pred}}+
\lambda_{\mathrm{std}}L_{\mathrm{std}}+
\lambda_{\mathrm{cov}}L_{\mathrm{cov}}+
\lambda_{\mathrm{time}}L_{\mathrm{time}}+
\lambda_{\mathrm{IDM}}L_{\mathrm{IDM}},
\]
with terms enabled and weighted according to the example configuration. The prediction term compares predicted and target visual/latent embeddings. The regularizer can be composed from the following.

### 6.1 Variance hinge

For feature dimension \(j\), compute batch standard deviation \(\sigma_j\). EB-JEPA's hinge term is
\[
L_{\mathrm{std}}=\frac1D\sum_{j=1}^{D}\max(0,\gamma-\sigma_j),
\]
where the implementation defaults to margin \(\gamma=1\) and adds \(10^{-4}\) under the standard-deviation square root. A feature with standard deviation above the margin receives no additional reward for increasing it under this term alone. The term penalizes low variance but does not prove semantic content or full rank.

### 6.2 Covariance decorrelation

For centered batch features \(Z\in\mathbb R^{B\times D}\),
\[
C=\frac{Z^\top Z}{B-1},\qquad
L_{\mathrm{cov}}=\operatorname{mean}_{i\ne j}(C_{ij}^2).
\]
The pinned library averages the squared off-diagonal covariance entries. This encourages feature dimensions to be decorrelated in the sampled batch. It does not force a Gaussian distribution or guarantee independence beyond second-order statistics.

### 6.3 Temporal similarity

For a representation sequence \(z_1,\ldots,z_T\),
\[
L_{\mathrm{time}}=\frac1{T-1}\sum_{t=1}^{T-1}
\operatorname{mean}\left((z_{t+1}-z_t)^2\right).
\]
This encourages neighbouring frames along a trajectory to have similar representations. That can smooth the representation manifold, but if over-weighted it can suppress rapid state changes or interactions. A smooth representation is not automatically a predictive representation.

### 6.4 Inverse dynamics

Given adjacent representations and an action label, an inverse dynamics model predicts the action:
\[
\hat a_t=q_\psi(z_t,z_{t+1}),\qquad
L_{\mathrm{IDM}}=\operatorname{MSE}(\hat a_t,a_t).
\]
This provides a reason for the representation to retain distinctions that help identify which action connected adjacent states. It complements temporal similarity, which by itself tends to reduce adjacent differences. The inverse-dynamics coefficient can be zero, and projection placement is configuration-dependent.

The EB-JEPA library combines such terms in a configurable regularizer whose coefficients, use of the first time slice versus all times, treatment of spatial positions as samples, and placement of projections are adjustable. A method explanation must state the chosen configuration rather than assume every regularizer is active in every experiment.

### 6.5 Parallel prediction, autoregressive prediction and planning

The pinned base class supports two different unrolls:

- **Parallel unroll:** predict many positions in parallel, while re-feeding ground-truth context for the left-hand context window during training.
- **Autoregressive unroll:** predict one next latent, append it to the existing context, and use predicted states as context for later steps.

The latter is closer to deployment: future observations are unavailable, so the predictor must consume its own predictions. Training loss on teacher-forced trajectories is not sufficient evidence that autoregressive rollout remains stable over long horizons.

In EB-JEPA's action-conditioned example, the predictor receives encoded actions and the planner searches for actions that make the predicted latent approach a goal latent. Its documented planning cost is based on distances between predicted and goal embeddings, optimized with methods such as MPPI or CEM. The default video example is a small Moving-MNIST setup; the action-conditioned example is a Two Rooms environment. These are pedagogical code paths, not the same scale or evidence base as V-JEPA 2 pretraining.

**Primary source:** [EB-JEPA paper](https://arxiv.org/abs/2602.03604); [pinned official repository](https://github.com/facebookresearch/eb_jepa/tree/966e61e9285b3a876f49b9774e9720d9a99a7925); [action-conditioned example](https://github.com/facebookresearch/eb_jepa/tree/966e61e9285b3a876f49b9774e9720d9a99a7925/examples/ac_video_jepa).

## 7. Losses that appear similar but are not equivalent

| Family or variant | Prediction source | Target | Additional constraints | Primary limitation |
|---|---|---|---|---|
| I-JEPA | Visible image context | Other image-block embeddings | Target EMA, block masks | Semantic sufficiency is not guaranteed by block prediction |
| V-JEPA | Visible video context | Space-time target embeddings | Target EMA, video mask geometry | Passive clip prediction does not condition on actions |
| V-JEPA 2-AC | State/video history plus action input | Action-conditioned future latent | Interaction-trajectory post-training | Generalization to unseen actions/interventions must be tested |
| V-JEPA 2.1 | Context and masked-token predictor outputs | Dense targets, including context/visible positions and intermediate layers | Deep self-supervision, distance-weighted context loss in inspected config | Recipe changes multiple factors and is computationally large |
| EB-JEPA video | Video sequence | Predicted latent sequence | Configurable variance/covariance penalties | Simple Moving-MNIST experiment does not establish general world-model quality |
| EB-JEPA action-conditioned | Observation history plus encoded actions | Next/future latent states | Variance, covariance, temporal similarity, optional inverse dynamics | Regularizer weights trade smoothness against action-relevant state variation |

All variants predict in a learned embedding space, but they differ in input conditions, target positions, target normalization, loss function, regularizers and training data. Treating them as a single loss formula hides important method details.

## 8. The world-model contract is stronger than JEPA loss

For a controlled system, let \(H_t\) be observation-action history, \(A_t\) the next action and \(Z_t=E(H_t)\) the representation. A sufficient predictive state should satisfy, on a stated population and action distribution,
\[
P(O_{t+1},R_t\mid H_t,A_t)
=P(O_{t+1},R_t\mid Z_t,A_t).
\]
A JEPA loss does not establish this conditional independence. It trains prediction of selected target embeddings under training masks and observed trajectories. Its target can intentionally discard pixel details; that is useful if discarded details are nuisance variables, harmful if they determine reward or dynamics.

Evidence should be layered:

1. **Optimization:** the self-supervised loss is finite and decreases.
2. **Representation statistics:** variance, effective rank, neighbourhood diversity and token geometry behave as expected.
3. **Latent prediction:** targets are predicted on held-out data, including multi-step rollout.
4. **State sufficiency:** histories with nearly identical latents are checked for different future distributions or rewards.
5. **Action conditioning:** the same history is tested under alternative actions, especially interventions uncommon in training.
6. **Planning:** a controller using the predictor succeeds in the real environment without exploiting model errors.

A pixel reconstruction score, linear probe, latent loss or successful rollout from teacher-forced training does not replace these tests.

## 9. Pinned source and recipe audit

### I-JEPA

- Paper: https://arxiv.org/abs/2301.08243
- Code pinned at facebookresearch/ijepa revision 52c1ae95d05f743e000e8f10a1f3a79b10cff048.
- Audit questions: exact context/target block construction, predictor position encoding, target feature normalization and EMA schedule. Check the config before claiming exact reductions or target normalization.

### V-JEPA and V-JEPA 2

- Original V-JEPA paper: https://arxiv.org/abs/2404.08471
- V-JEPA 2 paper: https://arxiv.org/abs/2506.09985
- Official repository pinned at facebookresearch/vjepa2 revision 204698b45b3712590f06245fbfba32d3be539812.
- The repository separates app/vjepa/, app/vjepa_2_1/ and action-conditioned V-JEPA 2-AC paths. Do not cite one source path as if all versions shared identical configs or loss details.

### V-JEPA 2.1

- Paper: https://arxiv.org/abs/2603.14482
- Trainer: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py
- Predictor: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/predictor.py
- Example config: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitg16/pretrain-256px-16f.yaml
- Confirmed: target encoder has no gradients; EMA follows the online optimizer update; target embeddings receive LayerNorm; loss exponent 1.0 means absolute-error loss; the example enables dense prediction and distance-weighted context prediction. The config contains placeholder data paths and calls for large multi-node hardware; it is source evidence, not a run performed in this project.

### EB-JEPA

- Paper: https://arxiv.org/abs/2602.03604
- Prediction and unroll implementation: https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/eb_jepa/jepa.py
- Regularizers: https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/eb_jepa/losses.py
- Small video example config: https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/examples/video_jepa/cfgs/default.yaml
- Action-conditioned world-model example: https://github.com/facebookresearch/eb_jepa/tree/966e61e9285b3a876f49b9774e9720d9a99a7925/examples/ac_video_jepa
- Confirmed: trainable JEPA adds prediction cost to a configurable regularizer and supports parallel/autoregressive rollout. Regularizers include feature variance hinge, off-diagonal covariance, temporal similarity and inverse dynamics. The example environment is Two Rooms, distinct from large-scale V-JEPA video pretraining.

This source audit identifies code behavior and version-specific config properties. It does not claim any pinned repository was executed here or that published results were reproduced.
