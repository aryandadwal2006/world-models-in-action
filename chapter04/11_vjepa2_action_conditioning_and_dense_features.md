# 11. V-JEPA 2 and V-JEPA 2.1: from video pretraining to controlled prediction

The move from a video encoder to a model that can be used for action selection is not a single architectural modification. It changes the training data, the predictor's inputs, the target distribution and the evaluation protocol. V-JEPA 2 makes this separation unusually clear: large-scale action-free video pretraining produces a visual representation, after which a new predictor is trained on robot trajectories with explicit action and end-effector-state inputs. V-JEPA 2.1 later changes the representation objective to improve dense spatial features. Its robot-planning experiments still require a separately trained action-conditioned predictor.

These stages should not be compressed into the claim that “V-JEPA 2 learns to control robots” or that dense features automatically create a world model. The useful lesson is how each stage changes the information available to the predictor and what additional evidence is needed before a model can be used for planning. (Assran et al., 2025; Mur-Labadia et al., 2026)

## 11.1 Scaling the action-free representation

V-JEPA 2 begins with an action-free video representation model. Its pretraining data combines large video sources with images. The paper describes the resulting VideoMix22M mixture as 22 million samples and more than one million hours of video, with ImageNet images included to increase visual coverage. An image is converted into a video sample by repeating it across the temporal dimension. This allows images and videos to be processed by a shared video encoder while preserving a consistent input format.

The encoder is trained with a masked-feature prediction objective. A video is patchified into spatiotemporal tokens, a subset is removed from the context input, and the online encoder processes the visible tokens. Learnable mask tokens carrying the positions of missing patches are concatenated with the visible embeddings and passed to a predictor. The predictor outputs features at the target positions. An EMA encoder processes the unmasked input and supplies the feature targets. The loss is an \(L_1\) regression loss on the masked tokens.

The high-level objective retains the structure of V-JEPA:

\[
\mathcal L_{\mathrm{masked}}
=
\frac{1}{|M|}
\sum_{i\in M}
\left\|
P_\phi(E_\theta(x),\Delta_y)_i
-
\operatorname{sg}(E_{\bar\theta}(y)_i)
\right\|_1,
\]

where \(M\) is the set of masked positions, \(x\) is the masked context view, \(y\) is the unmasked target view, and \(\Delta_y\) contains the positions of the missing tokens. The stop-gradient operator detaches target features in the loss; the EMA rule updates the target encoder's parameters. The formula is deliberately written at the level of the masked-token operation. The complete recipe also specifies data mixtures, temporal sampling, augmentations, architecture, optimization and model scale.

The distinction between masked prediction and action-conditioned prediction remains unchanged at this scale. The pretraining objective asks the encoder-predictor pair to infer missing video features from visible video context. It does not provide the model with a candidate intervention and ask what would happen under that intervention. An action-free video model can support motion understanding and human-action anticipation, but those evaluations are not evidence that it can predict the consequences of a robot's chosen control command.

The paper reports strong results on visual understanding and temporal anticipation, including 77.3% top-1 accuracy on Something-Something v2 and 39.7 recall-at-5 on Epic-Kitchens-100 for the reported V-JEPA 2 setting. These are task-specific results obtained with the evaluation protocols described in the paper. They do not measure the accuracy of an action-conditioned transition model.

## 11.2 Why video features alone do not identify action effects

Suppose a model observes a sequence \(x_{1:t}\) and predicts a future feature \(z_{t+1}\). From passive data, it can estimate a conditional relation of the form

\[
p(z_{t+1}\mid x_{1:t}).
\]

The distribution includes the actions that happened to be taken in the recorded trajectories, whether those actions are visible or not. It is not the same object as

\[
p(z_{t+1}\mid x_{1:t},a_t),
\]

where \(a_t\) is a specified action whose consequence is being evaluated.

To see the difference, consider a robot arm approaching a cup. The observed videos may mostly contain successful grasps. A passive predictor can learn that a particular hand configuration is often followed by the cup moving with the gripper. But it has not necessarily learned how the cup would move if the arm stopped, moved sideways, opened the gripper or pushed the cup. These alternatives may be rare or absent in the dataset. A model that is not given the candidate action cannot directly distinguish them as interventions.

This is not an argument that passive video is useless for learning dynamics. It can provide visual features, motion regularities and strong priors about the physical environment. It is an argument about identification: the training objective must expose the predictor to the variable whose effect it is meant to estimate. If control is the target use, the action must be represented in the model input and aligned with the observed transition.

## 11.3 V-JEPA 2-AC: freeze the encoder, learn a controlled predictor

V-JEPA 2-AC is a post-training stage built on top of the pretrained V-JEPA 2 encoder. The encoder is frozen. A new autoregressive predictor is trained to estimate future video features conditioned on past video features, control actions and robot end-effector states.

The paper studies a tabletop Franka Panda arm with a fixed external camera. Each sampled clip contains 16 frames at four frames per second. The corresponding end-effector state is a seven-dimensional vector containing three position coordinates, three orientation coordinates and a gripper-state coordinate. Actions are constructed from changes in end-effector state between adjacent frames. In this setup the model receives both the visual feature sequence and proprioceptive information about the robot.

The predictor is a transformer with approximately 300 million parameters and 24 layers. Its attention is block-causal: a token at a given time can attend to the visual features, action and end-effector state at that time and to information from previous time steps, but not to future inputs. This makes the predictor autoregressive with respect to the controlled trajectory. The input types are projected separately into the predictor's hidden dimension before being combined.

The training data are drawn from the raw DROID robot dataset. The paper uses roughly 62 hours of unlabeled robot video after discarding clips shorter than four seconds, leaving approximately 23,000 trajectories. “Unlabeled” here has a specific meaning. The model does not receive task labels, reward values, or success/failure annotations for these demonstrations. It does receive the video and the end-effector state signals required to construct actions and condition the transition predictor.

Let \(z_k=E(x_k)\) be the frozen visual feature map at time \(k\), \(s_k\) the end-effector state, and \(a_k\) the change in end-effector state between adjacent frames. The predictor receives the history

\[
(a_t,s_t,z_t)_{t\leq k}
\]

and predicts the next visual feature map. A teacher-forcing loss compares the prediction at each step with the encoded feature of the observed next frame:

\[
\mathcal L_{\mathrm{TF}}
=
\frac{1}{T}
\sum_{k=1}^{T}
\left\|
P_\phi\!\left((a_t,s_t,z_t)_{t\leq k}\right)
-z_{k+1}
\right\|_1.
\]

Teacher forcing supplies the observed preceding feature at every training step. At inference time, however, the model must consume its own predicted features when rolling forward. This creates a training-inference mismatch: a predictor trained only with observed histories can accumulate error when its own estimates become part of the next input.

V-JEPA 2-AC therefore adds a short rollout loss. The predictor is run autoregressively for two steps, and its final predicted representation is compared with the corresponding encoded target:

\[
\mathcal L_{\mathrm{rollout}}
=
\left\|
P_\phi(a_{1:2};s_1,z_1)-z_3
\right\|_1.
\]

The complete objective is the sum of teacher-forcing and rollout terms:

\[
\mathcal L_{\mathrm{AC}}
=
\mathcal L_{\mathrm{TF}}+\mathcal L_{\mathrm{rollout}}.
\]

The two-step rollout is not a guarantee of long-horizon accuracy. It trains the predictor through one recurrent step beyond the teacher-forced transition, making the model experience a limited version of its own prediction errors. Longer rollouts would increase both the difficulty of optimization and the chance that errors in early predictions distort later inputs.

## 11.4 From the predictor to a planning loop

Once the model predicts future feature maps under proposed actions, a planner can compare candidate action sequences. Let \(z_k\) be the current visual representation, \(s_k\) the current end-effector state, and \(z_g\) the representation of a goal image. For a candidate action sequence \(\hat a_{1:T}\), the model predicts a future representation. The paper uses a goal-conditioned energy of the form

\[
\mathcal E(\hat a_{1:T};z_k,s_k,z_g)
=
\left\|
P_\phi(\hat a_{1:T};s_k,z_k)-z_g
\right\|_1.
\]

The planner searches for an action sequence that reduces the discrepancy between the imagined future and the encoded goal. V-JEPA 2 uses the Cross-Entropy Method (CEM) to optimize the action sequence. At each iteration it samples candidate trajectories from a distribution over actions, evaluates their predicted feature-space outcomes, retains high-scoring candidates and updates the sampling distribution around them.

The system then executes only the first action, receives a new camera observation and robot state, and plans again. This is receding-horizon or model-predictive control. It prevents the controller from blindly executing an entire open-loop sequence when fresh observations are available.

The complete loop therefore contains at least five distinct elements:

1. An encoder maps current observations into a feature representation.
2. An action-conditioned predictor estimates future features under candidate controls.
3. A goal representation specifies the desired visual outcome.
4. A planner searches candidate action sequences using the model and a goal-conditioned energy.
5. The robot executes an initial action and the loop repeats with a new observation.

The prediction model is not the planner, and the energy is not the prediction loss used to train the encoder. The \(L_1\) feature-prediction loss teaches the predictor to match observed futures during training. The goal-conditioned energy ranks candidate action sequences during planning. The distinction is the same one established at the start of the chapter between prediction discrepancy, compatibility score and task cost.

## 11.5 What the physical planning result demonstrates

The V-JEPA 2 paper deploys the system zero-shot on Franka arms in two different laboratories. It evaluates reaching, grasping and pick-and-place tasks using image goals. The reported results are measured over a small number of task trials and several object or initial-position variations. The paper's claim is specifically that the pretrained video representation, when combined with robot-trajectory post-training and model-predictive control, supports these tasks in new environments without collecting environment-specific robot data there or adding task-specific training or reward labels for the deployment tasks.

This is significant evidence for the complete system. It is not evidence that the action-free encoder alone controls the robot. The action-conditioned predictor was trained on DROID robot trajectories, and the planner uses explicit visual goals. Nor does the experiment establish general-purpose control across arbitrary robots or environments.

The paper also identifies limitations. The model is sensitive to camera placement because action coordinates are defined relative to the robot base while the visual input comes from a camera. If the base is not visible, inferring the relation between the camera frame and action coordinates can be underdetermined. Long-horizon planning is limited by two different effects: autoregressive representation errors accumulate, and the number of candidate action trajectories grows rapidly with the planning horizon. A longer planning horizon can be necessary for non-greedy tasks, but it increases both model and search difficulty.

These limits are valuable because they reveal where the representation-to-control transition remains incomplete. A model may predict short-horizon features well but become unreliable when rolled forward. A planner may find actions that minimize the learned goal distance while exploiting errors in the predictor. A representation may ignore a visually small detail that determines whether a gripper actually closes around an object. Physical planning must be evaluated at the level of the full closed loop.

## 11.6 V-JEPA 2.1: why masked-token prediction is not enough for dense features

The next development addresses a different limitation. V-JEPA 2's masked-token objective supervises the target features at masked positions. The visible context tokens are inputs to the predictor, but they do not receive the same direct prediction loss. The V-JEPA 2.1 paper argues that this can encourage context tokens to aggregate information useful for predicting global masked targets without preserving sufficiently local information at each visible position.

The distinction is between **global information useful for prediction** and **local information accessible at a particular token**. A representation can contain enough information to predict a hidden region while individual visible tokens are not ideal descriptors of the local object part or spatial location they correspond to. This is not necessarily a failure for tasks evaluated on pooled features. It becomes important for dense tasks such as depth estimation, semantic segmentation, object correspondence and localization, where a feature at one position should retain information about the scene at that position.

V-JEPA 2.1 adds a context loss to the original masked-token prediction loss. The context loss supervises visible tokens, with a weight that depends on their spatial-temporal distance from the masked region. The total dense prediction objective can be written schematically as

\[
\mathcal L_{\mathrm{dense}}
=
\mathcal L_{\mathrm{predict}}
+
\mathcal L_{\mathrm{context}},
\]

with

\[
\mathcal L_{\mathrm{context}}
=
\frac{1}{|C|}
\sum_{i\in C}
\lambda_i
\left\|
P_\phi(E_\theta(x),\Delta_y)_i
-
\operatorname{sg}(E_{\bar\theta}(y)_i)
\right\|_1.
\]

Here \(C\) is the set of context positions and \(\lambda_i\) is a position-dependent weight. The target encoder processes the unmasked input. The original masked-token loss remains in place; the new term extends self-supervision to visible positions. The weighting is intended to encourage local consistency without making the context objective equivalent to an unconstrained identity mapping over all visible tokens.

This change creates a trade-off. In the paper's ablation, adding context loss improves dense prediction substantially, but initially reduces global recognition performance. On the reported ADE20K segmentation and NYUv2 depth evaluations, the context term improves dense metrics; at the same time, ImageNet and Something-Something classification accuracy fall. The paper's next component addresses that loss.

## 11.7 Deep self-supervision and multiple representation levels

V-JEPA 2.1 applies supervision at several intermediate encoder depths, not only at the final output. Representations from multiple blocks are normalized, combined and projected through a lightweight multilayer perceptron. The predictor produces corresponding multi-level outputs, and the masked-token and context-token losses are applied at these levels.

The motivation is that a final-layer objective can place most of the learning pressure on the representation after many attention and mixing operations. Earlier blocks may not receive an equally direct signal to retain spatial detail. Applying losses at several depths creates training signals throughout the encoder. It also gives the model a way to balance local structure with global semantics instead of relying on the final representation to satisfy both demands alone.

The ablation is instructive. Context loss alone raises dense-task performance but harms global recognition. Adding multi-level prediction recovers much of the recognition performance while preserving the dense improvement. The reported progression then adds a larger and more diverse image mixture, modality-specific tokenizers, model scaling and a high-resolution cooldown phase. These ingredients are not all the same kind of intervention: the context loss changes what is supervised, deep self-supervision changes where the signal enters the encoder, tokenizers change the input interface, and data/model scaling changes capacity and coverage.

For a reader, the most important point is not to memorize the sequence of recipe modifications. It is to understand the underlying tension. A representation trained to predict a hidden target may learn global context useful for that target. A representation used for dense prediction must also preserve spatially localized information at visible positions. Dense supervision changes the objective so that visible tokens themselves have a target to match.

## 11.8 Dense representations are not action conditioning

V-JEPA 2.1 is an image-and-video representation-learning recipe. Its dense prediction loss improves spatially and temporally structured features. It does not, by itself, add candidate actions to the visual prediction objective.

The V-JEPA 2.1 robot experiments make this distinction explicit. The authors train a frame-causal action-conditioned predictor following the V-JEPA 2-AC protocol, use the V-JEPA 2.1 video encoder to compute features, and train the approximately 300-million-parameter predictor on the raw DROID trajectories with teacher-forcing and two-step rollout losses. The model is then deployed zero-shot on a tabletop Franka Panda arm using image-goal model-predictive control. Thus, the representation encoder changes, but the action-conditioned predictor and planning interface remain separate trained components.

The reported grasp results require careful comparison. Under the matched planning configuration of 800 candidate samples, 10 CEM iterations, a one-step horizon and a three-second budget, the reported cup-grasp success rate rises from 60% with V-JEPA 2 features to 70% with V-JEPA 2.1 features. That is a ten-percentage-point increase in the matched table. The paper also reports an 80% V-JEPA 2.1 grasp result using 300 samples, 15 iterations, an eight-step horizon and a fourteen-second budget. The latter is a different planning configuration and should not be presented as a matched comparison with the V-JEPA 2 row.

The appropriate conclusion is limited but useful: in the reported setup, improved dense visual features can improve the spatial information available to a separately trained action-conditioned predictor and planner. The result does not show that dense feature prediction alone creates a controller, or that the same improvement will hold under all planning budgets, robots or tasks.

## 11.9 Three independent design questions

The progression from V-JEPA 2 to V-JEPA 2.1 separates three questions that are often collapsed.

**What representation should be learned from passive video?** The action-free pretraining objective determines which visual and temporal distinctions are rewarded. Masked-token prediction, dense context supervision, intermediate-layer losses, tokenizer design and data coverage all affect this representation.

**How should the model predict under a chosen action?** This requires an action-conditioned predictor whose inputs include aligned observations, control commands and, when relevant, proprioceptive state. The training trajectories must cover the state-action combinations over which the model will be used.

**How should the system choose and execute actions?** A planner needs a goal or cost, a candidate action parameterization, a search or optimization procedure, constraints and a feedback policy. Its performance depends on model accuracy, search budget and the consequences of model error.

Changing the encoder addresses the first question. Adding an action-conditioned predictor addresses the second. Choosing a cost and planning algorithm addresses the third. A result at one level does not automatically establish the others.

**Figure 11.1. V-JEPA 2 is a staged system.** Draw two clearly separated stages. Stage A pretrains an action-free video encoder and predictor from images and videos using masked feature targets. Stage B freezes the encoder and trains a new autoregressive predictor on robot observations, end-effector states and actions. Show the teacher-forcing and short rollout objectives only in Stage B.

**Figure 11.2. The V-JEPA 2.1 context loss adds a target for visible tokens.** Draw the same spatiotemporal token grid twice. In the first panel, show prediction loss only on masked tokens. In the second, retain the masked loss and add distance-weighted context loss on visible tokens. The figure should distinguish target coverage from action conditioning.

**Figure 11.3. Representation, dynamics and planning are separate interfaces.** Show the encoder producing current and goal features, an action-conditioned predictor mapping candidate controls to future features, and a planner minimizing a goal-conditioned cost before executing the first action and replanning. Keep the model-training loss and planner energy in separate boxes.

This progression provides a concrete example of how representation learning becomes part of a world-model system. The next section formalizes the distinction between passive prediction and prediction under an intervention, then examines the assumptions required before a learned transition model can be trusted for planning.
