# 12. From passive video representations to robot prediction

The transition from video representation learning to physical prediction requires more than adding time to the input. A video encoder can learn features that describe motion and interaction, yet still lack an explicit variable for the action whose consequence a controller wants to estimate. V-JEPA 2 makes the separation concrete: it first learns a visual representation from large-scale passive data, then trains an action-conditioned predictor on robot trajectories. V-JEPA 2.1 strengthens the spatial density of the representation, while retaining a separate controlled predictor for its robot-planning experiments.

The important progression is therefore not a sequence of model names. It is a separation of responsibilities: the encoder supplies visual features; the controlled predictor estimates consequences under actions; the planner evaluates candidate action sequences.

## 12.1 Why passive video is a useful starting point

A large video corpus contains motion, object interaction, scene change and many visual regularities. An encoder pretrained on such data can provide features that are useful for recognizing objects and anticipating events. This gives a robot predictor a strong visual input without requiring the robot trajectories themselves to teach every visual concept from scratch.

But passive video is generated under the actions taken by the people or systems in the recordings. A model trained on these observations learns relations in the observed trajectory distribution. It does not automatically learn how the scene would change under a different action from the same state.

Let \(X_{1:t}\) denote the observed history and \(Z_{t+1}\) a future feature. An action-free predictor estimates

\[
p(Z_{t+1}\mid X_{1:t}).
\]

A controlled predictor must estimate

\[
p(Z_{t+1}\mid X_{1:t},A_t),
\]

where \(A_t\) specifies the action whose consequence is being considered. The second distribution can change when the action changes while history remains fixed. The first has no explicit input representing that alternative.

Passive pretraining and action-conditioned prediction therefore solve different problems. The pretrained encoder can make the visual input easier to interpret. It cannot supply a missing action variable by itself.

## 12.2 V-JEPA 2-AC adds the control interface

V-JEPA 2-AC is a post-training stage within the V-JEPA 2 paper. It freezes the pretrained visual encoder and learns a separate autoregressive predictor using robot video, action and end-effector state. The predictor receives an aligned history of visual features and control information, then estimates future visual features. This is the central separation in the V-JEPA 2-AC post-training stage (Assran et al., 2025).

The robot data come from DROID demonstrations. The paper uses end-effector position, orientation and gripper state to define a robot state vector. Actions are derived from changes in this state between adjacent observations. The data are therefore not merely passive images: the predictor receives a control signal aligned with the transition it must predict.

Let \(E\) be the frozen visual encoder, \(x_t\) an image, \(s_t\) the end-effector state and \(a_t\) the action. Write

\[
z_t=E(x_t).
\]

A causal predictor estimates the next feature from the history of visual features and controls:

\[
\widehat z_{t+1}
=
P_\phi\big((z_i,s_i,a_i)_{i\leq t}\big).
\]

The predictor uses block-causal attention: tokens at a time step can attend to current and previous information, but not future observations. This makes the temporal conditioning appropriate for an autoregressive transition model.

The paper combines teacher-forced prediction and a short rollout objective. Under teacher forcing, the predictor receives observed preceding features at each step and is trained to match the next encoded observation:

\[
\mathcal L_{\mathrm{TF}}
=
\frac1T\sum_{t=1}^{T}
\left\|
P_\phi\big((z_i,s_i,a_i)_{i\leq t}\big)
-z_{t+1}
\right\|_1.
\]

At deployment, however, the model must consume its own previous predictions when rolling forward. Teacher forcing alone does not expose it to that input distribution. V-JEPA 2-AC therefore adds a two-step autoregressive rollout loss, in which a predicted feature is fed back to the predictor before the later prediction is compared with its encoded target:

\[
\mathcal L_{\mathrm{roll}}
=
\left\|
P_\phi(a_{1:2};s_1,z_1)-z_3
\right\|_1.
\]

The complete post-training objective is the sum of these terms. The two-step rollout reduces the gap between training and inference, but it does not guarantee long-horizon accuracy. Errors may still accumulate when the predictor is applied repeatedly.

This design illustrates why the control interface is a modelling decision. The visual encoder can remain frozen while a predictor learns how observed robot actions relate to subsequent feature changes. The target now comes from a later observation, and the action is explicitly part of the conditioning information.

## 12.3 Planning uses a different objective from prediction

Once a predictor can estimate future features under candidate actions, a planner can compare action sequences. Let \(z_t\) be the current visual feature, \(s_t\) the current robot state and \(z_g\) the feature of a goal image. For a proposed action sequence \(a_{t:t+H-1}\), the predictor estimates a future feature. A goal-conditioned energy can compare that estimate with \(z_g\):

\[
\mathcal E(a_{t:t+H-1})
=
\left\|
P_\phi(a_{t:t+H-1};s_t,z_t)-z_g
\right\|_1.
\]

This energy is used to rank candidate actions during planning. It is not the same loss used to train the predictor. The training loss compares predicted features with observed future features. The planning energy compares an imagined outcome with a desired goal representation.

V-JEPA 2 uses the Cross-Entropy Method (CEM) to search over candidate action sequences. It repeatedly samples action sequences, evaluates them through the predictor, retains high-scoring candidates and updates the sampling distribution around them. The controller executes an initial action, obtains a new observation and plans again. This receding-horizon loop limits reliance on one long open-loop prediction.

The separation is essential. The predictor estimates consequences. The goal representation defines what outcome is sought. The planning algorithm searches over controls. A feature-prediction loss alone does not provide a goal, a cost or a search procedure.

## 12.4 What the robot experiments establish

The V-JEPA 2 paper evaluates the complete system on tabletop Franka Panda tasks, including reaching, grasping and pick-and-place with image goals. The reported deployments include two laboratories and test whether the pretrained visual representation, robot-trajectory post-training and model-predictive controller transfer without task-specific robot training in the deployment environments.

The result is evidence about the combined system. It is not evidence that the action-free video encoder alone controls a robot. The controlled predictor is trained on DROID trajectories, and the planner uses an explicit goal and action search. Nor do the experiments establish general-purpose control across arbitrary robots, cameras or action spaces.

The limitations reveal why the interfaces matter. The action coordinates are defined relative to the robot base, while the image is obtained from a camera. If the relation between camera and robot coordinates is ambiguous, the predictor may have difficulty connecting visual features to actions. Long rollouts also accumulate prediction errors, while longer planning horizons increase the size and difficulty of the action search. Better visual features cannot remove every limitation in the control and planning interface.

## 12.5 V-JEPA 2.1 changes the density of visual supervision

V-JEPA 2.1 addresses a separate representation problem. In masked feature prediction, target tokens receive direct prediction loss, while visible context tokens primarily serve as inputs. A context token may therefore become useful for predicting distant masked targets without retaining sufficiently local information at its own position.

This matters for dense tasks such as depth estimation, segmentation and localization. A pooled representation can support scene recognition while discarding local spatial detail. Dense prediction requires the feature at a position to retain information about the corresponding part of the scene.

V-JEPA 2.1 adds a context loss on visible tokens. Let \(C\) denote visible positions and let \(d_{\min}(i,M)\) be the distance, measured in spatiotemporal mask blocks, from visible token \(i\) to the nearest masked block. The loss assigns a larger weight to tokens nearer the masked region:

\[
\lambda_i=\frac{\lambda}{\sqrt{d_{\min}(i,M)}}.
\]

The weighting emphasizes the visible features closest to the boundary between context and target. It encourages local consistency across that boundary without applying the same target constraint uniformly to every visible token. The full objective retains the masked-token prediction loss and adds the visible-context loss.

The method also applies deep self-supervision at three intermediate encoder blocks and the final block. The dense context objective and its weighting follow the V-JEPA 2.1 paper (Mur-Labadia et al., 2026). Intermediate features are combined and supervised alongside the final representation. The motivation is that local spatial information should not be required to survive only in the last encoder layer after many rounds of global feature mixing.

These changes affect where and which visual features are supervised. They do not add an action variable. V-JEPA 2.1 is still a video representation-learning method; its robot-planning experiments use a separately trained action-conditioned predictor.

## 12.6 Dense features can improve a controller without becoming a controller

The V-JEPA 2.1 robot experiments retain the action-conditioned predictor and planning interface used in V-JEPA 2-AC, while replacing the visual encoder with the denser representation. Under the matched configuration reported for cup grasping, the success rate rises from 60% with V-JEPA 2 features to 70% with V-JEPA 2.1 features. This is a ten-percentage-point difference under that particular comparison.

The paper also reports a higher V-JEPA 2.1 success rate under a different planning configuration with more CEM iterations, a longer horizon and a larger time budget. That result should not be presented as a matched comparison with the 60% or 70% rows. The search budget and horizon are part of the experiment.

The limited conclusion is that more spatially informative visual features can improve the information available to a separately trained action-conditioned predictor and planner. It does not show that dense masked prediction alone creates a physical transition model.

## 12.7 Three questions that should remain separate

The progression from V-JEPA 2 to V-JEPA 2.1 makes three design questions visible:

1. **What should the representation preserve?** The pretraining target, mask geometry, visible-token supervision, intermediate losses, data and tokenizer determine which visual distinctions are rewarded.
2. **How should the model predict under a chosen action?** A controlled predictor needs aligned observations, actions and relevant robot state. Its data must cover the state-action combinations on which it will be used.
3. **How should actions be selected?** A planner needs a goal or cost, an action parameterization, a search strategy, constraints and feedback.

A change to the encoder addresses the first question. Adding an action-conditioned predictor addresses the second. A planner addresses the third. Evidence at one level does not automatically establish competence at the others.

**Figure 4.19. V-JEPA 2 separates passive visual pretraining from controlled post-training.**

Description: A visual encoder is pretrained on passive images and videos. It is then frozen while a separate predictor learns from robot observations, end-effector state and aligned actions. A planner evaluates candidate controls using the resulting predictor and a goal representation.

**Figure 4.20. V-JEPA 2.1 adds supervision to visible tokens without adding actions.**

Description: The original masked prediction loss supervises withheld target tokens. V-JEPA 2.1 adds distance-weighted supervision to visible context tokens and applies losses at intermediate encoder depths. The action-conditioned predictor remains a separate component for robot planning.