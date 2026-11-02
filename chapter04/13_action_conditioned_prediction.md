# 13. Learning what an action does

A model trained on video learns relationships between observations. A model used for control must answer a different question: what would happen if the agent chose a particular action now? The distinction is between predicting an outcome under the behaviour recorded in a dataset and predicting the outcome under a specified intervention.

An action input is necessary for the second problem, but not sufficient. The data must contain information that identifies how outcomes vary with actions, and the representation must preserve the state variables that affect those outcomes.

## 13.1 Observed actions and chosen actions define different distributions

Let \(H_t\) denote the observation history available at time \(t\), \(A_t\) the action taken, and \(Y_{t+1}\) a future observation or representation. A passive dataset provides samples from a joint distribution such as

\[
p(H_t,A_t,Y_{t+1}).
\]

From it, a model can estimate the conditional distribution

\[
p(Y_{t+1}\mid H_t,A_t=a).
\]

This is a statement about examples in which the behaviour policy happened to choose \(a\) under histories like \(H_t\). A controller, however, wants the distribution that would result from setting the action to \(a\):

\[
p(Y_{t+1}\mid H_t,\operatorname{do}(A_t=a)).
\]

The intervention notation distinguishes fixing an action from selecting observed cases where that action occurred. The two distributions agree only under additional assumptions. If the action is selected using information about the state that also affects the future, conditioning on the action can select a biased subset of histories. A behaviour policy may choose to push an object only when it is already close to the gripper, for example. Observed pushes are then correlated with the geometry that made the push possible.

## 13.2 What must be identified

A common way to express the controlled problem is through a state \(S_t\) that summarizes the history:

\[
S_t=f_\theta(H_t).
\]

For this state to support controlled prediction, it should preserve the future distribution relevant to the action space:

\[
p(Y_{t+1}\mid H_t,A_t)
=
p(Y_{t+1}\mid S_t,A_t).
\]

This is a task-relative sufficiency requirement. If hidden velocity, contact state or gripper configuration affects the outcome of an action but is absent from \(S_t\), a predictor may give different answers for histories that it has represented identically.

Even a sufficient state does not automatically identify interventions from arbitrary observational data. At least three conditions must be considered.

**Consistency.** The recorded outcome under an observed action must correspond to the potential outcome associated with that action. The action representation must have a well-defined meaning and be aligned with the transition interval.

**Action support.** The dataset must contain relevant actions in the states where the model will be queried. If the behaviour policy never tries to move left from a particular state, the data provide no direct evidence about that transition. This is often called positivity or overlap: the actions to be evaluated must have nonzero probability under the data-collection policy in the relevant states.

**Conditional exchangeability.** After conditioning on the state information supplied to the predictor, action selection should contain no unobserved information that also changes the outcome. In a controlled simulator, one may know the complete state and intervention mechanism. In real robot data, unobserved contact, object motion or policy information can violate this assumption.

These conditions explain why recording actions beside video is not enough. Alignment supplies the input variable. Coverage and state sufficiency determine whether the resulting conditional model can support the counterfactual questions a planner will ask.

## 13.3 A controlled feature predictor

Let a visual encoder map the current observation to features \(Z_t=E(X_t)\). Let \(A_t\) be the control command and \(R_t\) any additional robot state needed to describe the actuator or end effector. A one-step transition predictor can be written as

\[
\widehat Z_{t+1}
=
F_\phi(Z_{\leq t},A_{\leq t},R_{\leq t}).
\]

For a Markov-style model, the history may be replaced by a learned state \(S_t\):

\[
\widehat S_{t+1}=F_\phi(S_t,A_t).
\]

Training on aligned trajectories compares predictions with the encoded observations that actually followed the recorded actions. For a feature loss \(\ell\),

\[
\mathcal L_{\mathrm{transition}}
=
\mathbb E
\left[
\ell\!\left(
F_\phi(S_t,A_t),
\operatorname{sg}(S_{t+1})
\right)
\right].
\]

The target may be a visual feature, a structured state vector or a combination. The stop-gradient indicates that the target representation is not updated by this transition loss; the precise target update depends on the chosen architecture. A predictor trained with teacher forcing receives the observed previous state at each step. If it will be rolled forward at inference, training should also examine the errors produced when predicted states become subsequent inputs.

The action-conditioned post-training of V-JEPA 2 is one concrete example of aligning robot end-effector changes with subsequent visual features (Assran et al., 2025). The action representation must match the control system. A discrete action label, a joint-position command, a velocity target and a Cartesian end-effector displacement describe different control interfaces. The time interval over which the action applies must also be explicit. A command observed at 10 Hz cannot be treated as though it were a continuous force applied over an unspecified interval.

## 13.4 Data collection determines which questions can be answered

Suppose a robot dataset consists mostly of successful grasps. The predictor may learn the visual transition that usually accompanies a grasp. It may not have enough evidence to predict what happens if the gripper closes too early, moves sideways or opens during contact. The model can interpolate between observed actions only to the extent that the data and architecture support that interpolation.

Coverage can be improved by collecting varied trajectories, randomizing safe control parameters or using a simulator to generate interventions. In real systems, exploration is constrained by collision risk, actuator limits and the possibility of damaging objects. The dataset must therefore be designed around the action space that the planner will search, not merely around the actions that were convenient to record.

A useful split holds out transitions by state and action, rather than randomly splitting adjacent video frames from the same trajectory. Random frame-level splits can place nearly identical contexts in both training and test sets, overstating generalization. Evaluation should ask whether predictions remain accurate for unseen state-action combinations that lie within the intended deployment support.

## 13.5 Testing the intervention claim

A transition predictor should be evaluated on questions that distinguish observation from intervention. In a simulator, one can hold the initial state fixed, apply several candidate actions and compare predicted outcomes with the corresponding simulated transitions. This tests whether changing the action changes the prediction in the correct way.

Other useful tests include one-step prediction error on held-out trajectories; error as a function of rollout horizon when predictions are fed back; performance on actions that are rare but still within the training support; sensitivity to changes in camera, object geometry or actuator calibration; and closed-loop success when a planner uses the predictor to select actions.

No single metric establishes causal correctness for every environment. A good one-step feature match can coexist with poor counterfactual prediction. A predictor can match the average outcome under a behaviour policy while failing under an action distribution chosen by an optimizer.

The essential requirement is that the model be tested on the conditional questions it will be asked at deployment. If the planner will compare several actions from the same state, evaluation must compare those alternatives, not only replay observed trajectories.

**Figure 4.21. Conditioning on an observed action is not automatically an intervention.**

Description: A behaviour policy selects actions using the current state and produces observed outcomes. An intervention fixes a candidate action while holding the starting state fixed. Identifying the intervened outcome requires aligned actions, adequate action support and sufficient control of confounding variables.