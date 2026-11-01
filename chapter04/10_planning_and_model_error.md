# 10. Planning with a learned predictive model

An action-conditioned predictor estimates how a representation may change under a specified control input. Planning adds another problem: choose a sequence of controls so that the predicted outcome satisfies an objective.

The predictor and the planner therefore have different responsibilities. The predictor estimates consequences. The planner compares those consequences and searches over possible actions. Their interaction creates capabilities that neither component has alone, but it also creates failure modes that are absent from ordinary representation evaluation.

## 10.1 From a one-step prediction to a candidate trajectory

Let \(z_t\) denote the current latent state, \(a_t\) a control input, and \(g_\phi\) an action-conditioned transition predictor. A deterministic one-step model has the form

\[
\widehat z_{t+1}=g_\phi(z_t,a_t).
\]

A model with observation history or proprioception can include those variables as additional inputs. For a candidate action sequence

\[
\mathbf a_{t:t+H-1}
=
(a_t,a_{t+1},\ldots,a_{t+H-1}),
\]

the model is iterated to produce a predicted latent trajectory:

\[
\widehat z_{t+k+1}
=
g_\phi(\widehat z_{t+k},a_{t+k}),
\qquad k=0,\ldots,H-1.
\]

This is a model rollout. It is not a prediction of the real environment at every intermediate step. It is the trajectory implied by repeated application of the learned transition function.

A planner evaluates the candidate sequence using a cost. A simple goal-conditioned objective is

\[
J(\mathbf a_{t:t+H-1})
=
d(\widehat z_{t+H},z_g)
+
\sum_{k=0}^{H-1}c_a(a_{t+k}),
\]

where \(z_g\) is a goal representation, \(d\) measures terminal mismatch, and \(c_a\) penalizes action magnitude or another control preference. A more general objective can accumulate state costs:

\[
J(\mathbf a)
=
\sum_{k=1}^{H}c_z(\widehat z_{t+k})
+
\sum_{k=0}^{H-1}c_a(a_{t+k})
+
c_T(\widehat z_{t+H}).
\]

The components have different meanings. The transition model predicts latent states. The goal or cost function evaluates them. The action penalty expresses a control preference. The planner searches over the sequence.

A low training prediction loss does not specify any of these planning choices. A latent predictor cannot decide by itself whether a future is desirable.

## 10.2 Image goals and latent distance

A common goal-conditioned setup supplies a target image \(o_g\). The current observation and goal image are encoded into representations,

\[
z_t=f_\theta(o_t),
\qquad
z_g=f_\theta(o_g).
\]

For each candidate action sequence, the transition model predicts a future representation \(\widehat z_{t+H}\). The planner can then minimize a goal discrepancy such as

\[
d(\widehat z_{t+H},z_g).
\]

This avoids requiring the model to decode every predicted state into an image before the planner can compare it with the goal. It can also make the objective insensitive to appearance variation that the encoder has learned to ignore.

But latent distance is only a proxy for physical success. If the encoder discards gripper aperture, contact state or a small obstacle, a low latent distance can correspond to an unsafe or unsuccessful physical state. The goal representation and the transition representation must preserve the variables needed to distinguish success from failure.

A visual goal can also be ambiguous. The same image may be reached through different object configurations, contact conditions or robot postures. Whether those states are interchangeable depends on the task. A goal encoder does not automatically resolve this ambiguity.

## 10.3 Searching over action sequences

The action space may be continuous and the sequence horizon long. Directly evaluating every possible sequence is usually infeasible. Planning therefore relies on an optimization or search procedure.

One option is gradient-based optimization. If the transition model and cost are differentiable, gradients can be propagated through the imagined rollout to the candidate action variables:

\[
\nabla_{\mathbf a}J(\mathbf a).
\]

The optimizer changes the action sequence to reduce predicted cost. This is possible because the learned model is a differentiable computation graph. It does not imply that gradients pass through the physical environment. The agent optimizes actions inside its model, executes a selected action in the environment, and then obtains a new observation.

A derivative-free alternative is the cross-entropy method (CEM). It maintains a distribution over candidate action sequences:

\[
\mathbf a^{(i)}\sim q_\psi(\mathbf a),
\qquad i=1,\ldots,N.
\]

Each sampled sequence is rolled through the learned model and assigned a cost. The method selects an elite set with low predicted cost and refits the sampling distribution to those candidates. Repeating this procedure concentrates samples around sequences that the model predicts to be promising.

CEM does not require gradients through the predictor, though it still depends on the predictor's accuracy. It can also search over actions whose parameterization is inconvenient for direct gradient descent. Its effectiveness depends on the sample count, number of iterations, distribution family, horizon and action bounds.

Neither gradient descent nor CEM is intrinsically a guarantee of good physical behaviour. They optimize the objective supplied to them under the model they are given. If that objective or model is wrong, a more effective optimizer may find a more convincing failure.

## 10.4 Receding-horizon feedback

A planner normally does not execute an entire optimized sequence open loop. It executes the first action or a short prefix, obtains a new observation, estimates the current state again, and replans.

At time \(t\), solve

\[
\mathbf a^*_{t:t+H-1}
=
\arg\min_{\mathbf a_{t:t+H-1}}
J_t(\mathbf a_{t:t+H-1}).
\]

Execute \(a_t^*\), observe the environment at \(t+1\), update the representation and solve a new optimization problem. This is the receding-horizon principle used in model-predictive control.

Replanning limits how long the system commits to a single predicted trajectory. It can compensate for disturbances and moderate model errors because the next plan starts from a fresh observation. It does not remove systematic bias. If the model repeatedly underestimates a collision risk or ignores a hidden state, replanning can repeatedly choose the same unsafe action.

Feedback is therefore a robustness mechanism, not a proof of safety.

## 10.5 Why planning can fail when prediction looks good

Prediction error is usually measured on trajectories sampled from a data distribution. A planner does not sample actions according to that same distribution. It actively searches for sequences that minimize the model's estimated cost. This creates a form of distribution shift.

Consider a model that is accurate on ordinary actions but slightly wrong in a particular region of the action space. A planner searches many candidates and may discover a sequence that exploits the error. The predicted trajectory appears to reach the goal cheaply, so the optimizer prefers it. In the real environment, the sequence fails.

This is often called model exploitation or model bias exploitation. The planner is not necessarily behaving irrationally. It is solving the optimization problem it was given. The problem is that the model's predictions are trusted outside the region where they have been validated.

Several effects make this more likely:

- **Horizon growth.** Small one-step errors compound when the model is iterated over many steps.
- **Action distribution shift.** The optimizer may choose actions that were rare or absent in the training trajectories.
- **Representation aliasing.** Two physical states that respond differently to an action may have similar latent representations.
- **Unmodelled state.** Proprioception, contact, friction or other hidden variables may be omitted.
- **Cost mismatch.** A goal-distance objective may reward a state that violates a safety or feasibility constraint.
- **Search pressure.** Evaluating many candidate sequences increases the chance of finding a sequence whose predicted value is optimistic because of model error.

A planning result must therefore report the model, training data, action representation, horizon, planner, objective, search budget, environment and execution protocol. Reporting only a representation benchmark or a one-step prediction score is insufficient.

## 10.6 Multiple futures and uncertainty

A deterministic transition model maps one context and action to one predicted future. Real environments may have multiple possible outcomes because of stochastic dynamics, partial observability, other agents or model uncertainty.

A more complete model represents a conditional distribution,

\[
p_\phi(z_{t+1:t+H}\mid z_{\le t},\mathbf a_{t:t+H-1}),
\]

or a set of alternative hypotheses. A latent variable can index possible futures, but it must be structured so that it selects meaningful alternatives rather than simply copying the target.

A planner using uncertain predictions has to decide how to evaluate risk. It might minimize expected cost,

\[
J_{\mathrm{mean}}(\mathbf a)
=
\mathbb E_{Z\sim p_\phi(\cdot\mid\mathbf a)}
[C(Z)],
\]

or it might penalize high-cost tails, worst-case outcomes or violations of a chance constraint. These choices produce different behaviour. Expected cost can prefer a sequence with a small probability of catastrophic failure if most outcomes are cheap. A risk-sensitive objective can prefer a more conservative action.

The uncertainty estimate itself must be evaluated. A stochastic predictor does not guarantee calibrated probabilities. An ensemble does not automatically capture all uncertainty. A multi-hypothesis model can still omit a mode. The planner's risk criterion is only as reliable as the predictive distribution it receives.

## 10.7 Safety is not a latent distance

Suppose a planner minimizes only the distance between a predicted latent and a goal latent. Unless the representation and objective explicitly preserve and penalize unsafe states, the optimizer has no reason to avoid them.

A manipulation system may need explicit constraints on collision, force, velocity, joint limits and gripper state. A navigation system may need obstacle constraints and a margin around forbidden regions. These constraints can be imposed in the cost, in the action parameterization, through a separate safety filter, or by a combination of mechanisms.

The role of a world model is to predict consequences that the planner can evaluate. It does not decide by itself which consequences are acceptable. Nor does a low energy or a low feature distance imply a safe state.

## 10.8 What a physical planning result demonstrates

A successful physical planning experiment demonstrates that a complete system worked under the reported conditions. The system includes at least:

1. an observation encoder and state representation;
2. a predictor conditioned on relevant actions and state variables;
3. training trajectories that support the requested predictions;
4. a goal representation and task cost;
5. a planner and a specified optimization budget;
6. an execution interface and feedback loop;
7. an environment and a measured success criterion.

The V-JEPA 2-AC experiments are an example of this system-level evidence. The visual encoder is pretrained from passive data, the action-conditioned predictor is post-trained on robot trajectories, and an image-goal planner searches candidate action sequences. The reported physical outcome belongs to that complete configuration. (Assran et al., 2025) It should not be attributed to the visual encoder alone.

A responsible evaluation should distinguish simulated from physical results, familiar from unfamiliar environments, one-step from multi-step training, and matched from unmatched planner budgets. Success rates should be read with the number of trials, task definition and uncertainty across trials. A single aggregate score cannot certify safety or generalization to all objects, robots and environments.

A predictive model becomes useful for action only when it can be queried with interventions and integrated into an objective, search procedure and feedback loop. The final section examines the larger architecture proposed to organize these capabilities, including memory, multiple levels of prediction and the distinction between fast reactive behaviour and deliberative planning.
