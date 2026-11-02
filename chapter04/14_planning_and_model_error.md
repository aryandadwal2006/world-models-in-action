# 14. Planning with a learned predictor

A transition model becomes useful for control when it can evaluate the consequences of candidate actions. The planner supplies the objective that distinguishes desirable futures from undesirable ones. Prediction and planning are linked through imagined trajectories, but they optimize different quantities.

## 14.1 From a one-step transition to a trajectory

Suppose the model receives a state \(S_t\) and an action \(A_t\), and predicts the next state:

\[
\widehat S_{t+1}=F_\phi(S_t,A_t).
\]

Applying the predictor repeatedly gives an imagined trajectory under an action sequence:

\[
\widehat S_{t+k+1}
=
F_\phi(\widehat S_{t+k},A_{t+k}).
\]

The model can therefore evaluate a sequence before the agent executes it. For a finite horizon \(H\), a general planning objective is

\[
\min_{A_{t:t+H-1}}
\mathbb E
\left[
\sum_{k=0}^{H-1}
c(\widehat S_{t+k},A_{t+k})
+
V_{\mathrm{term}}(\widehat S_{t+H})
\right].
\]

The immediate cost \(c\) evaluates states and actions during the imagined trajectory. The terminal value \(V_{\mathrm{term}}\) estimates the value beyond the explicit rollout. Without a terminal term, a finite-horizon planner can prefer a trajectory that makes quick progress during the search window but ends in a poor state just beyond it.

The objective is not the same as the feature-prediction loss. The predictor is trained to match observed transitions. The planner minimizes a task cost evaluated on predicted states. A low transition error does not imply that the task cost is low, and a low task cost under the model does not guarantee that the real environment will produce the imagined state.

## 14.2 Image goals and feature-space distance

In image-goal control, the goal may be an image \(X_g\) rather than a vector of physical coordinates. An encoder maps the current observation and goal image into features:

\[
Z_t=E(X_t),
\qquad
Z_g=E(X_g).
\]

A simple terminal cost is a feature discrepancy,

\[
c_{\mathrm{goal}}(\widehat Z_{t+H},Z_g)
=
\|\widehat Z_{t+H}-Z_g\|_1.
\]

This cost is useful only to the extent that the representation preserves distinctions relevant to the goal. If two visually different states are mapped to similar features, the planner may consider them interchangeable. If a feature distance ignores contact, collision or object orientation, it may reward an unsafe or incomplete outcome.

Goal matching is not a safety specification. Collision constraints, actuator limits and force limits must be represented separately when they matter. A latent distance can rank visual similarity; it does not certify that the path between two feature vectors is physically safe.

## 14.3 Searching over action sequences

The planner must solve an optimization problem over a sequence of actions. Gradient-based optimization is possible when the model and action parameterization are differentiable, but it can be sensitive to non-smooth constraints and model errors. Sampling-based methods search by evaluating candidate sequences.

The Cross-Entropy Method (CEM), used in V-JEPA 2, maintains a distribution over action sequences. It samples candidates, evaluates their predicted costs, selects an elite subset with the best scores, and updates the sampling distribution toward those candidates. Repeating this procedure concentrates samples in regions of action space that the model predicts to be promising.

The search budget matters. More samples and iterations can find lower-cost trajectories under the model, but the model may be inaccurate for the states and actions that the optimizer discovers. A stronger optimizer can therefore make real-world performance worse if it is allowed to exploit errors in the learned transition model.

The planner is optimizing the model's prediction, not the true environment. This is the source of model exploitation.

## 14.4 Receding-horizon feedback

An open-loop plan would execute the entire optimized action sequence without observing the environment again. Small model errors can then accumulate over the full horizon. A common alternative is receding-horizon control:

1. Observe the current environment and encode its state.
2. Optimize a candidate action sequence over a finite horizon.
3. Execute only the first action or short initial segment.
4. Receive a new observation and update the current state.
5. Replan from the new observation.

The next observation corrects some of the prediction error before the remaining actions are executed. This feedback loop does not make the model exact, and it does not guarantee safety. It limits how long the controller relies on one open-loop imagined trajectory.

The horizon remains a trade-off. A short horizon limits accumulated prediction error but may miss actions whose benefit appears later. A long horizon can represent delayed consequences, but the predictor must remain accurate through repeated application and the search becomes more demanding.

## 14.5 Why optimization can make model error worse

Let the true transition be \(F^\star\) and the learned model be

\[
F_\phi(s,a)=F^\star(s,a)+\epsilon_\phi(s,a),
\]

where \(\epsilon_\phi\) denotes prediction error in an appropriate state representation. If the planner searches for actions that minimize model-predicted cost, it may find regions where \(\epsilon_\phi\) makes the predicted outcome look unusually good.

This is not random evaluation error. The optimizer actively selects actions based on the model's outputs. Even if prediction errors are small on average under the training distribution, the minimum predicted cost across many candidates can be biased toward sequences with optimistic errors.

The effect intensifies with search. More samples increase the chance of finding a sequence that exploits a model defect. The optimizer does exactly what it is designed to do: it finds the best candidate according to the supplied model. The failure is that the model is not accurate enough over the region the optimizer explores.

Mitigations include collecting data near planner-proposed states, penalizing uncertain predictions, using ensembles to identify disagreement, constraining candidate actions to supported regions, and replanning frequently. Each helps only under assumptions. Ensemble agreement does not prove correctness if all models share the same bias. A penalty on uncertainty is useful only if the uncertainty estimate is calibrated enough to detect unfamiliar states. Restricting actions to the data support can reduce extrapolation but may also exclude useful behaviours.

## 14.6 Multiple futures and uncertainty

For a fixed state and action, the next state may be stochastic because of unobserved variables, sensor noise or genuinely random dynamics. A deterministic predictor trained with a pointwise loss estimates a statistic of the conditional target, such as a mean under squared error or a median under absolute error. It may not preserve several distinct outcomes.

For planning, uncertainty matters when different plausible futures have different costs. One option is to predict a distribution or several latent hypotheses and evaluate candidate actions across those outcomes. Another is to maintain an ensemble of transition models and examine their disagreement. These are different approximations to uncertainty; neither guarantees that all relevant futures have been represented.

A model that outputs one plausible feature sequence can still be useful in near-deterministic settings. In ambiguous settings, averaging incompatible futures can produce a feature vector that corresponds to no realizable state. The planner must then be designed around the output semantics of the predictor rather than treating every predicted vector as a physical state.

## 14.7 Prediction quality, task value and safety

Three evaluations should remain separate.

**Transition accuracy** measures whether the model predicts observed or intervened outcomes.

**Task value** measures whether the imagined outcome satisfies the goal or cost function.

**Safety** measures whether the trajectory respects constraints on collision, force, workspace, velocity or other hazards.

A model can be accurate while the planner chooses an undesirable action because its cost is wrong. A model can predict a goal-like final image while the intermediate trajectory collides with an obstacle. A trajectory can satisfy a learned feature cost while violating an unmodelled physical constraint.

This is why a world model does not replace a task specification or a safety layer. It supplies imagined consequences that other components use. Closed-loop deployment must evaluate the full system under the hazards and constraints of the actual environment.

**Figure 4.22. Receding-horizon planning executes an action and then replans.**

Description: A planner samples and evaluates action sequences using a learned transition model. It executes only the first action, obtains a new observation and repeats the optimization from the updated state.

**Figure 4.23. Search can exploit optimistic prediction errors.**

Description: Several candidate trajectories have similar true outcomes, but one is assigned an artificially low model-predicted cost. An optimizer selects that trajectory because it appears best under the model, even though it fails when executed in the environment.

**Figure 4.24. A point prediction can hide distinct plausible futures.**

Description: One state-action pair leads to several separated outcome modes. A deterministic point estimate may lie between them, whereas a distribution or multiple hypotheses can preserve the alternatives relevant to planning.