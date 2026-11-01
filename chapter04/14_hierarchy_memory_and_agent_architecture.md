# 14. Memory, hierarchy and an agent around the world model

An action-conditioned predictor supplies a model of how candidate actions may change a represented state. It still does not specify which actions should be tried, what outcomes count as desirable, how information from earlier observations should be recovered, or how behaviour should change when the task changes.

These responsibilities belong to the agent that uses the world model. LeCun's *A Path Towards Autonomous Machine Intelligence* proposes an architecture that makes them explicit. (LeCun, 2022) The paper is a research position and design proposal, not a report of a fully validated system implementing every component. Its value here is to clarify what must surround a predictive model if it is to support autonomous behaviour.

## 14.1 Six modules, six different responsibilities

The proposed architecture has six top-level modules:

1. **Perception** maps sensory observations into an internal state representation.
2. **World Model** predicts possible future states under imagined action sequences.
3. **Cost** assigns scalar energy to current and predicted states.
4. **Actor** proposes actions and action sequences.
5. **Short-Term Memory (STM)** stores and retrieves relevant past, current and predicted states, together with associated intrinsic costs.
6. **Configurator** configures the other modules for the task at hand.

The hierarchy inside Cost matters. The proposal places two components within the Cost module:

- **Intrinsic Cost** computes immediate intrinsic energy from a state. It represents basic drives or intrinsic quantities that are not learned in the same way as the critic.
- **Trainable Critic** predicts future intrinsic cost from states and experience.

Intrinsic Cost and the Critic are not additional top-level peers beside Perception, World Model, Cost, Actor, STM and Configurator. They are components of Cost.

This architecture separates several scalar functions that might otherwise be conflated. A JEPA prediction loss measures mismatch between a predicted and target representation. A compatibility energy scores whether a candidate configuration is compatible with a model. An agent-level cost evaluates whether a predicted state or trajectory is desirable. A critic estimates future cost. These scalars may interact, but they do not have the same meaning.

## 14.2 The information flow through the agent

At a high level, the agent operates as a closed loop.

An observation reaches Perception, which produces a state representation. The World Model uses that state, relevant memory and candidate actions to predict possible futures. Cost evaluates the predicted states. Actor uses those evaluations to select an action or action sequence. The selected action is sent to the environment, which produces a new observation. The loop repeats.

STM is not merely a record of every past frame. It provides an associative mechanism for retrieving information relevant to a current query. The World Model can use stored states when predicting or completing a state estimate, and the Critic can use retrieved experience and associated intrinsic costs to estimate future cost.

Configurator modulates the active modules according to the task. It can change what Perception emphasizes, what the World Model predicts, which costs matter and how the Actor behaves. The proposal treats this as an executive function, not simply as an extra task embedding appended to a fixed network.

The physical environment sits outside the learned computational graph. Gradients can be propagated through differentiable Perception, World Model, Cost and Actor components when optimizing imagined trajectories. The agent then executes an action in the real environment and receives a new observation. The architecture should not be interpreted as backpropagating gradients through the robot or the physical world itself.

## 14.3 Mode 1: reactive behaviour

Mode 1 selects an action directly from the current state, possibly using STM:

\[
a_t=A_\psi(s_t,m_t),
\]

where \(s_t\) is the perceived state and \(m_t\) denotes retrieved memory. The policy does not deliberately search through multi-step imagined action sequences for every decision.

Mode 1 can be computationally inexpensive at inference time. It can still be sophisticated, memory-dependent and task-configured. Its defining property is that action selection is reactive rather than the result of a fresh optimization over candidate trajectories through the World Model and Cost.

The World Model need not stop learning while the agent operates in Mode 1. It can continue to predict the next state, compare that prediction with the new observation, and use the observed transition as a learning signal. Reactive action selection and predictive learning are separate operations.

## 14.4 Mode 2: deliberate planning through imagined futures

Mode 2 uses the World Model as an internal simulator. A typical planning cycle is:

1. Perception estimates the current state and Cost evaluates its immediate energy.
2. Actor proposes an initial action sequence.
3. World Model predicts one or more possible state sequences under those actions.
4. Cost evaluates the predicted states or trajectory.
5. Actor modifies the proposed actions to reduce the predicted cost.
6. The system executes the first action or a short prefix.
7. A new observation updates the state estimate and the planning cycle is repeated.

For a deterministic model and an additive cost, the sequence objective can be written schematically as

\[
J(\mathbf a)
=
\sum_{k=1}^{H}C(\widehat s_{t+k})
+
\sum_{k=0}^{H-1}c_a(a_{t+k}).
\]

When the learned model and cost are differentiable, the system can optimize candidate actions by gradients through the imagined rollout. For discrete actions or non-differentiable objectives, it can use other search or planning procedures. The model is used to evaluate candidates internally; only a selected action is executed in the world.

Mode 2 is closely related to model-predictive control. Planning considers a finite horizon, but execution is followed by fresh observation and replanning. The distinction between an imagined action sequence and an executed action sequence is fundamental. The model may simulate many alternatives, but the physical environment experiences only the action that the Actor sends to it.

The proposal does not prescribe one universally correct optimizer. Gradient-based optimization, dynamic programming, tree search and sampling-based methods make different assumptions. Their effectiveness depends on the model, cost, action parameterization and available compute.

## 14.5 Distilling Mode 2 into Mode 1

Repeated deliberative search can be expensive. The proposal therefore allows the outputs of Mode 2 to train a reactive policy.

Let \(\widetilde a_t\) be an action selected by Mode-2 planning for state \(s_t\). A policy \(A_\psi\) can be trained to imitate these planner-generated actions:

\[
\mathcal L_{\mathrm{distill}}(\psi)
=
\mathbb E_{s_t}
\left[
D\!\left(\widetilde a_t,A_\psi(s_t)\right)
\right],
\]

where \(D\) is an action-space loss appropriate to the action representation. The resulting policy approximates the planner's decision with a direct state-to-action mapping.

This is a form of amortized inference. Mode 2 computes a relatively expensive answer for sampled states. Mode 1 learns a function that approximates those answers quickly. The learned policy can later operate reactively, or it can provide an initialization for another Mode-2 search.

The distillation target is not necessarily an expert action from an external teacher. It is an action that the agent's own model and cost considered desirable. Consequently, the distilled policy inherits the teacher planner's errors. If the World Model is biased or Cost is misaligned, policy distillation can make the same error faster and more consistently.

## 14.6 Short-Term Memory as associative state retrieval

A partially observed system may encounter the same physical configuration more than once without receiving identical observations. Relevant information may be absent from the current camera view, hidden behind an object, or distributed across observations separated in time.

An associative memory stores information in a form that can be retrieved by content. A query derived from the current state can retrieve a related past state and its associated value or intrinsic cost. The retrieved information can support state completion, prediction or evaluation.

This differs from simply appending a fixed number of recent frames to the current input. A temporal window retrieves information by adjacency in time. Associative memory retrieves information by similarity or a learned relation to the query. A past state can be relevant even if it is not the immediately preceding state.

The proposed memory stores information about past, current and predicted states, together with associated intrinsic costs. The World Model can query and update this memory. The Critic can retrieve past states and their observed costs while learning to estimate future intrinsic cost.

The paper leaves important implementation questions open: capacity, retrieval, write policy, key and value design, update rules and interference between stored states. These are research choices, not solved consequences of naming a module STM. The mathematical treatment of temporal state and recurrent memory belongs elsewhere in the book; the point here is the role associative retrieval plays in a complete predictive agent.

## 14.7 Hierarchical prediction and H-JEPA

A single predictor operating at one temporal resolution may have to represent both fine sensory changes and abstract long-term outcomes. A hierarchy separates these responsibilities.

Let \(s_t^{(0)}\) denote a fine-scale state representation. Higher levels \(s_t^{(1)},s_t^{(2)},\ldots\) describe increasingly abstract states or events over longer intervals. A level-\(k\) predictor may estimate an abstract target over horizon \(\Delta_k\):

\[
\widehat s_{t+\Delta_k}^{(k)}
=
F_k(s_{\le t}^{(k)},u_{t:t+\Delta_k}).
\]

The variable \(u\) at a high level need not be a physical motor command. It can encode a subgoal or a constraint that a lower-level model must satisfy. A high-level predictor may specify that an object should be moved to a region. A lower-level predictor then determines which sequence of intermediate states and physical actions can realize that subgoal.

This changes the planning problem. A high-level model can reason over a smaller number of abstract transitions, while lower-level models expand a subgoal into detailed predictions and actions. It can reduce the effective horizon and search complexity. However, abstraction is only useful if it preserves the constraints needed for execution. If a high-level representation discards a contact condition or obstacle, a lower-level controller may be unable to realize the proposed subgoal.

LeCun's H-JEPA discussion is a proposed direction for organizing prediction across levels and time scales. (LeCun, 2022) It should not be described as a single canonical architecture with a settled training algorithm. The specific representations, interfaces, temporal resolutions and methods for training levels remain design questions.

## 14.8 Multiple futures and latent variables

A deterministic predictor produces one future for a given context and action sequence. Yet the same action can lead to different outcomes because the system is partially observed, the environment is stochastic, or other agents intervene.

A latent variable \(z\) can index alternative futures:

\[
\widehat s_{t+1:t+H}
=
g_\phi(s_{\le t},\mathbf a,z).
\]

Different values of \(z\) can select different plausible continuations. In an energy-based formulation, the latent variable can select a configuration that reduces incompatibility between context and target. The latent should represent a restricted alternative, not an unrestricted copy of the target. Otherwise the model can transmit the answer through \(z\) and make prediction trivial.

A hierarchy can also represent uncertainty at several levels. A high-level model may retain alternative subgoals, while lower-level predictors estimate the detailed transitions associated with each subgoal. The agent can use Cost to rank these possibilities and avoid expanding every possible trajectory indefinitely.

This is a computational problem as well as a statistical one. If a model branches into many futures at each time step, the number of candidate trajectories can grow exponentially with horizon. The system must use structure, learned priors, costs or search procedures to allocate computation to the futures that matter.

The proposal does not prescribe one complete solution to uncertainty. Latent-variable models, ensembles, stochastic predictors and explicit distributions each represent different assumptions. A stochastic output is not automatically calibrated, and multiple hypotheses are not useful if they omit the dangerous or relevant mode.

## 14.9 The Configurator remains an open problem

The Configurator receives information from the other modules and changes how the system is organized for a task. LeCun describes it as executive control: it pre-configures Perception, World Model, Cost and Actor for the task, modulating the parameters and attention circuits of the modules it feeds. It can prime Perception to extract task-relevant information, configure the World Model for the current situation, and configure Cost to express the current goal. The Actor's policy is also described as configurable. The paper does not provide a finished algorithm for learning the Configurator.

This role is broader than selecting a task-specific output head. It is intended to configure how several components function together. Yet the proposal does not provide a finished learning algorithm for discovering these configurations or prove that such modulation will be stable. The design raises difficult questions about what a task description must contain, how modules should communicate, and how configuration should change as the environment changes.

The same restraint applies to the full architecture. The six-module design makes the responsibilities of an autonomous agent explicit, but it is not empirical evidence that one integrated implementation has solved them. The chapter's contribution is to show why a predictive representation is only one component in that system.

## 14.10 The role of a JEPA inside an agent

The architecture closes the argument developed throughout the chapter.

A representation-learning objective defines which views should agree and which target features should be predictable. A distributional constraint controls the geometry of the embedding space. An action-conditioned model predicts the consequences of candidate interventions. A cost or energy ranks the predicted states. An Actor searches or selects actions. Memory supplies information that is absent from the current observation. Hierarchy allows prediction and control to operate at different temporal scales. A Configurator adapts these mechanisms to a task.

No single feature-prediction loss provides all of these functions. The useful question is not whether a JEPA is itself an autonomous agent. It is what predictive information it supplies to an agent, and what other structures are required before that information can support reliable action.
