# 15. The agent around the world model

A transition predictor does not define a complete agent. The agent also needs a way to perceive the environment, choose objectives, select actions, retain relevant history and decide when to plan. LeCun's proposal for autonomous machine intelligence is useful because it makes these responsibilities explicit. It should be read as a research architecture and position paper, not as evidence that the complete system has been validated end to end. (LeCun, 2022)

## 15.1 Six responsibilities, not six interchangeable networks

The proposal describes six top-level modules:

**Perception** maps sensor observations into a representation of the current situation. Its output should expose the distinctions needed by prediction and control, rather than merely compressing the raw input.

**World Model** predicts how the represented situation may evolve. It may represent multiple possible futures and should be able to condition those futures on actions or other variables that affect the transition.

**Cost** evaluates predicted states and trajectories according to goals, preferences and constraints. It supplies the criterion by which imagined futures are compared.

**Actor** selects actions. It can implement a direct policy or participate in a deliberative planning process.

**Short-Term Memory** retrieves information from previous observations when the current input is insufficient to determine the relevant state.

**Configurator** specifies the task, subgoals and operating mode of the agent, and configures the other modules for the current objective.

The physical environment remains outside this internal module list. It generates observations and receives actions. This distinction prevents an architecture diagram from suggesting that the World Model directly changes the world.

The names do not imply that each module must be a separate neural network. A practical implementation may share parameters or combine components. The purpose of the decomposition is to separate responsibilities that have different objectives and failure modes.

## 15.2 Prediction and evaluation must remain separate

The World Model should answer what may happen under a proposed action. Cost should answer whether that predicted outcome is desirable. The distinction can be written as

\[
\widehat S_{t+1}=F_\phi(S_t,A_t),
\qquad
J(A_{t:t+H-1})
=
\sum_{k=0}^{H-1}c(\widehat S_{t+k},A_{t+k})
+
V_{\mathrm{term}}(\widehat S_{t+H}).
\]

The transition model \(F_\phi\) describes consequences. The cost \(c\) evaluates them. The same transition model can be reused for different tasks by changing the objective, provided the representation and action interface retain the information those tasks require.

LeCun places Intrinsic Cost and a Trainable Critic within the Cost module. Intrinsic Cost represents built-in preferences or constraints; the Trainable Critic estimates task-related cost from experience. They are subcomponents of Cost, not additional top-level modules. This separation allows some preferences to be specified independently of a learned task value.

The decomposition also exposes two distinct failure cases. If the model predicts the wrong future, a correct cost cannot repair the transition. If the model predicts accurately but the cost function rewards the wrong outcome, the planner can select an undesirable action. The two modules require separate evaluation.

## 15.3 Mode 1: direct reactive behaviour

Mode 1 maps a current representation directly to an action:

\[
A_t=\pi_\psi(S_t).
\]

It does not need to search over imagined futures at every decision. A trained reactive policy can act quickly and with relatively low computational cost. This is useful when the control loop must run at high frequency or when planning is too expensive for every step.

The limitation is that the policy must have learned an appropriate response for the states and tasks it encounters. It does not explicitly compare a new set of candidate trajectories when the goal changes. Its behaviour outside the training distribution depends on generalization rather than on an online search over predicted consequences.

## 15.4 Mode 2: deliberation through imagined futures

Mode 2 uses the World Model and Cost to evaluate candidate action sequences. It searches for actions whose predicted consequences have low cost. The planner can adapt to a new goal or a changed objective without retraining the reactive policy.

This flexibility has a computational price. Search requires repeated model evaluations, and the chosen action depends on the accuracy of the imagined trajectories. The planner may also exploit prediction error, as described in the preceding section. Mode 2 is not automatically more reliable than Mode 1; it is more explicit about the candidate futures it evaluates.

The two modes therefore solve different operating problems. Mode 1 is inexpensive at execution but may be less flexible. Mode 2 can deliberate over alternative futures but consumes more computation and depends on the model being reliable in the regions explored by the search.

## 15.5 Distilling deliberation into a reactive policy

A planner can generate training targets for a faster policy. Suppose Mode 2 chooses an action \(A^{\mathrm{M2}}(S)\) after searching from state \(S\). A reactive policy can be trained to imitate those decisions:

\[
\mathcal L_{\mathrm{distill}}
=
\mathbb E_{S\sim\mathcal D_{\mathrm{M2}}}
\left[
\|\pi_\psi(S)-A^{\mathrm{M2}}(S)\|_2^2
\right].
\]

The resulting Mode 1 policy approximates the planner over the states represented in the distillation data. It can reduce the computational cost of repeatedly solving the planning problem.

Distillation does not transfer a guarantee of correctness. If Mode 2 exploits model errors, the selected actions can be wrong, and the reactive policy can learn to reproduce them. If the deployment state distribution differs from the distillation distribution, Mode 1 may not reproduce the planner's behaviour reliably. The improvement is computational: decisions obtained through search are converted into a faster mapping. It is not a proof that the policy is optimal outside the sampled states.

## 15.6 Short-Term Memory as associative retrieval

The current observation may not contain enough information to infer the relevant state. An object can be temporarily occluded; an earlier action can determine whether a container is open; a task may require remembering which subgoal has already been completed.

The Short-Term Memory in LeCun's proposal is associative rather than merely chronological. A key derived from the current context is used to retrieve stored content associated with similar past situations. Abstractly,

\[
M_t=\operatorname{Retrieve}(K_t,\mathcal M),
\]

where \(K_t\) is the current retrieval key and \(\mathcal M\) is the memory. Retrieved content can be combined with the current representation before prediction or planning.

Associative retrieval is useful when the present observation is ambiguous but a past episode contains relevant information. It also introduces a failure mode: the key must retrieve memories based on task-relevant similarity. If the representation ignores the distinction that determines which memory is relevant, the system may retrieve a plausible but incorrect episode. Memory therefore inherits the invariances and omissions of the perception system.

The architecture leaves important implementation questions open. What should be stored? How should keys and values be formed? When should a memory be updated or forgotten? How should the agent distinguish a genuinely similar situation from one that only looks similar in the current representation? These are part of the memory problem, not details solved by assigning a module name.

## 15.7 Hierarchical prediction separates time scales

A single predictor operating at the frame or control-step level may be inefficient for tasks with long horizons. A hierarchy can predict at several temporal scales. A high-level predictor proposes an abstract subgoal or a compressed future segment; a lower-level predictor estimates the intermediate states and actions required to realize it.

Let \(S_t^{(0)}\) denote a low-level state representation. A higher-level representation can summarize a longer interval:

\[
S_k^{(1)}=G_1(S_{t:t+\Delta}).
\]

A high-level predictor can then estimate a temporally extended target:

\[
\widehat S_{k+1}^{(1)}
=
F_1(S_k^{(1)},A_k^{(1)}).
\]

A lower-level predictor refines this target into shorter-horizon states and actions. The abstraction reduces the need for one model to predict every low-level detail over the entire horizon.

The hierarchy also creates a consistency requirement. The high-level prediction must be realizable by the lower-level controller. If the high-level representation discards a constraint needed for execution, the lower-level model may be unable to realize the proposed subgoal. Hierarchical JEPA is therefore not simply a stack of identical encoders. Each level needs a target and temporal abstraction appropriate to its role, together with a relationship between levels that preserves executability.

## 15.8 The Configurator is an open problem

The Configurator is the least specified of the six modules. In the proposal, it configures the task, goals, constraints and relation between reactive and deliberative operation. It is intended to determine what the agent should work on and how the other modules should be arranged for that task.

That role raises questions which are not solved by drawing a box. Where do goals come from? How are constraints represented? How does the agent know when the task has changed? How are conflicts between intrinsic preferences and externally specified objectives resolved? What prevents the Configurator from selecting a convenient but inappropriate objective?

LeCun's proposal makes this responsibility explicit, which is useful for reasoning about a general agent. It does not supply a complete general algorithm for the Configurator or demonstrate that the six modules form a reliable autonomous system.

## 15.9 What a JEPA contributes

A JEPA contributes a representation and a predictive relation. In an action-free visual model, it can supply features that summarize a scene and support prediction of withheld or future representations. In an action-conditioned system, a separate predictor can estimate consequences under proposed controls. The agent still requires a cost, a policy or planner, feedback, memory and a mechanism for configuring the task.

The architectural conclusion is that representation prediction is one responsibility inside a larger agent. Its value depends on whether the representation preserves the distinctions needed by the predictor and whether the predictor remains reliable over the actions considered by the planner.

**Figure 4.25. The proposed agent separates six responsibilities.**

Description: Perception supplies representations to the World Model. The World Model predicts imagined states, which are evaluated by Cost. Actor selects actions, Short-Term Memory retrieves relevant past information, and Configurator specifies tasks and modes. Intrinsic Cost and Trainable Critic are shown inside Cost, not as additional top-level modules.

**Figure 4.26. Mode 1 executes a reactive policy; Mode 2 searches imagined futures.**

Description: Mode 1 maps a representation directly to an action. Mode 2 uses the World Model and Cost to compare action sequences. Planner-selected actions can train a cheaper Mode 1 policy, but distillation transfers the planner's behaviour only over the training distribution.

**Figure 4.27. Hierarchical prediction separates temporal scales.**

Description: A high-level predictor proposes an abstract subgoal or temporally extended state. A lower-level predictor refines it into intermediate states and actions. The hierarchy reduces the need for one predictor to represent every detail at every time scale.