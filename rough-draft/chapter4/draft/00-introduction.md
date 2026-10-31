# Chapter 4: Predicting in representation space

A world model must retain information that helps an agent anticipate what can happen next, distinguish consequences that matter, and evaluate possible actions before committing to them. Chapter 2 framed this as a state-representation problem. Chapter 3 examined how an encoder can learn useful features without state labels. This chapter turns to a more direct training signal: predict the representation of something that is not currently available to the model.

Joint-Embedding Predictive Architectures (JEPAs) encode a context and predict a target in a shared representation space. The target may be a hidden region of an image, a masked portion of video, a future frame, or a text embedding associated with a visual context. The prediction is evaluated in feature space rather than by reconstructing every pixel. The design changes what the model is asked to explain: pixel reconstruction rewards texture, lighting and fine detail even when these are not relevant to the prediction of interest. A latent predictor can instead focus on aspects of the target that are predictable from the context under the selected representation.

The word *predictive* needs care. In an image JEPA, a target region can be spatially withheld without being in the future. In a video JEPA, a target can be temporally later, spatially masked, or both. In an action-conditioned model, the predicted representation depends on a proposed action as well as the current observation. A JEPA is therefore not necessarily a temporal model, and a temporal JEPA is not automatically an action model.

The chapter proceeds in this order: the generic objective, image and video examples, distributional regularization, action-conditioned prediction, and finally the broader autonomous-agent proposal described by LeCun (2022). This progression teaches the predictive component before placing it inside a complete architecture.

## A world model is one component of an agent

An autonomous agent requires at least four capabilities that should not be conflated: a representation of the current situation; a model of how it may change; a way to assess outcomes against goals or costs; and a mechanism for selecting and executing actions. Memory and task configuration affect all four. A representation-learning objective may improve the first two, but does not by itself provide a reliable planner, cost function, controller, or memory system.

LeCun's *A Path Towards Autonomous Machine Intelligence* proposes one way these pieces could fit together. It is a position paper, not a report of one fully implemented system. Its six top-level modules are Perception, World Model, Cost, Actor, Short-Term Memory, and Configurator. The Cost module contains an Intrinsic Cost component and a Trainable Critic; these are not additional top-level modules. We preview the module names here and reserve the detailed relationships for Section 4.3, after action-conditioned prediction.

A representation that ignores the position of a moving object may be adequate for scene classification but inadequate for predicting a collision. A representation that preserves object identity but not velocity may support recognition while failing to distinguish histories with different futures. The training target, mask and evaluation protocol determine which distinctions are encouraged.

**Figure 4.1 - From an encoder to an agent.**

> **Figure description.** Arrange four stages from left to right: observation encoder, latent predictor, evaluator of predicted outcomes, and actor that selects an action. Beneath them, show memory and task configuration as cross-cutting support modules rather than layers in the prediction network. Distinguish internal information flow from the action sent to the environment and the new observation returned by it. The argument is that JEPA supplies a predictive representation mechanism, not the entire agent.

## Evidence boundaries

A mathematical result is valid under its stated assumptions. An empirical result belongs to the paper's data, architecture, training recipe and evaluation protocol. An architectural proposal may guide research, but is not established merely because its diagram is coherent.

I-JEPA demonstrates image representation learning with latent prediction. V-JEPA demonstrates feature prediction as a video pretraining objective. V-JEPA 2 separates action-free video pretraining from a later action-conditioned model trained with robot trajectories. LeJEPA develops a theory and objective for regularized embeddings. VISReg proposes a different regularizer separating scale and shape. These are related contributions, not interchangeable implementations of one fixed model.

Chapter 3 has already introduced contrastive learning, bootstrap methods, redundancy reduction, VICReg, masked reconstruction and collapse. We use those ideas as context rather than repeating their full derivations. The focus here is the predictive target, the encoder-predictor system, and the distributional constraints placed on its representation.
