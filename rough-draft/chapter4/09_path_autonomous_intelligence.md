# 09 — A Path Towards Autonomous Machine Intelligence

## 1. Why include this paper

Yann LeCun's “A Path Towards Autonomous Machine Intelligence” (version 0.9.2, 27 June 2022) is a position paper that proposes an architecture and research agenda for autonomous intelligent systems. It is not a conventional empirical paper validating one end-to-end implementation. It should be included because it gives the JEPA idea a larger systems context: a predictive world model is one component of an agent that also needs goals, memory, action selection and a mechanism for configuring its behaviour.

The paper itself explicitly frames the document as a proposal and a vision, not a traditional scholarly paper claiming priority for every constituent idea. The chapter should preserve that distinction.

## 2. The proposed architecture

The high-level architecture contains:
- **Perception:** converts incoming sensory observations into internal representations.
- **World model:** predicts likely future states or representations under candidate actions.
- **Cost modules:** assess whether predicted outcomes are desirable or undesirable.
- **Actor:** proposes actions intended to reduce predicted cost.
- **Critic:** estimates the quality of states or action outcomes and can help evaluate plans.
- **Intrinsic cost/motivation:** supplies internally generated objectives where external reward is sparse or absent.
- **Short-term memory:** retains information that is not present in the current observation.
- **Configurator:** selects or adjusts sub-models and objectives depending on context and task.

The world model is not the whole architecture. Its value comes from how the rest of the system queries it and uses its predictions.

## 3. Hierarchical prediction

The paper argues for hierarchical representations and prediction at multiple levels of abstraction and time horizon. A low-level model may predict short-term sensory changes; a higher-level model may predict more abstract outcomes over longer horizons.

This offers a useful conceptual bridge to:
- multiscale temporal models in Chapter 5;
- latent dynamics and hierarchical rollouts in Chapter 6;
- planning and value estimation in Chapter 7.

Do not turn the hierarchical JEPA idea into a claim that a specific universally accepted H-JEPA architecture already exists. Present it as a proposed design direction unless citing a concrete paper with a validated implementation.

## 4. What it contributes to the JEPA chapter

The paper helps answer an important objection: even a good representation predictor does not create autonomous behaviour on its own. The agent must have:
- a state representation;
- a model of how candidate actions change that state;
- goals or costs that evaluate possible outcomes;
- a mechanism to choose actions;
- memory for partially observed variables;
- a way to update plans as new observations arrive.

This is a more intellectually honest ending than presenting JEPA as synonymous with general intelligence.

## 5. What it does not prove

The paper does not:
- demonstrate one integrated autonomous system that realizes every proposed component;
- prove that all world models should use one particular JEPA recipe;
- establish that latent prediction alone supplies causal understanding;
- show that a hierarchical predictor automatically solves long-horizon planning;
- settle how intrinsic motivation, cost learning, configurators or memory should be implemented.

Keep the paper's role as a conceptual architecture and research programme.

## 6. Figure and prose plan

**Figure 4.15 — A predictive model inside an agent.** Reproduce the architecture conceptually using an original diagram: perception feeds a world model; the actor proposes actions; the model predicts; cost and critic evaluate; short-term memory supplies hidden context; configurator adjusts the active modules. Use distinct arrows for observation, action, prediction and evaluation. The figure must be an original synthesis, not a traced copy of the paper's illustration.

Use one short prose bridge:
- Chapters 1–2 introduced the world-model loop;
- Chapter 4 has shown how a JEPA learns a predictive representation;
- the architecture paper explains why prediction must sit inside a larger action-and-goal system;
- Chapters 5–7 develop temporal state, dynamics and planning in detail.

## 7. Bibliographic identity

Yann LeCun, *A Path Towards Autonomous Machine Intelligence*, version 0.9.2, 27 June 2022. OpenReview: https://openreview.net/forum?id=BZ5a1r-kVsf

The OpenReview PDF is the primary reading source. Cite it as a position paper and do not cite a blog summary in place of the original.
