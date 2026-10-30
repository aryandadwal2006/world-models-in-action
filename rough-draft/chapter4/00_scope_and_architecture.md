# 00 — Scope, reader contract and chapter handoff

## 1. What the reader should already know

Chapter 4 should assume the reader has encountered the following ideas.

### From Chapter 1
A world model is an internal predictive system used by an agent to reason about possible states and outcomes. Prediction and generation are related but distinct. A system can predict properties of a future without generating a full sensory observation. The chapter should briefly recall that distinction, not restate the entire introductory chapter.

### From Chapter 2
The reader has already seen the anatomy of a world model: observations, latent state, transition dynamics, reward/value and planning. More importantly, Chapter 2 has already introduced the state contract, temporal consistency, action grounding, planning utility and the limits of visual quality as a proxy for a useful model. Chapter 4 should use these concepts as tests for a JEPA representation.

### From Chapter 3
The reader already knows:
- self-supervised learning creates supervision from data;
- contrastive learning ranks positives against sampled negatives;
- BYOL and SimSiam use asymmetric prediction pathways;
- DINO matches teacher distributions;
- Barlow Twins and VICReg constrain feature statistics;
- masked methods differ in their target spaces;
- a representation can collapse completely or dimensionally;
- decorrelation is not the same as independence or semantic usefulness;
- probe results and pretraining losses do not establish predictive sufficiency;
- CPC predicts future latent candidates, but candidate retrieval does not demonstrate recursive rollout or planning.

The current Chapter 3 manuscript contains about 11,249 words, 17 inserted figures, one synthesis table and one short source-attributed official PyTorch VICReg listing. It already states explicitly that JEPA-based predictive representations are covered in Chapter 4.

## 2. What must not be repeated

Avoid:
- a second general introduction to self-supervised learning;
- a second full derivation of InfoNCE;
- method-by-method re-explanations of SimCLR, MoCo, BYOL, SimSiam, DINO and Barlow Twins;
- a full re-derivation of VICReg's variance and covariance terms;
- another long explanation of predictive sufficiency or aliasing;
- a general explanation of masked autoencoders;
- a generic survey of latent bottlenecks or projector heads.

A one-paragraph recap is justified only when a new method depends on the idea. For example, VICReg can be recalled in one compact section to establish the difference between second-order constraints and full distributional matching.

## 3. Chapter 4's unique conceptual contribution

Chapter 3 teaches that the target of an SSL objective determines which relations are rewarded. Chapter 4 asks a narrower and more consequential question:

**Can the system predict a representation of something it has not observed, using only a restricted context?**

This adds an explicit predictor over latent targets. For images the target may be a masked spatial region. For video it may be a space-time region. For controlled environments it may be a future representation conditioned on a selected action or action sequence.

That is the core transition:
- from matching two views to predicting withheld target features;
- from image features to spatiotemporal features;
- from passive future prediction to action-conditioned future prediction;
- from pretraining objectives to a model that can be tested inside a planning loop.

The final transition is not automatic. A JEPA loss does not itself supply a reward function, control constraints, a planner, an action interface or a guarantee of long-horizon correctness.

## 4. Vocabulary that should remain precise

| Term | Recommended meaning in this chapter | Avoid calling it |
|---|---|---|
| Context | The observations/tokens available to the prediction branch | The whole history unless it is actually supplied |
| Target | The selected representation that the predictor is trained to match | Ground truth state unless it is a measured state variable |
| Target encoder | The branch that computes target features | Always an EMA teacher |
| Predictor | Maps context features and target positions/conditions to predicted target features | A simulator unless it models transitions under actions |
| Mask | Defines which positions are input context and which are prediction targets | Merely data augmentation |
| Latent prediction | Prediction in a learned representation space | Guaranteed semantic or physical prediction |
| Action-conditioned | The action is an explicit input to the predictive model | Merely trained on videos that happen to contain actions |
| World model | A predictive model evaluated for downstream state/dynamics/planning use | Any encoder with a good representation score |

## 5. A three-level evidence ladder

The chapter should teach the reader to distinguish:

1. **Representation evidence:** frozen features support classification, retrieval, depth, anticipation or other downstream readouts.
2. **Predictive-model evidence:** the predictor matches held-out target features at a stated horizon, under a stated mask and distribution.
3. **Agent/planning evidence:** a model and planner produce successful action sequences in simulated or physical environments, under stated success criteria.

Evidence at one level does not prove performance at a higher level. Strong linear-probe accuracy does not establish action-conditioned dynamics. Low latent-prediction error does not establish safe plans. A planning result belongs to a complete model–planner–environment system, not to a feature encoder in isolation.

## 6. Proposed opening paragraph

A possible opening argument, to be rewritten in the author's final voice:

> Chapter 3 showed that a representation is shaped by the relationship an objective asks it to preserve. Contrastive methods identify matching views, non-contrastive methods match features or distributions, and masked methods predict withheld targets. A joint-embedding predictive architecture makes one further choice: instead of reconstructing the hidden observation, it predicts the hidden part's representation. This shifts the prediction target from pixels to learned features, but it does not remove the need to choose what those features should preserve. The rest of this chapter examines that choice, first for images and video, then for models conditioned on actions.

## 7. Chapter exit condition

At the end of Chapter 4, the reader should be able to say:

- what the JEPA predicts;
- which branch computes the target;
- which information the context branch can access;
- what the selected loss and mask actually constrain;
- which regularization method is being used;
- whether actions enter the predictive model;
- how the learned predictor would be evaluated before it is trusted for planning.

If these questions cannot be answered for a particular paper, describe that paper as an adjacent or emerging research line rather than forcing it into the core taxonomy.
