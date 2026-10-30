# 08 — Action-conditioned JEPA world models and planning

## 1. Passive prediction is not controlled prediction

A video model trained without action labels can learn powerful representations of what tends to happen in observed clips. A model used for planning must answer a different question:

**What is likely to happen if the agent selects this action?**

A simple controlled latent transition is:
\[
\hat z_{t+1}=g_\phi(z_t,a_t).
\]

Here \(a_t\) is an explicit action or intervention. Real models may condition on a sequence of actions, robot state, proprioception, action chunks or tokenized control signals. The important requirement is that the predictor receives information that distinguishes the candidate intervention.

Training on passive video does not automatically identify action effects. Correlations in observed trajectories can reflect the behaviour policy, environment dynamics, camera motion or hidden confounders. A model intended for control needs an action interface and data whose action/observation alignment supports the desired prediction.

## 2. V-JEPA 2-AC

V-JEPA 2-AC is the action-conditioned post-trained model associated with V-JEPA 2. The V-JEPA 2 paper separates:
- large-scale action-free video pretraining for the visual encoder;
- post-training on robot trajectory data to create a latent predictor conditioned on actions;
- planning from image goals using the resulting predictive model.

The paper reports zero-shot deployment on Franka arms in new labs/settings using image goals, without environment-specific robot data collection or task-specific training/calibration in those deployment environments. Preserve the exact scope of the experimental setup and never shorten this to “V-JEPA 2 can control any robot.”

The pedagogical distinction is central to the entire book: a generic representation encoder and an action-conditioned world model solve different problems.

## 3. Energy-based interpretation

An energy-based model assigns a scalar compatibility score to a candidate configuration, such as a predicted future representation under a context and action. Lower energy can mean greater compatibility under the learned model.

In latent planning, one can evaluate candidate actions or action sequences by the predicted latent outcome and a goal/cost function. An energy surface is not a policy by itself. It must be combined with:
- a candidate action parameterization;
- an optimizer or search procedure;
- constraints and action bounds;
- a termination/replanning policy;
- validation that low model energy corresponds to desirable outcomes in the real environment.

The official EB-JEPA repository is a teaching/reference library with image, video and action-conditioned examples and planning notebooks. Treat it as an implementation ecosystem and set of examples, not automatically as a single canonical peer-reviewed method with one universal recipe.

## 4. JEPA-WMs

Terver et al.'s “What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?” (arXiv:2512.24497; updated in 2026) is especially relevant to the book's main theme. It studies design choices that determine whether planning in learned representation spaces works in simulated and real robotic tasks. It compares JEPA-WM models with baselines including DINO-WM and V-JEPA 2-AC.

The official repository provides code, data and checkpoints for environments including DROID/RoboCasa and Meta-World, with additional benchmark variants. It is a useful source for:
- the model architecture;
- target and prediction depth;
- the planning objective;
- the relationship between encoder choice and planner performance;
- environment-specific configuration;
- failure cases and reproducibility constraints.

The paper's key lesson is experimental: success depends on the model architecture, objective and planning algorithm together. It is not a proof that latent-space planning is always better than pixel-space planning or that every JEPA representation is suitable for control.

## 5. What a planning demonstration must report

A reader should see the whole system:
1. observations and robot/environment state;
2. encoder producing current latent features;
3. candidate action sequence or action chunk;
4. action-conditioned predictor;
5. predicted future latents;
6. goal/cost evaluation;
7. action selection and execution;
8. fresh observation and replanning.

A responsible evaluation should report:
- number and diversity of training trajectories;
- action representation and temporal granularity;
- input resolution and frame rate;
- predictor horizon/depth;
- planner search/optimization budget;
- target/goal representation;
- success rate and uncertainty across seeds or trials;
- physical deployment details and failure analysis;
- whether the environment was seen during training.

## 6. Why latent planning might help, and why it might fail

**Potential benefit:** a representation can suppress irrelevant appearance variation and give the planner a smoother, lower-dimensional objective.

**Potential failure modes:**
- the encoder discards a small but action-critical feature;
- the predictor averages incompatible futures;
- action labels are misaligned with visual observations;
- the learned model is inaccurate under actions not seen during training;
- long-horizon errors compound;
- a planner exploits model errors by searching for action sequences that look good only to the model;
- the learned cost or goal representation is poorly aligned with physical success;
- the policy violates safety or actuation constraints.

This is a natural bridge to Chapter 6's rollout stability and Chapter 7's planning algorithms. Introduce these issues here, but do not repeat later chapters in full.

## 7. Recommended figure

**Figure 4.14 — From image-goal planning to executed action.** Show current observation, latent encoder, action-conditioned predictor, candidate action sequence, predicted future latent, goal distance, selected action, environment feedback and replanning. Distinguish a one-step model call from a complete closed-loop agent.

## 8. Sources

- Assran et al. (2025), V-JEPA 2, arXiv:2506.09985.
- Terver et al. (2025/2026), “What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?”, arXiv:2512.24497.
- Official V-JEPA 2 code: https://github.com/facebookresearch/vjepa2
- Official EB-JEPA examples: https://github.com/facebookresearch/eb_jepa
- Official JEPA-WMs code/data: https://github.com/facebookresearch/jepa-wms
