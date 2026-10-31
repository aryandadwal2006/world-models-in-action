# Chapter 4 design: JEPA, predictive representations and autonomous intelligence
**Revision:** 1 November 2026. **Status:** editorial blueprint, not chapter prose.

## 1. Decisions
- Retain the approved top-level TOC: 4.1 Why predict in latent space; 4.2 JEPA; 4.3 Action-conditioned predictive representations; 4.4 Implementing a miniature JEPA system; 4.5 Summary.
- Core: image/video JEPA, LeJEPA/SIGReg, VISReg, action conditioning, and LeCun's proposed autonomous-agent architecture.
- After regularization, discuss only VL-JEPA for 1.5–2 pages. At most one original figure. Mention A-JEPA, Audio-JEPA, Var-JEPA and VJEPA briefly in a paragraph or compact row. No detailed tutorials or proof work for them.
- Introduce energy/EBM compatibility in Chapter 4 because it is essential to LeCun's Cost module. Do not relocate the whole EBM theory here. Propose formal EBM foundations in Chapter 8 and cost-based action optimization in Chapter 7. The current TOC has no explicit EBM heading; this is a proposed addition.
- Explain the important concepts in A Path Towards Autonomous Machine Intelligence across Chapters 4–7. In Chapter 4, explain the six-module system, Mode 1/Mode 2, distillation from Mode 2 to Mode 1, intrinsic cost and critic, associative STM, latent-variable futures, H-JEPA, hierarchical planning and configurator.
- No new experiments, model training, benchmark replications or book-generated results. Use only paper-reported results and source-attributed reading of official implementations. At most 3–4 short code snippets/pseudocode. Code inspection is not reproduction.
- Target 65–72 printed pages. Let the proofs take the space they need, but do not pad with modality surveys.

## 2. Reader contract and book boundary
Chapters 1–2 already introduce world models, state, dynamics, memory, reward/value and planning. Chapter 3 already covers SSL, contrastive and non-contrastive learning, BYOL/SimSiam, DINO, Barlow Twins, VICReg, masked prediction, bottlenecks, collapse and predictive sufficiency. Do not repeat those foundations.

Chapter 4 adds the JEPA context/target/predictor graph; I-JEPA; V-JEPA, V-JEPA 2, 2-AC and 2.1; deep distributional regularization; a brief VL-JEPA example; action-conditioned prediction; and LeCun's proposed agent architecture.

Later homes:
- Ch. 5: persistent/recurrent temporal state, S4/Mamba; compare, but do not conflate, associative STM and SSM state.
- Ch. 6: latent transition functions, stochasticity, uncertainty, rollout stability, multistep prediction and hierarchical time scales.
- Ch. 7: costs, critic/value, actor, MPC, action-sequence optimization, search and hierarchical planning.
- Ch. 8: VAE/AR/GAN and add a proposed formal section on energy-based models: energy vs normalized probability, latent inference, partition function, EBM learning/inference and comparison with other generative objectives.
- Chs. 9–12: diffusion, interactive video, robotics, scaling and open problems.

## 3. Where EBMs belong
Chapter 4 should teach the distinction between (i) a predictive model that estimates a target embedding, (ii) an energy that scores compatibility of a candidate configuration, and (iii) a task cost that scores predicted outcomes for action selection. Energy is not automatically a normalized probability.

A latent prediction discrepancy may be interpreted as one possible energy:
\[
E_\theta(x,y,z)=D(s_y,\operatorname{Pred}_\phi(s_x,z)),
\]
where \(s_x,s_y\) are context/target embeddings and \(z\) is a latent configuration. Minimizing over \(z\) is not the same as probabilistic marginalization. Do not imply all JEPA recipes use one EBM objective.

Keep this conceptual bridge to about 2–3 pages. Chapter 7 develops cost-based planning; Chapter 8 should carry EBM foundations. EB-JEPA is a useful small-scale library/workshop-paper example, not the whole EBM field. Its paper reports CIFAR-10 representation learning, Moving-MNIST prediction and 97% success on its specific Two Rooms benchmark. Two Rooms is simulated/procedural, not physical robotics. The reported planner uses a simple latent-distance cost; learned task-specific costs and hierarchical models are listed as future directions. Attribute these results only to the exact paper and evaluation.

## 4. Proposed internal structure and pages

### 4.1 Why predict in latent space — 3–4 pp
Contrast reconstruction with latent prediction. Show that unpredictable appearance may be irrelevant for one task but crucial for another. The target encoder defines which distinctions the loss rewards; latent prediction is not universally superior.

### 4.2 JEPA — 34–39 pp
1. **Generic architecture/objective (5–6):** define context, target, encoders, predictor, positions and feature loss. Specify visibility and information flow. Label the equation generic, not universal.
2. **I-JEPA (3–4):** patches, context, multiple target blocks, position, target update and mask geometry. Distinguish frozen-feature evaluation from fine-tuning.
3. **V-JEPA (3–4):** spatiotemporal tokens/masks and feature prediction. Action-free video training does not identify the effects of chosen actions automatically.
4. **Target branches/MSE (2–3):** distinguish EMA, stop-gradient and other updates. Derive conditional mean under squared error; a multimodal mean may fall between plausible futures.
5. **V-JEPA 2 and 2-AC (4–5):** separate action-free video pretraining, robot-trajectory post-training with actions, and planning. Attribute planning to the complete model/planner/environment, not the visual encoder alone.
6. **V-JEPA 2.1 (2–3):** dense feature targets, token losses and intermediate-layer supervision, according to the paper. Distinguish this from action conditioning.
7. **Regularizers (13–16):**
   - VICReg recap at most one page: invariance, variance, covariance. Do not repeat Chapter 3's derivation.
   - LeJEPA (3–4): linear-probe bias/variance motivation; nonlinear radius-based k-NN/kernel probes; integrated-square-bias criterion; scalar covariance/total-variance constraint; theorem assumptions. State only the theorem's exact Gaussian optimum within its admissible distribution class and specified probe risk. Not universal optimality for every task.
   - SIGReg (3–4): characteristic functions, one-dimensional projections, Epps–Pulley discrepancy, numerical quadrature, minibatch effects and projection count. Cramér–Wold is all-directions; finite projections are sketches. Distinguish formal max-over-directions test from averaged training objective. Keep projection rates attached to smoothness assumptions.
   - VISReg (4–5): separate regularizer components: centering, per-dimension scale and shape. Prediction/invariance is a separate term combined with regularization. Explain detached standard-deviation denominator in shape normalization; this is not EMA/teacher. Explain sliced-Wasserstein via sorting and Gaussian quantiles. Derive sorting identity only for equal-weight/equal-size samples. Distinguish scalar scale derivative from full network parameter-gradient guarantees. Complexity: report projection cost in B,d,K and sorting separately.
   - Compare methods and limitations; no universal winner.
8. **VL-JEPA/other directions (2–2.5 total):** after regularization. VL-JEPA 1.5–2 pp: visual input plus text query predicts continuous text embedding; optional lightweight decoder generates text when needed. Contrast with autoregressive token prediction; one or two sourced results only. It is not an action-conditioned physical dynamics model. Optional original architecture figure. All other modalities/variational lines get only one paragraph/row.

### 4.3 Action conditioning and LeCun proposal — 17–20 pp
1. Passive video versus intervention (2–3): \(\hat z_{t+1}=g_\phi(z_t,a_t)\), action/observation alignment, demonstrations and why passive correlations do not identify action effects.
2. V-JEPA 2-AC, JEPA-WMs and EB-JEPA (3–4): distinguish data, action inputs, predictor, planner and evaluation. Do not mix simulated and physical evidence.
3. LeCun's architecture (5–6): explain its **six** top-level modules:
   - Perception: maps observations to a state representation.
   - World Model: predicts futures under imagined action sequences.
   - Cost: scalar energy/discomfort module containing Intrinsic Cost (imposed/non-trainable immediate intrinsic quantities) and Critic/Trainable Cost (predicts future intrinsic cost).
   - Actor: proposes/optimizes action sequences.
   - Short-Term Memory: stores current/predicted states and intrinsic costs.
   - Configurator: configures perception, model, cost and actor for the task.
   Intrinsic Cost and Critic are inside Cost, not extra top-level peers. Explain flow: percept→Perception; state→World Model/Actor/Cost; Actor proposals→World Model; predictions→Cost; Cost signal→Actor; selected first action→environment; new percept closes loop; STM read/write; Configurator modulation. Internal modules may be differentiable; the physical environment is not generally differentiable.
4. Mode 1/Mode 2 (2–3): Mode 1 reactive policy without rollout optimization at every decision; Mode 2 proposes action sequences, imagines futures, scores with Cost, optimizes/searches, executes first action(s), observes and replans (MPC-like). These are proposed modes, not validated cognitive science.
5. Mode-2-to-Mode-1 distillation (1): amortized inference from expensive plans to cheaper reactive skills.
6. Intrinsic cost, critic and STM (2–3): example memory tuple (time,state,intrinsic energy); critic predicts future intrinsic cost. Compare associative STM with SSM recurrence without teaching full memory mathematics here.
7. Multiple futures and H-JEPA (2–3): latent variables for possible outcomes; lower short-horizon/detail levels, higher abstract/long-horizon levels, and high-level subgoals refined into lower-level action sequences. Label hierarchy as proposal.

### 4.4 Implementation — 4–5 pp
Reframe the approved heading as official implementation reading, not author-run reproduction:
- generic pseudocode explicitly labelled;
- one official I-JEPA mask/target excerpt;
- one compact official EB-JEPA example;
- optionally one regularizer excerpt if exact source, paper correspondence and commercial-book license are verified;
- explain limitations of code inspection and checkpoints.
At most 3–4 snippets. No new experiments or benchmark claims.

### 4.5 Summary/exercises/references — 3–4 pp
Synthesize target choice, mask, regularization limits, VL-JEPA, action conditioning, energy/cost and the modules needed around prediction. State that LeCun's integrated architecture is a proposal.

## 5. LeCun concept coverage across the book
| Concept | Chapter 4 | Full development |
|---|---|---|
| Learning by observation, differentiable reasoning, hierarchy | Motivate architecture | Ch. 12 open problems |
| Perception/state | Module/flow | Ch. 2; Ch. 11 |
| Configurator | Purpose and open issue | Ch. 12 |
| World model/futures | JEPA bridge | Ch. 6; Ch. 10 |
| EBM/energy | Compatibility, energy vs probability | Ch. 8; planning cost in Ch. 7 |
| Cost, intrinsic cost, critic | Roles and hierarchy | Ch. 7 |
| Actor/action proposal | Proposal vs execution | Ch. 7 |
| Mode 1/Mode 2 and distillation | Explain concepts | Ch. 7 |
| Associative STM | What is stored/retrieved | Ch. 5 |
| Multiple futures | Motivation | Ch. 6 |
| H-JEPA and hierarchical planning | Proposed abstraction/subgoals | Ch. 6–7 |

Do not count a concept as covered merely because its name appears in a caption.

## 6. Figure plan
Aim for 18–21 original figures:
1. Reconstruction versus latent prediction.
2. Generic JEPA graph.
3. I-JEPA spatial masks.
4. V-JEPA spatiotemporal masks.
5. V-JEPA 2 pretraining versus 2-AC post-training.
6. V-JEPA 2.1 dense objective.
7. EMA/stop-gradient/update paths.
8. MSE conditional mean between target modes.
9. Same moments, different distributions.
10. LeJEPA theorem scope: probes, risk, scalar constraint, assumptions and optimum.
11. Cramér–Wold all directions versus finite sketch.
12. SIGReg: projections, characteristic functions, Epps–Pulley discrepancy and quadrature.
13. VISReg: centering, scale, shape and detached denominator.
14. One-dimensional sliced-Wasserstein sorted samples/quantiles.
15. Action-conditioned latent prediction.
16. Evidence map for V-JEPA 2-AC, JEPA-WMs and EB-JEPA.
17. **Mandatory LeCun brain/agent architecture.**
18. Mode 1 versus Mode 2.
19. Mode-2-to-Mode-1 distillation.
20. H-JEPA/hierarchical planning.
21. Optional VL-JEPA diagram only if it adds value.

**Brain/agent figure:** exactly six top-level modules: Perception, World Model, Cost, Actor, Short-Term Memory, Configurator. Draw Intrinsic Cost and Critic/Trainable Cost inside Cost. Show environment percept/feedback; Actor candidate actions→World Model; predicted states→Cost; cost/evaluation→Actor; selected action→environment; STM read/write; Configurator modulation. Use distinct arrows/legend for observation, state, action, prediction, scalar cost/gradient, memory and configuration. Do not show gradients through the physical environment. Brain silhouette may be an understated frame only. Create original artwork; do not trace/recolour the paper diagram.

Use Chapter 3 visual system: white, charcoal, restrained #3B6FB6, grayscale-safe, consistent linework, print-legible labels, landscape architecture canvas, no decorative effects, no figure number/title inside artwork. Captions about four printed lines or fewer.

## 7. Tables and code
Four or five tables: core image/video JEPA comparison; target-update mechanisms; VICReg/SIGReg/VISReg; passive video vs action-conditioned model vs full agent; LeCun's six modules. VL-JEPA/other modalities should not become a full survey table.

At most 3–4 code excerpts: generic pseudocode; official I-JEPA operation; official EB-JEPA example; optionally verified regularizer operation. Record exact commit/path/lines/license/edits/paper version. If commercial reproduction is not licensed, write original pseudocode. No full training loops.

## 8. Mathematical acceptance requirements
**Conditional mean:** for \(\mu(c)=E[H|C=c]\),
\[
E[\|H-a\|^2|c]=E[\|H-\mu\|^2|c]+\|a-\mu\|^2.
\]
The cross term vanishes. Require finite conditional second moment. Illustrate a multimodal target whose mean is between modes.

**LeJEPA:** read exact paper and appendices before stating Theorem 1. Transcribe the probes, integrated-square-bias functional, scalar covariance/total-variance constraint, admissible class, regularity assumptions, dimensions/constants and asymptotic regime. The theorem's Gaussian optimum is scoped to that criterion/class, not arbitrary downstream tasks. Do not write “Gaussian is universally optimal.”

**Cramér–Wold:** all one-dimensional projections in every direction identify the distribution. Finite sampled directions are only a sketch. Keep approximation rates attached to smoothness assumptions; distinguish formal max statistic from practical average.

**SIGReg:** verify the exact Epps–Pulley weight, bandwidth, quadrature, directions and reductions in the paper. Explain finite minibatch bias. A failure to reject is not proof of equality; a scalar test derivative is not a universal encoder-gradient guarantee.

**VISReg:** verify exact notation/reductions. Three regularizer terms are centering, scale and shape; prediction/invariance is separate. Show detach only at shape-normalization denominator. Sorted transport formula requires equal weights and equal sample counts. Give projection and sorting complexity separately. Do not infer full-network nonzero gradient from scalar derivative.

Audit dimensions, edge cases, gradient paths, loss scaling and all theorem assumptions before final figures. If unsupported, qualify or remove.

## 9. No-new-experiments policy
No new training, experiments, replications, data collection, fresh tables or author-generated results for this chapter.

Allowed: paper-reported results with exact attribution; official source inspection; short official snippets if license permits; original pseudocode; explanation of published experiments.

Not allowed: claim “we reproduced,” “our benchmark,” or “we trained”; treat code reading/checkpoint loading as reproduction; invent or interpolate results.

Use official I-JEPA for masks/updates, V-JEPA 2 for model/action interfaces, EB-JEPA for compact code-reading, LeJEPA for SIGReg. Old HaiyuWu/visreg now redirects to altos-labs/visreg; current canonical source was not accessible in this audit, so do not claim its code was inspected. JEPA-WMs is marked non-commercial in the current audit; do not copy its code into a commercial book without permission. Recheck all licenses at exact commit. For each excerpt record repo, commit, path, lines, license, changes, paper version and whether executed (planned: no).

## 10. Production and release gates
Follow Chapter 3 AsciiDoc numbering, anchors, equations and captions. Define symbols near first use; PyTorch only; no raw link appendix; no em dashes. Evidence order: primary paper, appendix, official code/config, official project page, later studies, then notes for intuition only.

The remote manuscript branch currently has no PDFs inside rough-draft/chapter4. The primary papers were checked through authoritative arXiv/OpenReview sources. Additional local but unpushed PDFs are not visible in the remote branch.

Release gates: freeze sources; verify every objective/update; audit LeJEPA theorem and SIGReg/VISReg equations; draft narrative within chapter boundaries; inspect official code/license only; approve brain diagram topology; check every regularizer figure against equations; build the PDF and inspect all pages.
