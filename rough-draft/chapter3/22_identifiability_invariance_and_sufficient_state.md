# Representation identifiability, invariance and sufficient state

This chapter-theory note explains why a self-supervised objective can learn a useful feature without uniquely identifying “the state of the world,” and why a low pretraining loss is insufficient evidence for a world-model representation.

## 1. Three distinct objects are easily conflated

Let:
- \(O_t\) be an observation, such as an image or sensor vector;
- \(H_t=(O_{0:t},A_{0:t-1})\) be the interaction history available at time t;
- \(S_t\) be an idealized underlying state, if one exists;
- \(Z_t=f_\theta(O_t)\), or \(Z_t=f_\theta(H_t)\), be the learned representation.

These objects need not have the same information content. An image is an observation generated from the state. A single image may be ambiguous about velocity or occluded objects. A history-aware encoder can disambiguate some of these variables. A feature encoder can also intentionally discard variables that do not help the pretext objective.

The word “state” should be reserved for a representation satisfying a stated predictive or decision-theoretic condition. Merely naming an encoder output \(z_t\), \(s_t\) or “latent state” does not make it Markov sufficient.

## 2. Predictive sufficiency as a conditional-independence statement

Suppose the target of a world model includes next observation \(O_{t+1}\) and reward \(R_t\), and the available control is \(A_t\). A history is predictively sufficient if

\[
P(O_{t+1},R_t\mid H_t,A_t)
=P(O_{t+1},R_t\mid Z_t,A_t)
\]

for the relevant histories and actions. Equivalently, under the distribution in question,

\[
(O_{t+1},R_t)\perp H_t\mid (Z_t,A_t).
\]

This is a strong property. It says that, once the latent and the candidate action are known, extra details of the past provide no additional information about the next observation or reward. It does not require the latent to reconstruct every pixel of the past. It does require preserving whatever aspects of the past change the predictive distribution.

For a latent transition model, one may require a kernel \(p(z_{t+1}\mid z_t,a_t)\) such that future latent dynamics depend on history only through \((z_t,a_t)\). For partially observable systems, the sufficient state may be a belief distribution over hidden states or another memory representation, not a single static frame embedding.

These definitions depend on the environment and action space. A representation sufficient for one set of actions or rewards may be insufficient for another task.

## 3. State aliasing: a constructive counterexample

Suppose two histories \(h_1,h_2\) map to the same latent:

\[
f(h_1)=f(h_2)=z.
\]

If there exists an action a and a measurable set of next observations B for which

\[
P(O_{t+1}\in B\mid H_t=h_1,A_t=a)
\ne
P(O_{t+1}\in B\mid H_t=h_2,A_t=a),
\]

then the latent aliases two histories that have different predictive consequences. No single transition distribution \(p(O_{t+1}\mid z,a)\) can equal both conditional distributions exactly. The latent is not predictively sufficient for this setting.

A simple physical example is a ball at the same position but with different velocities. A single-frame encoder may map them to similar or identical features if appearance dominates the loss. Under the same action, their next positions differ. A second example is two visually identical gripper configurations with different hidden contact forces; under the same commanded action, the outcomes can differ.

This establishes why a static image-representation loss cannot by itself guarantee a correct dynamics state: the objective must constrain the latent on distinctions that affect conditional futures.

## 4. Invariance is a quotient operation

Let a family of transformations G act on observations. A perfectly invariant encoder satisfies

\[
f(g\cdot x)=f(x), \qquad \forall g\in G.
\]

This implies that all observations in the same transformation orbit receive identical features. The representation effectively maps observations to equivalence classes under the relation

\[
x\sim_G x'
\quad\text{if}\quad x'=g\cdot x
\text{ for some }g\in G.
\]

In geometric language, invariance constructs a representation of the quotient space \(X/G\). This is a compact way to express the cost of invariance: distinctions within an orbit are deliberately collapsed.

The key assumption is that members of every orbit are equivalent for the downstream problem. In image classification, moderate changes in illumination or crop may preserve an object's class. For localization, the same crop can remove absolute position. For robotics, horizontal flipping can reverse handedness; color jitter can erase an object marker; temporal subsampling can remove velocity direction.

The transformation law is therefore part of the task definition. An augmentation is not “safe” in the abstract. It is safe only relative to the variables the downstream system must retain.

## 5. Data processing and information bottlenecks

If \(Z=f(X)\) is a deterministic representation, then for any target Y the data processing inequality gives

\[
I(Z;Y)\leq I(X;Y).
\]

A deterministic encoder cannot create information about Y that was absent from X. It can, however, make some information easier to access with a restricted decoder, improve robustness by discarding nuisance variation, or organize the information in a useful geometry.

The information bottleneck objective is often written

\[
L_{\mathrm{IB}}=I(Z;X)-\beta I(Z;Y),
\]

where the first term penalizes retaining too much input information and the second rewards retaining information about task target Y. This is an explanatory framework, not a universal practical SSL loss. In label-free pretraining, the target Y is not necessarily the downstream variable; methods instead create proxy targets from views, pairs, masks or teacher features.

This reveals an important limitation of maximizing dependence between two views. The objective may preserve information shared by the views, including nuisance information, and can discard information unique to one view—even if that unique information matters for the downstream task. More mutual information between augmentations does not automatically mean greater predictive sufficiency for the environment.

## 6. Why many latents can be equivalent

Suppose \(Z\) is a representation and \(\phi\) is a bijection on its support. Define

\[
Z'=\phi(Z).
\]

For any future predictor \(F\) in Z-space, define a conjugated predictor

\[
F'(z',a)=\phi\!\left(F(\phi^{-1}(z'),a)\right).
\]

Then the model can produce equivalent predictions in the transformed coordinates. This shows that even if a sufficient latent exists, its individual coordinates need not have unique physical meanings. The dynamics can be represented in many coordinate systems.

This invariance only applies when the downstream model class can represent the conjugated mapping and the loss permits the corresponding transformations. Specific SSL losses impose additional geometry: normalized dot-product objectives preserve certain orthogonal symmetries more readily than arbitrary invertible transformations; reconstruction heads may break some symmetries. Do not claim every SSL method is invariant to every invertible transform. The general point is that learned coordinates are not automatically identifiable without further constraints.

If interpretability is desired, additional assumptions or probes are required: known interventions, supervision, architectural priors, object-centric structure, disentanglement objectives or explicit physical constraints. These constraints can improve identifiability but can also inject bias.

## 7. Observational equivalence versus causal sufficiency

Two observations may be statistically similar under the training distribution but differ under intervention. A purely observational SSL objective sees correlations in sampled data. Unless the training signal includes action-conditioned outcomes or suitable interventions, it need not identify causal relationships that determine what happens after a chosen action.

Formally, observational prediction concerns distributions such as \(P(O_{t+1}\mid Z_t,A_t)\) under the data-collection policy. Planning can query action sequences that differ from the policy distribution, and a model can be exploitable off-distribution. Causal evaluation asks whether predictions remain valid under interventions on actions and relevant state factors, not just whether they fit logged trajectories.

This does not imply that SSL features cannot encode causal information. It means causal sufficiency is not entailed by a generic SSL loss. It must be supported by the data regime, modelling assumptions and intervention/generalization evidence.

## 8. A hierarchy of claims

Use the following order and do not skip levels without evidence:

1. **Optimization claim:** the loss decreases or the algorithm reaches a stable run.
2. **Representation-statistics claim:** features have nonzero variance, suitable rank or a specified distribution.
3. **Proxy-task claim:** the representation supports the pretext target or feature-matching relationship.
4. **Transfer claim:** features help a selected downstream benchmark under a stated protocol.
5. **Predictive-sufficiency claim:** relevant futures/rewards are conditionally predictable from latent and action.
6. **Action-conditioned dynamics claim:** the model predicts outcomes for selected actions, not merely passive sequence continuation.
7. **Planning claim:** a controller using model predictions achieves reliable real-environment performance without exploiting modelling errors.

Evidence at a lower level does not prove a higher-level property. A training curve cannot prove feature rank; rank cannot prove semantics; a linear probe cannot prove Markov sufficiency; passive video anticipation cannot prove action-conditioned control; a simulated planning score cannot by itself guarantee robust deployment.

## 9. Evaluation implications for the book

To evaluate the state contract:
- choose variables and future consequences before choosing probes;
- define the history/action distribution on which sufficiency is required;
- include pairs of histories that look alike but differ in velocity, contact, occlusion or hidden context;
- test same-action future distributions and multiple action choices;
- measure multi-step prediction, not only one-step candidate retrieval;
- hold out object configurations, domains or interventions;
- report whether the probe is linear, nonlinear, fine-tuned or trained with labels;
- treat reconstruction, representation rank and visualization as diagnostic evidence, not final proof.

The conclusion is not that one SSL family necessarily solves these requirements. It is that the representation contract determines what “good” means, and the choice of pretext objective determines which parts of that contract receive direct pressure during training.
