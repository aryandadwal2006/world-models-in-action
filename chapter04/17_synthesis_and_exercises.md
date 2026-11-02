# 17. What a JEPA gives a world model

The chapter began with an encoder and the question of how it can learn relations between parts of an observation. A JEPA answers by defining a context, a target and a feature space in which the target is predicted. The method moves the learning objective away from reproducing every pixel and toward matching a learned representation. That change is useful, but it does not by itself produce a physical state, an action-conditioned transition model or a complete planner.

## 17.1 What is the target?

I-JEPA selects spatial target blocks from an image. V-JEPA selects spatiotemporal target blocks from a video. VL-JEPA shows that the target representation can belong to another modality. These methods differ in their conditional prediction problems because they differ in what information is visible, what is withheld and what the target encoder represents.

The mask and query are part of the objective. A mask that makes the target easy may encourage shortcuts; a mask that removes task-relevant evidence may make the target unidentifiable. A loss is meaningful only after the sampling distribution and target representation have been specified.

## 17.2 What prevents a trivial solution?

Agreement alone admits a constant representation. An EMA target, stop-gradient or predictor changes the optimization dynamics but does not automatically prove non-collapse. Variance and covariance penalties constrain second-order geometry. SIGReg constrains projected distributions through characteristic functions. VISReg separates centre, scale and normalized shape.

These constraints address different mathematical properties. None establishes that the representation preserves the information required by a particular downstream control task. That stronger requirement is predictive sufficiency relative to a target, action set and time horizon.

## 17.3 Why does embedding geometry matter?

A covariance spectrum with weak directions can make linear readouts poorly conditioned. Local nonlinear estimators depend on where samples lie within the embedding space, leading to the Fisher-information functional under the assumptions studied by LeJEPA. Gaussian and isotropic optimality follow from distinct inequalities, and only for the risk functionals and constraints analysed.

SIGReg and VISReg make these distributional targets trainable through sampled one-dimensional comparisons. Their objectives improve control over embedding geometry, but a finite sketch does not certify exact Gaussianity and a Gaussian embedding is not automatically sufficient for physical prediction.

## 17.4 What changes when the system acts?

A passive video predictor estimates future features under the trajectory distribution in its data. An action-conditioned predictor must receive the candidate action and be trained on observations aligned with that action. Identifying its response requires state sufficiency, action support and assumptions that make observed transitions informative about interventions.

Planning then introduces another objective. The transition model predicts consequences; the cost evaluates desirability; the planner searches action sequences; feedback corrects the state estimate after execution. More search can exploit model error, and latent goal similarity is not a safety guarantee.

LeCun's proposed agent architecture adds Perception, World Model, Cost, Actor, Short-Term Memory and Configurator as distinct responsibilities. Mode 1 and Mode 2 separate fast reactive behaviour from deliberate planning. These are architectural proposals and research questions, not evidence that a complete general agent has been built.

## 17.5 Questions to carry forward

The most useful way to read a new JEPA paper is to reconstruct its information flow. What does the context encoder observe? What does the target encoder observe? How is the target selected? What does the predictor receive? What loss is applied, and which parameters receive its gradient? What distribution of contexts and targets is sampled? Finally, what downstream claim is supported by the evaluation?

Those questions keep four levels separate: representation learning, target prediction, controlled dynamics and planning. A result at one level may support the next, but it does not automatically establish competence at the others.

## Exercises

1. **Conditional prediction.** For a squared-error feature predictor, derive the conditional-mean solution. Construct a two-mode target distribution for which the conditional mean is not a likely target.

2. **Mask design.** Describe two masks for the same video. Explain what information each leaves available and what shortcut each may permit.

3. **Collapse.** Show that a constant representation minimizes a pure view-agreement loss. Give an example of a rank-deficient representation with nonzero variance in every coordinate.

4. **Predictive sufficiency.** Construct two histories that map to the same embedding but produce different future distributions under the same action. Explain why the representation is insufficient.

5. **Ridge bias.** Derive the coefficient shrinkage factor in the eigenbasis of the covariance matrix. Which task direction has the greatest shrinkage?

6. **OLS variance.** Under the homoskedastic linear model, derive the trace of the OLS coefficient covariance. Use Jensen's inequality to show why isotropy minimizes it at fixed trace.

7. **Local regression.** Starting from the kernel-regression expectation, identify where the density gradient enters the second-order expansion. State the smoothness assumptions needed for the expansion.

8. **SIGReg.** Explain why the Cramér–Wold theorem requires all directions, while a minibatch objective samples only finitely many. Distinguish the maximum directional test from the mean directional training loss.

9. **VISReg.** Explain why the scale loss and shape loss act on different quantities. What changes when the standard-deviation denominator is stop-gradient?

10. **Intervention.** Give an example in which \(p(Y\mid X,A=a)\) differs from \(p(Y\mid X,\operatorname{do}(A=a))\). State what additional assumptions would allow observational trajectories to identify the intervention distribution.

11. **Model exploitation.** Construct a situation in which an optimizer finds an action sequence with low predicted cost because the transition model is wrong. Explain why increasing the number of search samples can make the selected action worse in the real system.

12. **Agent architecture.** Separate the responsibilities of World Model, Cost and Actor. Explain what Mode 2 can provide to Mode 1 through distillation, and why this does not transfer a guarantee of correctness.

**Figure 4.29. Representation, prediction, dynamics and planning are separate interfaces.**

Description: The encoder determines what information is represented; the JEPA objective specifies a target to predict; the controlled transition model estimates consequences under actions; and the planner selects actions using a separate cost. No single feature loss supplies all four responsibilities.