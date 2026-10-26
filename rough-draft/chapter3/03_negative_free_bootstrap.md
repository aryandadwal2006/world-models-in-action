# Negative-free learning: bootstrap prediction and self-distillation

## 1. Why remove explicit negatives?

Contrastive learning supplies a direct non-collapse mechanism: the positive must outrank candidates considered negative. It also makes the training problem depend on the sampling set. A negative example may be another view of the same class, and candidate queues or large batches cost memory. A natural alternative is to make one branch predict another branch's representation without including explicit negative examples.

This does not make collapse disappear as a mathematical possibility. If both branches output a constant vector, an agreement-only objective can be small. The research question becomes how the complete optimization procedure avoids, escapes or makes collapsed solutions unattractive.

## 2. BYOL: an online predictor and a moving target

BYOL uses an online encoder fθ, an online projection head gθ and a predictor qθ. A target network fξ and projection head gξ produce the target representation for a different augmented view. The target parameters follow an exponential moving average of the online parameters rather than gradient descent. The target branch is stop-gradient.

For two views, one direction minimizes a normalized regression loss, often written as squared distance between normalized outputs or 2 minus twice their cosine similarity. The roles can be swapped to make the training loss symmetric. The predictor creates an asymmetry between the online prediction and the target representation; the target update makes targets evolve on a slower timescale.

Do not explain BYOL as “EMA mathematically forbids collapse.” It does not. A constant embedding is compatible with matching. The paper's experiments show that the complete recipe can learn useful representations without explicit negatives, while theory subsequent to the original paper studies which parameterizations and dynamics lead to collapse or non-collapse. EMA, predictor, normalization, data views and optimization jointly matter.

## 3. SimSiam: what stop-gradient changes

SimSiam removes the EMA target update. Both branches share the same encoder and projection parameters; one branch passes through a predictor, and the target of the other branch is detached. A symmetric loss swaps which branch is the target. The paper's controlled ablations argue that stop-gradient is central to its particular implementation.

Stop-gradient is a change to the vector field followed by gradient descent, not a semantic regularizer. In ordinary differentiable algebra, two branches may look nearly symmetric. Under automatic differentiation, however, only one side receives gradients through a particular loss term. This breaks the update symmetry. A proof that a collapsed point exists is not a proof that gradient descent converges to it. Conversely, an empirical success is not a universal theorem that stop-gradient alone suffices in arbitrary models.

The chapter should explain this distinction plainly. In the simple demonstration, show that a constant pair can minimize pure invariance regression, then compare the actual training diagnostics of a SimSiam-style predictor/stop-gradient setup to an explicitly variance-regularized objective. Avoid presenting the toy code as a reproduction of ImageNet SimSiam.

## 4. DINO: predict distributions over prototypes

DINO uses a student network and a teacher network that process different views and output probability distributions over learned output prototypes. The student distribution is trained to match the sharpened teacher distribution through cross-entropy. The teacher is updated by an exponential moving average of student parameters. Centering adjusts the teacher's output statistics; sharpening lowers the teacher distribution's entropy. These mechanisms interact: centering counters concentration on a small subset of prototypes, while sharpening encourages confident assignments.

A prototype here is a learned direction or output unit used to produce a distribution. The output distribution is not necessarily a human-interpretable class label. A student predicting a teacher distribution is a self-distillation objective: the target comes from a slowly updated model on another view, not from external annotations.

DINO's notable empirical result is that ViT features exhibit useful spatial structure, including object boundaries and segmentation-like grouping under certain visualizations/evaluations. Phrase this as an observed property of trained features, not as a guarantee that the architecture explicitly segments every image. DINOv1, DINOv2 and DINOv3 are a lineage, but the later versions add scale, data curation, registers and training/feature-stabilization changes; they are not a single unchanged algorithm.

## 5. DINOv2 and DINOv3: scaling is not just a larger model

DINOv2 demonstrates that carefully collected large-scale image data, teacher-student training and a robust recipe can yield general-purpose visual features that transfer across a broad collection of tasks. Its released models and data-engineering details are part of the contribution. A survey must not attribute all of this to “the DINO loss” as though data quality and scale were held constant.

DINOv3 (2025) further targets large-scale, high-resolution dense features. The paper reports that dense feature maps can degrade over long training schedules and introduces Gram anchoring to preserve their quality. This is an important correction to a simplistic view that more training automatically improves every representation. The current DINOv3 repository has its own license; code availability and weights must not be described as unrestricted open source without checking the license.

The current chapter's frontier section should use DINOv3 to illustrate that an objective works only as part of a training system: target generation, data, architecture, resolution, schedule and feature diagnostics matter.

## 6. A cautious theory map

The useful formal question is not “which single component prevents collapse?” but “what are the fixed points of this complete update and how stable are they?” For an online parameter vector θ and a target ξ, abstract the updates as

\[
\theta_{n+1}=\theta_n-\eta\,\nabla_\theta L(\theta_n,\xi_n),
\qquad
\xi_{n+1}=m\xi_n+(1-m)\theta_{n+1}.
\]

The teacher equation is low-pass filtering of online parameters. It slows target motion; it does not directly constrain feature covariance. Stop-gradient changes the gradient in the first equation. A predictor changes the parameterization that minimizes the loss. Normalization changes the set of reachable feature distributions. Different theories study different regimes of this dynamical system; a universal guarantee cannot be inferred from a name shared by multiple algorithms.

When reviewing theoretical results, record the exact assumptions: linear or nonlinear encoders, batch normalization or no normalization, population or finite-batch statistics, predictor linearity, target update rate, dimension and data covariance. Many apparent contradictions in collapse analyses arise because those details differ.

## 7. What this family buys and what it does not

Benefits: no explicit negative queue; flexible use of teacher target distributions; strong visual transformer backbones; natural support for multi-crop and dense patch features. Risks: complexity around the moving teacher, sharpening and centering; sensitivity to normalization and training recipe; additional target branch; collapse diagnostics needed; no direct guarantee of physical-state sufficiency or controlled dynamics.

## References

- Grill, J.-B. et al. (2020), Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning, arXiv:2006.07733.
- Chen, X. and He, K. (2021), Exploring Simple Siamese Representation Learning, arXiv:2011.10566.
- Caron, M. et al. (2021), Emerging Properties in Self-Supervised Vision Transformers, arXiv:2104.14294.
- Oquab, M. et al. (2023), DINOv2: Learning Robust Visual Features without Supervision, arXiv:2304.07193.
- Siméoni, O. et al. (2025), DINOv3: Self-supervised learning for vision at unprecedented scale, arXiv:2508.10104.
