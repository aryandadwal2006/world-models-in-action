# Figure register and production plan

The figures should explain the mathematical argument, not decorate a literature survey. A paper figure is preferred when it communicates a method faithfully and is suitable for the specific teaching point. Original diagrams are preferred when the book needs a derivation, a comparison or an explanatory transformation that is absent from the source paper.

No artwork is embedded in the Markdown sources at this stage. For every official figure candidate, verify the exact paper version, caption, figure number, author attribution and reuse terms before placing the image in a commercial book. A paper's citation or arXiv availability does not by itself establish artwork reuse permission. If permission or a compatible license is not clear, create an original diagram that explains the same concept without tracing the source artwork.

## Figure principles

- Every figure must teach one precise point that the text has already introduced.
- Prefer source-paper figures for method-specific architecture and experiment setup when the original visual is unusually clear.
- Prefer original figures for first-principles derivations, counterexamples, gradient paths, conceptual contrasts and planning failure mechanisms.
- If an official figure is used, preserve its scientific meaning. Do not silently crop away a limitation, change a label, or make a result look stronger than the paper reports.
- Original diagrams should be visually consistent: white background, dark neutral labels, restrained blue accent, grayscale-safe contrast, readable labels and simple linework. No title text or figure number inside the artwork.
- Captions should explain the argument, not merely identify the objects. The figure description is an art brief; the printed caption should remain concise.
- The figure register can change as the text develops. A figure that adds no independent explanatory value should be removed rather than retained to meet a count.

## Planned figures

| ID | Teaching purpose | Preferred source | Production decision |
|---|---|---|---|
| F01 | A compatibility energy over candidate targets, including multiple low-energy modes | Original | Draw a context, candidate-target space and an energy surface. Distinguish a score from a normalized density. |
| F02 | Energy score, Gibbs density and task cost are different objects | Original | Three panels with separate symbols and normalization/cost semantics. |
| F03 | Information flow through a JEPA: visible context, target encoder, target selection and predictor query | Original, informed by I-JEPA | I-JEPA's arXiv version is under the arXiv perpetual non-exclusive license, not an explicit CC BY license. Produce an independent information-flow diagram unless the publisher obtains permission. |
| F04 | Spatial, spatiotemporal, future and action-conditioned targets are different conditional problems | Original | Four aligned panels. Make explicit that a masked video target is not necessarily a future target. |
| F05 | Squared-error prediction of a multimodal target returns the conditional mean | Original | Two target modes with the mean between them. Include a feature-space version to avoid implying that latent means are always physically meaningful. |
| F06 | Constant collapse exactly satisfies an unconstrained agreement objective | Original | Two distinct inputs mapped to one constant vector; show zero agreement loss and zero input information. |
| F07 | Complete collapse versus low-rank/dimensional collapse | Original | Compare a point, a narrow subspace cloud and a full-rank cloud. Avoid using a visual as a substitute for the covariance derivation. |
| F08 | Same total covariance trace, isotropic versus anisotropic geometry | LeJEPA, Fig. 3 / corresponding appendix illustration as candidate | LeJEPA v3 is marked CC BY-SA 4.0. Reuse requires attribution and ShareAlike handling; otherwise create an independent geometry illustration. Do not state a same-feature-span assumption. |
| F09 | Ridge shrinkage in covariance eigen-directions | Original | Show a weak and strong eigen-direction with the corresponding ridge shrinkage factor. |
| F10 | OLS coefficient variance grows with inverse covariance eigenvalues | Original | Plot the reciprocal eigenvalue contribution and the fixed-trace Jensen argument. |
| F11 | Local regression neighbourhoods are pulled by the embedding-density gradient | Original | Show a query, a locally asymmetric sample density and a target function with its gradient. |
| F12 | The score field and Fisher-information functional | Original | Show \(\nabla\log p(z)\) as a vector field and relate average squared score magnitude to \(J(p)\). |
| F13 | Gaussian minimization of the Fisher-information functional at fixed covariance | Original | Two-step visual: fixed covariance selects a Gaussian; fixed scalar covariance constraint selects isotropic covariance. |
| F14 | All-direction Cramér–Wold identification versus finite random-direction sketching | LeJEPA, Fig. 2 as candidate | LeJEPA v3 is marked CC BY-SA 4.0. If reused, preserve attribution and ShareAlike requirements; otherwise create an independent projection diagram. |
| F15 | Same first two moments, different projected distribution shapes | LeJEPA, Fig. 5 as candidate | The paper's constructed distribution is relevant. Reuse only with permission; otherwise create an independent two-dimensional counterexample. |
| F16 | Empirical characteristic function and Epps–Pulley discrepancy | Original, informed by LeJEPA, Sec. 4.2.3 | Plot target and empirical characteristic functions, weight function and integrated squared discrepancy. Label the quadrature as numerical approximation. |
| F17 | Projection count, batch size and numerical quadrature in SIGReg | Original | Separate the roles of \(N\), \(M\) and frequency knots. This is a computation diagram, not a convergence theorem. |
| F18 | VISReg separates centre, scale and shape | Original | Three subpanels showing mean shift, scale contraction/expansion and non-Gaussian shape. |
| F19 | One-dimensional sliced-Wasserstein transport by sorting | Original | Projected samples, sorted order statistics and Gaussian target quantiles. State equal-weight/equal-cardinality assumption. |
| F20 | VISReg stop-gradient boundary separates the scale statistic from the shape normalization | Original | Draw the gradient paths explicitly. Distinguish detaching the standard-deviation denominator from an EMA target update. |
| F21 | Passive observation versus intervention-conditioned prediction | Original | Same history with two candidate actions and different predicted outcomes. Include the data-collection policy as context. |
| F22 | V-JEPA 2: passive visual pretraining followed by action-conditioned robot post-training | V-JEPA 2, overview/training figure as candidate | The supplied arXiv v1 page is marked CC BY 4.0. Reuse with attribution and license notice, after checking for third-party content. Preserve the robot post-training stage and action/proprioception inputs. |
| F23 | V-JEPA 2.1 dense visible/masked-token and deep supervision | Original, informed by V-JEPA 2.1 | No permissive artwork license was established from the supplied arXiv metadata. Create an original explanation unless permission is confirmed. Do not label V-JEPA 2.1 itself as action-conditioned. |
| F24 | A planner evaluates multiple candidate rollouts and executes only the first action | Original | Draw candidate latent trajectories, goal/cost evaluation, action selection, environment feedback and replanning. |
| F25 | A planner can exploit model error | Original | Show a candidate sequence with low predicted cost but an unsafe or unsuccessful real outcome. Distinguish model prediction from environment result. |
| F26 | LeCun's six-module agent architecture | Original, informed by LeCun 2022 | No commercial-reuse license was established for the OpenReview PDF. Draw an independent diagram from the textual module specification. Preserve exactly six top-level modules and show Intrinsic Cost and Trainable Critic inside Cost. |
| F27 | Mode 1, Mode 2 and planner-to-policy distillation | Original, informed by LeCun 2022 | Direct reactive path, deliberative planning loop and a teacher signal from Mode 2 to Mode 1. |
| F28 | Hierarchical prediction across time scales | LeCun, H-JEPA figure as candidate | Prefer the official hierarchy illustration if permitted; otherwise draw high-level subgoals being refined into lower-level states and actions. |
| F29 | A single deterministic future versus multiple latent futures | Original | Branch several plausible outcomes from the same history and action. Show a latent selector and a cost-based ranking step. |
| F30 | How the chapter's components fit together | Original | One compact closing diagram: representation, target prediction, distributional constraint, action-conditioned dynamics, cost, planning and feedback. |
| F31 | One I-JEPA training step, including tensor dimensions, mask positions and gradient flow | Original, informed by Assran et al. (2023) | Show patch tokens, context selection, full-image target encoding, target-position selection, positional query tokens, predicted features and the EMA update. Use solid arrows for forward computation and dashed arrows for parameter updates. |

## Figure-source rights notes

The accessible version metadata identifies these source terms:

- I-JEPA v3: arXiv.org perpetual non-exclusive license. This is not an explicit Creative Commons license for figure reuse.
- Original V-JEPA v1: CC BY 4.0.
- V-JEPA 2 v1: CC BY 4.0.
- LeJEPA v3: CC BY-SA 4.0.
- VL-JEPA v2: CC BY 4.0.
- V-JEPA 2.1 v3, VISReg v1 and LeCun v0.9.2: no permissive commercial artwork license was established from the accessible metadata during this review.

These metadata do not settle the rights status of every individual figure. Captions, third-party credits, publisher terms and any excluded material must be checked before publication. For the unresolved cases, an original diagram is the default; do not reproduce or trace the source artwork without permission.

## Official-figure candidates and limits

**I-JEPA.** The paper's architecture and mask figures are candidates for the context/target pathway and target geometry. The paper and archived implementation use a particular target-feature selection path. If the figure is reused or adapted, the caption must not imply that the target encoder processes only an isolated target crop when it actually processes the image before target positions are selected.

**V-JEPA.** The paper's spatiotemporal masking and feature-prediction figures are candidates for showing the extension from image tokens to video tokens. The figure must not imply that every masked target is a future frame or that video pretraining is action-conditioned.

**V-JEPA 2.** Its training-stage and robot-planning figures are candidates because the distinction between action-free pretraining and action-conditioned post-training is central. Any reused figure must preserve the robot-data stage, the action/proprioception inputs and the planner boundary.

**V-JEPA 2.1.** Its dense-feature objective is a candidate for the visible-token, masked-token and deep-supervision explanation. Do not reuse a robot-planning result figure without including the separate action-conditioned predictor and planner that produce the physical result.

**LeJEPA.** Its SIGReg projection schematic and linear/nonlinear probe illustrations are particularly useful. The official figure should be used only if its reuse terms are suitable; otherwise create original derivations. A re-drawn version must distinguish the all-directions theorem from the sampled projection sketch.

**VISReg.** The paper's objective schematic and gradient comparisons may be useful, but no reproduction permission is assumed. Original scale/shape diagrams are likely clearer for the derivations in this chapter.

**VL-JEPA.** The paper's architecture figure is a candidate only for the short cross-modal example. Do not let it expand into a separate survey of vision-language methods.

**LeCun.** The proposed six-module architecture and H-JEPA figures are important source diagrams. If reuse is not permitted, an original diagram must preserve the hierarchy and topology accurately. In particular, Cost is one top-level module; Intrinsic Cost and Trainable Critic are inside it. The physical environment must not appear inside the differentiable model graph.

## Production status

All entries above are descriptions and candidates, not completed artwork. No paper figure has been copied into the manuscript. The final figure set should be selected after the text is stable and the publisher's artwork and rights requirements are checked.
