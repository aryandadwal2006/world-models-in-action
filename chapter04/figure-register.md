# Chapter 4 figure register

All 34 figures are original, editable SVG diagrams stored in `images/chapter04/` and embedded in the reader-facing Markdown section sources. They explain mathematical derivations and information paths and are informed by cited papers, but are not copied or traced reproductions of paper artwork.

The diagrams were drawn rather than importing official paper figures because the supplied papers have different reuse terms, some figures contain third-party material, and the exact repository PDFs could not be materialized for local inspection in this environment. No paper figure is presumed reusable merely because it is publicly accessible. Any later publisher-selected official figure needs an independent check of its exact version, caption, authorship, third-party credits and commercial reuse terms.

## Figure sequence

| Figure | SVG asset | Teaching purpose | Source section |
|---|---|---|---|
| 4.1 | `F01_energy_landscape.svg` | Compatibility energy over multiple candidate targets | `chapter04/01...` |
| 4.2 | `F02_scalar_roles.svg` | Prediction discrepancy, compatibility score and task cost | `chapter04/01...` |
| 4.3 | `F33_general_prediction_flow.svg` | Context, query, predictor and target information paths | `chapter04/02...` |
| 4.4 | `F03_target_types.svg` | Spatial, spatiotemporal, later-time and action-conditioned targets | `chapter04/02...` |
| 4.5 | `F04_conditional_mean.svg` | Two-mode conditional distribution and squared-error mean | `chapter04/02...` |
| 4.6 | `F05_ijepa_information_flow.svg` | I-JEPA context and target encoder pathways | `chapter04/03...` |
| 4.7 | `F31_ijepa_mask_geometry.svg` | I-JEPA context and target block geometry | `chapter04/03...` |
| 4.8 | `F06_vjepa_spatiotemporal_mask.svg` | Video target tube across spatial and temporal positions | `chapter04/04...` |
| 4.9 | `F32_masked_video_vs_action.svg` | Masked video prediction versus action-conditioned prediction | `chapter04/04...` |
| 4.10 | `F07_collapse_modes.svg` | Complete collapse and rank collapse | `chapter04/05...` |
| 4.11 | `F08_covariance_geometry.svg` | Equal-trace isotropic and anisotropic embeddings | `chapter04/06...` |
| 4.12 | `F09_ridge_shrinkage.svg` | Ridge shrinkage across covariance eigen-directions | `chapter04/07...` |
| 4.13 | `F23_ols_inverse_spectrum.svg` | OLS coefficient variance and inverse eigenvalues | `chapter04/07...` |
| 4.14 | `F10_local_density_bias.svg` | Local-regression bias under asymmetric density | `chapter04/08...` |
| 4.15 | `F22_fisher_score_field.svg` | Density score field and Fisher-information functional | `chapter04/08...` |
| 4.16 | `F29_gaussian_optimum.svg` | Gaussian and isotropic covariance minimization | `chapter04/08...` |
| 4.17 | `F11_sigreg_projection.svg` | All-directions identification and finite SIGReg sketch | `chapter04/09...` |
| 4.18 | `F30_same_moments_different_shapes.svg` | Equal first two moments with different distribution shapes | `chapter04/09...` |
| 4.19 | `F24_epps_pulley.svg` | Empirical and Gaussian characteristic functions | `chapter04/09...` |
| 4.20 | `F12_visreg_components.svg` | VISReg centre, scale and shape decomposition | `chapter04/10...` |
| 4.21 | `F25_sliced_wasserstein.svg` | One-dimensional sorted-sample transport | `chapter04/10...` |
| 4.22 | `F26_visreg_gradient_paths.svg` | VISReg scale and shape gradient paths | `chapter04/10...` |
| 4.23 | `F34_vljepa_cross_modal_prediction.svg` | Visual context and textual query predict a text embedding; decoding is optional | `chapter04/11...` |
| 4.24 | `F13_vjepa2_stages.svg` | V-JEPA 2 passive pretraining, robot post-training and planning | `chapter04/11...` |
| 4.25 | `F14_vjepa21_dense_supervision.svg` | V-JEPA 2.1 masked and visible-context supervision | `chapter04/11...` |
| 4.26 | `F15_intervention.svg` | Behaviour-policy observations versus interventions | `chapter04/12...` |
| 4.27 | `F16_planning_loop.svg` | Receding-horizon planning and feedback | `chapter04/13...` |
| 4.28 | `F17_model_exploitation.svg` | Planner exploitation of transition-model error | `chapter04/13...` |
| 4.29 | `F27_multiple_futures.svg` | Alternative futures under one history and action | `chapter04/13...` |
| 4.30 | `F18_agent_architecture.svg` | The six top-level modules of a predictive agent | `chapter04/14...` |
| 4.31 | `F19_mode1_mode2.svg` | Reactive Mode 1 and deliberative Mode 2 | `chapter04/14...` |
| 4.32 | `F20_hierarchy_memory.svg` | Hierarchical prediction, memory and configuration | `chapter04/14...` |
| 4.33 | `F28_training_step.svg` | I-JEPA training computation and EMA update | `chapter04/15...` |
| 4.34 | `F21_chapter_synthesis.svg` | Representation, prediction, dynamics and planning interfaces | `chapter04/16...` |

## Production rules
- Preserve the distinction between a masked target and a future conditioned on an action.
- Do not imply the standard original V-JEPA mask is causal or past-only.
- Keep target-feature selection separate from target-encoder input.
- Keep prediction discrepancy, compatibility energy and task cost visually distinct.
- Show LeCun's six modules exactly; Intrinsic Cost and Trainable Critic belong inside Cost.
- Use restrained blue accent `#3B6FB6`, grayscale-safe labels and white backgrounds.
- Keep equations, titles and figure numbers in manuscript text, not baked into SVG artwork.
