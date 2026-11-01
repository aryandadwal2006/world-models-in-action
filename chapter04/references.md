# References and source inventory

This file identifies the exact versions supplied for Chapter 4 and the primary sources used in the text. Paper-specific claims should be cited to the version that supports them. Where a method is a stage within a larger paper rather than a separate publication, it is cited to that paper rather than treated as an independent work.

## Primary papers supplied for the chapter

### I-JEPA

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. (2023). “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv:2301.08243, version 3.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2301.08243v3.pdf\`.

Primary source: https://arxiv.org/abs/2301.08243

Official implementation: https://github.com/facebookresearch/ijepa

Use in this chapter: spatial context-to-target prediction, multiple target blocks, feature-space prediction, target-encoder pathway and the limits of image-only representation learning.

### Original V-JEPA

Bardes, A. et al. (2024). “Revisiting Feature Prediction for Learning Visual Representations from Video.” arXiv:2404.08471, version 1.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2404.08471v1.pdf\`.

Primary source: https://arxiv.org/abs/2404.08471

Official project and implementation sources are maintained by Meta FAIR through the V-JEPA research releases.

Use in this chapter: spatiotemporal feature masking and prediction, target/context information flow, video representation learning, and the distinction between masked video prediction and action-conditioned dynamics.

### V-JEPA 2 and V-JEPA 2-AC

Assran, M. et al. (2025). “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” arXiv:2506.09985, version 1.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2506.09985v1.pdf\`.

Primary source: https://arxiv.org/abs/2506.09985

Official implementation: https://github.com/facebookresearch/vjepa2

Use in this chapter: passive video pretraining, the separate action-conditioned post-training stage, robot trajectory data, proprioceptive inputs, image-goal planning and the scope of the reported physical deployments. V-JEPA 2-AC is treated as a model/stage within this paper, not as an unrelated publication.

### LeJEPA and SIGReg

Balestriero, R. and LeCun, Y. (2025). “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.” arXiv:2511.08544, version 3, 14 November 2025.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2511.08544v3 (1).pdf\`.

Primary source: https://arxiv.org/abs/2511.08544

Official project/code pointer: https://github.com/rbalestr-lab/lejepa

Use in this chapter: the conditions under which a predictor is justified, multi-view agreement, linear and nonlinear probe risk, the Fisher-information functional, Gaussian optimality under stated assumptions, Cramér–Wold projections, the Epps–Pulley statistic and SIGReg.

### VL-JEPA

Chen, D., Shukor, M., Moutakanni, T., Chung, W., Yu, L., Kasarla, T., Bolourchi, A., LeCun, Y., and Fung, P. (2025/2026). “VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.” arXiv:2512.10942, version 2.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2512.10942v2.pdf\`.

Primary source: https://arxiv.org/abs/2512.10942

Use in this chapter: one cross-modal example showing that the target representation need not be visual. VL-JEPA predicts a text embedding from visual context and can decode language when a text output is required. It is not treated as an action-conditioned physical dynamics model.

### V-JEPA 2.1

Mur-Labadia, L. et al. (2026). “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.” arXiv:2603.14482, version 3.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2603.14482v3_compressed.pdf\`.

Primary source: https://arxiv.org/abs/2603.14482

Official implementation: https://github.com/facebookresearch/vjepa2

Use in this chapter: dense visible- and masked-token supervision, deep self-supervision and the distinction between dense representation learning and action-conditioned robot prediction. Any planning comparison must retain the predictor and planner settings from the paper.

### VISReg

Wu, H., Balestriero, R., and Levine, M. (2026). “Variance-Invariance-Sketching Regularization for JEPA Training.” arXiv:2606.02572, version 1.

Supplied file: \`rough-draft/chapter4/papers_pdfs/2606.02572v1 (1).pdf\`.

Primary source: https://arxiv.org/abs/2606.02572

Project/source pointer: https://haiyuwu.github.io/visreg/

Use in this chapter: explicit separation of centre, scale and shape; stop-gradient in shape normalization; sliced-Wasserstein projections; scale-gradient analysis; compute dependence on batch size, embedding dimension and number of slices; and the paper's bounded experimental claims.

### Autonomous machine intelligence

LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. Version 0.9.2, 27 June 2022. OpenReview: BZ5a1r-kVsf.

Supplied file: \`rough-draft/chapter4/papers_pdfs/10356_a_path_towards_autonomous_mach.pdf\`.

Primary source: https://openreview.net/forum?id=BZ5a1r-kVsf

Use in this chapter: energy-based compatibility, latent variables for alternative futures, the six-module agent proposal, Mode 1 and Mode 2, Mode-2-to-Mode-1 distillation, associative short-term memory, hierarchical prediction and the Configurator. The document is identified as a position paper and research proposal.

## Source-review record and access limitation

The eight filenames above were mapped to the paper identities by their arXiv identifiers and title pages, rather than by relying on the filenames alone. The supplied PDFs remain in the repository and are not modified.

The GitHub connector used for repository access can enumerate binary PDF files and return their blob identifiers and sizes, but it does not expose their binary bytes to the text retrieval operation in this environment. The source review therefore used the matching versioned primary-paper text, PDF page renders where the exact version was accessible, and the paper's appendices and figures. The following distinctions are retained:

| Supplied PDF | Version-matched source review |
|---|---|
| I-JEPA v3 | Full paper and page-level PDF review |
| Original V-JEPA v1 | Full paper and page-level PDF review |
| V-JEPA 2 v1 | Exact-version full text, methods, results, appendices and references; binary PDF exceeded the viewer/connector transfer limit |
| LeJEPA v3 | Full 50-page PDF and appendices reviewed |
| VL-JEPA v2 | Full 15-page PDF and page-level review |
| V-JEPA 2.1 v3 | Exact-version full text and a complete paper reading record; large PDF binary could not be transferred through the connector |
| VISReg v1 | Version-matched extracted paper text, including method, objective, algorithm, experiments and references; the repository PDF binary could not be transferred through the connector |
| LeCun v0.9.2 | Full 48-page PDF and page-level review |

The exact binary contents of the V-JEPA 2, V-JEPA 2.1 and VISReg files could not be independently downloaded through the connected GitHub interface. The corresponding version-matched primary texts were reviewed instead. This record does not claim that the inaccessible binary bytes were extracted directly.

## Additional source and attribution notes

- Chapter 3 already provides the detailed background for VICReg, Barlow Twins, BYOL, SimSiam and related SSL methods. They are used here only when needed to explain why second-order statistics, branch asymmetry or view agreement are insufficient for the present argument.
- EB-JEPA is a library and set of examples rather than a substitute for the foundational primary papers. It may be cited for implementation context if the text discusses its examples, but its repository should not be treated as a theorem or as a universal energy-based JEPA recipe.
- Code and artwork licenses must be checked independently. A paper citation does not grant permission to reuse repository code or figures. No code or paper artwork is reproduced by the Markdown draft.
- Reported experimental numbers must retain the benchmark, split, readout, planner, action horizon, search budget and deployment conditions needed to interpret them. A representation score is not a planning score.
