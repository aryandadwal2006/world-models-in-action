# References and source inventory

This file identifies the exact versions supplied for Chapter 4 and the primary sources used in the text. Paper-specific claims should be cited to the version that supports them. Where a method is a stage within a larger paper rather than a separate publication, it is cited to that paper rather than treated as an independent work.

## Primary papers supplied for the chapter

### I-JEPA

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. (2023). “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv:2301.08243, version 3.

Supplied file: `rough-draft/chapter4/papers_pdfs/2301.08243v3.pdf`.

Primary source: https://arxiv.org/abs/2301.08243

Official implementation: https://github.com/facebookresearch/ijepa

Use in this chapter: spatial context-to-target prediction, multiple target blocks, feature-space prediction, target-encoder pathway and the limits of image-only representation learning.

### Original V-JEPA

Bardes, A. et al. (2024). “Revisiting Feature Prediction for Learning Visual Representations from Video.” arXiv:2404.08471, version 1.

Supplied file: `rough-draft/chapter4/papers_pdfs/2404.08471v1.pdf`.

Primary source: https://arxiv.org/abs/2404.08471

Official project and implementation sources are maintained by Meta FAIR through the V-JEPA research releases.

Use in this chapter: spatiotemporal feature masking and prediction, target/context information flow, video representation learning, and the distinction between masked video prediction and action-conditioned dynamics.

### V-JEPA 2 and V-JEPA 2-AC

Assran, M. et al. (2025). “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” arXiv:2506.09985, version 1.

Supplied file: `rough-draft/chapter4/papers_pdfs/2506.09985v1.pdf`.

Primary source: https://arxiv.org/abs/2506.09985

Official implementation: https://github.com/facebookresearch/vjepa2

Use in this chapter: passive video pretraining, the separate action-conditioned post-training stage, robot trajectory data, proprioceptive inputs, image-goal planning and the scope of the reported physical deployments. V-JEPA 2-AC is treated as a model/stage within this paper, not as an unrelated publication.

### LeJEPA and SIGReg

Balestriero, R. and LeCun, Y. (2025). “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.” arXiv:2511.08544, version 3, 14 November 2025.

Supplied file: `rough-draft/chapter4/papers_pdfs/2511.08544v3 (1).pdf`.

Primary source: https://arxiv.org/abs/2511.08544

Official project/code pointer: https://github.com/rbalestr-lab/lejepa

Use in this chapter: the conditions under which a predictor is justified, multi-view agreement, linear and nonlinear probe risk, the Fisher-information functional, Gaussian optimality under stated assumptions, Cramér–Wold projections, the Epps–Pulley statistic and SIGReg.

### VL-JEPA

Chen, D., Shukor, M., Moutakanni, T., Chung, W., Yu, L., Kasarla, T., Bolourchi, A., LeCun, Y., and Fung, P. (2025/2026). “VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.” arXiv:2512.10942, version 2.

Supplied file: `rough-draft/chapter4/papers_pdfs/2512.10942v2.pdf`.

Primary source: https://arxiv.org/abs/2512.10942

Use in this chapter: one cross-modal example showing that the target representation need not be visual. VL-JEPA predicts a text embedding from visual context and can decode language when a text output is required. It is not treated as an action-conditioned physical dynamics model.

### V-JEPA 2.1

Mur-Labadia, L. et al. (2026). “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.” arXiv:2603.14482, version 3.

Supplied file: `rough-draft/chapter4/papers_pdfs/2603.14482v3_compressed.pdf`.

Primary source: https://arxiv.org/abs/2603.14482

Official implementation: https://github.com/facebookresearch/vjepa2

Use in this chapter: dense visible- and masked-token supervision, deep self-supervision and the distinction between dense representation learning and action-conditioned robot prediction. Any planning comparison must retain the predictor and planner settings from the paper.

### VISReg

Wu, H., Balestriero, R., and Levine, M. (2026). “Variance-Invariance-Sketching Regularization for JEPA Training.” arXiv:2606.02572, version 1.

Supplied file: `rough-draft/chapter4/papers_pdfs/2606.02572v1 (1).pdf`.

Primary source: https://arxiv.org/abs/2606.02572

Project/source pointer: https://haiyuwu.github.io/visreg/

Use in this chapter: explicit separation of centre, scale and shape; stop-gradient in shape normalization; sliced-Wasserstein projections; scale-gradient analysis; compute dependence on batch size, embedding dimension and number of slices; and the paper's bounded experimental claims.

### Autonomous machine intelligence

LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. Version 0.9.2, 27 June 2022. OpenReview: BZ5a1r-kVsf.

Supplied file: `rough-draft/chapter4/papers_pdfs/10356_a_path_towards_autonomous_mach.pdf`.

Primary source: https://openreview.net/forum?id=BZ5a1r-kVsf

Use in this chapter: energy-based compatibility, latent variables for alternative futures, the six-module agent proposal, Mode 1 and Mode 2, Mode-2-to-Mode-1 distillation, associative short-term memory, hierarchical prediction and the Configurator. The document is identified as a position paper and research proposal.

## Additional background citations used in Chapter 4

These papers are cited briefly for mechanisms already introduced in Chapter 3. The Chapter 4 discussion is limited to the role each mechanism plays in the collapse argument.

- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). “A Simple Framework for Contrastive Learning of Visual Representations.” *Proceedings of the 37th International Conference on Machine Learning (ICML)*. arXiv:2002.05709.
- Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., and Joulin, A. (2020). “Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.” *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2006.09882.
- Grill, J.-B. et al. (2020). “Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.” *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2006.07733.
- Chen, X. and He, K. (2021). “Exploring Simple Siamese Representation Learning.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv:2011.10566.
- Zbontar, J. et al. (2021). “Barlow Twins: Self-Supervised Learning via Redundancy Reduction.” *Proceedings of the 38th International Conference on Machine Learning (ICML)*. arXiv:2103.03230.
- Bardes, A., Ponce, J., and LeCun, Y. (2022). “VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.” *International Conference on Learning Representations (ICLR)*. arXiv:2105.04906.

## Source-review record and access limitation

The eight supplied filenames were mapped to paper identities by matching their arXiv identifiers, version labels and title-page metadata. The original PDF files remain in `rough-draft/chapter4/papers_pdfs/` and have not been modified.

The GitHub repository connector can enumerate the PDF files and return their sizes and blob identifiers, but its file retrieval interface accepts UTF-8 text only and does not expose the binary bytes of these PDFs. The review therefore used the corresponding versioned primary-source text, including appendices and references where available. The access distinction is recorded below.

| Supplied PDF | Exact-version source examined | Limitation |
|---|---|---|
| I-JEPA v3 | Full arXiv v3 HTML text, including method, experiments and references; official implementation inspected for target-feature selection and mask handling | Repository PDF binary could not be downloaded through the connector; page images were not inspected |
| Original V-JEPA v1 | Full arXiv v1 HTML text, including method, masking variants, experiments and references | Repository PDF binary could not be downloaded through the connector; page images were not inspected |
| V-JEPA 2 v1 | Full arXiv v1 HTML text, including action-conditioned post-training, model architecture, objectives, planning, experiments and appendices | Repository PDF binary could not be downloaded through the connector; page images were not inspected |
| LeJEPA v3 | Full arXiv v3 HTML text, including theoretical results, proofs, SIGReg, experiments, appendices and references | The repository PDF binary could not be downloaded through the connector; the exact-version text was examined instead |
| VL-JEPA v2 | Full arXiv v2 HTML text, including architecture, objective, evaluations and references | Repository PDF binary could not be downloaded through the connector; page images were not inspected |
| V-JEPA 2.1 v3 | Full arXiv v3 HTML text, including dense objective, deep supervision, planning comparisons, experiments and references | Repository PDF binary could not be downloaded through the connector; page images were not inspected |
| VISReg v1 | Full arXiv HTML text available from the versioned abstract page, including core method equations, objective, complexity, experiments, appendices and references; official project page also checked | Direct PDF bytes were not available through the connector; page images were not inspected |
| LeCun v0.9.2 | Full text extracted from the OpenReview PDF source, including the six-module architecture, Mode 1/2, planning, memory, H-JEPA, Configurator and references | The repository's PDF bytes were not downloaded; the OpenReview PDF text was examined, but page images were not visually inspected |

The review consulted the exact-version source text available for each paper and checked the equations, methods, experimental claims and limitations used in this chapter. It is **not** a claim that every word of all eight repository PDF binaries has been read or that every page and figure has been visually inspected. The GitHub connector rejects these binary blobs as non-UTF-8 data, so the PDFs could not be materialized in this environment. Text extraction can omit figure labels, layout-dependent notation, footnotes or other material. A literal page-by-page PDF audit remains open and must be completed when the binary files are available through a PDF-capable file interface.

The paper-specific technical claims in the chapter are based on the primary sources listed above. Implementation claims are attributed to the official code or algorithm where relevant. A code path is not treated as an experimental reproduction, and no new experiments are claimed in this chapter.

## How the sources are used

The chapter follows the dependencies of the argument rather than the publication timeline. I-JEPA and V-JEPA establish what spatial and spatiotemporal feature prediction ask a model to do. LeJEPA supplies a statistical analysis of downstream estimator risk and a distributional regularizer. VISReg gives a distinct scale-and-shape decomposition. V-JEPA 2 separates action-free visual pretraining from action-conditioned robot post-training. V-JEPA 2.1 changes the density of feature supervision without removing the need for a controlled predictor. LeCun's position paper is used to discuss the responsibilities surrounding a world model, not as evidence that a complete six-module agent has been built. VL-JEPA is restricted to a short cross-modal example.

The central derivations are written independently in the chapter and then checked against the cited sources. A source's empirical result is retained only with enough context to distinguish its metric and protocol from the chapter's mathematical claims.

## Additional source and attribution notes

- Chapter 3 already provides the detailed background for VICReg, Barlow Twins, BYOL, SimSiam and related SSL methods. They are used here only when needed to explain why second-order statistics, branch asymmetry or view agreement are insufficient for the present argument.
- EB-JEPA is a library and set of examples rather than a substitute for the foundational primary papers. It may be cited for implementation context if the text discusses its examples, but its repository should not be treated as a theorem or as a universal energy-based JEPA recipe.
- Code and artwork licenses must be checked independently. A paper citation does not grant permission to reuse repository code or figures. No code or paper artwork is reproduced by the Markdown draft.
- Reported experimental numbers must retain the benchmark, split, readout, planner, action horizon, search budget and deployment conditions needed to interpret them. A representation score is not a planning score.
