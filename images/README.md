# Chapter 3 figures and source assets

Figure stems use the Manning chapter template's CH03_F##_Dadwal convention. The manuscript sets :imagesdir: ../images/; image paths in the AsciiDoc are relative to this directory.

## Artwork embedded in the manuscript

- **CH03_F01_Dadwal.svg**: objective-centred map of self-supervised learning signals, informed by Gui et al. (2024), Uelwer et al. (2025), and V-JEPA 2 (2025).
- **CH03_F02_Dadwal.svg**: original schematic of a useful representation and the aliasing problem.
- **CH03_F03_Dadwal.png**: rendered observations from the selected DeepMind Control Suite tasks.
- **CH03_F04_Dadwal.svg**: rotation prediction, colorization, and future-frame prediction from a common source observation.
- **CH03_F05_Dadwal.svg**: anchor, positive view, and negative views in a contrastive batch.
- **CH03_F06_Dadwal.svg**: NT-Xent as a temperature-scaled candidate-classification objective.
- **CH03_F07_Dadwal.png**: committed NT-Xent training-loss plot. The loss is still decreasing at the last epoch, so the figure does not show demonstrated convergence.
- **CH03_F08_Dadwal_Source.png**: current before/after PCA projections from the committed seed-0 run. **Known limitation:** the plot generator currently fits a separate PCA basis to each panel and does not include a clear legend. The panels must not be read as sharing coordinates. The generator is being corrected; this asset should be regenerated before production handoff.
- **CH03_F09_Dadwal.svg**: endpoint spread comparison from committed collapse-demo metrics, with a logarithmic axis that makes both the tiny attraction-only spread and contrastive spread visible. The caption/body explicitly limit what this statistic establishes.
- **CH03_F10_Dadwal.svg**: conceptual comparison of negative pairs, asymmetric training paths, and variance/covariance constraints.
- **CH03_F11_Dadwal_Source.png**: committed VICReg total-loss and component curves.
- **CH03_F12_Dadwal.svg**: masked autoencoder architecture; paired with CH03_F12_Dadwal_Source.png, which shows original inputs, masking, and recorded reconstructions.
- **CH03_State_Probe_Comparison.svg**: mean state-probe position/velocity R² with standard-deviation error bars, sourced from committed comparison outputs.
- **CH03_F13_Dadwal.svg**: latent-space prediction versus pixel reconstruction.
- **CH03_F14_Dadwal.png**: committed two-task latent-dimension sweep.
- **CH03_F15_Dadwal.svg**: CPC context and candidate-future latent scoring.
- **CH03_F16_Dadwal.png**: primary CPC future-latent retrieval plot from the committed evaluation.
- **CH03_CPC_Evaluation.svg**: complementary plot showing CPC state probes and horizon-wise evaluation metrics.
- **CH03_F17_Dadwal.svg**: encoder as one component of a larger control loop.
- **CH03_Aliasing_Diagnostic.svg**: single-frame versus two-frame probe results.

The PNGs listed above are based on committed experiment outputs, not recreated by hand. The vector schematics were made specifically for this manuscript. They are conceptual diagrams, not renderings of unpublished model results.

## Archived and source assets

- **CH03_F01_Dadwal_Reference_2024.png**: archived 2024 taxonomy image supplied as a reference; it is not inserted in the current manuscript.
- **CH03_F05_Dadwal_Source.png**: legacy image of view pairs, retained for provenance; the new batch-level schematic is used instead.
- The same-stem Markdown briefs are original art-production specifications retained for audit history. They are not referenced as live placeholders in the reader-facing AsciiDoc.
- **CH03_F09_Dadwal_PanelA.png** and **CH03_F09_Dadwal_PanelB.png**: earlier collapse panels retained for provenance; the logarithmic SVG summary is embedded instead.

## Figure readiness

All 17 numbered figure anchors resolve to real artwork; the manuscript no longer includes visible “figure to be produced” notes. Figure 3.8 remains the one known empirical-figure cleanup item because the existing PCA plot's two panels use independently fitted axes. That plot must be regenerated with a shared projection and a legend before treating the chapter as production-ready.
