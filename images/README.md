# Chapter 3 figures and source assets

Figure stems follow the Manning chapter template's `CH03_F##_Dadwal` convention. The manuscript sets `:imagesdir: ../images/`; image paths in the AsciiDoc are relative to this directory.

## Artwork embedded in the manuscript

- **Figure 3.1 — `CH03_F01_Dadwal.svg`**: objective-centred map of self-supervised learning signals, informed by Gui et al. (2024), Uelwer et al. (2025), and V-JEPA 2 (2025).
- **Figure 3.2 — `CH03_F02_Dadwal.svg`**: original schematic of a useful representation and the aliasing problem.
- **Figure 3.3 — `CH03_F03_Dadwal.png`**: rendered observations from the selected DeepMind Control Suite tasks.
- **Figure 3.4 — `CH03_F04_Dadwal.svg`**: corrected pretext-task comparison: rotation prediction, jigsaw prediction, and colorization.
- **Figure 3.5 — `CH03_F05_Dadwal.svg`**: composite schematic connecting anchor/positive/negative pairs to the NT-Xent candidate-classification pipeline.
- **Figure 3.6 — `CH03_F07_Dadwal.png`**: committed NT-Xent training-loss plot; the loss is still decreasing at the final epoch and does not demonstrate convergence.
- **Figure 3.7 — `CH03_Aliasing_Diagnostic.svg`**: single-frame versus two-frame state-probe results.
- **Figure 3.8 — `CH03_F09_Dadwal.svg`**: endpoint feature-spread comparison from committed collapse-demo metrics.
- **Figure 3.9 — `CH03_F10_Dadwal.svg`**: conceptual comparison of negative pairs, asymmetric training paths, and variance/covariance constraints.
- **Figure 3.10 — `CH03_F11_Dadwal.png`**: committed VICReg total-loss and component curves.
- **Figure 3.11 — `CH03_F12_Dadwal.svg`** plus **`CH03_F12_Dadwal_Source.png`**: MAE architecture and reconstruction examples.
- **Figure 3.12 — `CH03_State_Probe_Comparison.svg`**: mean state-probe position/velocity scores with seed-level standard-deviation error bars.
- **Figure 3.13 — `CH03_F13_Dadwal.svg`**: latent-space prediction versus pixel reconstruction.
- **Figure 3.14 — `CH03_F14_Dadwal.png`**: committed two-task latent-dimension sweep.
- **Figure 3.15 — `CH03_F15_Dadwal.svg`**: CPC context and candidate-future latent scoring. It is a retrieval experiment, not an action-conditioned physical rollout.
- **Figure 3.16 — `CH03_CPC_Evaluation.svg`**: consolidated CPC view of state probes, horizon-wise candidate separation, and top-1/top-5 retrieval rates. Top-1 rates are 95.0%, 94.1%, and 92.4% at horizons one through three; top-5 is 100% at each horizon under the stated in-batch protocol.
- **Figure 3.17 — `CH03_F17_Dadwal.svg`**: encoder as one component of a larger control loop; it does not itself model transitions or plan.

Empirical PNGs come from committed experiment outputs. Vector schematics are original conceptual illustrations.

## Retained but not embedded in this revision

- **`CH03_F06_Dadwal.svg`**: former standalone NT-Xent classification diagram; its content is folded into Figure 3.5.
- **`CH03_F08_Dadwal.png`** and its source copy: before/after PCA projection, dropped because the qualitative projection added little beyond the quantitative state probes.
- **`CH03_F16_Dadwal.png`**: former standalone CPC retrieval plot; its rates are stated in prose and summarized in Figure 3.16.
- **`CH03_F01_Dadwal_Reference_2024.png`**: archived 2024 taxonomy image, not embedded.
- **`CH03_F05_Dadwal_Source.png`**: legacy pair-illustration source, retained for provenance.
- **`CH03_F09_Dadwal_PanelA.png`** and **`CH03_F09_Dadwal_PanelB.png`**: earlier collapse panels retained for provenance.
- Same-stem Markdown files are original art-production briefs retained for audit history; they are not reader-facing placeholders.

## Readiness notes

All 17 numbered figure anchors resolve to artwork. Four numbered tables cover notation, anti-collapse mechanisms, objective trade-offs, and the Chapter 2 contract-to-evidence summary. Figure numbers follow source order, not the numeric suffix in asset filenames.
