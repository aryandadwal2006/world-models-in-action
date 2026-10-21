# Chapter 3 images and figure placeholders

All figure stems follow the attached Manning template: `CH03_F##_Dadwal`. The AsciiDoc source sets `:imagesdir: ../images/`, so image macros resolve relative to this directory.

## Assets used directly by the current composite AsciiDoc

- `CH03_F01_Dadwal.svg`: updated, editable taxonomy diagram, informed by the supplied 2024 reference and 2025 work on visual self-supervision.
- `CH03_F02_Dadwal.svg`: original encoder-contract and aliasing diagram.
- `CH03_F03_Dadwal.png`: actual DMC task renders from `ch03/figures/fig03_02_dmc_tasks.png`.
- `CH03_F07_Dadwal.png`: recorded NT-Xent curve from `ch03/figures/fig03_06_contrastive_loss.png`; panel labels should be polished for publication, and the curve is still falling at the last epoch.
- `CH03_F14_Dadwal.png`: recorded two-task latent-dimension sweep from `ch03/figures/fig03_14_bottleneck_sweep.png`.
- `CH03_F16_Dadwal.png`: primary matched-16-frame CPC retrieval plot from `ch03/figures/fig03_16_cpc_future_retrieval.png`.
- `CH03_State_Probe_Comparison.svg`: editable two-panel plot of mean position and velocity probe R² with standard-deviation error bars. Values are sourced from the chapter's committed comparison results.
- `CH03_Aliasing_Diagnostic.svg`: paired point-and-error-bar plot comparing the single-frame and two-frame linear probes.
- `CH03_CPC_Evaluation.svg`: combined CPC figure with per-variable static/context probe scores and horizon-wise InfoNCE loss and score margin.

The four empirical PNGs listed above are byte-for-byte copies of the repository's canonical runner outputs. They are not reconstructed by hand.

## Source assets retained for planned composites/replots

- `CH03_F01_Dadwal_Reference_2024.png`: compact archival copy of the image supplied with the manuscript request; reference only, not inserted as the current taxonomy.
- `CH03_F05_Dadwal_Source.png`: current view-pair image from `ch03/figures/fig03_04_view_pairs.png`; it lacks the required batch-level positive/negative schematic.
- `CH03_F08_Dadwal_Source.png`: current latent projection from `ch03/figures/fig03_07_latent_projection_seed_0.png`; replot with wrapped angle bins and episode-stratified sampling.
- `CH03_F09_Dadwal_PanelA.png`, `CH03_F09_Dadwal_PanelB.png`: recorded collapse outputs from `fig03_09_collapse_demo.png` and `fig03_09_objective_space.png`; the final composite requires a log y-axis for feature spread.
- `CH03_F11_Dadwal_Source.png`: committed VICReg force/loss plot; add the required concept schematic to make the final composite.
- `CH03_F12_Dadwal_Source.png`: committed MAE reconstruction output; combine it with the masked-autoencoder architecture schematic.

## Markdown production placeholders

The composite `.adoc` explicitly marks these slots as unfinished and links to their same-stem Markdown briefs:

`CH03_F04_Dadwal.md`, `CH03_F05_Dadwal.md`, `CH03_F06_Dadwal.md`, `CH03_F08_Dadwal.md`, `CH03_F09_Dadwal.md`, `CH03_F10_Dadwal.md`, `CH03_F11_Dadwal.md`, `CH03_F12_Dadwal.md`, `CH03_F13_Dadwal.md`, `CH03_F15_Dadwal.md`, and `CH03_F17_Dadwal.md`.

No generic SVG placeholder cards are shipped. The Markdown briefs identify the requested panels, exact data/source constraints, credit lines, and acceptance checks. Replace the `.md`-marked production notes in the AsciiDoc only after the corresponding publication-quality figure is ready.