# Chapter 3 figure and table production register

This is the production source of truth for the Chapter 3 figures and tables. The uploaded Word draft contains figure briefs and paths, not embedded figure images. Insert only committed repository artifacts or newly drawn diagrams specified below. Do not generate synthetic replacement observations, invent results, or infer missing numerical values. Use white backgrounds, fine dark line art, restrained hatching, sentence-case labels, and a grayscale-safe design consistent with Chapter 2. Do not encode distinctions by color alone. All plot panels must have readable axis labels, units where relevant, legends, uncertainty definitions, and task/configuration labels.

## Figure 3.1 — The encoder's contract

**Type:** New original illustration. **Placement:** Section 3.1, immediately after the aliasing example. **Aspect:** Landscape, approximately 2:1.

**Instruction:** Draw three panels across the top and a narrow strip across the bottom. Left: label “Observation ($x$)” and show a real-looking cart-pole frame. Separate “Relevant to the state” (cart position, pole angle, cart velocity, pole angular velocity) from “Potentially irrelevant detail” (lighting, floor texture, background noise). Middle: show an encoder $f_\theta$ that compresses the image, with relevant information passing through and nuisance details being discarded. Do not assign one literal latent coordinate to each physical variable. Right: show $z$, a vector of $d$ numbers, and annotate “preserve relevant information” and “discard unnecessary detail.” Bottom strip: two visually identical frames with different underlying cart velocity directions both map to the same representation dot. Label the failure “Different states, same representation: a distinction is lost.” Keep arrows and labels simple enough to survive grayscale reproduction.

**Caption:** Figure 3.1 The encoder’s contract

## Figure 3.2 — Rendered tasks from the Control Suite

**Type:** Repository output. **Asset:** `../ch03/figures/fig03_02_dmc_tasks.png`. **Generator:** `ch03/01_collect_dmc_data.py`.

**Instruction:** Use actual 64 × 64 RGB frames rendered from the DeepMind Control Suite. Do not substitute illustrations, stock photographs, or generated imagery. Lay out one frame each for “Cartpole balance,” “Finger spin,” and “Cheetah run.” Under the title, identify the evaluation-state dimensions as 4, 6, and 18 respectively. Update existing panel titles to sentence case; preserve aspect ratio and do not stretch frames. State in the caption that these are the book's own renders of the simulator. The first-use DMC tech-report citation belongs in a footnote in §3.1.

**Caption:** Figure 3.2 Rendered observations from the three tasks

## Figure 3.3 — Pretext tasks on one frame

**Type:** New redrawn diagram, adapted from Gui et al.; credit required. **Placement:** Section 3.2.

**Instruction:** Four equal panels in one row: source cart-pole frame; rotation prediction; jigsaw prediction; colorization. Reuse one actual 64 × 64 Control Suite frame. Upscale using nearest-neighbor only. Rotation panel: show 0°, 90°, 180°, and 270° copies with the four-class target. Jigsaw: split a frame into a 3 × 3 grid and show one selected shuffle, labelled “predict the permutation,” not all possible permutations. Colorization: pair a grayscale input with the corresponding target color channels or the source-color reconstruction. Make arrows point from transformed input to the task's target. Keep the image content consistent across panels.

**Credit:** Redrawn and adapted from Gui et al., *A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends* (2024); example images are the book's own Control Suite renders.

**Caption:** Figure 3.3 Three pretext tasks from one frame

## Figure 3.4 — Positive and negative views in a batch

**Type:** Recompose repository output into a conceptual schematic. **Source material:** `../ch03/figures/fig03_04_view_pairs.png`.

**Instruction:** Three columns. Left, show three distinct source frames, $x_1,x_2,x_3$. Middle, each passes through a box labelled “shift and intensity jitter,” producing two independent views per source and six numbered views. Right, select view 1 as anchor; connect it with a solid double arrow to view 2, its positive; connect dashed arrows from view 1 to views 3–6, grouped as negatives. Make the source identity mapping explicit. Do not use only the existing image-pair panel: it must communicate the in-batch candidate structure, not merely show two visually similar views.

**Caption:** Figure 3.4 Constructing positives and negatives in a batch

## Figure 3.5 — InfoNCE as classification

**Type:** New original diagram. **Placement:** Section 3.3.2.

**Instruction:** Four horizontal stages: (1) one anchor frame with representation $z_1$; (2) the positive and two negative candidates with cosine similarities 0.80, 0.10, and −0.30; (3) division by temperature $\tau=0.5$, yielding logits 1.6, 0.2, and −0.6; (4) softmax probabilities 0.737, 0.182, and 0.082. Draw probability bars in grayscale. Use a solid outline for the positive and hatching for negatives. Include the single-anchor loss $-\ln(0.737)\approx0.306$. State along the bottom: “Three-class classification: identify the anchor's positive.” Keep the example values consistent with the derivation in the text.

**Caption:** Figure 3.5 NT-Xent for one anchor

## Figure 3.6 — Contrastive training curve

**Type:** Repository output. **Asset:** `../ch03/figures/fig03_06_contrastive_loss.png`. **Metric source:** `../ch03/results/contrastive_loss_summary.json`.

**Instruction:** Use the committed three-seed mean and one-standard-deviation band; do not sketch a new curve from prose. Use a sentence-case title (or no title if the caption handles it), x-axis “Epoch,” y-axis “NT-Xent loss,” and a legend defining the band as standard deviation across three trained seeds. State that the final epoch is still decreasing and is the end of the chosen budget, not a demonstrated convergence point. Check legibility at book-column width.

**Caption:** Figure 3.6 NT-Xent training loss on Cartpole

## Figure 3.7 — Validation representation projection

**Type:** Replot required before publication. **Current source to update:** `../ch03/figures/fig03_07_latent_projection_seed_0.png`; plotting routine in `ch03/04_train_contrastive.py`.

**Instruction:** Regenerate after wrapping the pole angle into $(-\pi,\pi]$. Select validation samples evenly across validation episodes, not from a prefix. Use marker shapes for four angle bins (upright, two sideways bins, and hanging) and grayscale intensity for cart position. Explain that the two-dimensional projection is qualitative and is not a measurement of linear decodability. Add consistent panels for before and after training, identical axes, and a legend explaining all marker/gray encodings. Never use color as the only distinction.

**Caption:** Figure 3.7 PCA projections of validation representations

## Figure 3.8 — Reserved production slot

The design deliberately reserves this number as an optional slot merged into Figures 3.6/3.7 if page pressure requires it. **Do not insert a blank placeholder or cite a missing figure.** The prose and final book's figure list should omit this slot unless the author explicitly commissions additional evidence and then renumbers all following figures in one pass. The fixed current design register keeps downstream numbers 3.9–3.17 unchanged.

## Figure 3.9 — Collapse with and without negatives

**Type:** Combine two committed outputs into a two-panel figure. **Assets:** `../ch03/figures/fig03_09_collapse_demo.png` and `../ch03/figures/fig03_09_objective_space.png`. **Metrics:** `../ch03/results/collapse_demo_metrics.json`.

**Instruction:** Left panel: per-dimension encoder feature standard deviation over 15 epochs for attraction-only training and the contrastive condition, using the three-seed run. Use a logarithmic y-axis or a clearly separated inset so the attraction-only collapse near $7.79\times10^{-5}$ remains visible next to the contrastive feature spread around 2.42. Right panel: show the existing objective-space diagnostic. Preserve the recorded output; do not synthesize a new trajectory. Define error bands if shown and identify SD across trained seeds. The caption and surrounding prose should say “constant-like collapse in this setup,” not claim a theorem from one experiment.

**Caption:** Figure 3.9 Feature collapse with and without negatives

## Figure 3.10 — Siamese methods without negatives

**Type:** New redrawn schematic, adapted from Gui et al.; credit required.

**Instruction:** Three parallel panels. SimSiam: two augmented views pass through shared online encoder/projector paths; one branch includes the predictor and a stop-gradient target. BYOL: online encoder/projector/predictor compared against a slowly updated target encoder/projector; show the exponential-moving-average update on parameters, not gradients. Barlow Twins: two view embeddings feed a cross-correlation matrix; the diagonal is driven toward one and off-diagonal entries toward zero. Use solid versus dashed arrows to distinguish gradient flow and parameter updates. Avoid implying that these mechanisms are otherwise identical or that they guarantee success on every dataset.

**Credit:** Redrawn and adapted from Gui et al. (2024), with the three mechanisms simplified for comparison.

**Caption:** Figure 3.10 Three mechanisms for avoiding collapse

## Figure 3.11 — VICReg's three forces

**Type:** Composite: new original conceptual schematic plus recorded curves. **Asset:** `../ch03/figures/fig03_11_vicreg_forces.png`.

**Instruction:** Top or left: show three separately labelled forces on a pair of batch embeddings: invariance pulls paired views together; the variance hinge prevents each coordinate's standard deviation falling below the floor $\gamma$; covariance regularization suppresses off-diagonal correlations. Do not describe the variance term as maximizing all variance without bound. Bottom or right: preserve the repository-recorded per-term training curves, with each term labelled. Match the source plot's seeds and training budget; explicitly say this standalone VICReg run is not the matched-budget Table 3.4 comparison. Use line style and markers rather than color alone.

**Caption:** Figure 3.11 VICReg's invariance, variance, and covariance terms

## Figure 3.12 — Masked autoencoder and reconstructions

**Type:** Composite: new schematic plus existing model output. **Asset:** `../ch03/figures/fig03_12_mae_reconstruction.png`.

**Instruction:** Keep the repository's actual three-row, four-example reconstruction panel intact. Pair it with a newly drawn architecture panel: split a 64 × 64 frame into patches; mask 75% of patches; send only visible patches to the encoder; attach mask tokens and restore original patch order for the decoder; reconstruct pixel patches; calculate mean-squared error only on masked patches. Identify input, visible patches, masked positions, reconstruction and loss mask. Do not make the reconstruction look sharper or cleaner than the stored output, and do not imply that the decoder's pooled projection was trained by reconstruction loss.

**Credit:** Architecture schematic redrawn and adapted from He et al.'s masked autoencoder work and Gui et al. (2024); reconstruction examples are from the book's recorded run.

**Caption:** Figure 3.12 A masked autoencoder and its reconstructions

## Figure 3.13 — Joint embedding versus reconstruction

**Type:** New redrawn conceptual comparison, adapted from Gui et al.; credit required.

**Instruction:** Two equal panels. Left, “Joint embedding”: two augmented views of the same source pass through shared encoders (or online/target encoders where relevant), and the objective acts on the embeddings. Indicate that the loss is not pixel reconstruction. Right, “Reconstruction”: an image with masked patches goes through an encoder and decoder, and the objective compares predicted pixels against original pixels in masked regions. Annotate the information each objective pays to retain: shared view content versus recoverable visual detail. Avoid claiming that either family always yields a better physical state.

**Credit:** Redrawn and adapted from Gui et al. (2024).

**Caption:** Figure 3.13 Joint embedding and pixel reconstruction

## Figure 3.14 — Latent dimension sweep

**Type:** Repository output. **Asset:** `../ch03/figures/fig03_14_bottleneck_sweep.png`. **Metrics:** `../ch03/results/bottleneck_sweep_metrics.json`.

**Instruction:** Plot frozen-feature state-probe $R^2$ against latent dimension $d\in\{2,4,8,16,64,256\}$ for both Cartpole Balance and Cheetah Run. Show mean across the three trained seeds and a clearly labelled one-standard-deviation band; distinguish tasks by marker and line style, not color alone. Keep a common y-axis so the curves can be compared. Make the state dimensions explicit (4 for Cartpole, 18 for Cheetah). Do not force an expected plateau if the recorded Cheetah curve is still rising at the largest tested dimension. Caption and text must state the architecture, frame budget, and three-seed scope from the result artifact.

**Caption:** Figure 3.14 State-probe performance across latent dimensions

## Figure 3.15 — CPC predicts future-latent identity

**Type:** New original schematic, designed to pair visually with Chapter 4's JEPA figure.

**Instruction:** Draw a chronological row of frames $x_1,\ldots,x_t$ entering one shared visual encoder to produce $z_1,\ldots,z_t$. Only past/current latents enter a **unidirectional** GRU, which emits context $c_t$. From $c_t$, branch to three horizon-specific compatibility matrices $W_1,W_2,W_3$. Each branch scores the appropriate future latent against in-batch candidate latents using $f(z_{t+k},c_t)=\exp(z_{t+k}^{\top}W_kc_t)$. Use a vertical barrier or arrow labels to make clear that the context does not see future frames. Show negative candidates and an identification loss; do not draw an action input, physical-state decoder, or planner, because this experiment has none.

**Caption:** Figure 3.15 Contrastive predictive coding identifies a future latent

## Figure 3.16 — Held-out future-latent retrieval

**Type:** Repository output. **Asset:** `../ch03/figures/fig03_16_cpc_future_retrieval.png`. **Canonical metrics:** `../ch03/results/table_03_05_cpc_metrics.json`.

**Instruction:** Use the matched-length 16-frame run. Plot top-1 retrieval by future horizon 1, 2, and 3, with mean and one-standard-deviation error bars across five trained seeds. Show the 3.5% query-weighted in-batch chance level as a reference line. Label the metric “in-batch top-1 retrieval,” not global 200-way retrieval; evaluation batches contain six batches of 32 candidates and a final batch of eight, and the 200 windows come from five episodes, not 200 independent trajectories. Do not use the superseded recursive gap-rollout curve or the 25-frame sensitivity run as the primary figure.

**Caption:** Figure 3.16 CPC future-latent retrieval by horizon

## Figure 3.17 — Encoder in the agent loop

**Type:** New original schematic styled after Chapter 1, Figure 1.1.

**Instruction:** Retain the Chapter 1 observation→encoder→state→dynamics→reward/value/planner loop. Replace the blank encoder component with “static visual encoder,” input $x_t$ (or a short frame stack), and output latent $z_t$. Put the optional recurrent wrapper around/after the visual encoder and label its hidden state $h_t$ separately from physical state. Show dynamics and reward/value as later components, using dashed outlines or “next chapter” labels for placeholders. The figure must not imply that the current Chapter 3 encoder already predicts action-conditioned transitions or performs planning.

**Caption:** Figure 3.17 The learned encoder in the agent loop

---

## Table register and data requirements

### Table 3.1 — Self-supervised objective families

Rebuild a compact comparison table from Gui et al. (2024), crediting the survey. Columns: family; representative methods; what supplies the target/positive signal; anti-collapse mechanism; information rewarded; characteristic limitation. Include pretext prediction, contrastive (SimCLR/InfoNCE), non-contrastive joint embedding (SimSiam, BYOL), redundancy reduction (Barlow Twins), variance/invariance/covariance regularization (VICReg), masked reconstruction (MAE), and temporal contrastive prediction (CPC). This table is a conceptual taxonomy, not a performance ranking or reproduction of original paper-scale outcomes. Use the Verification column to link to the chapter's own measurements: probes, collapse spread, reconstruction panel, or future-latent retrieval, as appropriate.

### Table 3.2 — Notation

Place in §3.1. Include symbol, meaning, and where used. Minimum entries: $x$ observation/frame; $y$ privileged simulator state for evaluation; $f_\theta$ static visual encoder; $z$ latent representation; $d$ latent dimension; $t$ time index; $z_t$ frame latent; $c_t$ recurrent CPC context; $h_t$ optional wrapper hidden state (do not conflate with simulator state); $N$ source frames in a contrastive batch; $K$ negatives per anchor; $\tau$ temperature; $M$ masked-patch indicator; $W_k$ horizon-specific CPC compatibility matrix; $R^2$ held-out state-probe score. Distinguish model state from physical state in words.

### Table 3.3 — State probes with one frame and two-frame stacks

Use the recorded values from `../ch03/results/table_03_03_aliasing_diagnostic.json`. Data are mean ± SD over three seeds; state variables are cart position, pole angle, cart velocity, and pole angular velocity. Preserve circular-angle scoring metadata. The current values in the manuscript are:

| State variable | Single frame $R^2$ | Two-frame stack $R^2$ | Change |
|---|---:|---:|---:|
| Cart position | 0.714 ± 0.040 | 0.707 ± 0.010 | −0.007 |
| Pole angle | 0.612 ± 0.069 | 0.586 ± 0.064 | −0.026 |
| Cart velocity | 0.006 ± 0.004 | 0.009 ± 0.007 | +0.002 |
| Pole angular velocity | −0.367 ± 0.001 | −0.375 ± 0.006 | −0.007 |

Caption and text must make clear that this specific two-frame setup did not reliably recover velocity and that this is not a general impossibility claim.

### Table 3.4 — Method-family state-probe comparison

Present as **one numbered table with two panels**. Panel A: task-level aggregate scores (position and velocity); panel B: per-variable scores in state-vector order. Cite the current source `../ch03/results/table_03_04_objective_comparison.json`. Values below are mean ± population SD across three seeds. The table compares complete configurations, not losses in isolation: supervised, contrastive, and VICReg use convolutional encoders; MAE uses a patch-transformer encoder/decoder.

**Panel A, Cartpole and Finger Spin:**

| Task | Configuration | Position $R^2$ | Velocity $R^2$ |
|---|---|---:|---:|
| Cartpole Balance | Supervised-pretrained encoder + frozen-feature linear probe | 0.981 ± 0.000 | −0.273 ± 0.012 |
| Cartpole Balance | Contrastive (NT-Xent) | 0.672 ± 0.001 | −0.187 ± 0.005 |
| Cartpole Balance | VICReg | 0.649 ± 0.022 | −0.183 ± 0.006 |
| Cartpole Balance | Masked autoencoder | 0.475 ± 0.012 | −0.156 ± 0.005 |
| Finger Spin | Supervised-pretrained encoder + frozen-feature linear probe | 0.995 ± 0.001 | −0.114 ± 0.010 |
| Finger Spin | Contrastive (NT-Xent) | 0.714 ± 0.029 | −0.007 ± 0.002 |
| Finger Spin | VICReg | 0.679 ± 0.040 | −0.005 ± 0.004 |
| Finger Spin | Masked autoencoder | 0.383 ± 0.010 | −0.034 ± 0.001 |

**Panel B, per-variable scores:** keep every state coordinate in state-vector order, so Finger Spin has six entries. Its third position coordinate uses the separately recorded π-periodic score: supervised 0.999 ± 0.000, contrastive 0.996 ± 0.001, VICReg 0.941 ± 0.003, and MAE 0.939 ± 0.023 (mean ± population SD over three seeds). Mark these entries as π-periodic and exclude this coordinate from the aggregate position score because its absolute 2π hinge angle is not identifiable from a static render. Keep the panel caption and source note under Table 3.4, not as an unnumbered separate comparison.

**Essential label:** “Supervised-pretrained encoder + frozen-feature linear probe.” The supervised encoder was trained with state labels, then a separate frozen-feature probe was fitted. Do not mislabel the number as the trained head's held-out prediction score or as an “oracle” result.

### Table 3.5 — CPC state probes and future-latent retrieval

Present as **one numbered table with two panels**, not independent tables “3.5A” and “3.5B.” Panel A: static encoder vs. CPC context for current and next-state probes. Panel B: future-latent retrieval and CPC-native training loss/margin. Report mean ± SD over five trained seeds; windows are 200 non-overlapping episode-safe windows, 40 from each of five validation episodes, evaluated with in-batch candidates. Top-1 chance is query-weighted 3.5% (six batches of 32 plus one batch of eight), not 1/200.

**Panel A:**

| State variable | Static $z_t\to s_t$ | Context $c_t\to s_t$ | Static $z_t\to s_{t+1}$ | Context $c_t\to s_{t+1}$ |
|---|---:|---:|---:|---:|
| Cart position | 0.770 ± 0.033 | 0.805 ± 0.084 | 0.770 ± 0.033 | 0.806 ± 0.084 |
| Pole angle | 0.515 ± 0.037 | 0.560 ± 0.051 | 0.514 ± 0.039 | 0.559 ± 0.050 |
| Cart velocity | 0.013 ± 0.006 | 0.016 ± 0.006 | 0.014 ± 0.006 | 0.016 ± 0.006 |
| Pole angular velocity | −0.347 ± 0.017 | −0.348 ± 0.030 | −0.345 ± 0.017 | −0.346 ± 0.029 |

**Panel B:**

| Future horizon | Top-1 retrieval | Top-5 retrieval | InfoNCE loss | Positive–hardest-negative margin |
|---|---:|---:|---:|---:|
| 1 step | 95.04% ± 0.84 pp | 100% | 0.115 ± 0.016 | 12.96 ± 0.65 |
| 2 steps | 94.14% ± 0.72 pp | 100% | 0.133 ± 0.017 | 11.52 ± 0.54 |
| 3 steps | 92.44% ± 0.51 pp | 100% | 0.171 ± 0.018 | 9.50 ± 0.50 |

Interpretation must state: high future-latent retrieval is not physical-state accuracy, action-conditioned transition prediction, recursive rollout quality, or planning utility. Top-5 of 100% is within the small in-batch candidate sets, not an all-futures guarantee.

### Table 3.6 — What each objective pays to preserve

Use the existing synthesis table from the draft. Columns: objective; information rewarded; information it may discard; characteristic failure invited. This is a conceptual synthesis, not a leaderboard. Convert the separate Chapter 2 contract mapping table into bullets under §3.8 rather than giving it an extra numbered table.

---

## Production integrity checklist

- The Word upload has no embedded images. Confirm every final figure is inserted in layout; paths above are sources, not embedded assets.
- Use one numbered-table identity for Table 3.4 and Table 3.5, each with a panel A/B substructure; eliminate labels “Table 3.5A” and “Table 3.5B” as separate numbering.
- Table 3.1 and Table 3.2 are new additions and must appear in the manuscript. The current Word source did not contain either.
- Numbered listing registry is 3.1–3.13. The VICReg code excerpt is unnumbered supplemental material so it cannot collide with the required MAE Listing 3.10.
- Equations are Markdown display math, not literal square-bracketed Word placeholders. Keep equation numbers 3.1–3.20 in the approved sequence.
- Verify captions and all numeric claims against canonical JSON before production. Do not overwrite canonical outputs when rerunning scripts.