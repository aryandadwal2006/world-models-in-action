# Figure 3.15: The training representation and the downstream interface

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Do not render a figure number, title, caption, or manuscript prose inside the artwork. These are supplied by the book layout.

## Purpose

Explain two related but distinct choices: which tensor a self-supervised pretraining objective directly constrains, and which representation a later task consumes. Make clear that, in a common architecture with a projection head, the loss acts on \(z=g(h)\), while downstream use often selects backbone features \(h\). Then show three alternative interfaces for those backbone features: a pooled global vector, a spatial grid of patch features, or an aggregated temporal context. These are alternatives, not a ranking and not a pipeline that every method must follow.

## Exact information and notation

Use these symbols consistently:

- \(x\): input observation.
- \(f\): encoder or backbone.
- \(h=f(x)\): backbone representation.
- \(g\): optional projection head.
- \(z=g(h)\): objective-space representation when the projection head is used.
- \(\mathcal{L}_{\mathrm{pre}}(z)\): pretraining loss, explicitly attached to \(z\), not \(h\).
- \(h_1,\ldots,h_P\): patch-level features when the backbone yields spatial tokens.
- \(h_{\mathrm{global}}=\frac{1}{P}\sum_{i=1}^{P}h_i\): illustrative average-pooled global representation.
- \(c_t=a(h_{t-K:t})\): an illustrative temporal context formed by aggregating a history of backbone features. Treat \(a\) as a generic aggregation operation, not a prescribed architecture.
- \(q(\cdot)\): a generic downstream predictor, used only to show where the chosen interface goes.

The projector is optional. Label the upper pathway as a common recipe “when a projection head is used,” not as the universal structure of self-supervised learning. Do not imply that \(h\), \(z\), the global vector, the spatial grid, and the temporal context are the same tensor or have the same geometry.

## Composition

Use a clean, landscape, two-lane diagram with generous whitespace and a clear left-to-right flow.

### Upper lane: the pretraining objective

Place a small lane label at far left: **Pretraining: what the loss sees**.

Draw the main path:

\[
x \longrightarrow f \longrightarrow h \longrightarrow g \longrightarrow z
\]

Attach a separate loss box to \(z\), labeled \(\mathcal{L}_{\mathrm{pre}}(z)\). Use a small note above or below the projector path: “common recipe with a projection head.” Make the loss attachment unambiguous: it originates from \(z\), not \(h\). Distinguish the backbone and projector using separate boxes. Do not draw the loss as acting directly on both \(h\) and \(z\).

### Lower lane: the downstream interface

Place a small lane label at far left: **Downstream: what the predictor receives**.

Show a restrained branch from the backbone representation \(h\) to a central junction. Do not route this branch through \(z\) or the pretraining-loss box. From the junction, fan out into three evenly aligned alternatives, each with a short heading and an abstract, accurate mini-glyph:

1. **Global vector**. Show patch-feature tokens entering a pooling operator and becoming one compact vector. Label it \(h_{\mathrm{global}}=\frac{1}{P}\sum_i h_i\). Add a short note: “compact; patch indices are removed.”
2. **Spatial grid**. Show a small regular grid of feature cells, with a few cells labelled by index or a single notation \(\{h_i\}_{i=1}^{P}\). Add a short note: “retains spatial indexing.”
3. **Temporal context**. Show a short ordered strip of feature tokens over time entering a neutral aggregation block \(a\), and leaving as \(c_t=a(h_{t-K:t})\). Add a short note: “summarizes a feature history.”

Mark the three branches as **alternative interfaces**. Keep branch sizes visually balanced. Do not stack them as if a global vector is later converted into a grid and then into temporal context. Each branch represents a different choice made for a downstream task.

Route each alternative into a common, generic **downstream predictor** box \(q(\cdot)\) and then to a simple output \(\hat{y}\). A shared destination is acceptable if the graphic clearly shows that one interface is selected for a given task, rather than feeding all three tensors jointly by default. Do not add a specific task, benchmark, robotic scene, action input, or dynamics-model training pipeline.

## Visual hierarchy and typography

- Landscape canvas, approximately 3200 × 2000 px.
- White background. Dark charcoal text and outlines; grayscale-safe throughout.
- Use #3B6FB6 as the only restrained accent, mainly for the backbone \(h\), the bypass into downstream interfaces, and selected interface outlines. All diagram meaning must remain legible in grayscale.
- Flat 2D vector-like drawing, consistent line weights, straight connectors and simple arrowheads.
- Main lane labels and major node labels should be about 64–76 px at source size. Equations and secondary labels should be about 52–60 px. Small notes must not be reduced to infographic microtype.
- Use a highly legible sans-serif for labels and a consistent mathematical typeface for equations. Keep equations on one line where feasible.
- Use concise labels, balanced spacing, aligned alternative branches, and whitespace between the two lanes.
- No gradients, shadows, perspective, 3D effects, decorative illustrations, stock imagery, screenshots, Mermaid styling, watermarks, or ornamental framing.

## Exclusions and accuracy checks

- Do not render the figure number, title, caption, citation, chapter number, or surrounding manuscript prose.
- Do not describe the projector as mandatory. Not every method uses one.
- Do not imply the pretraining loss directly constrains every pairwise relation in \(h\), or that low pretraining loss establishes downstream usefulness.
- Do not imply a global vector, patch grid, and temporal context are interchangeable or universally ordered by quality.
- Do not imply a spatial grid guarantees correct geometry, or that temporal aggregation automatically preserves predictive information.
- Do not add method families, named algorithms, performance numbers, or unsupported dimensions.
- Do not depict an action-conditioned dynamics model or an action input. This figure concerns representation interfaces, not dynamics-model training.
- At final book-column size, every label, symbol, arrow, and branch distinction must remain legible.

## Caption

Figure 3.15. The training representation and the downstream interface.

## Production note

Create original explanatory artwork suitable for a technical machine-learning textbook. Prioritize mathematical faithfulness, readable print typography, and a clear distinction between the representation constrained during pretraining and the feature interface used downstream.