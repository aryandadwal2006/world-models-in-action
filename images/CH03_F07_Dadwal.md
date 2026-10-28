# Figure 3.5: Alignment and uniformity are distinct geometric requirements

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, panel letter, page number, or surrounding page layout in the artwork.

**Purpose:** Contrast two different geometric pressures on normalized representations: matched positive views should lie close together (alignment), while independent samples should not all crowd into a small region of the unit sphere (uniformity). Show that these diagnostics describe distinct properties and do not guarantee that the representation preserves task-relevant information.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px, with a white background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Aim for the visual precision of a strong machine-learning paper adapted for a Manning technical textbook: simple geometry, clean alignments, consistent line weights, and generous whitespace. At original resolution, principal labels should be approximately 64–76 px and equations and secondary notes approximately 52–60 px. Keep equations large and unobstructed; avoid tiny labels or dense paragraphs. The diagram must remain clear when reduced to printed book-column width.

## Composition

Use two main panels of equal visual importance, one titled **Alignment** and one titled **Uniformity**. Panel headings are explanatory labels, not figure metadata. The geometry is conceptual, not measured data.

### Left panel: alignment of positive pairs

Show a simple 2D conceptual projection of a normalized embedding space. Include several distinct pairs of points, each representing two augmented views of one source example. Use thin, restrained connector lines and short arrows indicating the pressure to reduce the distance within each positive pair. Highlight one matched pair with #3B6FB6; make other pairs visible in dark grayscale-safe line work.

Directly label the points or small groups “view” and “matched view”; distinguish matched pairs from unrelated points using connector patterns and layout, not color alone. Show the paired points becoming close in representation space, while preserving a clear indication that they are two separate observations.

Include this compact explanation:
“Alignment: matched views are close.”

Below the sketch, display the alignment measure exactly:
\[
\mathcal L_{\mathrm{align}}
=\mathbb E_{(x,x^+)\sim p_{\mathrm{pos}}}
\left\|f(x)-f(x^+)\right\|_2^\alpha,
\qquad \alpha>0.
\]

Keep the equation large enough to read at book width. The notation should clearly refer to positive pairs sampled from \(p_{\mathrm{pos}}\). Do not add a claim that this formula is optimized directly by every self-supervised method.

### Right panel: uniformity of independent samples

Show two small, clearly separated conceptual views of normalized embeddings:

1. **Crowded:** many independent sample points cluster in one region of a unit-circle / unit-sphere projection.
2. **More spread out:** the same number of independent sample points are distributed more evenly across the displayed sphere projection.

Use the same simple point symbols and matching scale in both sketches so the contrast is geometrical rather than decorative. Label them “crowded” and “more spread out.” Add a concise label such as “independent samples” to clarify that this diagnostic concerns pairs drawn independently from the data distribution, not the positive pairs shown in the alignment panel.

Represent the unit sphere as a simple 2D schematic projection or circle boundary; do not suggest that a general high-dimensional embedding literally lives in a two-dimensional plane. Keep all plotted points on the schematic normalized-space boundary or use a clear projection convention, consistently across both sketches. Avoid axes, ticks, fitted curves, or numerical measurements because this is a conceptual drawing, not experimental data.

Include this compact explanation:
“Uniformity: independent embeddings avoid excessive crowding.”

Below the sketches, display the uniformity measure exactly:
\[
\mathcal L_{\mathrm{uniform}}
=\log\mathbb E_{x,y\sim p_{\mathrm{data}}}
\exp\!\left(-t\left\|f(x)-f(y)\right\|_2^2\right),
\qquad t>0,
\]
with \(x\) and \(y\) independently sampled.

Add a short explanatory note near the equation:
“Nearby pairs contribute more strongly to the exponential; crowding raises the measure.”

Do not turn this into a detailed derivation. The role of the formula is to connect the geometry to the diagnostic measure.

## Bottom note: what the diagrams do not guarantee

Use a compact, legible note spanning the two panels, without a colorful callout box:

“Neither diagnostic guarantees semantic correctness: a representation can align views after a task-critical feature is erased, or spread out while encoding nuisance variation.”

Keep this note visually subordinate to the diagrams and equations, but large enough for print. Do not imply that alignment and uniformity form an exact or universal decomposition of all self-supervised objectives.

## Required conceptual accuracy

- The features are normalized, so their geometry is interpreted on the unit sphere; the 2D drawing is only a schematic projection.
- Alignment concerns the distance between positive / matched views sampled from \(p_{\mathrm{pos}}\).
- Uniformity concerns independent samples from \(p_{\mathrm{data}}\). Its measure uses \(\exp(-t\|f(x)-f(y)\|_2^2)\) inside a log-expectation, with \(t>0\).
- Nearby embeddings contribute strongly to the exponential term, making a crowded distribution increase this uniformity measure.
- Keep the direction of interpretation precise: a lower alignment measure indicates matched views are closer; a lower uniformity measure under the stated setup reflects less crowding.
- Alignment and uniformity are different geometric requirements, not a universal exact decomposition of every self-supervised loss.
- Neither diagnostic alone establishes semantic correctness, predictive sufficiency, or retention of task-critical information.
- Use only conceptual points. Do not invent numeric values, empirical results, dataset labels, performance scores, or claims of optimization.

## Visual style and restrictions

Use white background, dark grayscale-safe line work, and #3B6FB6 as the only accent. Flat 2D vector-like shapes, consistent point sizes, thin but print-safe connectors, clean geometry, generous whitespace, and restrained panel separators.

Do not use gradients, shadows, 3D effects, decorative watermarks, dashboards, elaborate icons, fake chart axes, unnecessary labels, ornamental backgrounds, or simulated page layout. No extra palette. Do not rely on color alone to show matched pairs or differences between panels.

## Acceptance checks

1. The alignment panel shows matched views and the pressure to reduce within-pair distance.
2. The uniformity panel contrasts crowding with a more spread-out distribution of independent normalized samples.
3. Both equations are accurate and legible, including the expectation distributions and parameters \(\alpha>0\) and \(t>0\).
4. The figure explains that nearby embeddings contribute strongly to the uniformity measure.
5. The note prevents readers from interpreting these diagnostics as a guarantee of semantic correctness or a universal decomposition of every SSL objective.
6. The geometric sketches are clearly conceptual and contain no invented empirical data.
7. The figure contains no figure number, title, or caption.

## Caption

Figure 3.5. Alignment and uniformity are distinct geometric requirements.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter's explanation and cited sources; do not invent empirical data.