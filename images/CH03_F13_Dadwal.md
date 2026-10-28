# Figure 3.8: Why agreement alone permits collapse

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, panel labels used by the book layout, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, figure title, caption, page number, or simulated book page in the artwork.

**Purpose:** Give a direct loss-level counterexample to the idea that matching representations from two views is sufficient for learning useful representations. Show two mappings that both achieve zero view-agreement loss: a constant mapping that erases all input distinctions, and a nonconstant mapping that keeps different source examples separated. The same objective value cannot distinguish these cases.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px. White background, dark charcoal line work and text, with #3B6FB6 as the only restrained accent. Use a precise, clean, flat 2D academic schematic style suitable for a Manning technical textbook. At original resolution, target approximately 64–76 px for main labels and 52–60 px for equations and explanatory notes. Use generous whitespace, clean alignment, consistent line weights, and readable mathematical notation. Everything important must remain legible at printed book-column width.

## Composition

Make a balanced two-panel comparison with a clear left-to-right logic:

- Left panel: **Constant mapping (collapsed)**.
- Right panel: **Nonconstant mapping (distinctions retained)**.

Use those as internal diagram headings only; no figure title or caption should appear.

Both panels must show distinct source examples, two views per source, and their encoded representations. Both panels must make the view-agreement loss equal to zero. The contrast between them is that the constant mapping preserves no distinction among source examples, whereas the nonconstant mapping does.

### Left panel: constant mapping

At the left edge, show three or four visibly different source examples labeled \(x_A\), \(x_B\), \(x_C\), and optionally \(x_D\). They may be simple geometric or abstract visual tiles. Each source produces two distinct views, such as \(x_i\) and \(x_i'\), connected to an encoder or mapping block.

Route every view to the same representation point or tile, labeled:
\[
z_i=z_i'=c \qquad \text{for every }i.
\]

Show multiple incoming arrows converging on the same single representation. The convergence must clearly indicate that different source inputs are mapped to the same constant vector. Do not merely draw several points close together: this is complete collapse, so all representations are exactly the same in the illustration.

Label the result:

“All inputs map to the same representation.”

Then show the agreement objective and its value:
\[
\mathcal L_{\mathrm{inv}}
=\frac{1}{N}\sum_{i=1}^{N}\|z_i-z_i'\|_2^2
=0.
\]

Use a subtle blue highlight for the common representation \(c\) or the loss value. Do not use a bright color-filled warning box.

Add a concise consequence label:
“View agreement is perfect, but input distinctions disappear.”

### Right panel: nonconstant mapping with agreement

Show the same number of distinct source examples, labeled \(x_A\), \(x_B\), \(x_C\), and optionally \(x_D\). Again, each source produces two views, \(x_i\) and \(x_i'\).

Map each pair of views to one shared representation for that source, but keep representations of different source examples visibly distinct:
\[
z_A=z_A', \qquad z_B=z_B', \qquad z_C=z_C',
\]
with \(z_A\ne z_B\), \(z_B\ne z_C\), and visibly distinct representation positions.

Use separate point or vector tiles for each source identity. Paired views may connect to the same point, but points for different source examples must remain separated. Do not imply a particular semantic meaning for these representation directions.

Show the same agreement objective:
\[
\mathcal L_{\mathrm{inv}}
=\frac{1}{N}\sum_{i=1}^{N}\|z_i-z_i'\|_2^2
=0.
\]

Add a concise label:
“Matched views agree; different sources remain distinguishable.”

The two panels should show the same objective value with different representational consequences. The central takeaway is that agreement alone permits both mappings.

## Shared comparison cue

Between the panels or beneath them, include a compact statement that is readable at book width:

“Same zero agreement loss; different information retained.”

This statement should tie together the counterexample without introducing a new assumption. It is an objective-level demonstration, not a report of a training run.

## Required conceptual accuracy

1. The figure illustrates the invariance-only objective:
   \[
   \mathcal L_{\mathrm{inv}}=\frac{1}{N}\sum_{i=1}^{N}\|z_i-z_i'\|_2^2.
   \]
2. In the collapsed case, all views from all source examples map to the same constant \(c\), so every term in the loss is zero.
3. In the nonconstant case, the two views from each source agree, so the same loss is zero, while distinct source examples retain distinct representations.
4. This is a counterexample about what the loss permits, not an empirical claim that every non-contrastive method collapses.
5. Do not imply the nonconstant representations are necessarily semantic or predictive. They only demonstrate that agreement can coexist with differences among examples.
6. Distinguish complete collapse (all outputs identical) from simply having nearby outputs. The left panel must show exact equality.
7. Do not introduce variance, covariance, predictor, stop-gradient, EMA, teacher/student, or other anti-collapse mechanisms. Those are covered by the following method sections.
8. Do not add invented numerical data or claims about convergence.

## Visual style and restrictions

Use a white background, grayscale-safe dark outlines and text, and #3B6FB6 as the only accent. Flat 2D shapes, simple distinct source tiles, clear mapping arrows, consistent point sizes, and spacious panels.

Use direct labels and math rather than a decorative legend. Ensure that source identity is traceable from input tiles to the relevant views and representation. Avoid tangled arrows and dense text.

Do not use gradients, shadows, 3D effects, Mermaid, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, or unnecessary icons. Do not depict a model training curve or pretend this is an empirical run.

## Acceptance checks

1. The constant panel clearly maps every distinct source and view to one identical representation \(c\).
2. The nonconstant panel maps paired views from each source to matching representations while keeping different sources distinct.
3. Both panels correctly display zero for the same invariance-only loss.
4. The contrast explains why agreement alone does not rule out collapse.
5. The figure is explicitly conceptual and does not imply every non-contrastive method collapses.
6. Mathematical notation is accurate and legible at printed book-column width.
7. The image contains no figure number, title, or caption.

## Caption

Figure 3.8. Why agreement alone permits collapse.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation; do not invent empirical results.