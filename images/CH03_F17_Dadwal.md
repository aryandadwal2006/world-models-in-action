# Figure 3.12: VICReg separates agreement, variance and covariance

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, panel labels used by the book layout, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout inside the artwork.

**Purpose:** Show how VICReg separates three complementary constraints: paired-view invariance, a per-dimension standard-deviation floor, and a penalty on off-diagonal within-view covariance. Make the failure mode addressed by each term distinct. The variance term prevents a constant representation from satisfying every term simultaneously under the stated positive-margin condition, but neither variance nor covariance guarantees semantic correctness.

**Format and print legibility:** Use a wide landscape canvas approximately 3500 × 2200 px, white background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Use a precise, flat 2D academic schematic suitable for a Manning technical textbook. At original resolution, target 64–76 px for principal labels and 52–60 px for equations and supporting notes. Keep formulas fully readable at printed book-column width. Use generous whitespace, consistent line weights, precise alignments, and a clear hierarchy.

## Composition

Use three equally legible panels arranged left to right, titled **Invariance**, **Variance**, and **Covariance**. Beneath or to the right, combine the three terms into the full VICReg objective. Each panel should show the statistic it acts on, the penalty it defines, and one brief limitation or explanation.

The panels are conceptually connected, but they must not be depicted as interchangeable operations on one generic matrix. The invariance term compares two views; the variance and covariance terms are computed separately on each view's batch.

### Panel 1: invariance between paired views

Show two aligned batches of feature vectors \(Z\) and \(Z'\), with rows corresponding to paired views of the same source examples. Use simple paired points or vector tiles to show that a row in \(Z\) should match its paired row in \(Z'\). Highlight one corresponding pair with a restrained blue accent. Add small connector lines between corresponding samples and label the relationship “paired views”.

Display the invariance term exactly:
\[
s(Z,Z')=\frac{1}{N}\lVert Z-Z'\rVert_F^2.
\]

Label it **Paired-view agreement**. Explain briefly:
“Reduces differences between matched representations.”

Do not replace the Frobenius norm with an elementwise average over both batch and feature dimensions. The chapter's definition sums squared differences over feature coordinates and averages over examples.

Do not imply that invariance by itself preserves differences between unrelated examples or prevents collapse.

### Panel 2: variance floor per feature dimension

Show a small conceptual view of several feature coordinates measured across a batch. Contrast:
- A nearly constant dimension with points tightly clustered around one value.
- A dimension with enough spread to reach or exceed the standard-deviation margin \(\gamma\).

Use a simple horizontal standard-deviation gauge or a small one-dimensional distribution sketch, but do not draw an empirical histogram or invented measured values. The point is that the penalty acts independently on each feature dimension's standard deviation across examples.

Display the variance term:
\[
v(Z)=\frac{1}{d}\sum_{j=1}^{d}
\max\!\left(0,\gamma-\sqrt{\mathrm{Var}(Z_j)+\epsilon}\right).
\]

Show that a dimension below the margin \(\gamma\) receives a positive hinge penalty, whereas a dimension that reaches the margin contributes zero to this term. Label the threshold \(\gamma\) and the measured standard deviation clearly. Use a simple hinge illustration only if it can remain correct and legible; otherwise use the threshold gauge.

Add a brief explanation:
“Penalizes dimensions whose standard deviation falls below \(\gamma\).”

Mention, if space permits:
“The term enforces a floor; it does not reward unlimited variance.”

The full objective contains \(v(Z)+v(Z')\); this panel may show \(v(Z)\) as the representative term, while the final objective must include both views.

### Panel 3: covariance penalty within a view

Show one batch of centered feature vectors from a single view and a small covariance matrix. Explicitly label it **Within-view covariance** to distinguish it from the cross-view correlation matrix used in Barlow Twins.

Use a simple two-coordinate example to contrast:
- **Redundant coordinates:** two varying coordinates move together, indicating high covariance.
- **Less linearly redundant coordinates:** the relationship between the coordinates is weaker, indicating a smaller off-diagonal covariance.

Do not claim this removes nonlinear dependence. The sketch is conceptual and is not measured data.

Display the covariance term:
\[
c(Z)=\frac{1}{d}\sum_{i\ne j}\mathrm{Cov}(Z)_{ij}^{2}.
\]

Show a small symmetric covariance matrix with its diagonal entries visually de-emphasized and the off-diagonal entries highlighted as the penalized cells. Directly label:
“Penalize off-diagonal covariance.”

If including the covariance definition, use:
\[
\mathrm{Cov}(Z)_{ij}
=\frac{1}{N-1}\sum_{b=1}^{N}
(z_{b,i}-\mu_i)(z_{b,j}-\mu_j).
\]
Only include it if readable; it is secondary to the covariance penalty and the distinction from cross-view correlation.

Add a brief explanation:
“Reduces linear redundancy between different dimensions.”

Do not imply that off-diagonal covariance penalties force independence or semantic factors.

## Full objective

Beneath the three panels, show the complete weighted-sum objective, prominently and accurately:

\[
L_{\mathrm{VICReg}}
=\lambda s(Z,Z')
+\mu\,[v(Z)+v(Z')]
+\nu\,[c(Z)+c(Z')].
\]

Label \(\lambda,\mu,\nu\) as the relative weights of the invariance, variance, and covariance terms. Use connectors or subtle grouping to map each objective term to its corresponding panel. Do not exchange or omit the weights, and do not merge the variance and covariance terms into a single “anti-collapse loss.”

### Shared takeaway

Include one compact annotation:
“Agreement, sufficient spread, and low linear redundancy are different constraints.”

A short limitation note can follow:
“These statistics do not guarantee that the retained variation is task-relevant.”

Keep both notes legible but subordinate to the equations and main panels.

## Optional conceptual counterexample

If the lower margin of the figure permits without crowding, include a small and simple two-coordinate cue:
- Two duplicate, varying coordinates can satisfy the variance floor but incur covariance penalty.
- A constant second coordinate can have little covariance with the first but incur a variance penalty.

Do not add numeric values. If this inset forces the main formulas or labels to shrink, omit it; the three panels and full objective take priority.

## Required conceptual accuracy

1. Invariance compares paired representations \(Z\) and \(Z'\) and averages the per-example squared feature-vector difference.
2. The variance term acts on each feature dimension's standard deviation over the batch. It uses the unsquared hinge \(\max(0,\gamma-\sqrt{\mathrm{Var}(Z_j)+\epsilon})\), averaged over dimensions.
3. A dimension at or above the margin contributes zero to the variance penalty; the term enforces a floor, not ever-increasing variance.
4. The full objective includes variance and covariance terms for both \(Z\) and \(Z'\).
5. The covariance term penalizes squared off-diagonal entries of the within-view empirical covariance matrix.
6. The covariance used for \(c(Z)\) is computed after centering feature columns. Its off-diagonal entries describe linear co-variation between dimensions within one view's batch.
7. Distinguish this within-view covariance from Barlow Twins' cross-view cross-correlation matrix.
8. A constant representation yields zero invariance loss and zero off-diagonal covariance, but a positive variance penalty when \(\gamma>\sqrt{\epsilon}\).
9. Variance and covariance terms are complementary, not interchangeable.
10. Neither adequate variance nor low linear covariance establishes semantic meaning, predictive sufficiency, or task-relevant state information.
11. Do not claim the method guarantees optimization success or universally prevents every undesirable representation.
12. Do not invent numerical measurements, empirical scatter plots, benchmark values, or training results.

## Visual style and restrictions

Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only accent. Use flat 2D shapes, aligned equations, direct labels, and ample whitespace.

Keep separate visual conventions for paired-view comparison, the per-dimension variance margin, and the within-view covariance matrix. Make it impossible to mistake the covariance panel for a cross-view correlation panel.

Do not use gradients, shadows, 3D effects, Mermaid, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, unnecessary icons, or fake data visualizations. Avoid dense matrices and complicated distributions. Every visual needs to clarify one of the three terms.

## Acceptance checks

1. All three panels represent different constraints and state what each penalizes.
2. The invariance term uses the per-example squared feature-vector difference and batch-average reduction.
3. The variance term includes the correct unsquared hinge, margin \(\gamma\), and standard deviation with \(\epsilon\).
4. The covariance term penalizes off-diagonal within-view covariance, not cross-view correlation.
5. The full objective has the correct weights and includes variance and covariance terms for both views.
6. The variance and covariance constraints are not presented as guarantees of semantic correctness.
7. All equations and labels remain readable at printed book-column width.
8. The artwork contains no figure number, title, caption, or invented empirical claim.

## Caption

Figure 3.12. VICReg separates agreement, variance and covariance.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.