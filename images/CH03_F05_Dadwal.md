# Figure 3.4: InfoNCE: candidate probabilities and gradients

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, panel letter, page number, or surrounding page layout in the artwork.

**Purpose:** Connect an illustrative three-candidate InfoNCE calculation to the cross-entropy gradient on each scaled similarity logit. The reader should be able to trace the same candidate order through similarities, temperature-scaled logits, probabilities, loss, and derivatives.

**Format and print legibility:** Create a wide landscape figure, approximately 3400 × 2000 px. Use a white background, dark charcoal text and outlines, and #3B6FB6 as the only restrained accent. The style should resemble a carefully typeset explanatory figure from a strong machine-learning paper adapted for a Manning textbook: crisp flat 2D, restrained, mathematically precise, and easy to read at printed book-column width. Use generous whitespace and aligned columns. At source resolution, target about 64–76 px for main labels and 52–60 px for equations, values, and secondary labels. Do not shrink equations to fit; allocate enough space. Ensure decimals, minus signs, subscripts, the indicator function, and gradient signs are all clearly legible.

## Composition

Use two clearly separated but visually connected panels, reading left to right. Keep the positive and two negatives in the same order throughout the full figure.

### Left panel: candidate classification

At the top, show one anchor icon or feature vector compared against exactly three candidates: one positive and two negatives. Label the similarities in the candidate order:
- positive: 0.80
- negative 1: 0.10
- negative 2: −0.30

Add a small, explicit note: “Illustrative values, not empirical results.”

Next, show temperature scaling with \(\tau=0.5\), giving the logits:
- positive: \(a_y=0.80/0.5=1.6\)
- negative 1: \(a_1=0.10/0.5=0.2\)
- negative 2: \(a_2=-0.30/0.5=-0.6\)

Keep similarities and logits in distinct rows or aligned columns, with headings that prevent readers from confusing the two.

Then show a three-entry softmax distribution, in the same candidate order:
- positive: \(p_y\approx0.737\)
- negative 1: \(p_1\approx0.182\)
- negative 2: \(p_2\approx0.082\)

Show that the probabilities are normalized across exactly these three candidates. Their displayed values sum to 1.001 only because of rounding; do not imply the unrounded probabilities fail to sum to one. Do not include the anchor itself in the candidate set.

Display the single-anchor cross-entropy:
\[
\mathcal{L}=-\log p_y=-\log(0.737)\approx0.306.
\]
Use the positive probability as the target probability. Do not omit the positive from the softmax normalization.

### Right panel: derivative with respect to each scaled logit

Use the same three candidate labels and order. At the top, display the general derivative:
\[
\frac{\partial\mathcal{L}}{\partial a_k}
=p_k-\mathbf{1}[k=y].
\]

Define the indicator clearly and briefly: it equals 1 for the positive index and 0 for a negative index.

Evaluate the derivative for each candidate:
- positive: \(0.737-1=-0.263\)
- negative 1: \(0.182\)
- negative 2: \(0.082\)

Retain appropriate approximate-equality notation or make clear that these values are rounded. Keep the negative derivative magnitudes distinct: the higher-probability negative receives the larger positive derivative.

Then show a concise update-direction cue corresponding to gradient descent:
- the negative derivative for the positive logit means gradient descent increases the positive logit;
- positive derivatives for both negative logits mean gradient descent decreases those logits.

Use labeled directional arrows or restrained up/down cues next to the values, but never use arrow direction as the only explanation. State the meaning in compact text: “Gradient descent raises the positive logit and lowers negative logits.”

Near the bottom of this panel, connect logits back to unscaled similarities:
\[
a_k=s_k/\tau
\qquad\Longrightarrow\qquad
\frac{\partial\mathcal{L}}{\partial s_k}
=\frac{1}{\tau}
\left(p_k-\mathbf{1}[k=y]\right).
\]
This must correctly communicate the additional factor \(1/\tau\) when differentiating with respect to the unscaled similarity. Keep this relationship visually distinct from the logit derivative; do not mislabel logit gradients as parameter updates.

## Required conceptual accuracy

- Treat the three numerical candidates as one illustrative anchor-level calculation, not measured or benchmark data.
- The logits are the similarities divided by temperature \(\tau=0.5\).
- The softmax includes the positive and both negatives, but excludes the anchor itself.
- The probability for the positive is approximately 0.737, yielding loss approximately 0.306.
- For cross-entropy, the derivative with respect to a scaled logit is \(p_k-\mathbf{1}[k=y]\).
- The positive-logit derivative is negative; under gradient descent, its logit increases. Negative-logit derivatives are positive; their logits decrease.
- The harder negative has the larger current probability and therefore the larger derivative with respect to its logit.
- The extra \(1/\tau\) factor applies to the derivative with respect to unscaled similarity, not to the already-scaled-logit formula.
- The figure explains derivatives with respect to logits and similarities. It must not suggest that these are already gradients with respect to network parameters. The representations and shared parameters receive accumulated gradients through the full computational graph.
- Do not imply that every negative is semantically unrelated.

## Visual restrictions

White background, dark grayscale-safe line work, and #3B6FB6 as the only accent. Use consistent candidate ordering and direct labels. Keep lines and separators subtle. Use no gradients, shadows, 3D effects, decorative watermarks, simulated page layout, charts with unlabeled axes, tiny legends, or extra color families. Avoid unnecessary iconography and long prose blocks inside the image.

## Acceptance checks

1. All values match the chapter's worked example: similarities 0.80, 0.10, −0.30; temperature 0.5; logits 1.6, 0.2, −0.6; probabilities approximately 0.737, 0.182, 0.082; loss approximately 0.306.
2. The softmax candidate set contains the positive and two negatives but not the anchor.
3. The gradient equation and all three derivative signs are correct.
4. The gradient-descent direction is explained correctly and is not confused with the sign of the derivative.
5. The relationship between the logit and similarity derivatives includes exactly the factor \(1/\tau\).
6. The text remains readable at book-column width, with no overlapping content.
7. The image contains no figure number, title, caption, or invented empirical claim.

## Caption

Figure 3.4. InfoNCE: candidate probabilities and gradients.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. All numeric values are the illustrative calculation given in the chapter. Do not invent empirical data.