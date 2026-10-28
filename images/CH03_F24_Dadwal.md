# Figure 3.14: MAE's visible-only encoder and masked-patch decoder

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout in the artwork.

**Purpose:** Explain the asymmetric design of a masked autoencoder (MAE): patchify an image, mask a large subset of patches, send only visible patch tokens through the encoder, then restore the original sequence positions in a smaller decoder using encoded visible tokens plus learned mask tokens. The decoder predicts pixel values, but the reconstruction loss is computed only at masked patch locations. The encoder representation, not the decoder reconstruction, is typically transferred to downstream tasks.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px, with a white background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Use a precisely aligned, flat 2D academic schematic suitable for a Manning technical textbook. At original resolution, target approximately 64–76 px for principal labels and 52–60 px for equations and secondary annotations. Maintain generous whitespace and consistent line weights. All patch labels, mask markers, sequence positions, and equations must remain readable at printed book-column width. Simplify unnecessary decoration rather than shrinking the type.

## Overall composition

Build a left-to-right pipeline with one clear branching distinction between the visible-only encoder input and the decoder's restored full-length sequence:

1. Image patchification.
2. Random masking.
3. Visible patch embeddings only enter the encoder.
4. The decoder receives encoded visible tokens and learned mask tokens inserted at missing positions, with positional information restored.
5. The decoder predicts patch pixels, but only masked patches contribute to the loss.
6. The encoder output is shown as the representation typically reused downstream.

Use one consistent patch grid throughout so the reader can track visible and masked positions. Make the figure's main idea immediately apparent:

**Mask tokens enter the decoder, not the encoder. The reconstruction loss is applied only to masked patches.**

## 1. Image patchification and masking

At the far left, show a simple image divided into a regular grid of patches. Label the operation **Patchify and embed**. The output should be a sequence or grid of patch tokens with their original positions identifiable.

Next, show random masking. Use a representative high mask ratio, and label it accurately as:
“Original MAE recipe: 75% of patches masked.”

Use neutral gray or a light diagonal pattern to indicate masked positions and dark outlines for visible patches. A restrained #3B6FB6 accent may highlight the visible-token route.

The 75% figure describes the original MAE recipe, not a universal optimum. Do not imply every MAE variant, dataset, resolution or task must use that ratio. Keep the patch examples conceptual; no measured results or photo collage is needed.

Label the two subsets:
- **Visible patches**
- **Masked positions**

## 2. Visible-only encoder

Draw a sequence containing only the visible patch tokens entering a box labeled:

**Encoder**

Place a clear annotation beside the encoder input:
“Visible tokens only.”

Do not draw mask tokens, placeholders representing masked content, or masked patch embeddings entering the encoder. The visible-token sequence should be shorter than the original full patch sequence.

Show the encoder outputs as encoded visible tokens. If helpful, label the output \(h_V\). Preserve the correspondence between their original locations so it is clear they will be placed back into the full sequence in the decoder.

Add a short explanation:
“Computes features only for visible patches.”

This is the asymmetric computational advantage: the encoder avoids processing the masked tokens.

## 3. Decoder sequence restoration

After the encoder, show a separate box labeled:

**Decoder input assembly**

Its inputs must be visibly distinct:
- Encoded visible tokens from the encoder.
- Learned mask tokens for positions that were masked.

Show the learned mask tokens being inserted at the missing positions, restoring the full sequence length and original positional order. Indicate that positional information is restored or supplied to the decoder. Do not imply that original image content from masked patches is inserted into the decoder.

Then route the reconstructed full-length token sequence into a smaller box labeled:

**Lightweight decoder**

The encoder and decoder should have different visual prominence: the encoder is the main feature extractor, while the decoder is smaller and reconstructs masked pixels. Do not use a fake scale comparison that implies exact model parameter counts; the labels are enough.

The decoder must receive both encoded visible tokens and learned mask tokens. It must not receive the original masked patch pixels as an input shortcut.

## 4. Masked-patch reconstruction and loss

Show the decoder producing pixel predictions at patch positions across the reconstructed sequence. Clearly distinguish:
- Predictions at masked positions, which are evaluated by the loss.
- Predictions at visible positions, which are ignored by the standard MAE reconstruction loss.

Use a small pair of aligned patch grids:
- Original patch targets.
- Decoder patch predictions.

Connect only the masked positions to the loss. Make visible positions visually faded or crossed out of the loss path, with the label:
“Visible-patch errors excluded.”

Display the patchwise mean-squared error:
\[
\ell_i=\frac{1}{P}\lVert x_i-\hat x_i\rVert_2^2,
\]
where \(P\) is the number of scalar pixel and channel values in patch \(i\).

Then show the masked-only reconstruction objective:
\[
L_{\mathrm{MAE}}=
\frac{\sum_i m_i\ell_i}{\sum_i m_i},
\qquad
m_i=
\begin{cases}
1,&\text{patch }i\text{ is masked},\\
0,&\text{otherwise}.
\end{cases}
\]

Keep the equations large and ensure the masking indicator is clearly defined. The formula averages the per-patch mean-squared error over masked patches only. The per-patch squared error is averaged over its scalar pixel and channel values first; the outer loss is then averaged over masked patches.

The loss path must never include visible patches. Do not display the loss as an average over every patch in the image.

## 5. Downstream representation interface

Below or at the end of the pipeline, show the encoder's output branching to a separate label:

**Encoder representation for downstream use**

Show an arrow from the encoded visible patch tokens or their aggregated representation to this label. Do not show an arrow from the decoder reconstruction to the downstream representation.

Add the concise note:
“Transfer the encoder, not the reconstruction decoder.”

This is a typical downstream transfer path, not a claim that every downstream task uses exactly the same pooling or token interface. Do not specify an unsupported aggregation operation. If the figure uses a single vector \(h\), label it “example pooled representation” and make clear it is schematic; otherwise keep the patch-token output.

## Visual hierarchy and arrow conventions

Use a clear visual grammar:
- Solid dark arrows: patch and token forward flow.
- #3B6FB6 accent: visible-token route or encoder feature output.
- Neutral mask shading: masked positions.
- Separate decoder assembly arrows: encoded visible tokens plus inserted learned mask tokens.
- Loss connectors: masked patches only.

Use labels and direct arrows in addition to color. The encoder path and decoder restoration path must not cross unnecessarily.

Make the encoder visibly consume fewer tokens than the decoder handles. The original full patch sequence is restored only after the encoder, in the decoder input assembly.

## Required mathematical and conceptual accuracy

1. MAE first divides an image into patches and selects visible versus masked positions.
2. The encoder processes only visible patches; masked tokens do not pass through the encoder.
3. The decoder input is assembled from encoded visible tokens plus learned mask tokens at the masked positions, with original positional ordering restored.
4. The decoder predicts pixel values at patch positions.
5. The standard reconstruction loss is computed only over masked patches.
6. The patch-level mean-squared error divides by the scalar pixel and channel values \(P\) within each patch; the outer objective averages that error over masked patches.
7. The 75% masking ratio is characteristic of the original MAE recipe, not a universal optimum.
8. The encoder representation, rather than decoder reconstructions, is typically transferred to downstream tasks.
9. Do not imply that pixel reconstruction guarantees a representation preserves every variable useful for downstream prediction.
10. Do not depict the decoder receiving the original masked pixels or mask tokens being processed by the encoder.
11. Do not add details from unrelated methods or Chapter 4 JEPA material.
12. Do not invent training metrics, empirical examples, or performance claims.

## Visual style and restrictions

Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent color. Use flat 2D shapes, crisp patch grids, direct labels, aligned equations, and generous whitespace.

Do not use gradients, shadows, 3D effects, Mermaid diagrams, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, or realistic photo collages. Avoid dense token grids or overly elaborate transformer blocks. The goal is to teach the encoder/decoder asymmetry and masked-only loss.

## Acceptance checks

1. Patchification and masking are clearly shown.
2. Only visible patch tokens enter the encoder.
3. Learned mask tokens are inserted after the encoder at missing positions.
4. Positional order is restored before the decoder processes the full sequence.
5. The decoder reconstructs pixels, but loss connectors lead only from masked patch predictions to target pixels.
6. The per-patch error and masked-only objective use the correct normalization.
7. The 75% masking ratio is labeled as the original recipe, not a universal rule.
8. The encoder representation is identified as the typical downstream interface, not the decoder reconstruction.
9. No figure number, title, caption, invented empirical value, or Chapter 4 content appears inside the artwork.

## Caption

Figure 3.14. MAE's visible-only encoder and masked-patch decoder.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.