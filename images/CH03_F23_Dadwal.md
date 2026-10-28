# Figure 3.13: One visible context, several masked-prediction targets

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, panel letters, page number, or simulated book layout in the artwork.

**Purpose:** Separate the mask from the choice of prediction target. Use one common masked input and one visible context, then branch into three alternative target spaces: pixel values, discrete visual-token IDs, and teacher-derived representations. The common input and mask stay fixed; what changes is the target that the predictor must recover. These are alternative methods, not a single model that uses every target mechanism.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px, with a white background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Use a carefully typeset, flat 2D academic schematic suitable for a Manning technical textbook. At original resolution, target approximately 64–76 px for principal labels and 52–60 px for equations and secondary annotations. Use consistent line weights, precise alignment, generous whitespace, and a clear hierarchy. All important labels must remain readable at printed book-column width.

## Overall composition

Build a left-to-right diagram with three stages:

1. A single input divided into visible and masked patches.
2. An encoder that processes only the visible context.
3. Three alternative branches that differ in the target space and loss.

Keep the mask fixed across the three branches. The figure's comparison should be about target construction, not different masking patterns, different input images, or different encoder architectures.

Label the common input-to-encoder path:
“Same visible context; same masked positions.”

Use a simple patch grid. Distinguish visible patches from masked patches with direct labels, outlines, and neutral shading. A restrained #3B6FB6 accent may mark the masked prediction positions, but the figure must remain clear in grayscale.

Do not imply that all three branches belong to a single model or share the same decoder, tokenizer, or teacher network.

## Common input and visible context

At the left, show one simple image-like patch grid split into:
- **Visible patches** \(V\)
- **Masked patches** \(M\)

Use the same grid structure in all three branches. Show only visible patches entering a block labeled **Encoder**. The masked positions should not enter the encoder as observed image content.

From the encoder, draw a shared conceptual representation labeled **Visible-context representation** or \(h_V\). This is a schematic common context, not a claim that all methods have identical encoders or internal features.

A small, readable annotation may state:
“The mask specifies where the target is withheld; the objective specifies what is predicted.”

## Branch A: pixel values

Label this branch:

**Pixel target**

Show the visible-context representation leading to a prediction head for the masked locations. At the masked positions, show the target as the original patch pixel values and the prediction as reconstructed pixel values.

Use a small masked-patch comparison: observed target patch versus predicted patch. Keep it abstract and schematic; do not use realistic photographs or invented model outputs.

Display a concise loss cue:
\[
\text{Masked-patch regression, e.g. squared error}
\]

If an equation is included, use:
\[
\ell_i=\frac{1}{P}\lVert x_i-\hat x_i\rVert_2^2,
\]
where \(x_i\) is the original masked patch, \(\hat x_i\) is its prediction, and \(P\) is the number of scalar pixel and channel values in that patch.

Label this target:
“Original pixel values at masked positions.”

MAE and SimMIM may appear as brief examples of pixel-target methods. Do not imply that their entire architectures or implementation details are identical.

## Branch B: discrete visual-token IDs

Label this branch:

**Discrete-code target**

Show the original image patches passing conceptually through a separate **Visual tokenizer** to produce discrete code IDs. The tokenizer path should supply target labels, not serve as the visible-context encoder's prediction output.

Show a masked location with a discrete target such as \(k_i\), represented by a simple token tile labeled **Code ID \(k_i\)**. The predictor assigns probabilities over possible codes.

Display the loss cue:
\[
\text{Cross-entropy over discrete code IDs}
\]
and, if room permits:
\[
\ell_i=-\log p_\theta(k_i\mid X_V).
\]

Label the target:
“Tokenizer code for each masked patch.”

BEiT may be named as an example. Make clear that the tokenizer is a target-construction component in this approach, not a component that must be shared by the other branches.

Do not use familiar category icons or semantic labels for code IDs. A code is a learned discrete target, not automatically a human-interpretable class.

## Branch C: teacher-derived features or distributions

Label this branch:

**Teacher-derived target**

Show a separate teacher path receiving an appropriate source view or image context and producing a target for the masked positions. Keep this teacher target path visually separate from the student's visible-context encoder.

Show two compact alternatives within this target family only if space allows:
- A **teacher distribution** over learned prototype outputs, such as the patch-level target in iBOT.
- A **continuous contextual feature** \(y_i^t\), such as the target used in data2vec.

Use direct labels to distinguish the two forms. Do not imply that iBOT and data2vec use the same target representation or the same loss.

For a teacher distribution, use the loss cue:
\[
\text{Cross-entropy with a teacher distribution}
\]

For a continuous teacher feature, use:
\[
\ell_i=\lVert \hat y_i-y_i^t\rVert_2^2.
\]

Label the target:
“Teacher-derived distribution or contextual feature.”

The teacher may be shown as a separate box or target-generation path, but do not impose a single shared teacher architecture across all teacher-target methods. Clearly show that the teacher supplies a target while the student prediction path processes the visible context.

Do not include any JEPA architecture, future-latent prediction, or action-conditioned predictive model in this figure.

## Shared comparison across branches

Align the three branch outputs horizontally or vertically so readers can compare the target types directly:

| Target family | What is predicted | Typical loss family |
|---|---|---|
| Pixel values | Original pixels at masked patches | Regression |
| Discrete codes | Tokenizer-generated code ID | Cross-entropy |
| Teacher targets | Distribution or contextual feature | Cross-entropy or regression, depending on target |

Render this information as compact aligned labels, not as a dense table if that reduces print legibility.

The target families must remain clearly distinguished. In particular:
- Pixel regression predicts numeric patch values.
- Discrete-code prediction assigns probability to the tokenizer's code.
- Teacher-based prediction matches a target distribution or continuous feature, depending on the method.

Add one concise takeaway:
“Same mask; different prediction target.”

## Required conceptual accuracy

1. The visible context and masked positions remain the same across the three conceptual branches.
2. Only visible input content enters the encoder in this schematic; the masked positions supply the locations where targets are predicted.
3. The figure compares alternative target-construction choices rather than one combined architecture.
4. Pixel-target methods predict original pixel values at masked positions and use a regression-style loss.
5. Discrete-code methods predict a learned tokenizer ID and use cross-entropy.
6. Teacher-derived methods may predict a distribution over prototypes or a continuous contextualized feature. Do not collapse these into one identical target or loss.
7. MAE and SimMIM may be examples of pixel reconstruction; BEiT is an example of discrete-token prediction; iBOT and data2vec are examples of teacher-derived targets. These are brief examples, not claims that their full methods are architecturally identical.
8. Do not imply that all methods use the same decoder, tokenizer, teacher, or target-generation network.
9. Do not imply that a discrete code is a human-defined semantic category.
10. Do not include JEPA architecture or other Chapter 4 material.
11. Do not invent empirical values, benchmark results, or performance comparisons.

## Visual style and restrictions

Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent. Flat 2D shapes, consistent patch grids, clearly labeled arrows, aligned branches, and generous whitespace.

Use the same masked input in all branches. Make target spaces visually distinct through their content and labels, not through additional colors. Keep any teacher path separate from the student encoder path.

Do not use gradients, shadows, 3D effects, Mermaid, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, realistic photo collages, or unnecessary icons. Avoid overly detailed architecture diagrams; the point is to compare the target, not all internals of every method.

## Acceptance checks

1. One common input is split into visible and masked patches.
2. The same visible context feeds three alternative target constructions.
3. The pixel branch predicts original masked-patch pixel values using a regression-style loss.
4. The discrete-code branch predicts tokenizer-generated code IDs using cross-entropy.
5. The teacher-target branch distinguishes distributions from continuous contextual features.
6. The diagram does not imply all methods share the same decoder, tokenizer, or teacher.
7. The figure includes no JEPA architecture or Chapter 4 technical material.
8. All text and mathematical notation remain legible at printed book-column width.
9. No figure number, title, caption, or invented empirical result appears inside the artwork.

## Caption

Figure 3.13. One visible context, several masked-prediction targets.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.