# Figure 3.1: The representation contract

**Purpose:** A compact reminder of Chapter 2's predictive-state contract, contrasting the distinctions a useful representation must preserve with the proxy relation a self-supervised loss optimizes.

## Copy-ready image-generation prompt

Create a publication-quality, print-ready academic schematic for a technical machine-learning textbook published by Manning. The figure must look like a carefully typeset explanatory figure from a strong machine-learning paper, adapted for a serious textbook: mathematically faithful, restrained, readable at printed book width, and designed to teach one precise idea. Do not make an infographic, poster, slide, dashboard, or decorative illustration.

### Canvas and print typography

- Landscape canvas, approximately 8:5 aspect ratio, high resolution (target 2400 × 1500 px or larger).
- White background; crisp vector-like geometry; flat 2D only.
- Use dark charcoal/black line work, muted grayscale fills, and restrained Manning blue (`#3B6FB6`) only for meaningful highlights, the shared latent, and the two predictive paths.
- Use one professional sans-serif family consistently. Keep strokes, arrows, box corners, spacing, and alignment consistent.
- The final image will be scaled to fit a book page. Use large labels: key labels roughly equivalent to 10–11 pt at print size, secondary labels no smaller than 8.5–9 pt. At 2400 px canvas width, aim for about 48–56 px for key labels and at least 40–44 px for every secondary label or mathematical expression. Do not use footnotes, tiny annotations, compressed text, or multi-line paragraphs inside the artwork. Leave generous whitespace around all text.
- Every word, subscript, symbol and arrow label must be sharp, correctly spelled, and readable. If the layout becomes crowded, remove decorative detail rather than reduce type size.

### Editorial boundary

**Draw only the diagram. Do not render a figure number, figure title, caption, descriptive prose, source note, running header, or page number inside the image.** The manuscript supplies the number and caption separately. The caption in the manuscript is “The representation contract.” Do not write “Figure 3.1” anywhere inside the artwork.

### Composition

Use two balanced panels across the canvas, with a small, quiet conceptual footer spanning the width. Preserve a clear left-to-right reading order. Each panel should have a short heading, a visual example, an encoder/representation path, and the relevant implication for future prediction. Avoid decorative framing or unnecessary icons.

**Left panel: desirable invariance to nuisance variation**

1. Heading: “Nuisance variation”.
2. Show three small, simple observations of the same underlying scene or physical configuration. Keep object position, geometry, orientation and relevant state exactly the same. Vary only irrelevant appearance factors such as illumination, background shade, or mild colour/contrast.
3. Label these inputs as “Same target-relevant state”. Make the appearance differences obvious but subtle enough that they cannot be mistaken for changes in the physical state.
4. Route all three observations through a single encoder block labelled `fθ` (use a legible theta, not the letters “theta” unless needed for font compatibility). Their paths converge to one blue latent box labelled `z`.
5. Show one shared future/target distribution downstream, not three different outcomes. Add the concise relation `p(y | x₁) = p(y | x₂)` with legible subscripts. Label the implication “Same target distribution”.
6. The visual should convey that ignoring appearance is appropriate only because those appearance changes do not alter the target distribution.

**Right panel: possible failure from over-invariance**

1. Heading: “Possible aliasing failure”.
2. Show two short histories, each as a minimal two- or three-frame sequence. Both end with an object at the same visible position, but the preceding frames indicate different motion: one is moving right and the other left. Use simple object positions and directional arrows, not detailed realistic scenes. This makes the velocity difference explicit without asserting that a single frame can never contain velocity cues.
3. Clearly label the difference “Target-relevant distinction”. Do not imply that the two histories are identical.
4. Route both histories through an overly invariant encoder labelled `fθ`. Make the two paths merge into the same blue latent box `z`, with `fθ(h₁) = fθ(h₂) = z`.
5. From the shared latent, show that the model cannot distinguish the two different future outcomes. Draw two separate possible future distributions/outcome paths above or below the shared latent, and label them “Different future distributions”. Include `p(y | h₁) ≠ p(y | h₂)`.
6. Make the merge visually read as an information-loss failure, not a success. Mark it with a restrained warning cue such as a thin red-free dark dashed outline or small cross-hatch at the merge. Do not introduce red, green, gradients, 3D, or alarm-style symbols.
7. The panel must be labelled as a **possible** failure, not an inevitable property of every encoder or every single-frame representation.

**Bottom conceptual footer**

Place two distinct, short text blocks, separated by whitespace or a thin rule:

- Label: “Predictive contract”
  Text: “Preserve distinctions that change the future distribution.”
- Label: “Proxy objective”
  Text: “Optimizes the relationship specified by the training loss.”

The two labels must be visually distinct and must not be merged into one category. Use the same typography as the panels; the explanatory sentences must stay comfortably readable.

### Mathematical and conceptual accuracy

- Use `x₁, x₂` for observations in the left panel and `h₁, h₂` for histories in the right panel.
- The left equation states equal target distributions for the illustrated nuisance variants.
- The right equations state that the encoder merges histories while their conditional future distributions differ. This is the defining inconsistency being illustrated.
- Do not suggest that the latent coordinate `z` corresponds to a physical variable.
- Do not claim all appearance variation is nuisance, that velocity can never be inferred from a single image, or that representation learning methods always collapse histories.
- Do not depict the figure as a new taxonomy, theorem, or empirical result. It is a conceptual diagram summarizing the state contract and aliasing from the preceding chapter.
- Keep the two panels symmetric in scale and visual weight. The desirable case should look clean and valid; the failure case should be clear but not visually overdramatic.

### Exclusions

No figure number, title, caption, explanatory paragraphs beyond the two footer sentences, bibliographic references, citations, URLs, logos, watermarks, screenshots, photorealistic texture, gradients, shadows, 3D, isometric perspective, dashboard widgets, decorative legends, unnecessary axes, or unrequested method names.

### Acceptance check

At print width, a reader should immediately understand: nuisance changes can share a representation when the target distribution stays the same; merging histories that imply different futures violates predictive sufficiency; and a proxy training loss does not automatically guarantee the predictive contract.
