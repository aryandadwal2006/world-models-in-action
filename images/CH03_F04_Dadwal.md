# Figure 3.3: Constructing positives and negatives

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render any figure number, title, caption, panel-letter, page number, or surrounding book text in the artwork.

**Purpose:** Explain the data-level construction of a contrastive candidate set before introducing the mathematical details of InfoNCE. The central teaching point is that positive and negative labels come from the sampling rule and source identity, not necessarily from semantic classes.

**Format and print legibility:** Make a landscape figure around 3200 × 2000 px, with a balanced two-part composition that remains easy to follow at normal printed book-column width. White background. Use dark charcoal outlines and text, with #3B6FB6 as the only restrained accent for the anchor-to-positive relation and important selection cues. The image should look like a precise figure from a strong machine-learning paper adapted for a Manning technical textbook: flat vector-like shapes, aligned elements, even line weights, clean spacing, and restrained visual hierarchy. Size the main labels generously. At the supplied resolution, aim for roughly 64–76 px for the principal labels and 52–60 px for secondary labels and notes; mathematical symbols must be at least as legible as secondary labels. Avoid tiny source IDs, footnotes, thin strokes, and dense text. Check all elements as if the image will be scaled down to a printed book page.

## Composition

Organize the figure from left to right.

**Left section: source examples and two views.** Show three or four distinct source observations, labeled by source identity, such as x_A, x_B, x_C, and x_D. Each source branches into two independently augmented views. Label the views clearly, for example x̃_A^(1) and x̃_A^(2), and show the transformation step with simple paired arrows. Use a small label above this part: “Sample two views per source”. The examples can be abstract, simple visual tiles with distinct shapes or scenes. They should communicate different sampled observations without implying that they have known class labels. Do not use realistic people or recognizable copyrighted characters.

**Right section: one anchor and its candidate set.** Select view x̃_A^(1) as the anchor. Place its paired view x̃_A^(2) prominently as the positive candidate. Use one clear solid blue arrow from the anchor to the positive, with the label “positive: same source A”. Alongside it, place views sampled from other source identities, such as x̃_B^(1), x̃_C^(1), and x̃_D^(1), as negative candidates. Use dashed charcoal arrows from the anchor to these candidates, with one shared label “negatives by the sampling rule”. The anchor, positive, and negative candidates must be spatially distinct and unmistakable. The examples in the candidate set should be shown as views, not mislabeled as class categories.

Add one compact annotation under the candidate set: “Source identity supplies the pair label; no class label is required.” Add a second, visually separate caution note near the negative candidates: “A sampled negative can still be semantically related.” If it fits legibly without clutter, mention “e.g., nearby video frames” as an example, not as the only possible failure case.

## Required conceptual accuracy

- A positive is the second independently transformed view of the same sampled source instance.
- A negative is another candidate treated as non-matching by the particular sampling procedure. Do not claim that every negative is semantically unrelated, visually dissimilar, or in a different semantic class.
- The drawing must distinguish the algorithm's operational label from the true semantic relationship. In particular, a different source identity does not guarantee a different object, category, or underlying state.
- Keep the figure focused on pair and candidate construction. Do not include an InfoNCE softmax, logits, gradients, temperature, or an equation for the contrastive loss; the following figure covers candidate classification and its mathematics.
- Do not imply that every contrastive method uses exactly this negative-sampling scheme. Present this as the common two-view, instance-level construction used to introduce the idea.
- A reader should understand the figure without relying on color alone. Keep anchor, positive, and negatives distinguishable by direct labels and solid-versus-dashed connections as well as restrained color.

## Visual restrictions

White background, dark grayscale-safe line work, and #3B6FB6 as the only accent color. Flat 2D, crisp vector-like edges, no gradients, shadows, 3D effects, glossy icons, decorative borders, infographic badges, unnecessary legends, watermark, or simulated page layout. Avoid long sentences inside the drawing. Use no ornamental background or extra color palette.

## Acceptance checks

1. A reader can trace the pipeline from source instance to two views and then identify the anchor's candidate set.
2. The positive is visibly tied to the same source as the anchor; the negatives are visibly tied to other source identities.
3. The figure explicitly warns that “negative” means negative under the sampling rule, not necessarily semantically unrelated.
4. All essential labels remain readable at book-column width.
5. The artwork contains no figure number, title, caption, or claims about empirical results.

## Caption

Figure 3.3. Constructing positives and negatives.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or a screenshot of code. Keep every visual claim consistent with the chapter's explanation and cited sources; do not invent empirical data.