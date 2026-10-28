# Figure 3.2: Augmentation defines the invariances rewarded by training

**Purpose:** Explain how the augmentation distribution becomes an implicit assumption about which input differences the representation should ignore, and why that assumption can be helpful or harmful.

## Copy-ready image-generation prompt

Create a publication-quality, print-ready academic schematic for a technical machine-learning textbook published by Manning. It should resemble a precise explanatory figure from a strong machine-learning conference paper, designed for a technical textbook page. Prioritize mathematical correctness, a strong explanatory hierarchy, and effortless reading at book-column width. Do not make a poster, slide, dashboard, playful infographic, or decorative illustration.

### Canvas and typography

- Landscape canvas, approximately 8:5 aspect ratio, at least 2400 × 1500 pixels.
- White background, crisp vector-like shapes, flat 2D geometry, no texture.
- Use dark charcoal/black line work and grayscale-safe fills. Use restrained Manning blue (#3B6FB6) only for the two representation paths, the consistency connection, and selected emphasis.
- Use one professional sans-serif font family. Keep line weights, arrowheads, spacing, and alignment consistent.
- At the final printed size, every label must remain clearly readable. As a starting point for a 2400-pixel-wide source image, use approximately 48–56 px for panel labels and key terms, and never below 40–44 px for secondary labels or equations. These are source-image guidelines, not a substitute for inspecting the exported figure at actual book width.
- Equations and labels must be sharply rendered, correctly spelled, and not crowded. Prefer fewer, larger labels over small explanatory text. No footnotes or microtext.
- Preserve generous whitespace between the mechanism, the two comparison cases, and all labels.

### Editorial boundary

**Render the diagram only. Do not put a figure number, figure title, caption, page number, explanatory prose, bibliography, source note, watermark, or running header inside the image.** The manuscript supplies the number and caption separately. Do not write “Figure 3.2” anywhere inside the artwork.

### Composition and reading order

Create one clear left-to-right pipeline across the upper half, followed by two side-by-side outcome panels in the lower half. The reader should follow the pipeline first and then compare a suitable augmentation with an unsuitable one.

#### Upper half: how augmentation enters the objective

1. At the far left, draw one compact source observation, labelled **Source observation**. Use a very simple grayscale scene containing a stationary geometric object and stable spatial context; avoid photorealistic details.
2. From that source, branch into two parallel paths. Label each path **Independent transform**. Show that the transforms are sampled separately, not that a single deterministic crop is duplicated.
3. Use small transformation symbols or simple thumbnail treatments to indicate that the sampled operations can include a crop/translation and a mild intensity perturbation. Label the transform source **Augmentation distribution** and annotate it with (T_1,T_2sim p(T)). Do not imply that these are the only valid transformations.
4. Label the resulting images **View 1** and **View 2**, with the notation
   [
   	ilde{x}^{(1)}=T_1(x), qquad 	ilde{x}^{(2)}=T_2(x).
   ]
5. Route both views through two copies of the same encoder, each labelled (f_	heta). Use a small shared-weight cue, such as identical blue outlines and a thin connector labelled **shared parameters**. Do not draw two separately learned encoders.
6. Label the outputs (z_1) and (z_2). Connect them to a consistency-objective node labelled **View consistency**. Include the concise equation
   [
   L_{mathrm{cons}}=mathbb{E}_{x,T_1,T_2}left[d!left(f_	heta(T_1x),f_	heta(T_2x)ight)ight].
   ]
   If space is constrained, keep the full equation legible by placing it in a clean, wide band below the pipeline. Never reduce font size to fit.
7. Include a concise annotation: **The transform distribution defines the preferred invariances.** The phrase must describe a training pressure, not an absolute guarantee that the encoder becomes exactly invariant.

The pipeline must visibly encode:
One source → independently sampled transformations → two views → shared encoder → representations encouraged to agree.

#### Lower-left panel: target-preserving transformations

Panel heading: **Target-preserving variation**

1. Show two or three versions of the same simple scene.
2. Vary only mild lighting/intensity or background appearance, and a crop that changes framing but retains all of the content necessary for the target.
3. Keep object identity, target-relevant location/geometry and necessary context intact. Make clear that these changes are acceptable only for the task illustrated.
4. Show the two views mapping to nearby representation points (z_1) and (z_2), joined by a short blue connector.
5. Label the implication **Agreement is useful when the target is unchanged**.
6. Avoid suggesting that every crop or intensity transformation is generally safe.

#### Lower-right panel: task-invalid transformation

Panel heading: **Task-critical information removed**

1. Use a similar source scene so that the comparison is direct. Give the scene one clearly identifiable object or spatial cue that is necessary to the target task.
2. Show a mild crop in one view that retains the cue, and an aggressive crop in the other view that removes it. Make the missing content visually obvious; do not use red or a dramatic warning icon.
3. Label the omitted cue **Task-relevant content** and mark that it is absent from the aggressive crop.
4. Show both resulting views still entering the same shared encoder and the same view-consistency objective. The loss is still asking their representations to agree even though one view no longer contains sufficient evidence for the target.
5. Use a restrained dashed boundary or dark cross-hatch to indicate the missing cue, not an additional colour.
6. Label the implication **Consistency can reward a harmful invariance**.
7. Do not imply that the encoder must always discard the cue or that every aggressive crop necessarily causes failure. The diagram illustrates an objective-design risk.

### Visual hierarchy

- The augmentation pipeline is the dominant element.
- The two lower panels must be visually balanced, so the reader compares valid and invalid transform choices directly.
- Use the same object design and scene geometry across the lower panels to keep the target-relevant difference clear.
- Place equations on clean white space, not over drawings or on curved paths.
- Use direct labels rather than a legend where possible.
- Every arrow must have an unambiguous source and destination.

### Mathematical and conceptual constraints

- The transform variables (T_1) and (T_2) are independently sampled from a chosen augmentation distribution (p(T)).
- The consistency loss rewards agreement across the views produced by that distribution. It does not directly test whether the transformations preserve the downstream target.
- The diagram must distinguish an *ideal exact invariance* from the practical pressure produced by stochastic augmentations. Do not depict the augmentation distribution as automatically defining a mathematical equivalence relation.
- Do not imply that crop and intensity perturbations are always appropriate; their suitability depends on the downstream task.
- Do not claim that stochastic augmentations guarantee exact equality (f_	heta(Tx)=f_	heta(x)). If exact invariance is mentioned, label it as an idealized condition for a specified transformation set.
- Do not introduce a taxonomy of augmentation types, performance claims, benchmark results, or additional methods.
- Do not repeat Figure 3.1's full predictive-state/aliasing explanation. This figure focuses specifically on how augmentation shapes the training signal.

### Exclusions

No figure number, title, caption, citations, URLs, logos, watermarks, screenshots, gradients, shadows, 3D effects, perspective, decorative visual metaphors, microtext, excessive annotation, or unrequested method names.

### Acceptance criteria

At printed book width, the reader should be able to identify the source observation, two independently sampled views, the shared encoder, the consistency objective, the augmentation distribution, and the difference between target-preserving and target-damaging transformations. The central takeaway should be immediate: augmentation is part of the learning objective because it specifies which differences the model is trained to ignore.
