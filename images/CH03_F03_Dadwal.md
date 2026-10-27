# Figure 3.2: Augmentation defines the invariances rewarded by training

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** The figure number and title at the top, and the caption below, are editorial metadata. Do not render any of them inside the artwork; the book layout supplies figure number and caption separately.

**Typography:** Size labels and equations for print at book-column width, using lettering about 1–2 pt larger than screen-first infographic defaults.

## Purpose

Show that augmentation is part of the learning objective, not neutral preprocessing.

## Composition

Source observation, two independently sampled transformed views, then the representation that the consistency loss encourages the encoder to produce. Include one failure case where an aggressive crop removes task-critical information.

## Required content

Use crop and intensity perturbations as examples, not universal choices. Show the transform distribution as the source of a preference for invariance. If exact equivalence classes are mentioned visually, mark them as an idealized condition requiring exact invariance over a specified transformation set, not as an automatic property of stochastic augmentations.

## Exclusions

No claim that the same augmentations are valid for all tasks or modalities; no claim that the sampled distribution is automatically a mathematical equivalence relation.

## Acceptance checks

Distinguish transformations from the semantic assumptions they encode. The failure case makes clear that a loss can reward ignoring information removed by the view construction.

## Caption

Figure 3.2. Augmentation defines the invariances rewarded by training.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Flat 2D only; no gradients, 3D effects, Mermaid, watermarks or decorative labels.