# Figure 3.16: The training representation and the downstream interface

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Join two distinctions that the prose treats separately: the feature optimized by the pretraining loss, and the representation interface selected for a later predictor.

## Composition

Use two clearly separated horizontal lanes.

**Pretraining lane:** observation `x` enters encoder `f` to produce backbone features `h`; a projector `g` maps `h` to objective-space features `z`, and the self-supervised loss acts on `z`. Label this as one common recipe, not a universal architecture.

**Downstream lane:** show the exported backbone features `h` represented in one of three forms: a pooled global vector, a spatial grid of patch features, or a temporal context produced by aggregating a history of features. Route the chosen interface to a generic later predictor. Separate the three options with "alternative interfaces" rather than a ranking.

## Required content

- Make explicit that the pretraining loss is applied to `z=g(h)` in methods that use a projection head, while downstream evaluation often exports `h`.
- A global vector is compact but removes explicit patch indexing; a grid retains spatially indexed features; temporal context depends on history and an aggregation mechanism.
- These are interface choices, not a quality hierarchy. Not every method uses a projector, and not every downstream task needs a spatial grid or temporal context.
- Keep the action input on the later predictor's transition interface only; Chapter 3 does not train an action-conditioned dynamics model.

## Exclusions

No robotics illustration, no planning result, no dimensions that imply universal optimality, and no claim that a spatial grid or temporal context is always superior.

## Acceptance checks

The two lanes must not imply that `z`, a pooled vector, a patch grid and a temporal context are the same tensor. Distinguish backbone, projector, objective loss, aggregation and later prediction.

## Caption

Figure 3.16. The training representation and the downstream interface.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Use flat 2D artwork; no gradients, 3D effects, Mermaid or decorative watermarks. Every label must remain readable at book-column width.