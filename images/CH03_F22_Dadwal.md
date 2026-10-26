# Figure 3.22: The JEPA context-target prediction architecture

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Bridge to Chapter 4 by contrasting latent prediction with pixel reconstruction.

## Composition

Context encoder, target encoder and predictor.

## Required content

Split image/video into context C and target Y. Context sees C only; target encoder produces target embeddings from less-corrupted input with stopped gradients. Predictor takes context features and target positions to predict target embeddings. Show EMA for the usual recipe and note passive versus action-conditioned prediction.

## Exclusions

No pixel decoder, explicit negatives, planner or sufficiency guarantee.

## Acceptance checks

Label context, target, predictor, stop-gradient and EMA.

## Caption

Figure 3.22. The JEPA context-target prediction architecture.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.