# Figure 3.10: Momentum contrast: query encoder, key encoder and queue

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain the momentum encoder and dictionary queue.

## Composition

Query path, target/key path, EMA arrow and FIFO queue.

## Required content

For MoCo v1/v2, the gradient-trained query encoder outputs q. The paired view passes through a stop-gradient key encoder to k+. Compare q with k+ and queued earlier keys. Draw an EMA arrow from query parameters to key parameters and enqueue/dequeue arrows for features. Add a clearly secondary side note: MoCo v3 removes the FIFO queue and uses current-batch momentum targets gathered across workers, with an online predictor.

## Exclusions

No gradients into the key encoder; do not confuse EMA with queue operations.

## Acceptance checks

Label query, key, positive, queued negatives and EMA separately.

## Caption

Figure 3.10. Momentum contrast: query encoder, key encoder and queue.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.