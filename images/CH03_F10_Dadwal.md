# Figure 3.10: Momentum contrast: query encoder, key encoder and queue

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain the momentum encoder and dictionary queue.

## Composition

Query path, target/key path, EMA arrow and FIFO queue.

## Required content

The gradient-trained query encoder outputs q. The paired view passes through a stop-gradient key encoder to k+. Compare q with k+ and queued earlier keys. Draw an EMA arrow from query parameters to key parameters and enqueue/dequeue arrows for features.

## Exclusions

No gradients into the key encoder; do not confuse EMA with queue operations.

## Acceptance checks

Label query, key, positive, queued negatives and EMA separately.

## Caption

Figure 3.10. Momentum contrast: query encoder, key encoder and queue.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.