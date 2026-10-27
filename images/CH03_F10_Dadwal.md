# Figure 3.6: Momentum contrast: query encoder, key encoder and queue

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain the defining MoCo v1/v2 mechanism: a gradient-trained query encoder compares its output with a momentum-encoder key and a queue of previously encoded keys.

## Composition

One main pathway with query branch, stop-gradient key branch, EMA parameter update and a FIFO queue. The main figure depicts MoCo v1/v2 only.

## Required content

The query encoder receives gradients and produces `q`. The paired view passes through a stop-gradient key encoder to produce the positive key `k+`. The query is scored against that key and queued earlier keys. Draw the EMA parameter update from query encoder parameters to key encoder parameters; separately show the queue's enqueue/dequeue operations.

Mark the queue contents as features produced by earlier parameter values, so readers can see why the momentum encoder reduces abrupt drift but does not eliminate staleness.

## Exclusions

Do not draw gradients into the key encoder; do not conflate EMA updates with queue operations. Do not add MoCo v3 as a second mini-architecture; the prose explains that v3 removes the FIFO queue.

## Acceptance checks

Query, positive key, queued negatives, stop-gradient, EMA parameter update and queue operations are individually labelled and visually distinct.

## Caption

Figure 3.6. Momentum contrast: query encoder, key encoder and queue.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Use flat 2D artwork: no gradients, 3D effects, Mermaid, watermarks or decorative arrows.