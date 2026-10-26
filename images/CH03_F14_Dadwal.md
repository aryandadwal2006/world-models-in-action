# Figure 3.14: BYOL and SimSiam: similar predictors, different target updates

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Compare architecture and gradient paths without conflating the methods.

## Composition

Two parallel architecture panels.

## Required content

SimSiam: shared encoder/projector, predictor and stop-gradient target, with symmetric loss. BYOL: online encoder/projector/predictor predicts EMA target encoder/projector; target has no gradients. Draw EMA only for BYOL and show prediction direction.

## Exclusions

Stop-gradient is not a collapse theorem; do not draw an EMA teacher in SimSiam.

## Acceptance checks

Predictor, detached target and EMA update are distinct.

## Caption

Figure 3.14. BYOL and SimSiam: similar predictors, different target updates.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.