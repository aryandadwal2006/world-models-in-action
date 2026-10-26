# Figure 3.19: VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain the move from second-order constraints to distribution-shape regularization.

## Composition

Three columns: moments, characteristic-function sketches and sorted projected distributions.

## Required content

VICReg: variance floor/off-diagonal covariance. SIGReg: random projections against a standard-normal characteristic function. LeJEPA: mean global-view embedding of the same instance plus SIGReg. VISReg: keep the same-image mean of global views (the view-invariance target) distinct from the per-view minibatch mean (the regularizer's center penalty); show scale, detached-scale normalization, and mean-squared sorted Gaussian quantile errors.

## Exclusions

Attribute Gaussian optimality to LeJEPA's assumptions; finite projections do not prove exact equality.

## Acceptance checks

Distinguish characteristic-function discrepancy from Wasserstein quantile matching.

## Caption

Figure 3.19. VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.