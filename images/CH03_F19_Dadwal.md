# Figure 3.19: VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain the move from second-order constraints to distribution-shape regularization.

## Composition

Three columns: moments, characteristic-function sketches and sorted projected distributions.

## Required content

VICReg: variance floor/off-diagonal covariance. SIGReg: random projections against a standard-normal characteristic function. LeJEPA: mean global-view embedding of the same instance plus SIGReg. VISReg: centering, explicit scale, stop-gradient scale normalization and sorted projections matched to Gaussian quantiles by sliced Wasserstein.

## Exclusions

Attribute Gaussian optimality to LeJEPA's assumptions; finite projections do not prove exact equality.

## Acceptance checks

Distinguish characteristic-function discrepancy from Wasserstein quantile matching.

## Caption

Figure 3.19. VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.