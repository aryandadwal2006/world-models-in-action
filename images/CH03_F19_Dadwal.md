# Figure 3.13: VICReg, SIGReg and VISReg constrain different aspects of feature distributions

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Compare the mathematical object constrained by three regularizers without suggesting that they are interchangeable methods or that any distributional constraint establishes semantic sufficiency.

## Composition

Three aligned columns: second-order moments, characteristic-function discrepancies on random projections, and sorted projected samples matched to Gaussian quantiles. Use a shared set of simple sample features only as a visual aid, not as data.

## Required content

**VICReg:** variance floor and off-diagonal covariance penalties. **SIGReg:** random one-dimensional projections compared with the standard-normal characteristic function. **VISReg:** scale normalization, sorted random projections and mean-squared matching to standard-normal quantiles, plus the minibatch-centering penalty. Keep the per-view minibatch center separate from the same-image global-view mean used for view invariance.

Add a small, secondary note that LeJEPA combines prediction with SIGReg; do not present it as a fourth regularizer or as equivalent to VISReg. State that finite projections and finite batches approximate the distributional constraints.

## Exclusions

No performance ranking; no claim that finite projections prove exact Gaussianity; no claim zero covariance is full independence. Do not attribute the LeJEPA theory result to every downstream task.

## Acceptance checks

Characteristic-function discrepancy and sorted-quantile matching are visually distinct. The scale, shape and center terms in VISReg are labelled without conflating the regularizer's minibatch mean with the same-image prediction target.

## Caption

Figure 3.13. VICReg, SIGReg and VISReg constrain different aspects of feature distributions.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Flat 2D only; no gradients, 3D effects, Mermaid, watermarks or decorative labels.