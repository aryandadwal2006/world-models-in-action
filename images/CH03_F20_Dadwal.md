# Figure 3.20: DeepCluster and SwAV: pseudo-labels and prototypes

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain how clustering creates training targets and how offline and online assignment differ.

## Composition

Two side-by-side learning loops.

## Required content

DeepCluster: encode, cluster offline, assign pseudo-labels, train classifier, refresh clusters. SwAV: two views, prototype assignments, balanced assignment in the cited recipe, cross-view assignment prediction and encoder/prototype updates.

## Exclusions

Clusters are not ground truth; Sinkhorn is not universal to every recipe.

## Acceptance checks

Assignment timing and dependency on current features are explicit.

## Caption

Figure 3.20. DeepCluster and SwAV: pseudo-labels and prototypes.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.