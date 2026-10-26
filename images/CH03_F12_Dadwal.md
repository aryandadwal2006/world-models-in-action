# Figure 3.12: CPC: causal context predicts future latent candidates

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Show temporal prediction without implying recursive dynamics or action-conditioned planning.

## Composition

Past observations to encoder to causal context to horizon-specific score heads.

## Required content

Show z_1...z_t flowing to a unidirectional context c_t. Each horizon k scores positive z_(t+k) with z_(t+k)^T W_k c_t against sampled candidate latents. Keep future targets outside context and show no action input or planner.

## Exclusions

No recursive latent rollout or physical-state decoder.

## Acceptance checks

Past/current versus future must be unmistakable.

## Caption

Figure 3.12. CPC: causal context predicts future latent candidates.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.