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

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.