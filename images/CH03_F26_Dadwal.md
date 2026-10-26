# Figure 3.26: Masked prediction and diffusion learn different conditional tasks

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Clarify the boundary between masked representation learning and generative denoising.

## Composition

Two corruption panels with target and output.

## Required content

Masked path predicts withheld tokens/features from visible context. Diffusion path shows x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon and a denoiser predicting epsilon at time t, followed by reverse sampling.

## Exclusions

Do not imply a denoising loss alone specifies every sampler or likelihood bound.

## Acceptance checks

Label target positions versus injected noise.

## Caption

Figure 3.26. Masked prediction and diffusion learn different conditional tasks.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.