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

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.