# Figure 3.6: How the InfoNCE gradient acts on candidate scores

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Make derivative signs intuitive after the loss is derived.

## Composition

One positive and several negatives with derivative strips.

## Required content

Show dL/da_k=p_k-1[k=y]. The positive derivative is p_y-1; negative derivatives are p_k. A hard negative with larger p_k receives a larger score derivative than an easy negative. Include the 1/tau factor when a_k=s_k/tau.

## Exclusions

Do not confuse score derivatives with parameter update directions.

## Acceptance checks

Label the derivative as with respect to logits and define the indicator.

## Caption

Figure 3.6. How the InfoNCE gradient acts on candidate scores.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.