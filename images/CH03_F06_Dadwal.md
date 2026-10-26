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

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key arrows or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels; component and equation labels specified by this brief are required. Keep every visual claim consistent with the manuscript and cited sources; do not invent empirical data.