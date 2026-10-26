# Figure 3.9: SimCLR's two-view encoder and projection head

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Show which output is optimized and which representation is normally retained downstream.

## Composition

Two symmetric branches with shared parameters.

## Required content

One image forks into two augmentations; both pass through the same encoder f and projector g. Apply NT-Xent to z=g(f(view)); label h=f(view) as the backbone representation commonly retained downstream. Show shared weights and positive-pair candidate arrows.

## Exclusions

Do not draw independently parameterized encoders or apply the loss to h when describing the original recipe.

## Acceptance checks

Distinguish h from z and show the positive pair.

## Caption

Figure 3.9. SimCLR's two-view encoder and projection head.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.