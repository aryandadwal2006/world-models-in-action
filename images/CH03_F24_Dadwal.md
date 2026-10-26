# Figure 3.24: MAE's visible-only encoder and masked-patch decoder

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain asymmetric architecture and loss placement.

## Composition

Patchification to random masking to visible-only encoder to mask-token decoder to reconstruction.

## Required content

Show patch embedding; high mask ratio such as 75% in the original recipe; encoder handles visible patches only; decoder inserts mask tokens/restores positions and predicts pixels; loss is on masked patches only. Mark encoder features as typical downstream representation.

## Exclusions

Do not feed mask tokens through the encoder or apply standard MAE loss to visible patches.

## Acceptance checks

Visible-token flow and masked-only loss are unmistakable.

## Caption

Figure 3.24. MAE's visible-only encoder and masked-patch decoder.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.