# Figure 3.5: InfoNCE as candidate classification

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Connect similarity, temperature scaling, softmax and cross-entropy.

## Composition

Embeddings to scaled similarities to probabilities to positive-index loss.

## Required content

Use an illustrative example: similarities 0.80, 0.10, -0.30; temperature 0.5; logits 1.6, 0.2, -0.6; probabilities about 0.737, 0.182, 0.082; loss -log(p_positive). Clearly label the values illustrative, not empirical.

## Exclusions

Do not omit temperature or denominator; do not present the values as experiment data.

## Acceptance checks

Probabilities sum to one within rounding; distinguish logits from probabilities.

## Caption

Figure 3.5. InfoNCE as candidate classification.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.