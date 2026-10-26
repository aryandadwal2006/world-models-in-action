# Figure 3.15: DINO teacher-student cross-view distillation

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Show original DINO and distinguish later family recipes.

## Composition

Teacher and student columns with crop inputs and prototype distributions.

## Required content

Student processes multiple crops; teacher global crops. Both output prototype logits. Center and sharpen teacher logits before softmax; stop gradients through teacher probabilities; update teacher parameters by EMA. Note DINOv2's masked-patch distillation/KoLeo additions and DINOv3's configuration-dependent Gram anchoring.

## Exclusions

No negative samples; centering is not Sinkhorn; Gram anchoring is not universal to DINOv3.

## Acceptance checks

Original DINO is visually separable from later extensions.

## Caption

Figure 3.15. DINO teacher-student cross-view distillation.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.