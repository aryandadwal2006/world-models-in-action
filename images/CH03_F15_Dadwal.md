# Figure 3.10: DINO teacher-student cross-view distillation

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Explain original DINO's teacher-student distribution-matching mechanism and distinguish centering from sharpening.

## Composition

Two columns for student and teacher, connected to prototype-logit distributions. Show multiple student crops and global teacher crops, with a clear stop-gradient/EMA pathway.

## Required content

Both branches produce logits over learned prototypes. The teacher distribution is centered and sharpened before it becomes the target; the student is trained by cross-entropy against that target. Show that teacher probabilities are detached and teacher parameters are updated by EMA. Label student temperature and teacher temperature separately if both are displayed.

## Exclusions

No negative samples, no claim that prototypes are ground-truth classes, and no DINOv2 or DINOv3 add-on panels. Do not equate centering with Sinkhorn balancing. Later recipe details remain in prose.

## Acceptance checks

The original DINO path is self-contained. The target distribution, stop-gradient, centering, sharpening and EMA update are not conflated.

## Caption

Figure 3.10. DINO teacher-student cross-view distillation.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Flat 2D artwork; no gradients, 3D effects, Mermaid, watermarks or decorative labels.