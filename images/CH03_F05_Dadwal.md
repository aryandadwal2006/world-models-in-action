# Figure 3.4: InfoNCE: candidate probabilities and gradients

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Connect cosine similarities, temperature scaling, softmax cross-entropy, the worked loss and the gradient with respect to candidate logits in one explanatory figure.

## Composition

A wide two-part schematic. The left side shows one anchor against its positive and two negatives, followed by scaled logits, softmax probabilities and the positive-index cross-entropy. The right side shows the derivative with respect to each scaled logit, using the same candidate order.

## Required content

Use the illustrative values from the prose: similarities 0.80, 0.10 and -0.30; temperature 0.5; logits 1.6, 0.2 and -0.6; probabilities about 0.737, 0.182 and 0.082; and loss `-log(p_positive)`, approximately 0.306. Clearly label all values illustrative, not empirical.

Include `dL/da_k = p_k - 1[k=y]`. Show the positive derivative `p_y - 1` and the two negative derivatives `p_k`. Under gradient descent, the positive logit is increased and negative logits are decreased. If `a_k=s_k/tau), show the additional `1/tau` factor when differentiating with respect to unscaled similarity.

## Exclusions

Do not omit the positive from the softmax denominator or include the anchor itself as a candidate. Do not imply every negative is semantically unrelated. Do not confuse logit derivatives with parameter-update directions.

## Acceptance checks

Probabilities sum to one within rounding; logits and probabilities are visibly distinct; gradient signs and indicator are defined; all text is legible at book-column width.

## Caption

Figure 3.4. InfoNCE: candidate probabilities and gradients.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work and #3B6FB6 as the only restrained accent. Use flat 2D artwork: no gradients, 3D effects, Mermaid, watermarks or decorative arrows.