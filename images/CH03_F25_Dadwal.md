# Figure 3.25: Discrete tokens and teacher features are different masked targets

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Distinguish BEiT, iBOT and data2vec beyond a general masked-prediction label.

## Composition

Three columns with different target generators.

## Required content

BEiT: pretrained tokenizer supplies discrete IDs. iBOT: teacher prototype distributions for masked patches plus class-level distillation in its recipe. data2vec: contextual teacher hidden features, often aggregated from upper layers, predicted from masked student input.

## Exclusions

Do not call iBOT targets fixed codebook IDs or data2vec targets pixels.

## Acceptance checks

Label IDs, distributions and continuous hidden vectors separately.

## Caption

Figure 3.25. Discrete tokens and teacher features are different masked targets.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and legibility at print width. Keep visual claims consistent with the manuscript and cited sources; do not introduce invented empirical data.