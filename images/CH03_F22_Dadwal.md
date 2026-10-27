# Figure 3.29: A training loss and a probe answer different questions

**Status:** Detailed production brief only. No artwork is included in this branch.

## Purpose

Show why a successful pretraining objective is evidence about its chosen target, not a universal certificate that the representation preserves all information needed by a downstream world model.

## Composition

Use a clear left-to-right progression with three main stages:

1. **Pretraining objective:** a view-pair or masked input enters an encoder; a visible loss symbol marks agreement, candidate ranking, or target reconstruction. Label the output of this stage "fit to the chosen target".
2. **Representation diagnostics:** show three compact tests, labelled "linear probe", "retrieval / feature geometry", and "reconstruction". Each should point to the specific claim it tests, not to a universal quality score.
3. **Target-specific prediction:** show the frozen representation entering a predictor for a named target and horizon. Label the conclusion "evidence for this target and protocol", not "guaranteed sufficient".

At the bottom, add one restrained conclusion: "No single metric certifies predictive sufficiency." Keep the diagram legible at book-column width. Use simple arrows and labels, not decorative icons or charts with invented numeric values.

## Required content

- Distinguish a training loss from downstream probes and direct target-specific prediction.
- Make it explicit that every diagnostic has a scope: objective fit, linear accessibility, selected feature geometry, reconstruction of a chosen target, or prediction of a specified target.
- Show that the target, observation history, prediction horizon and readout protocol determine what the evaluation can support.
- Indicate that a direct prediction test is stronger evidence for the named target than an unrelated pretraining metric, but is still not a universal guarantee.

## Exclusions

No JEPA architecture, no benchmark scores, no invented measurements, no single numerical "representation quality" score, and no claim that a probe can prove information is absent.

## Acceptance checks

- A reader can distinguish all three stages without reading the caption.
- The figure does not imply that linear-probe failure proves information is absent.
- The final target-specific test is clearly scoped to its target and protocol.
- All text remains readable at final book size.

## Caption

Figure 3.29. A training loss and a probe answer different questions.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent for key paths or highlights. Important distinctions must remain legible in grayscale. Use flat 2D artwork: no gradients, 3D effects, Mermaid, decorative watermarks, leaderboards or unnecessary labels. Keep every visual claim consistent with the manuscript; do not invent empirical data.
