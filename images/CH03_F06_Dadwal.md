# Figure 3.6: NT-Xent as a classification problem

**Status:**new original. No publication image is included for this slot; this Markdown file is the production placeholder.

**Expected publication filename:** `CH03_F06_Dadwal.png` (the SVG source is optional only if the final art is best kept as vector).

**Production brief**

Four stages: anchor representation z1; positive and two negative candidates with cosine similarities 0.80, 0.10, -0.30; division by tau=0.5 gives logits 1.6, 0.2, -0.6; softmax probabilities 0.737, 0.182, 0.082. Show bars in grayscale and the single-anchor loss -ln(0.737) ≈ 0.306. Bottom line: three-class classification, identify the anchor’s positive.

**Caption:** Figure 3.6 NT-Xent as a classification problem

**Acceptance checks**

- Use the actual committed data/output where specified; never redraw empirical values by eye or synthesize observations.
- White background; fine dark line art; grayscale-safe distinctions; sentence-case labels; legible at book-column width.
- Verify all labels, reported metric definitions, the caption and any source credit against the manuscript and repository artifact before replacing the placeholder.