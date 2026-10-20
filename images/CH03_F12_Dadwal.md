# Figure 3.12: Masked autoencoder and reconstructions

**Status:**composite: redraw plus committed model output. No publication image is included for this slot; this Markdown file is the production placeholder.

**Expected publication filename:** `CH03_F12_Dadwal.png` (the SVG source is optional only if the final art is best kept as vector).

**Production brief**

Compose a new architecture schematic with the real reconstruction panel from CH03_F12_Dadwal_Source.png. Show 64x64 source frame -> patchify -> hide 75% of patches -> encode visible tokens -> insert mask tokens / restore order -> decoder predicts masked patches. The loss applies only to masked patches. Make the encoder representation path and pixel reconstruction path visibly distinct. Architecture concept is adapted from Gui et al. (2024) and He et al. (2022); rendered observations are our own DMC renders.

**Caption:** Figure 3.12 Masked autoencoder and reconstructions

**Acceptance checks**

- Use the actual committed data/output where specified; never redraw empirical values by eye or synthesize observations.
- White background; fine dark line art; grayscale-safe distinctions; sentence-case labels; legible at book-column width.
- Verify all labels, reported metric definitions, the caption and any source credit against the manuscript and repository artifact before replacing the placeholder.