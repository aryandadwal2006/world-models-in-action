# Figure 3.15: CPC identifies a future latent

**Status:**new original. No publication image is included for this slot; this Markdown file is the production placeholder.

**Expected publication filename:** `CH03_F15_Dadwal.png` (the SVG source is optional only if the final art is best kept as vector).

**Production brief**

Draw frames x1...xt through a shared encoder producing z1...zt. Only past/current latents enter a unidirectional GRU, which emits context ct. Branch from ct to horizon-specific W1, W2, W3 compatibility heads. Each head scores candidate future latents by exp(z_(t+k)^T W_k c_t). Separate future targets visually from causal context; include negative candidates and identification loss. Do not show action input, a state decoder, or planner. Pair visual style with Chapter 4 JEPA figure.

**Caption:** Figure 3.15 CPC identifies a future latent

**Acceptance checks**

- Use the actual committed data/output where specified; never redraw empirical values by eye or synthesize observations.
- White background; fine dark line art; grayscale-safe distinctions; sentence-case labels; legible at book-column width.
- Verify all labels, reported metric definitions, the caption and any source credit against the manuscript and repository artifact before replacing the placeholder.