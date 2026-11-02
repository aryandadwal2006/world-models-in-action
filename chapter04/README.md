# Chapter 4: Predicting in representation space

This directory contains the canonical, section-by-section source of Chapter 4. Each numbered section is a separate Markdown file. The manuscript is being rewritten to improve its explanations, mathematical progression and continuity with Chapter 3.

## Governing argument

A representation becomes useful to a world model when the system can predict information absent from the current context. JEPA formulates this as prediction in a learned feature space. The target definition determines what must be preserved; the distribution of embeddings affects how downstream estimators use those features; and explicit action conditioning is required before passive prediction becomes a model of controlled consequences.

## Source files

The 17 numbered Markdown files are the reader-facing sources. `references.md` is the bibliography source, `source-audit.md` records the primary-source review and its access limitations, and `figure-register.md` contains text-only figure briefs.

## Figure policy

Chapter 4 contains **no inserted figures and no image assets**. Where a figure would materially clarify an argument, the section contains only a numbered caption and a short description. No SVG, PNG, raster, or other image embed may be added until the prose is complete and figure artwork is explicitly requested.

## Editorial rules

- Define mathematical objects before using them; derive the equations that carry the argument.
- Explain why each assumption is needed and what changes when it is relaxed.
- Distinguish a training objective from the statistical quantity it approximates and from the downstream task it is meant to support.
- Distinguish spatial masking, spatiotemporal masking, future prediction and action-conditioned prediction.
- Do not infer causality, physical sufficiency, calibrated uncertainty or planning competence from low representation loss.
- Scope theorems to their assumptions and empirical results to their actual protocols.
- Explain papers only where their mechanism advances the chapter's argument; do not turn the chapter into a publication survey.
- Use Chapter 3's incremental explanatory clarity without copying its section structure.
- Keep author-facing process commentary out of reader-facing text.
