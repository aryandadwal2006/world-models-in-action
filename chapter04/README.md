# Chapter 4: Predicting in representation space

This directory contains the canonical, section-by-section reader-facing source of Chapter 4. Each numbered section is a separate Markdown file. The chapter follows the explanatory progression of the argument rather than a catalogue of papers.

## Governing argument

A representation becomes useful to a world model when the system can predict information absent from the current context. JEPA formulates this as prediction in a learned feature space. The target definition determines what must be preserved; the distribution of embeddings affects how downstream estimators use those features; and explicit action conditioning is required before passive prediction becomes a model of controlled consequences.

The chapter develops this progression in five connected stages: define the context-target prediction problem; understand the representation and its degeneracies; derive why embedding geometry matters to downstream estimation; examine practical distributional regularizers; and move from passive video features to interventions, planning and an agent architecture.

## Source files

The 17 numbered Markdown files are the reader-facing sources. `references.md` is the bibliography source, `source-audit.md` records the primary-source review and access limitations, and `figure-register.md` indexes the text-only figure briefs.

## Figure policy

Chapter 4 contains **no inserted figures and no image assets**. It has 29 numbered figure briefs, each written in its section as a caption followed by a short description. These briefs explain what a future figure would show and why it matters. No SVG, PNG, raster or other image embed may be added until the prose is complete and figure artwork is explicitly requested.

## Manuscript statistics

The current section sources contain approximately 22,800 words, 144 display equations, 245 inline mathematical expressions and 29 text-only figure briefs. These counts are source-level checks, not a substitute for the final typeset page count.

## Editorial and technical rules

- Define mathematical objects before using them; derive the equations that carry the argument.
- Explain why each assumption is needed and what changes when it is relaxed.
- Distinguish a training objective from the statistical quantity it approximates and from the downstream task it is meant to support.
- Distinguish spatial masking, spatiotemporal masking, future prediction and action-conditioned prediction.
- Do not infer causality, physical sufficiency, calibrated uncertainty or planning competence from low representation loss.
- Scope theorems to their assumptions and empirical results to their actual protocols.
- Explain papers only where their mechanism advances the chapter's argument; do not turn the chapter into a publication survey.
- Use Chapter 3's incremental explanatory clarity without copying its section structure.
- Keep author-facing process commentary out of reader-facing text.
