# World Models in Action

Book companion code for learning how world models represent observations, predict dynamics, and support decisions.

## Start here

**Readers should begin with the short Chapter 3 examples:** [`ch03/minimal/`](ch03/minimal/README.md). They are deliberately small and focus on the mechanism being taught.

The longer scripts directly under `ch03/` are the research/reference implementation. They retain experiment configuration, sampling safeguards, diagnostics, metrics, and figure generation so that the reported results can be audited. They are not intended to be read linearly in a book chapter.

## Chapter 3 evidence

The reference experiments compare contrastive learning, VICReg, masked reconstruction, and temporal CPC. Their saved measurements are under `ch03/results/`. The temporal missing-observation test is a negative and seed-sensitive result; see [the publication-readiness audit](ch03/publication_readiness_audit.md) before repeating or quoting any number.

The minimal scripts are teaching implementations and **do not reproduce the exact checkpoints or metrics** in the reference artifacts. Use the reference scripts and committed JSON files for exact experimental reproduction.
