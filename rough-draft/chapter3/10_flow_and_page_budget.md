# Proposed reader-facing flow and page budget

## Editorial decision

Do not make the current chapter a sequence of empirical probes with small method descriptions attached. Keep the implementation and the evidence as support, not the spine. The spine is conceptual: each family solves a difficulty exposed by the preceding objective.

## Proposed narrative sequence

### Opening: why representation learning is a choice (3 pages)
Begin with two images of the same physical scene where one is cropped, one is color-shifted, or time changes the state. Ask which changes should be ignored and which must be retained. Establish the state-representation contract without repeating all of Chapter 2. Explain that objective design is a formal choice about what input relation becomes easy.

### Section 3.1: build a map of objective families (3 pages)
Present a single taxonomy figure. Distinguish views, candidate discrimination, teacher prediction, feature statistics, prototypes and masked targets. Use historical pretext tasks only to explain target construction, not as an encyclopedia.

### Section 3.2: derive contrastive learning (7–8 pages)
Define positive/negative samples; derive softmax cross-entropy and InfoNCE; derive the gradient; discuss temperature, false negatives, batch size, augmentation and the conditional MI-bound interpretation with assumptions and ceiling. SimCLR is the main instantiation; MoCo is the architectural response to dictionary size; CPC is a short lineage note. One tiny numerical experiment supports the derivation.

### Section 3.3: learn without explicit negatives (5 pages)
Move to BYOL and SimSiam; show target branch / predictor / stop-gradient; explain collapse as fixed point versus observed training dynamics. Then DINO shows student-teacher distributions and prototype outputs. DINOv2/v3 are a scale/data/frontier note, not extra algorithms to derive line by line.

### Section 3.4: constrain statistics directly (4 pages)
Derive Barlow's cross-correlation target and VICReg's invariance/variance/covariance decomposition. One compact equation comparison table. Second tiny demonstration compares constant embeddings under invariance-only, Barlow-style and VICReg-style constraints.

### Section 3.5: prototypes and pseudo-labels (2–3 pages)
DeepCluster explains offline clustering; SwAV explains online cross-view assignment and balancing. DINO is referenced as overlapping family, not taught twice.

### Section 3.6: mask the input, choose the target (5 pages)
Start with a general masked prediction equation. Contrast BERT/GPT context restriction briefly; MAE pixel targets; BEiT discrete tokens; iBOT teacher patch targets; ELECTRA and data2vec as short sidebars. A target-space figure should carry most of the comparison.

### Section 3.7: one framework, multiple modalities (2–3 pages)
CLIP/SigLIP and speech SSL examples. No independent deep dives: table compares source of positive pairs, target representation, loss family and limitation.

### Section 3.8: frontier and boundary (2 pages)
DINOv3 for dense features and scaling; compact JEPA note as the handoff to the next conceptual stage; distinguish passive pretraining from action-conditioned systems. One paragraph excludes diffusion/flow matching as generative objectives treated elsewhere in the book.

### Section 3.9: objective selection as an audit (2 pages)
A table maps objective family to required data structure, main technical burden, likely failure and what evidence is needed. Include a world-model-specific checklist: velocity/occlusion, actions, state sufficiency, dynamic consistency, feature transfer.

### Summary / exercises / references (3–4 pages)
Exercises require derivations and reasoning more than multi-hour experiments. Give solutions or selected derivations in the appendix. References cover seminal and frontier papers while directing readers to official implementations.

## Target length

A first composed theoretical draft should aim for roughly 32–38 pages, then be trimmed only after the argument works. Do not use experiments to inflate the chapter. If it naturally remains longer because a derivation needs space, prioritize explanatory completeness over a hard page count. The unbounded dossier is the place for method-by-method depth; the reader-facing chapter should preserve a coherent flow rather than every detail.

## Editorial guardrails

- Open each method section with the question it answers.
- Put the full objective before naming it a solution.
- Define target, context and update rule at first use.
- Separate claims by original authors, observations from released implementations, and our conceptual synthesis.
- Each figure must teach a mechanism; do not use architecture name collages.
- Do not reuse the current empirical plots as visual filler.
- Keep one consistent notation across all methods.
- Keep CPC to a short historical bridge.
- No universal rankings across heterogeneous papers.
- Every frontier claim receives a paper citation and an as-of date.
