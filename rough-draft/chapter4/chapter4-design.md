# Chapter 4 design: JEPA and Predictive Representation Learning

**Book:** *World Models in Action*  
**Publisher:** Manning  
**Target:** 60–75 printed pages for the core chapter; 50 pages is the minimum viable version. A deeper optional theory/frontier treatment can take the chapter to 90–100 pages.  
**Status:** detailed editorial specification, research synthesis and production blueprint (31 October 2026).

---

## 1. Chapter purpose

Chapter 4 is the book's hinge between learning a representation and learning a world model. Chapter 3 explains how self-supervised objectives construct useful features, and why neither low training loss nor anti-collapse regularization proves that a state contains the information required for future prediction. Chapter 4 introduces a more direct learning signal: infer the representation of an unobserved region, frame or future from a restricted context.

The chapter must answer five questions:

1. What is predicted in a Joint-Embedding Predictive Architecture (JEPA), and what is deliberately not reconstructed?
2. How do context encoders, target encoders, predictors, masks and target updates work in the main image and video architectures?
3. What mathematical problems remain after replacing pixel reconstruction with latent prediction, especially collapse, uncertainty and the geometry of the embedding distribution?
4. What changes when a predictive representation is trained on video and then conditioned on actions?
5. What extra ingredients turn a representation predictor into a world model that can support planning?

The thesis should be more careful than “JEPAs learn the meaning of the world.” A JEPA learns to predict a selected target representation under a selected context and sampling distribution. That can reduce pressure to model unpredictable pixel detail, but the learned representation is still shaped by the target encoder, mask policy, temporal sampling, architecture, regularizer and data. A useful world model additionally needs the right state variables, action-conditioned transitions, appropriate uncertainty handling, and evaluation in the environment where planning will occur.

### Reader promise

By the end, a reader who knows the foundations in Chapters 1–3 should be able to:
- draw and explain a generic JEPA training graph;
- distinguish I-JEPA, V-JEPA, V-JEPA 2 and V-JEPA 2.1;
- explain how a target encoder and predictor are used without assuming all JEPAs use the same update rule;
- derive the conditional-mean consequence of squared-error latent prediction;
- explain how VICReg, SIGReg and VISReg differ in the statistical properties they constrain;
- describe the additional data and model components required for action conditioning;
- identify why a strong representation benchmark is not the same as successful physical planning;
- inspect an official implementation without mistaking it for a complete reproducibility package.

---

## 2. Placement in the book and dependency map

**Before this chapter**
- Chapter 1 establishes the world-model idea, the distinction between prediction and generation, and the broad perception–state–prediction–planning loop.
- Chapter 2 defines the anatomy of a world model: observations, latent state, transition dynamics, reward/value, planning, memory and evaluation.
- Chapter 3 establishes the representation-learning problem and the limits of self-supervised proxies. It already derives or explains contrastive objectives, BYOL/SimSiam, DINO, Barlow Twins, VICReg and masked prediction.

**This chapter adds**
- the JEPA architecture as a specific predictive representation-learning design;
- image and video JEPA instances;
- distributional regularization beyond the VICReg recap;
- a route from action-free video features to action-conditioned latent prediction;
- one carefully scoped bridge to physical planning and LeCun's broader proposed architecture.

**After this chapter**
- Chapter 5 can discuss SSMs and persistent temporal memory without having to explain the JEPA loss again.
- Chapter 6 can formalize latent transition models, uncertainty and imagined rollouts.
- Chapter 7 can explain planning algorithms, model-predictive control and value estimation.
- Chapters 8–10 can compare latent prediction with generative reconstruction, diffusion and interactive video simulation.

**Boundary rule:** Chapter 4 explains what a JEPA learns and how an action-conditioned predictor can be used. Chapter 6 owns general dynamics modelling and rollout error. Chapter 7 owns detailed planning algorithms. Chapter 9 owns diffusion mathematics. Chapter 11 owns broad robotics and sim-to-real deployment.

---

## 3. Recommended final structure

Keep the publisher's approved TOC spine, with carefully named subsections. Do not replace the chapter's identity with a survey of every paper that contains “JEPA” in its title.

### 4.1 Why predict in latent space? (approximately 4–6 pages)

Open with two futures that are visually different in irrelevant ways but equivalent for the prediction at hand, and a counterexample where two histories look similar but imply different futures. Use these to introduce the central trade-off:
- pixel reconstruction can spend capacity on stochastic texture and exact appearance;
- latent prediction asks for the target encoder's representation;
- the target encoder decides which distinctions the loss treats as relevant;
- discarding detail can be beneficial or harmful depending on the downstream target.

Introduce the terms context, target, embedding, predictor and joint embedding with one intuitive example. Do not repeat the general history of SSL, the InfoNCE derivation, or the entire representation-sufficiency discussion from Chapter 3.

### 4.2 JEPA (Joint-Embedding Predictive Architecture) (approximately 28–34 pages)

This is the mathematical and architectural center. Recommended sequence:

**4.2.1 The generic architecture**
- context encoder, target encoder, predictor;
- context and target masks/positions;
- target features computed from a less restricted view;
- prediction loss in feature space;
- the difference between train-time target branches and inference-time encoder use.

**4.2.2 I-JEPA: spatial prediction from image context**
- image patch tokens and context/target blocks;
- multiple target blocks and positional information;
- EMA target branch as used in the method;
- why the mask geometry matters;
- what the image benchmark results do and do not establish.

**4.2.3 V-JEPA: prediction across space and time**
- video token grids with temporal and spatial coordinates;
- spatiotemporal target masks;
- temporal continuity and motion-sensitive targets;
- action-free pretraining, frozen evaluation and limitations of recognition benchmarks.

**4.2.4 The target branch is not one universal recipe**
- stop-gradient vs parameter update;
- EMA teacher vs symmetric training;
- predictor placement and target normalization;
- masks, target layers and output granularity;
- why method-specific implementation details must be preserved.

**4.2.5 V-JEPA 2: from internet video to physical prediction**
- the action-free V-JEPA 2 encoder pretraining stage;
- the separate action-conditioned post-training of V-JEPA 2-AC using robot trajectory data;
- the role of action tokens/conditioning and the latent predictor;
- image-goal planning and deployment in environments without task-specific robot data collection;
- do not attribute the planning result to the action-free encoder in isolation.

**4.2.6 V-JEPA 2.1: dense predictive features**
- dense rather than only pooled/global representations;
- loss on visible and masked tokens;
- deep self-supervision across intermediate layers;
- image/video tokenizers and spatial-temporal grounding;
- explain why dense feature quality matters for correspondence, localization and manipulation;
- keep it distinct from V-JEPA 2-AC: 2.1 is a representation recipe, not simply a new action-conditioned planner.

### 4.3 Action-conditioned predictive representations (approximately 8–10 pages)

Use the transition from observation-only prediction to intervention:
\[
\hat z_{t+1}=g_\phi(z_t,a_t)
\]
as the simplest illustration, then explain why real models may predict multiple tokens, patches, time steps or distributions.

Cover:
- passive video: what tends to happen next;
- action-conditioned video: what tends to happen after a specified action;
- observational correlations vs action-dependent consequences;
- demonstrations, robot data, action encodings and temporal alignment;
- what is identifiable from passive data and what requires interventions or action labels;
- uncertainty and multimodal futures;
- a simple imagined-trajectory cost, but defer MPC algorithms and trajectory optimization details.

Introduce JEPA-WMs and EB-JEPA here or in a short subsection. Distinguish the energy used to score a predicted latent from a complete control policy.

### 4.4 Regularizing predictive embeddings: from VICReg to LeJEPA and VISReg (approximately 12–16 pages)

Although the current TOC nests this in 4.2, the prose can use a clearly marked subsection to keep the chapter readable. If TOC changes are permitted later, consider a named subheading “The geometry of predictive embeddings.”

**VICReg recap (1–2 pages maximum):** Chapter 3 already derives invariance, variance and covariance. State the design question that remains: matching low-order moments does not specify a full probability distribution.

**LeJEPA and SIGReg (5–7 pages):**
- downstream-risk motivation and the specific assumptions under which an isotropic Gaussian target is derived;
- characteristic functions and one-dimensional projections;
- Cramér–Wold: exact equality requires all directions, whereas training uses a finite stochastic sketch;
- Epps–Pulley-style normality discrepancy and its numerical approximation;
- how the predictive term and SIGReg term are combined in LeJEPA;
- what the paper claims about reducing reliance on EMA/stop-gradient heuristics;
- practical limits, batch effects and finite-sketch variability.

**VISReg (5–7 pages):**
- why scale and distribution shape should be discussed separately;
- variance as scale control and sliced-Wasserstein as a shape discrepancy;
- one-dimensional optimal transport and sorting;
- random directional projections;
- stop-gradient boundaries and gradient behaviour as reported by the authors;
- computational scaling in terms of batch size, dimension and number of projections;
- compare VICReg/SIGReg/VISReg without calling any one method a universal solution.

### 4.5 Implementing a miniature JEPA system (approximately 5–7 pages)

Build a source-grounded reader walkthrough around the smallest faithful concept that can be understood on modest compute. Do not pretend this is a reproduction of I-JEPA or V-JEPA at paper scale.

Recommended teaching progression:
1. Prepare two views or a context/target mask from a small image batch.
2. Compute context features.
3. Compute target features under the correct target-branch update rule.
4. Predict target positions/features.
5. Compute the latent loss with the correct reduction and stop-gradient behaviour.
6. Add one regularizer as a separately explained extension.
7. Report diagnostics that separate prediction loss, collapse and downstream accessibility.

The only code excerpts in the printed chapter should come from official paper repositories, or be clearly labeled as pseudocode. The reader-facing chapter must not silently convert pseudocode into a purported faithful reproduction. Use perhaps 3–5 short excerpts, each chosen to explain one non-obvious operation. All full commands, environment setup and file paths belong in the repository companion notes or appendix.

### 4.6 Summary, exercises and references (approximately 4–5 pages)

End with a synthesis:
- JEPA changes the target space, not the requirement to choose a useful target.
- Video adds time, but time alone does not make a model action-aware.
- Action conditioning requires actions and a model that can use them.
- Distributional regularization shapes embeddings but does not certify their semantic or predictive sufficiency.
- A world model is an evaluated predictive system, not a representation loss in isolation.

---

## 4. Page budget and expansion policy

| Section | Core pages | Optional expansion |
|---|---:|---:|
| Opening and motivation | 4–6 | 1 |
| Generic JEPA objective and architecture | 6–7 | 2 |
| I-JEPA | 4–5 | 1 |
| V-JEPA | 4–5 | 1 |
| V-JEPA 2 and V-JEPA 2.1 | 6–7 | 2 |
| Target dynamics and collapse caveats | 2–3 | 2 |
| VICReg bridge, LeJEPA and SIGReg | 6–8 | 4 |
| VISReg | 5–6 | 3 |
| Other modalities and variational lines | 3–4 | 4 |
| Action-conditioned models and planning bridge | 6–8 | 3 |
| Path to autonomy | 2–3 | 2 |
| Implementation | 5–6 | 1 |
| Summary, exercises, references | 4–5 | 1 |
| **Total** | **57–74** | **27** |

**Minimum:** do not cut below 50 printed pages.  
**Preferred:** 65–72 pages.  
**90–100 page edition:** add the optional theory notebooks, primary-paper method comparisons, uncertainty discussion and implementation/source audit. Do not reach 100 pages by expanding every method into a repetitive survey card.

Page counts are editorial estimates, not a promise that every manuscript page maps one-to-one into the final Manning layout. Measure after the first AsciiDoc build.

---

## 5. Teaching arc

The chapter should feel like one argument, not a list of acronyms:

1. Pixel targets ask for exact appearance.
2. A latent target asks for a chosen representation of the hidden region.
3. A predictor must infer that target from restricted context.
4. The target architecture and mask policy decide what information is rewarded.
5. The embedding can still be trivial or badly shaped, so the training recipe needs a suitable anti-collapse mechanism.
6. VICReg makes scale and redundancy constraints explicit; LeJEPA/SIGReg and VISReg study stronger distributional constraints.
7. Video extends prediction across time, but action-conditioned prediction is a different problem.
8. V-JEPA 2-AC and JEPA-WM work show how latent prediction can enter physical planning, while also exposing the gap between representation quality and a working controller.
9. LeCun's proposed architecture places JEPA-like predictive models inside a broader system with goals, costs, memory, actors and critics.

Every new section should answer: **what limitation of the previous formulation motivates this next design choice?**

---

## 6. The figure plan

Aim for **18–22 original figures**, only where a drawing does more than text. A possible register is specified in 11_figures_tables_exercises.md. The most important figures are:

1. pixel reconstruction vs latent prediction;
2. generic JEPA training graph with context encoder, target encoder and predictor;
3. context and target masks in I-JEPA;
4. space-time target masks in V-JEPA;
5. what changes between action-free V-JEPA 2 pretraining and V-JEPA 2-AC post-training;
6. how V-JEPA 2.1 makes the target dense;
7. target-branch update paths, comparing representative variants;
8. deterministic squared-error prediction and the conditional-mean solution;
9. VICReg, SIGReg and VISReg as distinct statistical constraints;
10. random projection and characteristic-function sketching;
11. one-dimensional sorting for sliced-Wasserstein distance;
12. action-conditioned latent prediction;
13. a model-predictive planning loop at the level of a conceptual bridge;
14. JEPA-WMs evaluation: model, planner and environment must be assessed together;
15. LeCun's configurable predictive architecture.

Do not create one giant “JEPA family tree” that implies every method is the same architecture. A small method map can classify by target modality, prediction support, regularizer and action conditioning.

---

## 7. Tables and code listings

Target **4–6 tables** and **3–5 short code listings**.

Tables:
- method comparison for I-JEPA / V-JEPA / V-JEPA 2 / V-JEPA 2.1;
- target-branch update and predictor comparison;
- VICReg / SIGReg / VISReg comparison;
- action-free vs action-conditioned world models;
- adjacent modality/variational methods;
- evidence matrix distinguishing representation benchmarks, dynamics prediction and planning success.

Code listings:
- one small generic context/target forward path, clearly identified as algorithmic pseudocode unless copied from an official implementation;
- one official I-JEPA or V-JEPA mask/predictor operation;
- one official SIGReg/VISReg regularization operation only after source and license check;
- one short V-JEPA 2 frozen encoder loading example or config fragment;
- possibly one planner-side latent prediction call from official JEPA-WMs, only if it explains a central concept.

No full training loop, no giant model definition, no fake reproduction of internet-scale pretraining, and no listing copied from unofficial forks.

---

## 8. Proof and derivation programme

Keep proofs in the main chapter only when they clarify a design choice. Use boxed “Derivation” elements or a short appendix/dossier link for the longer details.

Main-text derivations:
1. generic JEPA regression objective and the role of stop-gradient;
2. conditional-mean optimum of squared-error prediction;
3. EMA as an exponentially weighted parameter history;
4. VICReg's covariance penalty vs full distributional constraints;
5. characteristic function of a standard normal and the projection argument;
6. one-dimensional empirical \(W_2^2\) as squared distance between sorted samples / quantiles;
7. action-conditioned predictive sufficiency with the action held fixed.

Optional extended derivations:
- a precise version of LeJEPA's downstream-risk assumptions and theorem;
- Epps–Pulley statistic and the selected quadrature;
- random direction estimator variance and projection-count trade-offs;
- gradients of the exact VISReg scale and shape losses as implemented;
- deterministic predictor failure under multimodal future distributions;
- why a planner cannot recover information that the encoder discarded.

Do not present a derivation from the author's notes as a theorem from the primary paper unless the statements and assumptions match. The proof ledger is in 10_claims_audit_and_proofs.md.

---

## 9. Manning production and style rules

Follow the current Chapter 3 manuscript style and the Manning AsciiDoc template in the must-read folder:

- Chapter-level heading and numbering attributes should match Chapter 3, with chapter number 4 and unique IDs. Preserve the established counters and stable cross-references.
- Use latexmath blocks for displayed equations and define every symbol before or immediately after first use.
- Use numbered figure, listing and table anchors with meaningful captions. Reference each major figure in the prose before or after placement.
- Captions should explain the visual argument in approximately four lines or less (the current Chapter 3 editorial budget is about 340 characters). Do not duplicate the caption in an adjacent paragraph.
- Put original schematics in the established images directory only after the brief is approved. Artwork should not contain a printed figure number, title or caption.
- Sidebars should make one specific distinction or give one compact intuition. Do not turn each method into a sidebar.
- Include a single concise comparison table where a reader needs to compare architectures, not multiple redundant taxonomies.
- Code lines in ordinary listings should remain at or below 76 characters; annotated lines at or below 55 characters; sidebar code at or below 70 characters. Wrap long statements and use concise variable names only where they are unambiguous.
- Use original diagrams; no copied paper screenshots or figures. Check figure, table and code licenses independently. The fact that a paper is open access does not automatically license its artwork, repository or weights.
- Use calm textbook prose, no unsupported superlatives, no generic “breakthrough” language, no rhetorical overclaiming.
- Keep citations in one consolidated bibliography, with author–year attributions in prose as in Chapter 3. Do not add a separate links section.
- Use PyTorch examples only, consistent with the book's current code policy.
- Build and validate with the repository's existing manuscript scripts. A PDF build is an author-side preview, not Manning's final production typesetting.

**Style access limitation:** the PDFs for Chapters 1 and 2 and the Manning Manual of Style are present in agent_memory_mustread, but the repository connector exposed their binary contents in a form that could not be rendered or text-extracted in this pass. The present plan therefore uses the TOC, the complete Chapter 3 AsciiDoc, the current manuscript README and figure register, and the checked-in Manning AsciiDoc template/style comments. Before typesetting, verify any publisher-specific details against the local PDF manual.

---

## 10. Scientific and editorial rules

1. Never say “JEPA predicts the future” without specifying the target, context, horizon, modality and whether actions are inputs.
2. Never claim that pixel-level unpredictability is always irrelevant. Some tasks require texture, force cues, small objects or contact state.
3. Never claim that Gaussian embeddings are universally optimal. Attribute the result to the exact risk analysis and assumptions of LeJEPA.
4. A finite set of random projections is an estimator, not a proof of exact distribution matching.
5. A regularizer that penalizes collapse does not guarantee semantic information, causal state, Markov sufficiency or controllability.
6. EMA and stop-gradient are not interchangeable and not universally required in all JEPA variants.
7. Separate V-JEPA 2's action-free pretraining from V-JEPA 2-AC's action-conditioned post-training.
8. Separate V-JEPA 2.1's dense-feature recipe from the control/planning stack.
9. Label A-JEPA, Audio-JEPA, VL-JEPA, Var-JEPA and VJEPA as related lines, not as if they all share the same training recipe.
10. Distinguish the JEPA-WMs paper's reported planning results from a general proof that latent-space planning always works.
11. Present A Path Towards Autonomous Machine Intelligence as a position paper and architectural proposal, not as a validated deployed system.
12. Use current source versions and date-sensitive benchmark numbers. Avoid leaderboard claims unless the benchmark, split, metric and publication date are stated.

---

## 11. Drafting workflow and release gates

**Gate A — source verification**
- Read the primary paper for every method in the core table.
- Check the exact loss, target construction, masking and update rules against official code/config.
- Record paper version, date, commit or tag, license and limitations.

**Gate B — narrative**
- Draft the generic architecture before discussing specific architectures.
- Place each paper in the teaching argument and identify the limitation it addresses.
- Remove any explanation already taught in Chapter 3 unless the chapter needs a one-paragraph recap.

**Gate C — scientific audit**
- Check all theorem assumptions, objective normalizations and complexity claims.
- Label empirical claims by source and benchmark.
- Use a “claim not supported” register for all overstatements found in notes.

**Gate D — visual and production**
- Approve figure briefs before generating artwork.
- Inspect all labels at printed book-column width.
- Verify all cross-references, equation numbers, listing line widths, captions and bibliography.
- Run manuscript structural verification, build the PDF and inspect every page.

**Gate E — reader test**
A reader should be able to explain why latent target prediction differs from reconstruction, why V-JEPA 2-AC is a different stage from V-JEPA 2 pretraining, why SIGReg/VISReg do not prove sufficiency, and which additional elements are needed for an agent to plan. If they cannot, revise the explanation before adding more papers.
