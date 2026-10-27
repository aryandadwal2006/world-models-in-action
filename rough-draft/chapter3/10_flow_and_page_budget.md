> **Status (27 October 2026): historical editorial plan, superseded by the composed manuscript.** The current source and section order live in `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`; the current register lists 18 figures after the figure-and-prose audit. Use this file only for background, not as a checklist or authority for figure counts, code listings, JEPA coverage or page budget.

# Chapter 3 final-draft specification

## Editorial purpose

Teach the main self-supervised representation-learning objectives from their mathematical targets and optimization mechanisms. The chapter should follow one clear route: foundations, contrastive learning, non-contrastive objectives, masked prediction, latent bottlenecks, predictive sufficiency and official implementation references.

The prose should be direct and theoretical. This is not a broad survey of every SSL paper, an experiment report, a robotics chapter or a homegrown reimplementation tutorial.

## Scope decisions

### Full explanations

- **Contrastive learning:** two-view construction, candidate-set convention, NT-Xent/InfoNCE, softmax and cosine gradients, temperature, false negatives, mutual-information interpretation and its assumptions, SimCLR, MoCo and a short CPC bridge.
- **BYOL and SimSiam:** prediction loss, stop-gradient, EMA update, shared versus target parameters, and the distinction between existence of a collapsed solution and the behavior of a particular training dynamic.
- **DINO family:** teacher/student probability distributions, cross-entropy, prototypes, centering, sharpening, multi-crop, and the distinction between DINO, DINOv2 and DINOv3.
- **Barlow Twins and VICReg:** cross-correlation toward identity versus explicit invariance, variance and covariance terms; explain what the statistics constrain and what they do not.
- **Masked prediction:** context/target contract, MAE's visible-only encoder and masked-only reconstruction loss, then BEiT, iBOT and data2vec as contrasting target spaces.
- **Latent bottlenecks and sufficiency:** projector versus encoder output, global vector versus patch grid, predictive sufficiency, aliasing and the limits of probes.

### Short references only

DeepCluster, SwAV, CLIP/SigLIP, BERT/GPT/ELECTRA and speech SSL receive brief comparisons to illustrate how target construction changes. JEPA gets a bridge to Chapter 4. Generative modelling belongs in later chapters.

## Length and status

The editorial target remains 45–50 finished book pages, but this must be measured from the rendered PDF and Manning-style layout, not estimated from word count. The current source contains about 8,900 words, six diagrams, one comparison table, equations, exercises and references. It deliberately contains no custom code listings. If the rendered chapter is materially shorter than the target, expand only derivations or mathematical explanations that are genuinely too compressed; do not restore experiment filler or add method summaries solely to occupy pages.

## Figures

The six authored diagrams are:

1. Objective-family map: CH03_T01_SSL_Landscape.svg.
2. InfoNCE candidates and gradients: CH03_T02_InfoNCE.svg.
3. Comparison of core objective mechanisms: CH03_T03_MethodMechanisms.svg.
4. Masked prediction target families: CH03_T04_TargetSpaces.svg.
5. MAE asymmetric encoder-decoder: CH03_F12_Dadwal.svg.
6. Representation contract and predictive aliasing: CH03_F02_Dadwal.svg.

The diagrams are original mechanism explanations, not pasted screenshots. The manuscript references each figure before its anchor.

## Code policy

The manuscript has zero custom code listings. The official-repository table directs readers to authors' code for exact normalization, update rules, masking protocols and training settings. Do not add local training recipes or present a compact educational variant as a reproduction.

## Final checks

- Render through manuscript/build-pdf.sh using the repository's GitHub Actions workflow.
- Verify SVG paths, equation rendering, caption numbering, references and page geometry.
- Confirm the six figure anchors and single objective-comparison table match the validator.
- Inspect the produced PDF's page count before deciding whether further theoretical expansion is needed.
