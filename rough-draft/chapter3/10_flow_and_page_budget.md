# Chapter 3 reader-facing flow and page budget

## Editorial decision

The chapter is a foundations chapter for the world-model book, not a general survey of all self-supervised learning. Preserve the book TOC and make each section prepare the next:

1. Self-supervised objectives construct different training targets.
2. Contrastive learning is derived in enough detail to understand its mechanics.
3. Masked prediction changes the target and the information available as context.
4. A latent bottleneck determines which representation reaches the later model.
5. A compact state must preserve distinctions needed for prediction.
6. A small encoder implementation makes the objectives executable without pretending to reproduce paper-scale results.

## Revised page target

**Target: 45–50 finished book pages.** This estimate assumes that the 20 planned architecture/working diagrams are eventually drawn and placed. The current manuscript contains descriptions, not images; it should not be treated as a final paginated PDF yet.

| TOC section | Target pages | Scope |
|---|---:|---|
| 3.1 Self-supervised learning foundations | 14 | Objective contract; intuitive BYOL, SimSiam, Barlow Twins, VICReg and DINO family explanations; only short mentions of DeepCluster, SwAV, CLIP, BERT/GPT and speech SSL |
| 3.2 Contrastive learning | 8 | InfoNCE/NT-Xent derivation, gradient, temperature, false negatives, candidate count, SimCLR, MoCo, short CPC bridge |
| 3.3 Masked prediction | 5 | General mask/context/target contract; MAE as main method; BEiT, iBOT and data2vec as target-space alternatives; BERT/GPT only as brief analogies |
| 3.4 Latent bottlenecks | 4 | Projection head vs encoder output; global vector vs patch grid; compression and information loss |
| 3.5 Compact state representations | 5 | State-relevant invariances, aliasing, probes, temporal consistency, action conditioning and evidence limits |
| 3.6 Building a simple latent encoder | 4 | Two implementations only: verify InfoNCE mechanics, then build/run the compact MAE-style masked predictor; official source repositories for canonical recipes |
| Summary, exercises and references | 5 | Derivation and reasoning exercises, concise synthesis, cited papers and repository pointers |
| **Total target** | **45** | Allow adjustment within the 45–50-page range after figure layout and Manning-template pagination |

Do not pad the chapter with result plots or extra methods to meet the target. The designed diagrams, method explanations, equations, tables, sidebars, listings and exercises should earn the space.

## Selection decisions

### Explain intuitively in the foundations section

- **BYOL:** online encoder/projector/predictor, detached target, EMA target update and why the update path matters.
- **SimSiam:** shared encoder, predictor and stop-gradient; explain why its update path differs from BYOL.
- **Barlow Twins:** diagonal cross-view alignment and off-diagonal redundancy reduction in the batch cross-correlation matrix.
- **VICReg:** invariance, variance floor and covariance penalty; distinguish constant-feature prevention from semantic usefulness.
- **DINO:** teacher/student probability distributions, prototypes, centering, sharpening and multi-crop prediction.
- **DINOv2/DINOv3:** explain the broader training systems and the emphasis on patch/dense features, without claiming their reported differences isolate one loss change.

These are intuition-and-mechanism treatments, not five extra implementations or independent empirical surveys.

### Mention briefly

- **DeepCluster:** offline clustering creates pseudo-labels, then the encoder predicts them.
- **SwAV:** online balanced prototype assignments are predicted across views.
- **CLIP/SigLIP:** image-text pairing, with one short note on softmax versus sigmoid loss.
- **BERT/GPT:** masked-token versus causal next-token targets, solely as a concise illustration of context choice.
- **Speech SSL:** wav2vec 2.0, HuBERT and data2vec as examples of quantized, cluster-derived and teacher-feature targets.
- **CPC:** a short historical bridge to temporal discrimination, with an explicit warning that future-latent retrieval is not action-conditioned rollout.
- **JEPA family:** hand off the detailed derivation and named variants to Chapter 4.
- **VAE/GAN/autoregressive generation/diffusion:** leave substantive treatment to Chapters 8–9.

## Two implementation exercises

1. **InfoNCE verification.** Use the loop-based reference and vectorized objective to compare loss and gradients. State candidate convention and sampling assumptions. Do not turn a nominal InfoNCE quantity into an unconditional mutual-information claim.
2. **Compact MAE-style masked predictor.** Use the local educational runner/module to illustrate patchification, random masking, visible-only encoding, mask-token restoration, decoder prediction and masked-only loss. Compare the architecture to the official MAE repository. Explicitly label the local model as reduced-scale and non-reproduction.

Use the official upstream submodules as the source of truth for paper architecture, update rules, training recipe, and version-specific settings. A source inspection or pinned repository is not evidence that the original paper's results were reproduced. Existing collapse/BYOL/SimSiam/Barlow/VICReg experiments can support a short mathematical note or editorial verification, not extra training tutorials or a method-ranking figure.

## Figure policy

Exactly 20 figure briefs are placed in the composed AsciiDoc manuscript. They are **descriptions only** at this stage. No image is generated, no external figure is pasted, and no experiment/data-analysis chart is planned as a chapter figure. The figure briefs cite the primary paper/repository to consult and specify an original teaching schematic that should be drawn later.

- Figures 1–9: learning-signal map and core mechanism diagrams in Section 3.1.
- Figures 10–12: SimCLR, InfoNCE gradient and MoCo in Section 3.2.
- Figures 13–15: masked target spaces, MAE architecture and target variants in Section 3.3.
- Figures 16–17: projection vs exported feature and global vector vs patch grid in Section 3.4.
- Figures 18–19: state aliasing and the world-model evidence audit in Section 3.5.
- Figure 20: implementation/evaluation path in Section 3.6.

All final diagrams should work in grayscale and use line style/shape/labels rather than color alone. Avoid gradients, 3D, screenshots and architecture-name collages. A figure must explain a mechanism, not decorate a section.

## Editorial guardrails

- Keep the TOC's seven substantive sections and avoid renaming or front-loading Chapter 4's JEPA material.
- Every method explanation starts with the target and update rule, then gives intuition and limitation.
- Separate the paper's claim, what the released source code does, what the local experiment verifies, and editorial synthesis.
- Keep one notation for \(x\), \(h\), \(z\), target encoder, batch and loss reductions.
- Check equation tags after any section reordering.
- In particular, distinguish the original VICReg per-example squared-vector-norm reduction from an educational elementwise MSE reduction; coefficients are not interchangeable without rescaling.
- State the output tensor used by downstream probes; pretraining loss on a projection \(z\) is not automatically an evaluation of encoder feature \(h\).
- Do not claim state sufficiency, action grounding, long-horizon rollout or planning utility from a self-supervised loss or a linear probe alone.
- Complete the primary-paper/source citation and AsciiDoc-to-PDF pagination audit before release.
