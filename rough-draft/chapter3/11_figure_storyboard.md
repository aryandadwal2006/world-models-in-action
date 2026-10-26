# Chapter 3 figure storyboard

## Production status

The composed manuscript on \`manuscript-theory\` contains **20 numbered figure briefs**, with no image macros and no generated or pasted artwork. Each brief specifies the mechanism, primary paper or official repository to consult, and what an original book schematic should teach. These placeholders are part of the working manuscript, not final artwork.

Draw figures in a consistent Manning-style visual system. Use grayscale-safe contrasts, arrows, labels, solid/dashed lines and shape differences. Avoid gradients, 3D, data-analysis plots, screenshot reproduction and architecture-name collages. The official papers are reference sources; redraw mechanisms as original explanatory diagrams.

## Figure list in manuscript order

| No. | Figure subject | What it must explain | Source basis |
|---:|---|---|---|
| 1 | Map the training signals | Candidate discrimination, bootstrap targets, batch statistics, prototypes and masked targets; show hybrid methods without suggesting disjoint families | SimCLR, BYOL, SimSiam, Barlow Twins, VICReg, DINO, SwAV, MAE |
| 2 | BYOL architecture | Online encoder/projector/predictor, detached target branch and separate EMA update | BYOL paper; \`google-deepmind/deepmind-research/byol\` |
| 3 | SimSiam architecture | Shared encoder, online predictor and stop-gradient target; no target EMA | SimSiam paper; \`facebookresearch/simsiam\` |
| 4 | DINO teacher/student | Global/local crops, prototype distributions, centering, sharpening and teacher EMA | DINO paper; \`facebookresearch/dino\` |
| 5 | DINO family progression | DINO self-distillation; DINOv2's added masked-patch/training components; DINOv3 dense-feature focus/Gram anchoring; not an ablation chart | DINO, DINOv2 and DINOv3 papers/repositories |
| 6 | Barlow Twins correlation matrix | Identity target; diagonal cross-view agreement and off-diagonal redundancy penalty | Barlow Twins paper; \`facebookresearch/barlowtwins\` |
| 7 | VICReg loss branches | Paired-feature invariance, variance floor and covariance penalty as distinct terms | VICReg paper; \`facebookresearch/vicreg\` |
| 8 | Objective pathway comparison | Contrastive candidate loss vs feature prediction vs batch statistics vs teacher distribution | Original synthesis of core papers |
| 9 | CLIP dual encoder | Image/text embeddings, pairwise similarity matrix and bidirectional matched-pair objective | CLIP paper; \`openai/CLIP\` |
| 10 | SimCLR pipeline | Two augmentation paths, shared encoder/projection head and batchwise contrastive loss; distinguish \(h\) from \(z\) | SimCLR paper; \`google-research/simclr\` |
| 11 | InfoNCE gradient | Anchor, positive, negatives, temperature, softmax and \(q_j-\mathbf{1}[j=p]\) gradient | InfoNCE derivation; SimCLR paper |
| 12 | MoCo queue | Query encoder, momentum key encoder, positive key, FIFO queue and EMA update | MoCo paper; official MoCo repositories |
| 13 | Masked target spaces | Same mask, different pixel/code/teacher targets; tiny language context comparison | MAE, BEiT, iBOT, data2vec, BERT, GPT |
| 14 | MAE architecture | Patchify/mask, visible-only encoder, restore mask tokens, lightweight decoder and masked-only loss | MAE paper; \`facebookresearch/mae\` |
| 15 | Target families for masked prediction | MAE RGB pixels; BEiT discrete codes; iBOT teacher distributions; data2vec contextual features | MAE, BEiT, iBOT, data2vec papers/repos |
| 16 | Projection head vs exported feature | \(h=f(x)\), \(z=g(h)\), pretraining loss on \(z\), downstream use often on \(h\) | SimCLR, BYOL, Barlow Twins, VICReg papers |
| 17 | Global vector vs patch grid | Pooling trades spatial structure for compactness; patch grid preserves local correspondence | Original schematic; DINOv2/DINOv3 feature interfaces |
| 18 | State aliasing | Same current image but different prior motion/future; single-frame bottleneck vs history-aware state | Original world-model teaching example |
| 19 | State-evidence audit | Feature spread, probes, temporal consistency, dynamics and action-conditioned evaluation answer different questions | Chapter 2 evaluation contract |
| 20 | Implementation path | Data view/mask → encoder/target → objective → distinct downstream checks; separate local mechanics from official recipe | Original code-companion workflow |

## Placement

- Section 3.1: Figures 1–8, plus the brief CLIP diagram in Figure 9.
- Section 3.2: Figures 10–12.
- Section 3.3: Figures 13–15.
- Section 3.4: Figures 16–17.
- Section 3.5: Figures 18–19.
- Section 3.6: Figure 20.

## Copyright and claim discipline

Use official paper figures to verify the computation graph and labels, but do not directly paste screenshots or closely trace a published figure as book art. Create an original diagram of the method's essential mechanics and cite the paper. A repository link establishes where to inspect code, not that the book ran the original method. Do not make an experiment chart from the small DMC runs substitute for a method architecture diagram.
