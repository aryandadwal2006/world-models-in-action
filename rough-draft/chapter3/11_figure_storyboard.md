# Figures and visual storyboard for the theory-first chapter

## Figure A: Map the learning signal, not the publication chronology

A wide branching diagram beginning with data and a target-construction stage. Branch into:
- pair/candidate discrimination (SimCLR, MoCo, CLIP/SigLIP);
- online-target prediction / self-distillation (BYOL, SimSiam, DINO);
- feature-statistics constraints (Barlow Twins, VICReg);
- prototype assignment (DeepCluster, SwAV; DINO overlaps);
- masked input prediction (BERT/MAE/BEiT/iBOT/ELECTRA/data2vec);
- contextual/latent predictive representations (data2vec, JEPA family).

Use dotted connector lines where a method spans categories. Include a note that categories overlap and the diagram is a teaching map. Do not portray one family as a linear replacement for the previous family.

## Figure B: One InfoNCE anchor and its candidate logits

Show anchor z_i, positive z_j and several negative candidates. All candidates pass through a similarity function and temperature; softmax gives a distribution over candidate indices; cross-entropy rewards the positive index. Add a gradient annotation: positive logit gradient p_j-1, each negative logit gradient p_k. The point is to connect the equation with the gradient rather than depict a generic neural network.

## Figure C: Architectures with the same input views but different target paths

Use four aligned rows with identical inputs (two augmented views), then show:
- SimCLR: shared encoder/projection → candidate softmax including negative views;
- BYOL/SimSiam: online branch and predictor → detached target representation, with EMA on BYOL target only;
- DINO: student probability distribution matched to teacher distribution over prototypes;
- Barlow/VICReg: paired embeddings → batch cross-correlation or variance/covariance statistics.

A minimal legend must define stop-gradient and EMA. Do not claim DINO is simply the same as BYOL; the row contrasts target forms, not equivalent losses.

## Figure D: The target space changes the task

Three visual patches with a masked region, each linked to a target:
- pixel RGB patch and an MSE reconstruction target;
- discrete code ID and a cross-entropy target (BEiT-style);
- contextual feature vector / teacher distribution (iBOT/data2vec/JEPA-style).
Add a separate language strip for masked token versus causal next token. Emphasize that identical masks can train different predictions.

## Figure E: Objective contract table rendered as a compact decision map

Axes: positive-pair construction; target space; negatives required; explicit anti-collapse mechanism; main compute burden; known failure modes; what downstream test is necessary. This can be a table in prose rather than a graphic if readability suffers. Do not create a dense unreadable poster.

## Grayscale and accessibility

All figures must remain legible in grayscale. Use solid/dashed/dotted lines, shapes, text labels and line weights rather than color alone. Prefer SVG with embedded text, no gradients, no 3D and no screenshots of external papers. Captions should state the pedagogical point and clarify when a figure is an original schematic synthesized from cited methods.
