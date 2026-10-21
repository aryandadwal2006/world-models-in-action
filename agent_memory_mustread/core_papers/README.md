# Core papers for Chapter 3

This folder keeps a small, directly relevant reading set beside the manuscript. The three PDFs listed as mirrored below are downloaded unmodified from their official source pages and are distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). Attribution and source links are recorded here. This permission applies to the paper files, not automatically to third-party figures, code, model weights, or datasets discussed in those papers.

## Mirrored papers

### SimCLR
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. “A Simple Framework for Contrastive Learning of Visual Representations.” *Proceedings of ICML 2020*, PMLR 119, pp. 1597–1607.

- Local file: `chen-etal-2020-simclr.pdf`
- Original paper page: https://proceedings.mlr.press/v119/chen20j.html
- Official PDF: https://proceedings.mlr.press/v119/chen20j/chen20j.pdf
- License: CC BY 4.0. PMLR publication agreement: https://proceedings.mlr.press/pmlr-license-agreement.html

What matters for the manuscript: the augmentation pipeline defines the positive-pair invariances; the nonlinear projection head is distinct from the representation used for downstream tasks; and the reported results depended on training scale and number of steps. The toy Cartpole experiments in the book do not reproduce SimCLR's ImageNet setting.

### Barlow Twins
Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. “Barlow Twins: Self-Supervised Learning via Redundancy Reduction.” *Proceedings of ICML 2021*, PMLR 139, pp. 12310–12320.

- Local file: `zbontar-etal-2021-barlow-twins.pdf`
- Original paper page: https://proceedings.mlr.press/v139/zbontar21a.html
- Official PDF: https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf
- License: CC BY 4.0. PMLR publication agreement: https://proceedings.mlr.press/pmlr-license-agreement.html

What matters for the manuscript: the method matches the cross-correlation matrix of two views to the identity. The diagonal term aligns corresponding coordinates; the off-diagonal term reduces cross-coordinate redundancy. This is not the same mechanism as stop-gradient, target-network averaging, or VICReg's explicit variance floor.

### V-JEPA 2
Mido Assran et al. “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” arXiv:2506.09985.

- Local file: `assran-etal-2025-vjepa2.pdf`
- arXiv record: https://arxiv.org/abs/2506.09985
- Official PDF: https://arxiv.org/pdf/2506.09985
- License listed in arXiv metadata: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

What matters for the manuscript: distinguish an action-free visual representation learned from large-scale video from a later action-conditioned predictor trained with interaction data. The reported planning results belong to the full staged system, not to self-supervised representation learning alone.

## Additional core papers (link to the authoritative copy)

These are indexed here for reading but are not mirrored in this folder. Open access by itself is not enough to assume that every redistribution is permitted; check each paper's current arXiv or publisher license before adding a local copy.

- **Contrastive Predictive Coding (CPC)** — van den Oord, Li, and Vinyals (2018): https://arxiv.org/abs/1807.03748
- **SimSiam** — Chen and He (2021): https://arxiv.org/abs/2011.10566
- **BYOL** — Grill et al. (2020): https://arxiv.org/abs/2006.07733
- **VICReg** — Bardes, Ponce, and LeCun (2022): https://arxiv.org/abs/2105.04906
- **Masked Autoencoders (MAE)** — He et al. (2022): https://arxiv.org/abs/2111.06377
- **I-JEPA** — Assran et al. (2023): https://arxiv.org/abs/2301.08243
- **A survey on self-supervised learning** — Gui et al. (2024): https://arxiv.org/abs/2301.05712

## Reading checks for the manuscript

As each method is discussed, verify four separate claims against the paper: (1) the objective's actual target and sampling assumptions; (2) which architectural asymmetries or regularizers the method requires; (3) what is directly measured in the original experiments; and (4) what does *not* follow from a low training loss or high retrieval score. Do not transfer claims about natural-image benchmarks to the small-control-suite experiment without stating the change in data, architecture, and evaluation.
