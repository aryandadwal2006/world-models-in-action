# Chapter 3 figure and table production crosswalk

**Source:** manuscript/CH03_Dadwal_World_Models_in_Action.adoc on manuscript-theory. Numbering follows anchor order in that source.

## Figures

| Figure | Prompt file | Subject |
|---:|---|---|
| 3.1 | images/CH03_F01_Dadwal.md | Self-supervised objectives grouped by their training signal |
| 3.2 | images/CH03_F02_Dadwal.md | The representation contract |
| 3.3 | images/CH03_F03_Dadwal.md | Augmentation defines the invariances rewarded by training |
| 3.4 | images/CH03_F04_Dadwal.md | Constructing positives and negatives |
| 3.5 | images/CH03_F05_Dadwal.md | InfoNCE as candidate classification |
| 3.6 | images/CH03_F06_Dadwal.md | How the InfoNCE gradient acts on candidate scores |
| 3.7 | images/CH03_F07_Dadwal.md | Alignment and uniformity are distinct geometric requirements |
| 3.8 | images/CH03_F08_Dadwal.md | A contrastive score, an energy model and a diffusion process |
| 3.9 | images/CH03_F09_Dadwal.md | SimCLR's two-view encoder and projection head |
| 3.10 | images/CH03_F10_Dadwal.md | Momentum contrast: query encoder, key encoder and queue |
| 3.11 | images/CH03_F11_Dadwal.md | Historical contrastive objectives change the positive relation |
| 3.12 | images/CH03_F12_Dadwal.md | CPC: causal context predicts future latent candidates |
| 3.13 | images/CH03_F13_Dadwal.md | Why agreement alone permits collapse |
| 3.14 | images/CH03_F14_Dadwal.md | BYOL and SimSiam: similar predictors, different target updates |
| 3.15 | images/CH03_F15_Dadwal.md | DINO teacher-student cross-view distillation |
| 3.16 | images/CH03_F16_Dadwal.md | Barlow Twins: the cross-correlation matrix |
| 3.17 | images/CH03_F17_Dadwal.md | VICReg separates agreement, variance and covariance |
| 3.18 | images/CH03_F18_Dadwal.md | Core non-contrastive mechanisms at a glance |
| 3.19 | images/CH03_F19_Dadwal.md | VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions |
| 3.20 | images/CH03_F20_Dadwal.md | DeepCluster and SwAV: pseudo-labels and prototypes |
| 3.21 | images/CH03_F21_Dadwal.md | Different modalities create different self-supervised targets |
| 3.22 | images/CH03_F22_Dadwal.md | The JEPA context-target prediction architecture |
| 3.23 | images/CH03_F23_Dadwal.md | One visible context, several masked-prediction targets |
| 3.24 | images/CH03_F24_Dadwal.md | MAE's visible-only encoder and masked-patch decoder |
| 3.25 | images/CH03_F25_Dadwal.md | Discrete tokens and teacher features are different masked targets |
| 3.26 | images/CH03_F26_Dadwal.md | Masked prediction and diffusion learn different conditional tasks |
| 3.27 | images/CH03_F27_Dadwal.md | The projection head, global vector and local feature grid |
| 3.28 | images/CH03_F28_Dadwal.md | Aliasing is a failure of predictive sufficiency |
| 3.29 | images/CH03_F29_Dadwal.md | An encoder can provide a vector, spatial tokens or temporal context |

Each source placeholder points to a detailed prompt. Artwork is intentionally absent; do not infer numbering from legacy experiment filenames. Prompts use the book's #3B6FB6 accent, white background and grayscale-safe flat 2D line work.

## Tables and listings

| Item | Count | Purpose |
|---|---:|---|
| Numbered tables | 1 | Objective-family synthesis |
| Numbered listings | 2 | Short source excerpts from official SimCLR and VICReg implementations |

The end implementation-reference table is removed. Method-specific implementation links sit beside relevant methods; scholarly sources remain in the bibliography.

## Checks

```bash
python3 manuscript/verify-manuscript.py
cd manuscript && ./build-pdf.sh
```

The structural verifier checks anchors, figure-brief correspondence, table/listing counts, one bibliography heading and absence of rendered artwork. The PDF preview still requires manual inspection of equations, captions, placeholder boxes, code listings and page breaks.
