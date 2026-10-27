# Chapter 3 figure, table and listing crosswalk

**Current source:** `manuscript/CH03_Dadwal_World_Models_in_Action.adoc` on `manuscript-theory`.  
**Status:** current crosswalk for the theory-first manuscript as of 27 October 2026. This file is synchronized with `images/README.md`; the anchor order in the AsciiDoc source determines printed numbering. The legacy `F##` stems are stable asset names, not printed figure numbers.

## Figure register

| Printed figure | Production brief | Caption |
|---:|---|---|
| 3.1 | `images/CH03_F01_Dadwal.md` | Self-supervised objectives grouped by their training signal |
| 3.2 | `images/CH03_F02_Dadwal.md` | The representation contract |
| 3.3 | `images/CH03_F03_Dadwal.md` | Augmentation defines the invariances rewarded by training |
| 3.4 | `images/CH03_F04_Dadwal.md` | Constructing positives and negatives |
| 3.5 | `images/CH03_F05_Dadwal.md` | InfoNCE as candidate classification |
| 3.6 | `images/CH03_F06_Dadwal.md` | How the InfoNCE gradient acts on candidate scores |
| 3.7 | `images/CH03_F07_Dadwal.md` | Alignment and uniformity are distinct geometric requirements |
| 3.8 | `images/CH03_F08_Dadwal.md` | A contrastive score, an energy model and a diffusion process |
| 3.9 | `images/CH03_F09_Dadwal.md` | SimCLR's two-view encoder and projection head |
| 3.10 | `images/CH03_F10_Dadwal.md` | Momentum contrast: query encoder, key encoder and queue |
| 3.11 | `images/CH03_F11_Dadwal.md` | Historical contrastive objectives change the positive relation |
| 3.12 | `images/CH03_F12_Dadwal.md` | CPC: causal context predicts future latent candidates |
| 3.13 | `images/CH03_F13_Dadwal.md` | Why agreement alone permits collapse |
| 3.14 | `images/CH03_F14_Dadwal.md` | BYOL and SimSiam: similar predictors, different target updates |
| 3.15 | `images/CH03_F15_Dadwal.md` | DINO teacher-student cross-view distillation |
| 3.16 | `images/CH03_F16_Dadwal.md` | Barlow Twins: the cross-correlation matrix |
| 3.17 | `images/CH03_F17_Dadwal.md` | VICReg separates agreement, variance and covariance |
| 3.18 | `images/CH03_F18_Dadwal.md` | Core non-contrastive mechanisms at a glance |
| 3.19 | `images/CH03_F19_Dadwal.md` | VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions |
| 3.20 | `images/CH03_F20_Dadwal.md` | DeepCluster and SwAV: pseudo-labels and prototypes |
| 3.21 | `images/CH03_F21_Dadwal.md` | Different modalities create different self-supervised targets |
| 3.22 | `images/CH03_F23_Dadwal.md` | One visible context, several masked-prediction targets |
| 3.23 | `images/CH03_F24_Dadwal.md` | MAE's visible-only encoder and masked-patch decoder |
| 3.24 | `images/CH03_F25_Dadwal.md` | Discrete tokens and teacher features are different masked targets |
| 3.25 | `images/CH03_F26_Dadwal.md` | Masked prediction and diffusion learn different conditional tasks |
| 3.26 | `images/CH03_F27_Dadwal.md` | The projection head, global vector and local feature grid |
| 3.27 | `images/CH03_F29_Dadwal.md` | An encoder can provide a vector, spatial tokens or temporal context |
| 3.28 | `images/CH03_F28_Dadwal.md` | Aliasing is a failure of predictive sufficiency |
| 3.29 | `images/CH03_F22_Dadwal.md` | A training loss and a probe answer different questions |

## Tables and listings

| Printed object | Source anchor | Caption or purpose |
|---|---|---|
| Table 3.1 | `ch3-table-T01` | Self-supervised objectives compared by the signal they construct |
| Listing 3.1 | `ch3-listing-vicreg` | Three VICReg terms in the official PyTorch implementation (source excerpt) |

The current composed chapter contains **29 figure placeholders, one numbered table, and one numbered listing**. The single listing is a short source-attributed excerpt from the official PyTorch VICReg implementation; the manuscript does not contain a SimCLR listing or a standalone implementation tutorial. The one table is a conceptual method-family synthesis, not the historical experiment-comparison table referenced by some names under `ch03/results/`.

## Production rules

- Use `images/README.md` and the individual briefs as the artwork production source. Do not infer printed numbering from filenames.
- Artwork is not yet present; placeholders remain in the manuscript until each original figure is drawn and approved.
- Use white backgrounds, grayscale-safe dark line work, and `#3B6FB6` as the only restrained accent. Flat 2D only; no gradients, 3D effects, Mermaid, or decorative watermarks.
- Preserve equation signs, target/gradient arrows, and method-specific distinctions. Do not invent measurements or turn conceptual drawings into empirical claims.
- Before replacing a placeholder, check mathematical fidelity, figure/caption agreement, legibility at book-column width, and page flow in the rebuilt PDF.
- Figure 3.26 and Figure 3.27 have different jobs: the former compares the projection head and output interfaces; the latter shows how vector, spatial and temporal representations may feed later prediction.

## Build and verification

From the repository root:

```bash
python3 manuscript/verify-manuscript.py
cd manuscript && ./build-pdf.sh
```

The verifier checks manuscript structure and placeholder counts. A successful build does not replace visual review of the inserted artwork and final page layout.