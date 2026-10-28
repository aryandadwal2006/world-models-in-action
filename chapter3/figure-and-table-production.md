# Chapter 3 figure, table and listing crosswalk

**Current source:** `manuscript/CH03_Dadwal_World_Models_in_Action.adoc` on `manuscript-theory`. **Status:** synchronized after the Chapter 4 scope and figure-numbering correction, 28 October 2026. Printed numbering follows source anchor order; legacy `F##` stems are stable asset names, not printed numbers.

| Printed figure | Production brief | Caption |
|---:|---|---|
| 3.1 | `images/CH03_F02_Dadwal.md` | The representation contract |
| 3.2 | `images/CH03_F03_Dadwal.md` | Augmentation defines the invariances rewarded by training |
| 3.3 | `images/CH03_F04_Dadwal.md` | Constructing positives and negatives |
| 3.4 | `images/CH03_F05_Dadwal.md` | InfoNCE: candidate probabilities and gradients |
| 3.5 | `images/CH03_F07_Dadwal.md` | Alignment and uniformity are distinct geometric requirements |
| 3.6 | `images/CH03_F10_Dadwal.md` | Momentum contrast: query encoder, key encoder and queue |
| 3.7 | `images/CH03_F12_Dadwal.md` | CPC: causal context predicts future latent candidates |
| 3.8 | `images/CH03_F13_Dadwal.md` | Why agreement alone permits collapse |
| 3.9 | `images/CH03_F14_Dadwal.md` | BYOL and SimSiam: similar predictors, different target updates |
| 3.10 | `images/CH03_F15_Dadwal.md` | DINO teacher-student cross-view distillation |
| 3.11 | `images/CH03_F16_Dadwal.md` | Barlow Twins: the cross-correlation matrix |
| 3.12 | `images/CH03_F17_Dadwal.md` | VICReg separates agreement, variance and covariance |
| 3.13 | `images/CH03_F23_Dadwal.md` | One visible context, several masked-prediction targets |
| 3.14 | `images/CH03_F24_Dadwal.md` | MAE's visible-only encoder and masked-patch decoder |
| 3.15 | `images/CH03_F27_Dadwal.md` | The training representation and the downstream interface |
| 3.16 | `images/CH03_F28_Dadwal.md` | Aliasing is a failure of predictive sufficiency |
| 3.17 | `images/CH03_F22_Dadwal.md` | A training loss and a probe answer different questions. No single metric certifies predictive sufficiency. |

Table 3.1 compares objective signals. Listing 3.1 is a short source-attributed excerpt from official PyTorch VICReg. The revised chapter contains 17 figure placeholders, one table and one listing.

Use the individual briefs and this crosswalk to create original 2D figures with white backgrounds, grayscale-safe dark line work and restrained `#3B6FB6`. Check every equation/arrow against the text and inspect the rebuilt PDF at book-column width. No artwork is yet approved.


## Artwork production constraints

- Do not render figure numbers, titles, captions or prose descriptions inside artwork; the book layout supplies the number and caption separately.
- Use print-sized lettering, about 1–2 pt larger than a typical screen-first infographic. All labels must remain readable at book-column width.
- Use white backgrounds, grayscale-safe dark line work and restrained `#3B6FB6`; flat 2D only, with no gradients, 3D effects, Mermaid or decorative watermarks.
