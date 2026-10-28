# Chapter 3 figure production briefs

**Status (28 October 2026):** final artwork-complete register for the revised `manuscript-theory` source. The composed chapter has **17 inserted figures**, one numbered synthesis table and one short VICReg listing.

Printed numbering follows anchor order in `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`. The Markdown brief stems are retained legacy identifiers; the PNG filenames `CH03_F01_Dadwal.png` through `CH03_F17_Dadwal.png` follow printed figure order.

## Figure register

| Printed figure | Production brief (legacy ID) | Artwork (printed order) | Caption |
|---:|---|---|
| 3.1 | `images/CH03_F02_Dadwal.md` | `images/CH03_F01_Dadwal.png` | The representation contract |
| 3.2 | `images/CH03_F03_Dadwal.md` | `images/CH03_F02_Dadwal.png` | Augmentation defines the invariances rewarded by training |
| 3.3 | `images/CH03_F04_Dadwal.md` | `images/CH03_F03_Dadwal.png` | Constructing positives and negatives |
| 3.4 | `images/CH03_F05_Dadwal.md` | `images/CH03_F04_Dadwal.png` | InfoNCE: candidate probabilities and gradients |
| 3.5 | `images/CH03_F07_Dadwal.md` | `images/CH03_F05_Dadwal.png` | Alignment and uniformity are distinct geometric requirements |
| 3.6 | `images/CH03_F10_Dadwal.md` | `images/CH03_F06_Dadwal.png` | Momentum contrast: query encoder, key encoder and queue |
| 3.7 | `images/CH03_F12_Dadwal.md` | `images/CH03_F07_Dadwal.png` | CPC: causal context predicts future latent candidates |
| 3.8 | `images/CH03_F13_Dadwal.md` | `images/CH03_F08_Dadwal.png` | Why agreement alone permits collapse |
| 3.9 | `images/CH03_F14_Dadwal.md` | `images/CH03_F09_Dadwal.png` | BYOL and SimSiam: similar predictors, different target updates |
| 3.10 | `images/CH03_F15_Dadwal.md` | `images/CH03_F10_Dadwal.png` | DINO teacher-student cross-view distillation |
| 3.11 | `images/CH03_F16_Dadwal.md` | `images/CH03_F11_Dadwal.png` | Barlow Twins: the cross-correlation matrix |
| 3.12 | `images/CH03_F17_Dadwal.md` | `images/CH03_F12_Dadwal.png` | VICReg separates agreement, variance and covariance |
| 3.13 | `images/CH03_F23_Dadwal.md` | `images/CH03_F13_Dadwal.png` | One visible context, several masked-prediction targets |
| 3.14 | `images/CH03_F24_Dadwal.md` | `images/CH03_F14_Dadwal.png` | MAE's visible-only encoder and masked-patch decoder |
| 3.15 | `images/CH03_F27_Dadwal.md` | `images/CH03_F15_Dadwal.png` | The training representation and the downstream interface |
| 3.16 | `images/CH03_F28_Dadwal.md` | `images/CH03_F16_Dadwal.png` | Aliasing is a failure of predictive sufficiency |
| 3.17 | `images/CH03_F22_Dadwal.md` | `images/CH03_F17_Dadwal.png` | A training loss and a probe answer different questions. No single metric certifies predictive sufficiency. |

## Production rules

- Original teaching schematics only; no pasted paper figures or implementation screenshots.
- White background, grayscale-safe dark line work and `#3B6FB6` as the only restrained accent. Flat 2D only; no gradients, 3D effects, Mermaid or decorative watermarks.
- Worked values are illustrative, not experiment results.
- Do not render figure numbers, titles, captions or prose descriptions inside artwork; the book layout supplies the number and caption separately.
- Use print-sized lettering, about 1–2 pt larger than a typical screen-first infographic, and check every label at book-column width.
- The deleted family map is not to be replaced with a new taxonomy. The method table is a navigation aid, not a claim of mutually exclusive families.
