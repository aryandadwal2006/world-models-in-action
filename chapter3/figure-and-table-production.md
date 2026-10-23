# Chapter 3 figure and table crosswalk

**Updated:** 23 October 2026  
**Scope:** Printed figure/table numbering for the test-branch Chapter 3 manuscript. Asset names retain their historical stems after consolidation.

The canonical reader-facing source is `manuscript/CH03_Dadwal_World_Models_in_Action.adoc`. The test-branch PDF is an author-side preview, not final Manning production typesetting.

## Numbered figures

| Printed figure | Asset in `images/` | Subject |
|---|---|---|
| 3.1 | `CH03_F01_Dadwal.svg` | Objective-centred map of self-supervised training signals |
| 3.2 | `CH03_F02_Dadwal.svg` | Representation contract and aliasing |
| 3.3 | `CH03_F03_Dadwal.png` | Rendered observations from the three Control Suite tasks |
| 3.4 | `CH03_F04_Dadwal.svg` | Rotation, jigsaw, and colorization pretext tasks |
| 3.5 | `CH03_F05_Dadwal.svg` | Composite view: anchor/positive/negative pairs and NT-Xent classification |
| 3.6 | `CH03_F07_Dadwal.png` | Recorded NT-Xent training loss; it is still decreasing at the final epoch |
| 3.7 | `CH03_Aliasing_Diagnostic.svg` | Single-frame versus two-frame state-probe comparison |
| 3.8 | `CH03_F09_Dadwal.svg` | Feature-spread diagnostic for attraction-only and NT-Xent objectives |
| 3.9 | `CH03_F10_Dadwal.svg` | Anti-collapse mechanisms, shown as a conceptual comparison |
| 3.10 | `CH03_F11_Dadwal.png` | Recorded VICReg loss and component curves |
| 3.11 | `CH03_F12_Dadwal.svg` plus `CH03_F12_Dadwal_Source.png` | MAE architecture and recorded reconstruction examples |
| 3.12 | `CH03_State_Probe_Comparison.svg` | Mean held-out state-probe scores |
| 3.13 | `CH03_F13_Dadwal.svg` | Latent-space prediction versus pixel reconstruction |
| 3.14 | `CH03_F14_Dadwal.png` | State-probe performance across latent dimensions |
| 3.15 | `CH03_F15_Dadwal.svg` | CPC context and candidate-future latent scoring |
| 3.16 | `CH03_CPC_Evaluation.svg` | Combined CPC state probes, retrieval rates, and horizon-wise candidate-separation metrics |
| 3.17 | `CH03_F17_Dadwal.svg` | Encoder as a component in the larger control loop |

Figure 3.5 combines the former positive/negative pair and NT-Xent classification diagrams. The PCA panel and the standalone CPC retrieval plot are no longer embedded; their source assets remain in the repository.

## Numbered tables

| Printed table | Anchor in the AsciiDoc | Subject |
|---|---|---|
| 3.1 | `ch3-table-T01` | Notation used in the chapter |
| 3.2 | `ch3-table-T06` | Anti-collapse mechanisms and limitations |
| 3.3 | `ch3-table-T05` | Representation objectives and their trade-offs |
| 3.4 | `ch3-table-T07` | Chapter 2 state contract, evidence, and remaining questions |

## Listings and supporting assets

The source contains ten numbered code-listing anchors after redundant SimSiam/BYOL and Barlow Twins code snippets were removed. The core mathematical spine remains: the NT-Xent derivation, InfoNCE bound and gradient, collapse counterexample, VICReg terms, MAE reconstruction loss, latent-dimension experiment, and CPC's bounded retrieval interpretation.

Empirical plots originate from committed experiment outputs; vector schematics are original conceptual illustrations.

## Quality checks

From a complete checkout:

```bash
python manuscript/verify-manuscript.py
```

The verifier checks chapter structure, unique anchors, numbering counts, listing captions, image paths, and Summary placement. After building, inspect the rendered pages manually; passing structural checks is not a replacement for PDF proofing.
