# Chapter 3 figure production briefs

**Status:** Prompt-only artwork directory for manuscript-theory. No PNG, JPEG, WebP, SVG or other rendered artwork belongs here until generated and approved.

The composed AsciiDoc chapter has 29 numbered figure placeholders. Printed order follows anchors in the chapter source, not legacy experiment filenames. Each CH03_FNN_Dadwal.md file contains purpose, composition, required visual content, exclusions, acceptance checks and caption.

## Figure register

| Figure 3.1 | CH03_F01_Dadwal.md | Self-supervised objectives grouped by their training signal |
| Figure 3.2 | CH03_F02_Dadwal.md | The representation contract |
| Figure 3.3 | CH03_F03_Dadwal.md | Augmentation defines the invariances rewarded by training |
| Figure 3.4 | CH03_F04_Dadwal.md | Constructing positives and negatives |
| Figure 3.5 | CH03_F05_Dadwal.md | InfoNCE as candidate classification |
| Figure 3.6 | CH03_F06_Dadwal.md | How the InfoNCE gradient acts on candidate scores |
| Figure 3.7 | CH03_F07_Dadwal.md | Alignment and uniformity are distinct geometric requirements |
| Figure 3.8 | CH03_F08_Dadwal.md | A contrastive score, an energy model and a diffusion process |
| Figure 3.9 | CH03_F09_Dadwal.md | SimCLR's two-view encoder and projection head |
| Figure 3.10 | CH03_F10_Dadwal.md | Momentum contrast: query encoder, key encoder and queue |
| Figure 3.11 | CH03_F11_Dadwal.md | Historical contrastive objectives change the positive relation |
| Figure 3.12 | CH03_F12_Dadwal.md | CPC: causal context predicts future latent candidates |
| Figure 3.13 | CH03_F13_Dadwal.md | Why agreement alone permits collapse |
| Figure 3.14 | CH03_F14_Dadwal.md | BYOL and SimSiam: similar predictors, different target updates |
| Figure 3.15 | CH03_F15_Dadwal.md | DINO teacher-student cross-view distillation |
| Figure 3.16 | CH03_F16_Dadwal.md | Barlow Twins: the cross-correlation matrix |
| Figure 3.17 | CH03_F17_Dadwal.md | VICReg separates agreement, variance and covariance |
| Figure 3.18 | CH03_F18_Dadwal.md | Core non-contrastive mechanisms at a glance |
| Figure 3.19 | CH03_F19_Dadwal.md | VICReg, SIGReg/LeJEPA and VISReg constrain different aspects of feature distributions |
| Figure 3.20 | CH03_F20_Dadwal.md | DeepCluster and SwAV: pseudo-labels and prototypes |
| Figure 3.21 | CH03_F21_Dadwal.md | Different modalities create different self-supervised targets |
| Figure 3.22 | CH03_F22_Dadwal.md | The JEPA context-target prediction architecture |
| Figure 3.23 | CH03_F23_Dadwal.md | One visible context, several masked-prediction targets |
| Figure 3.24 | CH03_F24_Dadwal.md | MAE's visible-only encoder and masked-patch decoder |
| Figure 3.25 | CH03_F25_Dadwal.md | Discrete tokens and teacher features are different masked targets |
| Figure 3.26 | CH03_F26_Dadwal.md | Masked prediction and diffusion learn different conditional tasks |
| Figure 3.27 | CH03_F27_Dadwal.md | The projection head, global vector and local feature grid |
| Figure 3.28 | CH03_F28_Dadwal.md | Aliasing is a failure of predictive sufficiency |
| Figure 3.29 | CH03_F29_Dadwal.md | An encoder can provide a vector, spatial tokens or temporal context |

## Production rules

- Create original teaching schematics; do not paste paper figures or screenshots of implementations.
- Use a white background, restrained monochrome line work, explicit labels/arrows, grayscale-safe patterns and a consistent book visual style.
- Mathematical examples must match the chapter. Worked values are illustrative, not experiment results.
- Replace a placeholder only after mathematical, caption, line-weight and page-width checks.
- Figures explain mechanisms; they do not prove semantic correctness, predictive sufficiency, action-conditioned dynamics or planning utility.
