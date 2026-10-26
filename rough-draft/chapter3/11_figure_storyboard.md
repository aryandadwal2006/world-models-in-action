# Chapter 3 figure storyboard

## Production status

The composed manuscript on manuscript-theory contains six original SVG diagrams and no placeholders. Each diagram explains a mechanism used in the chapter. No figure is a pasted paper screenshot or an empirical result plot.

Use a consistent Manning-style visual system: grayscale-safe contrasts, arrows, labels, solid/dashed lines and distinct shapes. The official papers verify the mechanisms; the book diagrams explain them in original artwork.

## Figure list in manuscript order

| No. | Source asset | Subject | What it explains |
|---:|---|---|---|
| 1 | images/CH03_T01_SSL_Landscape.svg | Objective-family map | Candidate discrimination, bootstrap targets, batch statistics, prototypes and masked prediction; families can overlap |
| 2 | images/CH03_T02_InfoNCE.svg | InfoNCE candidate classification | Anchor, positive and negative set, temperature, softmax probabilities and gradient direction |
| 3 | images/CH03_T03_MethodMechanisms.svg | Core objective comparison | Contrastive candidate loss, BYOL/SimSiam prediction pathways, DINO teacher distribution, Barlow Twins/VICReg batch constraints |
| 4 | images/CH03_T04_TargetSpaces.svg | Masked target families | Pixel regression, discrete-code prediction and teacher-feature targets from the same masked input |
| 5 | images/CH03_F12_Dadwal.svg | MAE architecture | Visible patches enter the encoder; mask tokens and encoded visible patches enter the decoder; loss applies to masked patches |
| 6 | images/CH03_F02_Dadwal.svg | Representation contract | Nuisance invariance versus distinctions that must remain separable for different target futures |

## Placement

- Figure 1: foundations.
- Figure 2: contrastive objective and gradient derivation.
- Figure 3: after Barlow Twins/VICReg, as a synthesis of the core mechanisms.
- Figure 4: general masked prediction and target-space distinction.
- Figure 5: after the MAE loss and architecture explanation.
- Figure 6: compact state representations and predictive sufficiency.

## Copyright and claim discipline

Use papers to verify the computation graph and notation, but do not paste paper screenshots. The SVGs are original conceptual teaching diagrams. A repository link establishes the source of the official code; it does not establish that the book ran or reproduced the method. Keep that distinction clear in captions and prose.
