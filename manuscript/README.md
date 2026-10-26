# Chapter 3 manuscript and PDF build

This folder contains the composed Chapter 3 AsciiDoc manuscript and the author-side PDF workflow. The chapter is theory-led: it derives the central objectives, explains their assumptions and limitations, and points to official implementations when source-level detail is needed. It does not include custom training runs or homegrown reproductions of the research methods.

## Build a PDF

From this folder, use Linux or WSL2 with Ruby 3.1+, the native libraries required by asciidoctor-mathematical, and Poppler installed:

    gem install bundler
    ./build-pdf.sh

The build produces CH03_Dadwal_World_Models_in_Action.pdf. It is an author-side preview, not Manning's final production typesetting. The GitHub Actions workflow builds and uploads the preview from manuscript-theory.

## Figures

The manuscript uses six original SVG teaching diagrams, all stored in the repository's images folder:

- CH03_T01_SSL_Landscape.svg — objective-family map.
- CH03_T02_InfoNCE.svg — candidate classification and gradient.
- CH03_T03_MethodMechanisms.svg — contrastive, bootstrap, teacher and statistical objectives.
- CH03_T04_TargetSpaces.svg — alternative masked-prediction targets.
- CH03_F12_Dadwal.svg — MAE's visible-patch encoder and masked-patch decoder.
- CH03_F02_Dadwal.svg — representation contract and predictive aliasing.

The verifier checks Manning-style anchors, figure/table counts, referenced local assets and heading structure. The diagrams are original explanatory schematics, not pasted paper images.

## Code and canonical implementations

The chapter contains no custom code listings. It links to official repositories for SimCLR, MoCo, BYOL, SimSiam, DINO/DINOv2/DINOv3, Barlow Twins, VICReg, MAE, DeepCluster, SwAV and CLIP. Those repositories are source references for the original methods, not evidence that this book independently implemented or reproduced them. Readers can follow the derivations without setting up a training environment.
