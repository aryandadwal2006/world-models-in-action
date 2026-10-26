# Chapter 3 manuscript and PDF build

This folder contains the composed Chapter 3 AsciiDoc and its author-side PDF preview. The theory-first revision organizes self-supervised learning by training signals and architecture choices rather than by a long sequence of empirical experiments. Detailed paper-by-paper derivations and supporting notes are kept in rough-draft/chapter3/.

## Build a PDF

From this folder, use Linux or WSL2 with Ruby 3.1+, the native libraries required by asciidoctor-mathematical, and Poppler installed:

    gem install bundler
    ./build-pdf.sh

The build produces CH03_Dadwal_World_Models_in_Action.pdf. It is an author-side preview, not Manning's final production typesetting. The GitHub Actions workflow builds and uploads the preview from manuscript-theory.

## Figures

The theory figures are original SVG teaching schematics:
- CH03_T01_SSL_Landscape.svg — family map.
- CH03_T02_InfoNCE.svg — candidate classification and gradients.
- CH03_T03_MethodMechanisms.svg — contrastive, bootstrap, teacher and statistical mechanisms.
- CH03_T04_TargetSpaces.svg — alternate targets for masked inputs.

The verifier checks their presence, Manning-style anchors, object references and expected figure/table counts.

## Scope and implementation

The chapter uses only two lightweight demonstrations from the existing code companion: an InfoNCE calculation and a collapse/objective diagnostic. Official paper implementations are linked and pinned as submodules on main-extra; the chapter itself does not vendor third-party source code.
