# Chapter 3 final-draft and publication gate

Status at 2026-10-26: a theory-first source draft is composed on manuscript-theory. It uses six original diagrams, mathematical explanations, theory exercises and official-code links rather than local experiments. It still requires a successful PDF build and final equation, citation and pagination review before it should be called print-ready.

## Scope now fixed

The chapter teaches the main objective families rather than surveying every self-supervised-learning paper. Core derivations cover InfoNCE/NT-Xent, BYOL, SimSiam, DINO, Barlow Twins, VICReg and MAE. DeepCluster, SwAV, CLIP/SigLIP, language objectives and speech SSL are brief target-construction comparisons. JEPA receives a short bridge to Chapter 4.

The chapter contains no bespoke training experiments and no local reimplementations. The official-source table points to original repositories so readers can inspect implementation details without implying reproduction.

## Final publication checks

| Gate | Current position | Required action |
|---|---|---|
| Mathematical correctness | The main objectives and update mechanisms are written in the manuscript; equations have been revised | Compare each core equation to its primary paper, especially InfoNCE sampling assumptions and VICReg reductions |
| Scope and prose | Robotics detours, custom experiments, figure placeholders and duplicate method summaries have been removed | Read once for directness and ensure no old prose survived |
| Figures | Six original SVGs are referenced; no placeholders remain | Confirm SVG rendering and cross-reference/caption numbering in PDF |
| Official sources | Core methods link to their authors' repositories | Verify links and paper metadata; a source link is not evidence of reproduction |
| AsciiDoc structure | Manning-style anchors and equation tags are present | Run manuscript/verify-manuscript.py via the PDF workflow |
| PDF production | The existing author-side build workflow is configured | Confirm the latest manuscript-theory workflow succeeds; inspect equation layout, figures, page geometry and total page count |
| Length | The editorial goal remains 45–50 finished book pages | Measure the generated PDF; expand theory only if needed, never add experimental filler to reach a page count |

## Release boundary

The theory-first source draft is ready for focused final review, but it is not declared print-ready until the PDF build, local-asset checks, equation rendering, bibliography and page count are verified. The broader research dossier remains a separate resource and is not required to be exhaustively audited for the chapter draft.
