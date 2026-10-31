# Manuscript sources and PDF previews

This directory contains author-side manuscript sources and PDF previews. The PDFs are working previews, not Manning's final production typesetting.

## Chapter 3

The Chapter 3 manuscript and preview are:

- `CH03_Dadwal_World_Models_in_Action.adoc`
- `CH03_Dadwal_World_Models_in_Action.pdf`

Its artwork is stored in the repository's `images/` directory. The Chapter 3 source/build workflow is maintained separately from Chapter 4.

## Chapter 4

The canonical reader-facing source is the section-by-section Markdown in `rough-draft/chapter4/draft/`. The consolidated AsciiDoc and PDF in this directory are generated outputs:

- `CH04_Dadwal_World_Models_in_Action.adoc`
- `CH04_Dadwal_World_Models_in_Action.pdf`

Do not edit the generated AsciiDoc as the primary source. Edit the appropriate Markdown section, then run the assembler and checks.

Chapter 4 currently uses 21 figure descriptions/placeholders. The figure artwork has intentionally not been created or embedded yet. Its detailed figure briefs are part of the section Markdown.

## Build Chapter 4

Use Linux or WSL2 with Python 3, Ruby 3.1+, Pandoc, Poppler, the native dependencies required by `asciidoctor-mathematical`, and Bundler. From the repository root:

```bash
python3 manuscript/assemble-ch04.py
python3 manuscript/verify-ch04-manuscript.py
cd manuscript
gem install bundler
bundle install
bash ./build-ch04-pdf.sh
```

The Chapter 4 GitHub Actions workflow runs the same assembly and verification steps when its source files change on the `manuscript` branch. On a successful build, it uploads the PDF artifact and commits the generated AsciiDoc/PDF to that branch. It checks the first page, math rendering, figure descriptions, embedded serif font, Letter page size and other PDF properties.

## Editorial and code boundaries

Keep reader-facing prose free of raw URLs; use author–year citations and a consolidated reference list. Separate paper-reported results from any experiments independently reproduced for the book. Short code excerpts must be attributed to the official implementation and checked against its license. The Chapter 4 pseudocode is explanatory, not a claim of reproducing large-scale training or published robot results.
