# Manuscript sources and PDF previews

These are author-side sources and working previews, not Manning's final production typesetting.

## Chapter 3
The Chapter 3 manuscript and preview remain `CH03_Dadwal_World_Models_in_Action.adoc` and `CH03_Dadwal_World_Models_in_Action.pdf`. Their source and artwork are unchanged by the Chapter 4 work.

## Chapter 4
The canonical reader-facing source is the 16-section Markdown collection in the repository-root `chapter04/` directory. Its 33 original explanatory SVG figures are stored in `images/chapter04/`. Research PDFs, notes and the source audit remain in `rough-draft/chapter4/`; these are supporting materials, not the reader-facing chapter source.

Generated outputs in this directory:
- `CH04_Dadwal_World_Models_in_Action.adoc`
- `CH04_Dadwal_World_Models_in_Action.pdf`

Do not edit generated AsciiDoc as the primary source. Edit the appropriate Markdown section in `chapter04/`, then run the assembler and checks.

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

The Chapter 4 Actions workflow runs when `chapter04/**` or `images/chapter04/**` changes on `manuscript`. A successful run uploads a preview and commits the generated AsciiDoc/PDF beside the source files. Checks cover section count, equations, image paths, figure captions, embedded serif font and US Letter page size.

## Editorial boundaries
Keep reader-facing prose free of raw URLs; use author–year citations and a consolidated reference list. Distinguish paper-reported results from independently reproduced experiments. Code excerpts must be attributed and checked against the official implementation's license. The chapter's pseudocode is explanatory, not a claim of reproducing large-scale training or published robot results.
