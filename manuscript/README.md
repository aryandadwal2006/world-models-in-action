# Chapter 3 manuscript and PDF build

This folder contains the composed AsciiDoc source, the author-side PDF preview, and the build/verification workflow on `manuscript-theory`. The chapter distinguishes the training target and loss from evidence about downstream representation quality. Paper-reported results are separated from results independently reproduced for this book; the short code excerpt is attributed to the official PyTorch implementation.

## Build and validate

Use Linux or WSL2 with Ruby 3.1+, the native dependencies required by `asciidoctor-mathematical`, and Poppler. From the repository root, the GitHub Actions workflow performs the same checks:

```bash
python3 manuscript/verify-manuscript.py
cd manuscript
gem install bundler
./build-pdf.sh
```

The build produces `CH03_Dadwal_World_Models_in_Action.pdf`. This is an author-side preview, not Manning's final production typesetting. The workflow checks the PDF text, first-page geometry, embedded serif font, US Letter size, and the presence of all 17 figure captions.

## Figure artwork

All 17 current figures are inserted in printed order. The artwork files are `images/CH03_F01_Dadwal.png` through `images/CH03_F17_Dadwal.png`. The Markdown production briefs use retained legacy identifiers; `images/README.md` maps each brief to its printed figure and actual PNG file. Figure number and caption remain manuscript metadata and are not baked into the artwork.

## Code and official implementations

The chapter contains one short source-attributed excerpt from the official PyTorch VICReg implementation. It is not a complete trainer, an alternate reimplementation, or evidence of independent reproduction. The manuscript intentionally contains no outbound hyperlinks; methods are attributed in prose and identified in the consolidated bibliography.
