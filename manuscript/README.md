# Chapter 3 manuscript and PDF build

This folder contains the composed AsciiDoc chapter and author-side PDF workflow. It derives objectives from target construction, loss, gradient/update mechanism, assumptions and limitations. It contains no custom training runs and does not imply official source links equal independent reproduction.

## Build and validate

Use Linux or WSL2 with Ruby 3.1+, the native dependencies required by asciidoctor-mathematical, and Poppler:

```bash
gem install bundler
python3 verify-manuscript.py
./build-pdf.sh
```

The build produces CH03_Dadwal_World_Models_in_Action.pdf. It is an author-side preview, not Manning's final typesetting.

## Figure placeholders

The chapter includes 29 conceptual figure placeholders. Each has a detailed brief in the images directory. The directory intentionally contains no rendered artwork; the Markdown briefs are the production source until the author generates and approves each figure.

## Code and official implementations

The chapter includes two short source excerpts from pinned official SimCLR and VICReg revisions. They are not complete trainers, alternate reimplementations or evidence of reproduction. Implementation links occur beside methods, with scholarly citations in one bibliography. The main-extra branch maintains upstream submodules for full recipes, dependencies and licensing.
