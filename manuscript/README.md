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

The chapter contains one short source-attributed excerpt from the official PyTorch VICReg implementation. It is not a complete trainer, an alternate reimplementation or evidence of reproduction. The manuscript intentionally contains no outbound hyperlinks; methods are attributed in prose and identified in the consolidated bibliography.
