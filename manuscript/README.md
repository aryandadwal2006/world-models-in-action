# Chapter 3 manuscript and PDF build

`CH03_Dadwal_World_Models_in_Action.adoc` is the composed Chapter 3 source. Its chapter/section anchors, caption macros, image stems, opener, and numbering follow the supplied Manning AsciiDoc template. The `chapter3/` folder remains the code companion source.

The standalone PDF is an **author-side preview**, not a claim to reproduce Manning's proprietary production typesetting. `manning-preview-theme.yml` adapts the visible typography and page proportions of the supplied Chapter 1/2 PDFs: US Letter, a Times-compatible serif face, red section headings, and no standalone title cover, automatic table of contents, or running footer. The template file is a structural AsciiDoc scaffold, not an Asciidoctor PDF theme, so the preview style is implemented separately.

## Build a PDF

The repeatable build writes `CH03_Dadwal_World_Models_in_Action.pdf` into this same `manuscript/` directory. The AsciiDoc headings remain unnumbered in source; the build inserts the chapter-relative numbering once, avoiding duplicate labels.

### Linux (including WSL2) or macOS

Install Ruby 3.1+ and the system libraries needed by the `asciidoctor-mathematical` extension. On Debian/Ubuntu, the extension commonly needs `build-essential`, `cmake`, `pkg-config`, `bison`, `flex`, `fonts-lyx`, `libxml2-dev`, `libxslt1-dev`, `libgmp-dev`, `libffi-dev`, `libcairo2-dev`, `libpango1.0-dev`, `libgdk-pixbuf-2.0-dev`, and `libglib2.0-dev`. Then run from this directory:

```bash
gem install bundler
./build-pdf.sh
```

### Windows

The STEM extension uses native libraries and does not currently build natively on a standard Windows Ruby installation. Use WSL2 with the Linux steps above, or let the repository's `Chapter 3 manuscript PDF` GitHub Actions workflow build the PDF. The workflow uploads the generated PDF as an artifact; no local Ruby setup is required to retrieve it.

## Figure conventions and status

The `.adoc` uses `:imagesdir: ../images/` and figure stems `CH03_F##_Dadwal`, following the supplied template. Numbering is limited to the first two numbered heading levels; deeper subheads remain unnumbered, as required by the Manning manual. `sectnumoffset-treeprocessor.rb` restores chapter-relative section numbers when this single chapter is compiled outside the complete book. `images/` contains the two editable vector diagrams, the committed empirical PNGs available for direct use, source panels for planned composites, the original 2024 taxonomy image as an archive, and same-stem Markdown briefs for figures still requiring final artwork or replotting. In the `.adoc`, pending visuals are marked by production notes that link to the corresponding `.md` brief; no generic SVG placeholder cards are treated as finished illustrations.

The supplied taxonomy image was from 2024. The current Figure 3.1 is an updated objective-centred vector synthesis informed by Gui et al. (2024), Uelwer et al. (2025), and V-JEPA 2 (2025). It replaces the planned family-comparison Table 3.1; the notation table is now Table 3.1 and subsequent tables are renumbered consistently. The 2024 image remains in `images/` for reference only.

## Device note in the chapter

The manuscript keeps device selection to one sentence: the runners select CUDA when PyTorch detects a compatible GPU and otherwise use CPU. Full setup and rendering instructions belong in `../ch03/README.md`. MuJoCo/OpenGL rendering is separate from PyTorch training, and cached observations avoid repeating it.

## Core paper references

`../agent_memory_mustread/core_papers/` contains unmodified PDFs for SimCLR, Barlow Twins, and V-JEPA 2, with attribution and licensing details in its `README.md`. The same index links to CPC, SimSiam, BYOL, VICReg, MAE, and I-JEPA without mirroring files whose redistribution terms have not been verified.

## Validation

Run `python verify-manuscript.py` from this directory after a full repository checkout. The `--allow-unfetched-assets` option is only for deliberately partial copies that lack empirical PNG assets. On a push to `manuscript`, GitHub Actions verifies the figure paths, builds the PDF beside the AsciiDoc source, checks that the first page contains the book title, chapter title, and opener without a separate cover/TOC, uploads a workflow artifact, and commits the refreshed PDF into this folder. The `[pdf-generated]` commit is excluded from rebuilds to prevent a workflow loop. This remains a manuscript working draft: finish the Markdown figure briefs before production handoff.