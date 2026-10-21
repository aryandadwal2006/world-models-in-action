# Chapter 3 manuscript and PDF build

`CH03_Dadwal_World_Models_in_Action.adoc` is the composed Chapter 3 source. Its chapter/section anchors, caption macros, image stems, opener, and numbering follow the supplied Manning AsciiDoc template. The `chapter3/` folder remains the code companion source.

The standalone PDF is an **author-side preview**, not Manning's proprietary production typesetting. The source has been audited against the supplied `agent_memory_mustread/CH03_Dadwal_World_Models_in_Action.adoc` scaffold: document/chapter title anchors, unique section anchors, caption and counter conventions, figure/listing/table anchors, heading hierarchy, and chapter-relative numbering. The preview theme is separate because the supplied `.adoc` is a structural scaffold, not an Asciidoctor PDF theme. The font, page proportions, heading colour, and first-page spacing are tuned against the supplied Chapter 2 PDF; the new style verifier checks measurable first-page coordinates as well as text, font embedding, page size, and absence of an automatic table of contents.

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

The `.adoc` uses `:imagesdir: ../images/` and stable figure asset stems, following the supplied template. Numbering follows the template hierarchy: chapter title, main sections, and subsections are numbered; the next level of subhead remains unnumbered. `sectnumoffset-treeprocessor.rb` restores chapter-relative section numbers when this single chapter is compiled outside the complete book. The composed chapter now has 20 numbered figure anchors and two actual table anchors. Three content plots previously mislabelled as tables have been converted to figures, and the main-text captions number them in document order. The image README maps stable filenames to the printed figure numbers. All visible figure prompts have been replaced with artwork or committed experiment plots.

The supplied taxonomy image was from 2024. The current Figure 3.1 is an updated objective-centred vector synthesis informed by Gui et al. (2024), Uelwer et al. (2025), and V-JEPA 2 (2025). It replaces the planned family-comparison Table 3.1; the notation table is now Table 3.1 and subsequent tables are renumbered consistently. The 2024 image remains in `images/` for reference only.

## Device note in the chapter

The manuscript keeps device selection to one sentence: the runners select CUDA when PyTorch detects a compatible GPU and otherwise use CPU. Full setup and rendering instructions belong in `../ch03/README.md`. MuJoCo/OpenGL rendering is separate from PyTorch training, and cached observations avoid repeating it.

## Core paper references

`../agent_memory_mustread/core_papers/` contains unmodified PDFs for SimCLR, Barlow Twins, and V-JEPA 2, with attribution and licensing details in its `README.md`. The same index links to CPC, SimSiam, BYOL, VICReg, MAE, and I-JEPA without mirroring files whose redistribution terms have not been verified.

## Validation

Run `python verify-manuscript.py` from this directory after a full repository checkout. The verifier checks publisher-style heading anchors, opener length, duplicate IDs, the final Summary position, all 20 figures, both true tables, and all image paths. The `--allow-unfetched-assets` option is only for deliberately partial copies that lack empirical PNG assets. On a push to `manuscript`, GitHub Actions verifies all 75 level-2-to-level-4 section headings have unique Manning-style anchors, checks figure paths and figure slots, builds the PDF beside the AsciiDoc source, tests first-page title and chapter-heading geometry against the supplied sample, checks the embedded serif font, page size and absence of a generated TOC, uploads a workflow artifact, and commits the refreshed PDF into this folder. The `[pdf-generated]` commit is excluded from rebuilds to prevent a workflow loop. This remains an author-side manuscript preview, not Manning's final typeset proof. The latest PDF workflow must pass its style checks and the generated figure-asset commit must be inspected before calling this revision production-ready.