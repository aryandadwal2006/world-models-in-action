# Chapter 3 manuscript and PDF build

`CH03_Dadwal_World_Models_in_Action.adoc` is the single composed Chapter 3 source. It was assembled from the section-level Markdown in `../chapter3/`, while preserving the heading, caption, cross-reference, and image naming conventions from the supplied Manning AsciiDoc template. The existing `chapter3/` folder is intentionally unchanged.

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

The `.adoc` uses `:imagesdir: ../images/` and figure stems `CH03_F##_Dadwal`, following the supplied template. `sectnumoffset-treeprocessor.rb` restores chapter-relative section numbers when this single chapter is compiled outside the complete book. `images/` contains the two editable vector diagrams, the committed empirical PNGs available for direct use, source panels for planned composites, the original 2024 taxonomy image as an archive, and same-stem Markdown briefs for figures still requiring final artwork or replotting. In the `.adoc`, pending visuals are marked by production notes that link to the corresponding `.md` brief; no generic SVG placeholder cards are treated as finished illustrations.

The supplied taxonomy image was from 2024. The current Figure 3.1 is an updated objective-centred vector synthesis informed by Gui et al. (2024), Uelwer et al. (2025), and V-JEPA 2 (2025). It replaces the planned family-comparison Table 3.1; the notation table is now Table 3.1 and subsequent tables are renumbered consistently. The 2024 image remains in `images/` for reference only.

## Device note in the chapter

The manuscript explains that `worldmodels.train.get_device()` selects CUDA automatically when PyTorch detects a CUDA GPU, otherwise CPU; it includes a direct verification snippet. This applies to PyTorch training only. MuJoCo/OpenGL rendering is a separate dependency, and cached data avoids repeated rendering.

## Validation

Run `python verify-manuscript.py` from this directory after a full repository checkout. The `--allow-unfetched-assets` option is only for deliberately partial copies that lack empirical PNG assets. On a push to `manuscript`, GitHub Actions verifies the figure paths, builds the PDF beside the AsciiDoc source, uploads a workflow artifact, and commits the refreshed PDF into this folder. The `[pdf-generated]` commit is excluded from rebuilds to prevent a workflow loop. This remains a manuscript working draft: finish the Markdown figure briefs before production handoff.