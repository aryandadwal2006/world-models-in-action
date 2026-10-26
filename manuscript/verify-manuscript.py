#!/usr/bin/env python3
"""Validate Chapter 3 structure and the prompt-only figure register."""
from pathlib import Path
import re
import sys

here = Path(__file__).resolve().parent
source_path = here / "CH03_Dadwal_World_Models_in_Action.adoc"
image_dir = here.parent / "images"
source = source_path.read_text(encoding="utf-8")
lines = source.splitlines()
errors = []

required = (
    "= World Models in Action\n", ":chapter: 3", ":sectnums:", ":sectnumlevels: 3",
    ":imagesdir: ../images/", ":stem: latexmath", "[.book-title]", "This chapter covers",
    "predictive sufficiency", "InfoNCE", "SimCLR", "MoCo", "CPC", "BYOL", "SimSiam",
    "DINOv3", "Barlow Twins", "VICReg", "SIGReg", "LeJEPA", "VISReg",
    "DeepCluster", "SwAV", "MAE", "BEiT", "iBOT", "data2vec", "BERT", "CLIP",
    "SigLIP", "wav2vec 2.0", "energy-based", "diffusion", "References",
    "\\lambda_{\\mathrm{scale}}", "\\lambda_{\\mathrm{shape}}", "\\lambda_{\\mathrm{center}}",
    "same-image global-view center", "per-view minibatch mean",
)
for item in required:
    if item not in source:
        errors.append(f"Missing required topic or structure: {item}")
if not source.startswith("= World Models in Action\n"):
    errors.append("Document title must be the first line.")
try:
    opener = lines.index("This chapter covers")
    bullets = []
    for line in lines[opener + 1:]:
        if line.startswith("* "):
            bullets.append(line[2:].strip())
        elif line.strip():
            break
    if not 3 <= len(bullets) <= 5:
        errors.append(f"Expected 3–5 opener bullets; found {len(bullets)}.")
    if sum(len(x) for x in bullets) > 360 or any(len(x) > 90 for x in bullets):
        errors.append("Chapter opener exceeds concise bullet limits.")
except ValueError:
    errors.append("Missing chapter opener.")
for forbidden in ("figure to be produced", "artwork pending", "@@figure_", "production slot", "== Official implementation references"):
    if forbidden.casefold() in source.casefold():
        errors.append(f"Obsolete marker/section remains: {forbidden}")

delimiters = {"++++", "----", "****", "____", "===="}
inside = None
headings = []
for number, line in enumerate(lines, start=1):
    if inside:
        if line.strip() == inside:
            inside = None
        continue
    if line.strip() in delimiters:
        inside = line.strip()
        continue
    level = len(line) - len(line.lstrip("="))
    if level and len(line) > level and line[level].isspace():
        title = line[level:].strip()
        if level >= 5:
            errors.append(f"Heading exceeds level 4 at line {number}.")
        if level in (2, 3, 4):
            headings.append((number, level, title))
            expected = {2: "[#unique_section_ch3_", 3: "[#unique_section2_ch3_", 4: "[#unique_section3_ch3_"}[level]
            j = number - 2
            while j >= 0 and not lines[j].strip():
                j -= 1
            previous = lines[j] if j >= 0 else ""
            if title != "References" and (not previous.startswith(expected) or "caption=" not in previous):
                errors.append(f"Missing Manning-style anchor at line {number}: {title}")
            if line.endswith("."):
                errors.append(f"Heading ends in a period at line {number}.")

anchor_ids = [line[2:line.index(",")] for line in lines if line.startswith("[#") and "," in line]
if len(anchor_ids) != len(set(anchor_ids)):
    errors.append("Duplicate anchor IDs detected.")
figures = [line for line in lines if line.startswith("[#ch3-figure-")]
tables = [line for line in lines if line.startswith("[#ch3-table-")]
listings = [line for line in lines if line.startswith("[#ch3-listing-")]
if len(figures) != 29:
    errors.append(f"Expected 29 figures; found {len(figures)}.")
if len(tables) != 1:
    errors.append(f"Expected one synthesis table; found {len(tables)}.")
if sum(line == "[.figure-placeholder%unbreakable]" for line in lines) != 29:
    errors.append("All 29 figure placeholder boxes must be kept together across pages.")
if 'options="header,unbreakable"' not in source:
    errors.append("The synthesis table must stay together when it fits on one page.")
if len(listings) != 2 or sum(line.startswith("[source,") for line in lines) != 2:
    errors.append(f"Expected two short source listings; found {len(listings)} anchors.")
if re.search(r"^image::", source, flags=re.MULTILINE):
    errors.append("Image directives remain in prompt-only manuscript.")
if re.search(r"\[[^\]]+\]\(https?://", source):
    errors.append("Markdown-style links remain; use native AsciiDoc link macros.")
if r"\(" in source or r"\)" in source:
    errors.append("Markdown-style inline math delimiters remain; use latexmath macros.")
if len(re.findall(r"^== References$", source, flags=re.MULTILINE)) != 1:
    errors.append("Expected one consolidated References heading.")
briefs = sorted(image_dir.glob("CH03_F[0-9][0-9]_Dadwal.md"))
if len(briefs) != 29:
    errors.append(f"Expected 29 prompts; found {len(briefs)}.")
for number in range(1, 30):
    if not (image_dir / f"CH03_F{number:02d}_Dadwal.md").is_file():
        errors.append(f"Missing brief for Figure 3.{number}.")
for path in image_dir.iterdir():
    if path.is_file() and path.suffix.lower() != ".md":
        errors.append(f"Rendered artwork found in prompt-only directory: {path.name}")
if errors:
    print("Chapter 3 validation FAILED:")
    for error in errors:
        print(" - " + error)
    sys.exit(1)
print(f"Chapter 3 validation passed: {len(headings)} headings, {len(figures)} figure placeholders/prompts, 1 table, 2 source listings, no rendered artwork.")
