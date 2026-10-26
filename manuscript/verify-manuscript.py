#!/usr/bin/env python3
"""Validate the theory-first Chapter 3 AsciiDoc and its authored figure assets."""
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
    ".The principle behind representation learning", "static visual encoder", "full world model",
    "InfoNCE", "BYOL", "SimSiam", "DINOv3", "Barlow Twins", "VICReg", "DeepCluster",
    "SwAV", "MAE", "BERT", "CLIP", "wav2vec 2.0", "References",
)
for item in required:
    if item not in source:
        errors.append(f"Missing required structure or topic: {item}")

if not source.startswith("= World Models in Action\n"):
    errors.append("Document title must be the first source line.")

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
    if sum(len(s) for s in bullets) > 360 or any(len(s) > 90 for s in bullets):
        errors.append("Chapter opener exceeds the concise bullet limits.")
except ValueError:
    errors.append("Missing 'This chapter covers' opener.")

for forbidden in ("figure to be produced", "artwork pending", "@@figure_", "production slot"):
    if forbidden in source.lower():
        errors.append(f"Production placeholder remains: {forbidden}")

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
        text = line[level:].strip()
        if level >= 5:
            errors.append(f"Heading exceeds level 4 at line {number}: {line}")
        if level in (2, 3, 4):
            headings.append((number, level, text))
            expected = {2: "[#unique_section_ch3_", 3: "[#unique_section2_ch3_", 4: "[#unique_section3_ch3_"}[level]
            previous = lines[number - 2] if number > 1 else ""
            if not previous.startswith(expected) or "caption=" not in previous:
                errors.append(f"Missing Manning-style anchor for heading at line {number}: {text}")
            if line.endswith("."):
                errors.append(f"Heading ends in a period at line {number}: {text}")

anchor_ids = []
for line in lines:
    if line.startswith("[#") and "," in line:
        anchor_ids.append(line[2:line.index(",")])
seen = set()
for aid in anchor_ids:
    if aid in seen:
        errors.append(f"Duplicate anchor ID: {aid}")
    seen.add(aid)

for kind, prefix in (("figure", "ch3-figure-"), ("table", "ch3-table-")):
    anchors = [(i, line) for i, line in enumerate(lines) if line.startswith(f"[#{prefix}")]
    for number, (idx, _line) in enumerate(anchors, start=1):
        if f"{kind} 3.{number}" not in "\n".join(lines[:idx]):
            errors.append(f"{kind} 3.{number} is not mentioned before its anchor.")

figure_count = sum(line.startswith("[#ch3-figure-") for line in lines)
table_count = sum(line.startswith("[#ch3-table-") for line in lines)
if figure_count != 4:
    errors.append(f"Expected 4 theory figures, found {figure_count}.")
if table_count != 1:
    errors.append(f"Expected one objective comparison table, found {table_count}.")

for match in re.finditer(r"^image::([^\[]+)\[", source, flags=re.MULTILINE):
    target = (image_dir / match.group(1)).resolve()
    if not target.is_file():
        errors.append(f"Missing referenced image asset: {match.group(1)}")

if "The runners select CUDA automatically" in source or "five trained seeds" in source:
    errors.append("Old experiments-first manuscript text remains.")

if errors:
    print("Chapter 3 validation FAILED:")
    for error in errors:
        print(f" - {error}")
    sys.exit(1)

print(f"Chapter 3 validation passed: {len(headings)} section headings, {figure_count} figures, {table_count} table, all local images present.")
