#!/usr/bin/env python3
"""Validate Chapter 3's Manning-oriented source structure and figure assets."""

from pathlib import Path
import argparse
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--allow-unfetched-assets",
    action="store_true",
    help="Allow the four canonical empirical PNGs omitted from a deliberately partial checkout.",
)
args = parser.parse_args()

here = Path(__file__).resolve().parent
adoc = here / "CH03_Dadwal_World_Models_in_Action.adoc"
img = here.parent / "images"
source = adoc.read_text(encoding="utf-8")
lines = source.splitlines()
errors = []

for required in (
    "= World Models in Action",
    ":chapter: 3",
    ":sectnums:",
    ":sectnumlevels: 3",
    ":imagesdir: ../images/",
    ":stem: latexmath",
    "[.book-title]",
    "This chapter covers",
    ".Three components, three different jobs",
    ".The principle behind representation learning",
    "static visual encoder",
    "full world model",
):
    if required not in source:
        errors.append(f"Missing required Manning structure/content: {required}")

if not source.startswith("= World Models in Action\n"):
    errors.append("Document title must be the first source line.")

try:
    opener = lines.index("This chapter covers")
    bullet_lines = []
    for line in lines[opener + 1 :]:
        if line.startswith("* "):
            bullet_lines.append(line[2:].strip())
        elif line.strip() and not line.startswith("* "):
            break
    if not 3 <= len(bullet_lines) <= 5:
        errors.append(f"Chapter opener has {len(bullet_lines)} bullets; expected 3–5.")
    if sum(len(item) for item in bullet_lines) > 360:
        errors.append("Chapter-opener bullets exceed the approximate length guidance.")
    for item in bullet_lines:
        if len(item) > 90:
            errors.append(f"Chapter-opener bullet is too long: {item}")
except ValueError:
    pass

for pattern, message in (
    (r"Figure to be produced", "Unfinished figure production note remains in manuscript."),
    (r"Artwork pending|production slot|production plot", "Production-only artwork status remains in manuscript."),
    (r"@@FIGURE_[A-Z0-9_]+@@", "Unexpanded figure marker remains in manuscript."),
    (r"\b(?:previous|next|following|earlier|later) chapter\b", "Cross-chapter navigation wording remains in manuscript."),
    (r"\bchapter\s+(?:1|2|4)\b", "Explicit cross-reference to another chapter remains in manuscript."),
    (r"\b(?:royalt(?:y|ies)|contract|paid to keep|pay(?:s|ing)? to preserve)\b", "Contract/pay metaphor remains in manuscript."),
):
    if re.search(pattern, source, re.IGNORECASE):
        errors.append(message)

# The Manning scaffold uses anchors and reference captions on section headings.
# Scan only actual headings: code, passthrough math, examples, sidebar content,
# and admonition bodies can contain lines that resemble headings.
fenced_delimiters = {"++++", "----", "****", "____", "===="}
inside_delimiter = None
section_numbers_disabled = False
heading_records = []
for line_number, line in enumerate(lines, start=1):
    if line.strip() == ":sectnums!:":
        section_numbers_disabled = True

    if inside_delimiter is not None:
        if line.strip() == inside_delimiter:
            inside_delimiter = None
        continue

    if line.strip() in fenced_delimiters:
        inside_delimiter = line.strip()
        continue

    match = re.match(r"^(=+)\\s+(.*)$", line)
    if not match:
        continue

    level = len(match.group(1))
    heading_text = match.group(2)
    if level >= 5:
        errors.append(
            f"Heading at line {line_number} exceeds the allowed section hierarchy: {line}"
        )
    if len(line) > 120:
        errors.append(f"Overlong heading at line {line_number}: {line[:120]}...")
    if line.endswith("."):
        errors.append(f"Heading at line {line_number} ends in a period: {line}")

    if level not in (2, 3, 4):
        continue

    heading_records.append((line_number, level, heading_text, section_numbers_disabled))
    previous = lines[line_number - 2] if line_number > 1 else ""
    expected_prefix = {
        2: "[#unique_section_ch3_",
        3: "[#unique_section2_ch3_",
        4: "[#unique_section3_ch3_",
    }[level]
    if not previous.startswith(expected_prefix):
        errors.append(
            f"Heading at line {line_number} lacks the Manning-style {level}-level anchor: "
            f"{heading_text}"
        )
    elif "caption=" not in previous:
        errors.append(f"Heading at line {line_number} anchor has no caption: {heading_text}")
    elif not section_numbers_disabled and level == 2:
        if "reftext={chapter}.{counter:section}" not in previous or \
           "caption='{chapter}.{section}'" not in previous:
            errors.append(
                f"Section anchor does not follow the template counter/caption convention "
                f"at line {line_number}: {heading_text}"
            )
    elif not section_numbers_disabled and level == 3:
        if "reftext={chapter}.{section}.{counter:subsection}" not in previous or \
           "caption='{chapter}.{section}.{subsection}'" not in previous:
            errors.append(
                f"Subsection anchor does not follow the template counter/caption convention "
                f"at line {line_number}: {heading_text}"
            )

# IDs are link targets; duplicates silently redirect readers to the wrong place.
anchor_ids = re.findall(r"^\\[#([^,\\]]+)", source, re.MULTILINE)
seen_ids = set()
for anchor_id in anchor_ids:
    if anchor_id in seen_ids:
        errors.append(f"Duplicate AsciiDoc anchor ID: {anchor_id}")
    seen_ids.add(anchor_id)

summary_heading = "== Summary"
if summary_heading not in lines:
    errors.append("Missing final Summary section.")
else:
    summary_idx = lines.index(summary_heading)
    if any(re.match(r"^=+\s", line) for line in lines[summary_idx + 1 :]):
        errors.append("Summary is not the final section heading.")
    summary_bullets = [line for line in lines[summary_idx + 1 :] if line.startswith("* ")]
    if len(summary_bullets) < 4:
        errors.append(f"Summary has only {len(summary_bullets)} top-level bullets.")
    if not lines[-1].strip().startswith("* "):
        errors.append("Summary must be the last content, with no trailing paragraph or production note.")

for n in range(1, 18):
    stem = f"CH03_F{n:02d}_Dadwal"
    if not re.search(rf"CH03_F{n:02d}_Dadwal", source):
        errors.append(f"Missing figure reference: {stem}")
    assets = [
        path
        for ext in ("png", "svg", "jpg", "jpeg")
        for path in img.glob(f"{stem}*.{ext}")
    ]
    if not assets and not (args.allow_unfetched_assets and n in (3, 7, 14, 16)):
        errors.append(f"Missing final figure asset: {stem}")

for match in re.finditer(r"^image::([^\[]+)\[", source, re.MULTILINE):
    target = match.group(1).strip()
    if not (img / target).exists():
        if args.allow_unfetched_assets and target in {
            "CH03_F03_Dadwal.png",
            "CH03_F07_Dadwal.png",
            "CH03_F14_Dadwal.png",
            "CH03_F16_Dadwal.png",
        }:
            continue
        errors.append("Unresolved image macro target: " + target)

for i, line in enumerate(lines, start=1):
    if re.match(r"^(?:TIP|WARNING|SIDEBAR|DEFINITION)\s+[—–-]", line):
        errors.append(f"Unformatted callout heading at line {i}: {line[:90]}")

if errors:
    print("\n".join(f"ERROR: {error}" for error in errors))
    sys.exit(1)

print("OK: Manning-oriented opener, heading hierarchy, final Summary, callouts, and all figure paths are consistent.")
