#!/usr/bin/env python3
"""Validate Chapter 3's Manning-oriented source structure and figure assets."""

from pathlib import Path
import argparse
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

# The Manning opener is a short overview list, not a paragraph wall.
try:
    opener = lines.index("This chapter covers")
    bullets = []
    for line in lines[opener + 1 :]:
        if line.startswith("* "):
            bullets.append(line[2:].strip())
        elif line.strip():
            break
    if not 3 <= len(bullets) <= 5:
        errors.append(f"Chapter opener has {len(bullets)} bullets; expected 3–5.")
    if sum(len(item) for item in bullets) > 360:
        errors.append("Chapter-opener bullets exceed the approximate length guidance.")
    for item in bullets:
        if len(item) > 90:
            errors.append(f"Chapter-opener bullet is too long: {item}")
except ValueError:
    pass

# Keep production notes, cross-chapter navigation and contract/pay metaphors
# out of the reader-facing text.
lower = source.lower()
checks = (
    ("figure to be produced", "Unfinished figure production note remains in manuscript."),
    ("artwork pending", "Production-only artwork status remains in manuscript."),
    ("production slot", "Production-only artwork status remains in manuscript."),
    ("@@figure_", "Unexpanded figure marker remains in manuscript."),
    ("previous chapter", "Cross-chapter navigation wording remains in manuscript."),
    ("next chapter", "Cross-chapter navigation wording remains in manuscript."),
    ("following chapter", "Cross-chapter navigation wording remains in manuscript."),
    ("earlier chapter", "Cross-chapter navigation wording remains in manuscript."),
    ("later chapter", "Cross-chapter navigation wording remains in manuscript."),
    ("royalty", "Contract/pay metaphor remains in manuscript."),
    ("royalties", "Contract/pay metaphor remains in manuscript."),
    ("paid to keep", "Contract/pay metaphor remains in manuscript."),
    ("pays to preserve", "Contract/pay metaphor remains in manuscript."),
)
for needle, message in checks:
    if needle in lower:
        errors.append(message)

# Parse actual AsciiDoc headings, excluding all delimited blocks. This avoids
# mistaking display-math lines beginning with '=' for document headings.
fenced_delimiters = {"++++", "----", "****", "____", "===="}
inside_delimiter = None
section_numbers_disabled = False
heading_records = []

def heading_level(line):
    count = len(line) - len(line.lstrip("="))
    if count and len(line) > count and line[count].isspace():
        return count
    return 0

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

    level = heading_level(line)
    if not level:
        continue

    heading_text = line[level:].strip()
    if level >= 5:
        errors.append(f"Heading at line {line_number} exceeds the allowed section hierarchy: {line}")
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
            f"Heading at line {line_number} lacks the Manning-style {level}-level anchor: {heading_text}"
        )
    elif "caption=" not in previous:
        errors.append(f"Heading at line {line_number} anchor has no caption: {heading_text}")
    elif not section_numbers_disabled and level == 2:
        if "reftext={chapter}.{counter:section}" not in previous or "caption='{chapter}.{section}'" not in previous:
            errors.append(f"Section anchor does not follow the template counter/caption convention at line {line_number}: {heading_text}")
    elif not section_numbers_disabled and level == 3:
        if "reftext={chapter}.{section}.{counter:subsection}" not in previous or "caption='{chapter}.{section}.{subsection}'" not in previous:
            errors.append(f"Subsection anchor does not follow the template counter/caption convention at line {line_number}: {heading_text}")

# Reject duplicate anchors without depending on a fragile nested-bracket regex.
anchor_ids = []
for line in lines:
    if line.startswith("[#") and "," in line:
        anchor_ids.append(line[2 : line.index(",")])
seen_ids = set()
for anchor_id in anchor_ids:
    if anchor_id in seen_ids:
        errors.append(f"Duplicate AsciiDoc anchor ID: {anchor_id}")
    seen_ids.add(anchor_id)

# Enforce the Manning Manual of Style section hierarchy: an individual
# heading level must not have exactly one child heading, and a heading must
# begin with regular explanatory prose rather than another heading or block.
heading_children = {record[0]: 0 for record in heading_records}
heading_stack = []
for record in heading_records:
    while heading_stack and heading_stack[-1][1] >= record[1]:
        heading_stack.pop()
    if heading_stack:
        heading_children[heading_stack[-1][0]] += 1
    heading_stack.append(record)

for line_number, level, heading_text, _disabled in heading_records:
    if level in (2, 3) and heading_children[line_number] == 1:
        errors.append(
            f"Heading at line {line_number} has only one child heading, which violates "
            f"the Manning hierarchy rule: {heading_text}"
        )

for line_number, _level, heading_text, _disabled in heading_records:
    if heading_text == "Summary":
        continue  # The manual explicitly allows the Summary to begin with bullets.
    index = line_number  # zero-based index of the line immediately after the heading
    while index < len(lines) and (
        not lines[index].strip() or lines[index].strip().startswith("[#")
    ):
        index += 1
    if index >= len(lines):
        errors.append(f"Heading at line {line_number} has no following prose: {heading_text}")
        continue
    following = lines[index].strip()
    if heading_level(following):
        errors.append(
            f"Stacked headings at lines {line_number} and {index + 1}: {heading_text}"
        )
    elif following.startswith((
        "[latexmath]", "image::", "[source", "[NOTE]", "[TIP]", "[WARNING]",
        "[IMPORTANT]", "[CAUTION]", "|===", "* ", "- ", "."
    )):
        errors.append(
            f"Heading at line {line_number} is followed by a non-prose block instead of a sentence: "
            f"{heading_text}"
        )

# Section titles use sentence case, with the first word after a colon capitalized.
for line_number, _level, heading_text, _disabled in heading_records:
    if ":" in heading_text:
        after_colon = heading_text.split(":", 1)[1].lstrip()
        if after_colon and after_colon[0].isalpha() and not after_colon[0].isupper():
            errors.append(
                f"Heading at line {line_number} does not capitalize the word after a colon: {heading_text}"
            )

# Figure, table, and listing references must appear in the prose before their
# numbered objects, as required by the publisher's style guide.
for kind in ("figure", "table", "listing"):
    anchors = [
        (idx, line)
        for idx, line in enumerate(lines)
        if line.startswith(f"[#ch3-{kind}-")
    ]
    for object_number, (idx, _line) in enumerate(anchors, start=1):
        reference = f"{kind} 3.{object_number}"
        if reference not in "\n".join(lines[:idx]):
            errors.append(
                f"{kind.capitalize()} 3.{object_number} is not mentioned in the text before its anchor."
            )

# Manning code-listing captions must fit on one line (65 characters including
# the number) and must not end with a period.
listing_anchors = [
    idx for idx, line in enumerate(lines) if line.startswith("[#ch3-listing-")
]
for listing_number, idx in enumerate(listing_anchors, start=1):
    title_index = idx + 1
    while title_index < len(lines) and not lines[title_index].strip():
        title_index += 1
    if title_index >= len(lines) or not lines[title_index].startswith("."):
        errors.append(f"Listing 3.{listing_number} has no caption immediately after its anchor.")
        continue
    caption = lines[title_index][1:].strip()
    if len(f"Listing 3.{listing_number} {caption}") > 65:
        errors.append(
            f"Listing 3.{listing_number} caption exceeds the 65-character Manning limit: {caption}"
        )
    if caption.endswith("."):
        errors.append(f"Listing 3.{listing_number} caption ends in a period: {caption}")

# All running references use lowercase, unlike the display captions.
for line_number, line in enumerate(lines, start=1):
    if any(marker in line for marker in ("Figure 3.", "Table 3.", "Listing 3.")):
        errors.append(
            f"Use lowercase in running references to figures, tables, and listings "
            f"(line {line_number}): {line.strip()}"
        )

figure_anchor_count = sum(line.startswith("[#ch3-figure-") for line in lines)
table_anchor_count = sum(line.startswith("[#ch3-table-") for line in lines)
if figure_anchor_count != 17:
    errors.append(f"Expected 17 numbered figures after consolidation; found {figure_anchor_count}.")
if table_anchor_count != 4:
    errors.append(f"Expected four numbered tables after adding anti-collapse and contract-evidence tables; found {table_anchor_count} table anchors.")
listing_anchor_count = sum(line.startswith("[#ch3-listing-") for line in lines)
if listing_anchor_count != 10:
    errors.append(f"Expected 10 numbered code listings after removing redundant method snippets; found {listing_anchor_count} listing anchors.")

summary_heading = "== Summary"
if summary_heading not in lines:
    errors.append("Missing final Summary section.")
else:
    summary_idx = lines.index(summary_heading)
    if any(heading_level(line) for line in lines[summary_idx + 1 :]):
        errors.append("Summary is not the final section heading.")
    summary_bullets = [line for line in lines[summary_idx + 1 :] if line.startswith("* ")]
    if len(summary_bullets) < 4:
        errors.append(f"Summary has only {len(summary_bullets)} top-level bullets.")
    for bullet in summary_bullets:
        text = bullet[2:]
        sentence_boundaries = sum(
            1
            for index in range(len(text) - 2)
            if text[index] in ".!?"
            and text[index + 1].isspace()
            and text[index + 2].isupper()
        )
        sentence_count = sentence_boundaries + 1
        if sentence_count > 2:
            errors.append(
                f"Summary bullet has {sentence_count} sentences; the Manning guide recommends one or two: {bullet}"
            )
    if not lines[-1].strip().startswith("* "):
        errors.append("Summary must be the last content, with no trailing paragraph or production note.")

retired_figure_stems = {
    "CH03_F06_Dadwal",  # folded into Figure 3.5
    "CH03_F08_Dadwal",  # low-evidence PCA projection removed
    "CH03_F16_Dadwal",  # CPC retrieval plot folded into Figure 3.16
}
for n in range(1, 18):
    stem = f"CH03_F{n:02d}_Dadwal"
    if stem not in source and stem not in retired_figure_stems:
        errors.append(f"Missing figure reference: {stem}")
    assets = [
        path
        for ext in ("png", "svg", "jpg", "jpeg")
        for path in img.glob(f"{stem}*.{ext}")
    ]
    # Retired assets may remain for provenance; only live figure slots must be embedded.
    if stem not in retired_figure_stems and not assets and not (
        args.allow_unfetched_assets and n in (3, 7, 14, 16)
    ):
        errors.append(f"Missing final figure asset: {stem}")

for line in lines:
    if line.startswith("image::") and "[" in line:
        target = line[len("image::") : line.index("[")].strip()
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
    if line.startswith(("TIP —", "TIP -", "WARNING —", "WARNING -", "SIDEBAR —", "SIDEBAR -", "DEFINITION —", "DEFINITION -")):
        errors.append(f"Unformatted callout heading at line {i}: {line[:90]}")

if errors:
    print("\n".join(f"ERROR: {error}" for error in errors))
    sys.exit(1)

print("OK: Manning anchors, nesting and prose flow, numbered object callouts, listing captions, final Summary, 17 figures, four tables, ten listings, and all live figure paths are consistent.")
