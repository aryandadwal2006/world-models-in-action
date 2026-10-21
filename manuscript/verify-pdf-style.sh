#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdf="CH03_Dadwal_World_Models_in_Action.pdf"
test -s "$pdf"

first_page="$(pdftotext -f 1 -l 1 -layout "$pdf" -)"
printf '%s\n' "$first_page"

for required in "World Models in Action" "Representation learning before generation" "This chapter covers"; do
  if ! grep -Fq "$required" <<<"$first_page"; then
    echo "ERROR: first PDF page is missing: $required" >&2
    exit 1
  fi
done

all_text="$(pdftotext -layout "$pdf" -)"
if grep -Fqi "Table of Contents" <<<"$all_text"; then
  echo "ERROR: standalone chapter PDF unexpectedly contains an auto-generated table of contents." >&2
  exit 1
fi

if ! pdffonts "$pdf" | grep -qi "LiberationSerif"; then
  echo "ERROR: the configured serif font was not embedded in the PDF." >&2
  exit 1
fi

# The supplied Chapter 2 PDF is the closest visual reference for this opener.
# Check title and chapter-heading coordinates against its visible top-page
# proportions; this catches a document that passes text checks but floats the
# title too high or leaves an excessive gap before the chapter heading.
bbox_file="$(mktemp)"
trap 'rm -f "$bbox_file"' EXIT
pdftotext -f 1 -l 1 -bbox "$pdf" "$bbox_file"
python3 - "$bbox_file" <<'PY'
from html.parser import HTMLParser
import sys

class Words(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.active = None

    def handle_starttag(self, tag, attrs):
        if tag == "word":
            values = dict(attrs)
            # pdftotext can emit empty or non-positioned word elements for
            # control characters; ignore those rather than failing the check.
            if "xmin" in values and "ymin" in values:
                self.active = {
                    "text": "",
                    "x": float(values["xmin"]),
                    "y": float(values["ymin"]),
                }
            else:
                self.active = None

    def handle_data(self, data):
        if self.active is not None:
            self.active["text"] += data

    def handle_endtag(self, tag):
        if tag == "word" and self.active is not None:
            self.items.append(self.active)
            self.active = None

parser = Words()
parser.feed(open(sys.argv[1], encoding="utf-8").read())
items = parser.items
tokens = [item["text"].strip() for item in items]

def sequence_index(sequence, start=0):
    for idx in range(start, len(tokens) - len(sequence) + 1):
        if tokens[idx:idx + len(sequence)] == sequence:
            return idx
    return None

title_idx = sequence_index(["World", "Models", "in", "Action"])
heading_idx = sequence_index(
    ["Representation", "learning", "before", "generation"],
    (title_idx or 0) + 4,
)
if title_idx is None or heading_idx is None:
    raise SystemExit("ERROR: title/heading sequence is not extractable from the first page.")

title_y = min(word["y"] for word in items[title_idx:title_idx + 4])
# The extracted sequence starts at the words after the visible section number.
# Include that number when comparing left edges and measuring the heading top.
heading_start_idx = heading_idx
if heading_idx > 0 and tokens[heading_idx - 1].endswith("."):
    heading_start_idx = heading_idx - 1
heading_y = min(word["y"] for word in items[heading_start_idx:heading_idx + 4])
title_x = items[title_idx]["x"]
heading_x = items[heading_start_idx]["x"]
gap = heading_y - title_y

print(
    f"First-page geometry: book-title y={title_y:.1f} pt, "
    f"chapter-title y={heading_y:.1f} pt, gap={gap:.1f} pt, "
    f"left edges={title_x:.1f}/{heading_x:.1f} pt"
)
if not 68 <= title_y <= 84:
    raise SystemExit("ERROR: book title sits outside the publisher-sample top-page range.")
if not 110 <= heading_y <= 130:
    raise SystemExit("ERROR: chapter title sits outside the calibrated publisher-sample vertical range.")
if not 38 <= gap <= 56:
    raise SystemExit("ERROR: title-to-chapter-heading spacing differs from the calibrated publisher-sample range.")
if abs(title_x - heading_x) > 2:
    raise SystemExit("ERROR: book title and chapter heading are not left-aligned.")

PY

if pdfinfo "$pdf" | grep -Eqi '^Page size:.*A4'; then
  echo "ERROR: preview PDF is A4; the chapter references are US Letter." >&2
  exit 1
fi

echo "OK: first-page opener, no auto TOC, serif font, and Letter page size verified."
