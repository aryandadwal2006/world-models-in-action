#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdf="CH04_Dadwal_World_Models_in_Action.pdf"
test -s "$pdf"
first="$(pdftotext -f 1 -l 1 -layout "$pdf" -)"
for term in "World Models in Action" "Predicting in representation space" "This chapter covers"; do
  grep -Fq "$term" <<<"$first" || { echo "ERROR: first page missing: $term" >&2; exit 1; }
done
all="$(pdftotext -layout "$pdf" -)"
if grep -Eqi 'stem [0-9a-f]{24,}' <<<"$all"; then echo "ERROR: STEM placeholder rendered." >&2; exit 1; fi
for n in $(seq 1 29); do
  grep -Eq "Figure 4\\.$n([.]|[[:space:]]|$|[-–])" <<<"$all" || { echo "ERROR: missing figure caption 4.$n" >&2; exit 1; }
done
if grep -Fqi "Table of Contents" <<<"$all"; then echo "ERROR: unexpected table of contents." >&2; exit 1; fi
if ! grep -Fq "Description:" <<<"$all"; then echo "ERROR: figure descriptions are missing." >&2; exit 1; fi
if pdfimages -list "$pdf" | awk 'NR > 2 && NF { found=1 } END { exit found ? 0 : 1 }'; then
  echo "ERROR: image artwork is embedded in the Chapter 4 PDF." >&2
  exit 1
fi
pdffonts "$pdf" | grep -qi "LiberationSerif" || { echo "ERROR: Liberation Serif not embedded." >&2; exit 1; }
if pdfinfo "$pdf" | grep -Eqi '^Page size:.*A4'; then echo "ERROR: expected US Letter." >&2; exit 1; fi
echo "OK: opener, 29 text-only figure briefs, zero embedded images, embedded serif font and Letter page verified."
pdfinfo "$pdf" | grep -E '^(Pages|Page size|File size):'
