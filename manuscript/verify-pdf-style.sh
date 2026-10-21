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

if pdfinfo "$pdf" | grep -Eqi '^Page size:.*A4'; then
  echo "ERROR: preview PDF is A4; the chapter references are US Letter." >&2
  exit 1
fi

echo "OK: first-page opener, no auto TOC, serif font, and Letter page size verified."
