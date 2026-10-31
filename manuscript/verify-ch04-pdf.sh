#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdf="CH04_Dadwal_World_Models_in_Action.pdf"
test -s "$pdf"
first_page="$(pdftotext -f 1 -l 1 -layout "$pdf" -)"
for required in "World Models in Action" "Predicting in representation space" "This chapter covers"; do
  grep -Fq "$required" <<<"$first_page" || { echo "ERROR: first page missing: $required" >&2; exit 1; }
done
all_text="$(pdftotext -layout "$pdf" -)"
for n in $(seq 1 21); do
  grep -Eq "Figure 4\\."$n"([[:space:]]|$|[-–])" <<<"$all_text" || {
    echo "ERROR: missing figure description 4."$n >&2; exit 1;
  }
done
if grep -Fqi "Table of Contents" <<<"$all_text"; then
  echo "ERROR: unexpected automatic table of contents." >&2; exit 1
fi
pdffonts "$pdf" | grep -qi "LiberationSerif" || { echo "ERROR: Liberation Serif not embedded." >&2; exit 1; }
if pdfinfo "$pdf" | grep -Eqi '^Page size:.*A4'; then
  echo "ERROR: expected US Letter size." >&2; exit 1
fi
echo "OK: opener, 21 placeholders, embedded serif font and Letter page verified."
pdfinfo "$pdf" | grep -E '^(Pages|Page size|File size):'
