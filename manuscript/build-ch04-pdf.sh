#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v ruby >/dev/null || { echo "Ruby 3.1+ is required" >&2; exit 1; }
command -v bundle >/dev/null || { echo "Install Bundler with: gem install bundler" >&2; exit 1; }
command -v pdftotext >/dev/null || { echo "Install poppler-utils" >&2; exit 1; }
python3 ./assemble-ch04.py
python3 ./verify-ch04-manuscript.py
bundle install
bundle exec asciidoctor-pdf \
  -r ./sectnumoffset-treeprocessor.rb \
  -r asciidoctor-pdf-mathjax \
  -a doctype=article \
  -a notitle \
  -a toc! \
  -a sectnumlevels=3 \
  -a pdf-theme=manning-preview-theme.yml \
  -a pdf-themesdir=. \
  -a stem=latexmath \
  -a allow-uri-read \
  -o CH04_Dadwal_World_Models_in_Action.pdf \
  CH04_Dadwal_World_Models_in_Action.adoc
bash ./verify-ch04-pdf.sh
echo "Created and checked CH04_Dadwal_World_Models_in_Action.pdf."
