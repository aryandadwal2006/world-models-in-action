#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v ruby >/dev/null || { echo "Ruby 3.1+ is required" >&2; exit 1; }
command -v bundle >/dev/null || { echo "Install Bundler with: gem install bundler" >&2; exit 1; }
command -v pdftotext >/dev/null || { echo "Install poppler-utils to verify the first page" >&2; exit 1; }
bundle install
bundle exec asciidoctor-pdf \
  -r ./sectnumoffset-treeprocessor.rb \
  -r asciidoctor-mathematical \
  -a doctype=article \
  -a notitle \
  -a toc! \
  -a sectnumlevels=3 \
  -a pdf-theme=manning-preview-theme.yml \
  -a pdf-themesdir=. \
  -a stem=latexmath \
  -a mathematical-format=svg \
  -a allow-uri-read \
  -o CH03_Dadwal_World_Models_in_Action.pdf \
  CH03_Dadwal_World_Models_in_Action.adoc
bash ./verify-pdf-style.sh
echo "Created and checked CH03_Dadwal_World_Models_in_Action.pdf."
