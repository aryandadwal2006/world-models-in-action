#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v ruby >/dev/null || { echo "Ruby 3.1+ is required" >&2; exit 1; }
command -v bundle >/dev/null || { echo "Install Bundler with: gem install bundler" >&2; exit 1; }
bundle install
bundle exec asciidoctor-pdf \
  -r ./sectnumoffset-treeprocessor.rb \
  -r asciidoctor-mathematical \
  -a stem=latexmath \
  -a mathematical-format=svg \
  -a allow-uri-read \
  -o CH03_Dadwal_World_Models_in_Action.pdf \
  CH03_Dadwal_World_Models_in_Action.adoc
echo "Created CH03_Dadwal_World_Models_in_Action.pdf in this directory."
