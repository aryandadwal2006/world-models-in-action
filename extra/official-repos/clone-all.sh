#!/usr/bin/env bash
set -euo pipefail
# Run from the root of this branch. Git will check out the exact pinned revisions.
git submodule update --init --recursive
