#!/usr/bin/env python3
from pathlib import Path
import argparse, re, sys

parser = argparse.ArgumentParser()
parser.add_argument('--allow-unfetched-assets', action='store_true', help='Allow four canonical empirical PNGs that are linked in GitHub but not downloaded into this local workspace.')
args = parser.parse_args()
here = Path(__file__).resolve().parent
adoc = here / 'CH03_Dadwal_World_Models_in_Action.adoc'
img = here.parent / 'images'
s = adoc.read_text(encoding='utf-8')
errors = []
if not s.startswith('= World Models in Action'):
    errors.append('Missing document title')
for required in [':chapter: 3', ':imagesdir: ../images/', ':stem: latexmath', 'static visual encoder', 'full world model']:
    if required not in s:
        errors.append('Missing required header/content: ' + required)
for n in range(1, 18):
    stem = f'CH03_F{n:02d}_Dadwal'
    if (img / (stem + '.png')).exists() or (img / (stem + '.svg')).exists() or (img / (stem + '.md')).exists():
        continue
    if args.allow_unfetched_assets and n in (3, 7, 14, 16):
        continue
    errors.append('Missing figure asset or Markdown placeholder: ' + stem)
markers = re.findall(r'@@FIGURE_[A-Z0-9_]+@@', s)
if markers:
    errors.append('Unexpanded figure markers: ' + ', '.join(markers))
for m in re.finditer(r'^image::([^\[]+)\[', s, re.M):
    target = m.group(1).strip()
    if not (img / target).exists():
        if args.allow_unfetched_assets and target in {'CH03_F03_Dadwal.png', 'CH03_F07_Dadwal.png', 'CH03_F14_Dadwal.png', 'CH03_F16_Dadwal.png'}:
            continue
        errors.append('Unresolved image macro target: ' + target)
if errors:
    print('\n'.join('ERROR: ' + e for e in errors))
    sys.exit(1)
print('OK: document structure and all local/intentional figure slots are consistent.')