#!/usr/bin/env python3
"""Assemble Chapter 4 section Markdown into a Manning-style AsciiDoc source."""
from __future__ import annotations
import re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DRAFT=ROOT/"rough-draft"/"chapter4"/"draft"
OUT=Path(__file__).resolve().parent/"CH04_Dadwal_World_Models_in_Action.adoc"
ORDER=["00-introduction.md","4.1-why-predict-in-latent-space.md","4.2-jepa-objective.md",
"4.2.1-i-jepa.md","4.2.2-v-jepa.md","4.2.3-v-jepa-2-and-2.1.md",
"4.2.4-regularization-before-distributional-tests.md","4.2.5-lejepa-sigreg.md",
"4.2.6-visreg.md","4.2.7-adjacent-modal-and-variational-jepas.md",
"4.3-action-conditioned-prediction-and-planning.md",
"4.3.1-path-towards-autonomous-intelligence.md",
"4.3.2-mode-one-mode-two-and-hierarchical-planning.md",
"4.4-reading-a-miniature-jepa-system.md","4.5-summary-and-exercises.md"]
def transform(name):
    lines=(DRAFT/name).read_text(encoding="utf-8").splitlines()
    if name=="00-introduction.md":
        if lines and lines[0].startswith("# "): lines=lines[1:]
        lines=[("**"+line[3:].strip()+"**") if line.startswith("## ") else line for line in lines]
        return "\n".join(lines).strip()+"\n"
    first=next((i for i,x in enumerate(lines) if x.startswith("# ")),None)
    if first is None: raise SystemExit(f"{name}: missing section heading")
    m=re.match(r"^(4(?:\.\d+){1,2})\s+(.+)$",lines[first][2:].strip())
    if not m: raise SystemExit(f"{name}: invalid numbered heading")
    number,title=m.groups()
    shift=number.count(".")-1
    for i,line in enumerate(lines):
        hm=re.match(r"^(#{1,6})\s+(.+)$",line)
        if hm:
            marks,content=hm.groups()
            if i==first: content=title
            lines[i]="#"*min(6,len(marks)+shift)+" "+content
    return "\n".join(lines).strip()+"\n"
def pandoc(markdown):
    r=subprocess.run(["pandoc","--from=markdown+tex_math_dollars+tex_math_single_backslash",
                      "--to=asciidoc","--wrap=none"],input=markdown,text=True,capture_output=True,check=True)
    return r.stdout.strip()
def add_anchors(adoc):
    out=[]; n=0
    for line in adoc.splitlines():
        if re.match(r"^={2,4}\s+\S",line):
            title=re.sub(r"^=+\s+","",line).strip()
            if title.casefold()!="references":
                n+=1
                slug=re.sub(r"[^a-z0-9]+","_",title.casefold()).strip("_")[:44]
                out.append(f"[#unique_section_ch4_{n:03d}_{slug}]")
        out.append(line)
    return "\n".join(out).strip()+"\n"
def main():
    missing=[n for n in ORDER if not (DRAFT/n).is_file()]
    if missing: raise SystemExit("Missing draft files: "+", ".join(missing))
    intro=pandoc(transform(ORDER[0]))
    sections="\n\n".join(pandoc(transform(n)) for n in ORDER[1:])
    ref=(DRAFT/"references.md").read_text(encoding="utf-8").splitlines()
    if ref and ref[0].startswith("# "): ref=ref[1:]
    ref=[("**"+x[3:].strip()+"**") if x.startswith("## ") else x for x in ref]
    references=pandoc("\n".join(ref))
    front="""= World Models in Action
:chapter: 4
:sectnums:
:sectnumlevels: 3
:figure-caption: Figure {chapter}
:listing-caption: Listing {chapter}
:table-caption: Table {chapter}
:sectnumoffset: 3
:leveloffset: 1
:imagesdir: ../images/
:table-number:
:figure-number:
:listing-number:
:!section:
:!subsection:
:!figure:
:!listing:
:!table:
:stem: latexmath

[.book-title]
World Models in Action

[#unique_chapter4_title_ID,reftext={chapter},caption='{chapter}']
= Predicting in representation space

This chapter covers

* The JEPA objective and the transition from image to video prediction
* I-JEPA, V-JEPA, V-JEPA 2, and action-conditioned latent world models
* Collapse prevention, LeJEPA/SIGReg, and VISReg
* LeCun's proposed architecture for autonomous machine intelligence

"""
    result=front+intro+"\n\n"+add_anchors(sections)+"\n== References\n\n"+references+"\n"
    OUT.write_text(result,encoding="utf-8")
    print(f"Assembled {len(ORDER)} sections into {OUT} ({OUT.stat().st_size:,} bytes)")
if __name__=="__main__": main()
