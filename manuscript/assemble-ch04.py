#!/usr/bin/env python3
"""Assemble the canonical root chapter04 section sources."""
from __future__ import annotations
import re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"chapter04"
OUT=Path(__file__).resolve().parent/"CH04_Dadwal_World_Models_in_Action.adoc"
ORDER=["01_compatibility_and_energy.md","02_the_prediction_problem.md","03_ijepa_spatial_prediction.md","04_vjepa_spatiotemporal_prediction.md","05_degeneracy_and_collapse.md","06_predictive_sufficiency_and_geometry.md","07_linear_downstream_risk.md","08_nonlinear_risk_and_fisher_information.md","09_sigreg_and_distribution_sketching.md","10_visreg_scale_and_shape.md","11_vjepa2_action_conditioning_and_dense_features.md","12_action_conditioned_prediction.md","13_planning_and_model_error.md","14_hierarchy_memory_and_agent_architecture.md","15_reading_a_jepa_training_step.md","16_synthesis_and_exercises.md"]
def clean_headings(md):
    out=[]; fence=None
    for line in md.splitlines():
        s=line.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            mark=s[:3]
            if fence is None: fence=mark
            elif fence==mark: fence=None
            out.append(line); continue
        if fence is None and re.match(r"^\s*#{1,6}\s+",line):
            line=re.sub(r"^(\s*#{1,6}\s+)\d+(?:\.\d+)*\.?\s+",r"\1",line)
        out.append(line)
    return "\n".join(out).strip()+"\n"
def math(md):
    out=[]; active=False
    for line in md.splitlines():
        s=line.strip()
        if s==r"\[":
            if active: raise SystemExit("nested display math")
            active=True; out.append("$$")
        elif s==r"\]":
            if not active: raise SystemExit("unmatched display math closer")
            active=False; out.append("$$")
        elif not active: out.append(line.replace(r"\(","$").replace(r"\)","$"))
        else: out.append(line)
    if active: raise SystemExit("unclosed display math")
    return "\n".join(out)
def convert(md):
    r=subprocess.run(["pandoc","--from=markdown+tex_math_dollars","--to=asciidoc","--wrap=none"],input=math(md),text=True,capture_output=True,check=True)
    s=re.sub(r"latexmath:\[\$+(.*?)\$+\]",lambda m:"latexmath:["+m.group(1)+"]",r.stdout.strip(),flags=re.S)
    lines=s.splitlines(); out=[]; waiting=False; inside=False; block=[]
    def emit_math_block(body):
        body="\n".join(body).strip()
        # Pandoc may preserve Markdown math delimiters inside a raw
        # AsciiDoc latexmath block. They are delimiters, not TeX operators.
        if body.startswith("$$"): body=body[2:].lstrip()
        if body.endswith("$$"): body=body[:-2].rstrip()
        # Pandoc may also wrap a one-line inline expression in a raw block.
        elif body.startswith("$") and body.endswith("$"):
            body=body[1:-1].strip()
        out.extend(["++++",body,"++++"])
    for line in lines:
        t=line.strip()
        if t=="[latexmath]": waiting=True; out.append(line); continue
        if t=="++++":
            if waiting: inside=True; waiting=False; block=[]
            elif inside: emit_math_block(block); inside=False; block=[]
            else: out.append(line)
            continue
        if inside:
            if t in (r"\[",r"\]"): continue
            block.append(line); continue
        out.append(line)
    if inside: raise SystemExit("unclosed Pandoc latexmath block")
    if waiting: raise SystemExit("Pandoc latexmath block missing opening delimiter")
    s="\n".join(out).replace("image::../images/chapter04/","image::chapter04/")
    return re.sub(r"(image::chapter04/[^\[]+)\[([^\]]*)\]",lambda m:m.group(1)+"["+m.group(2)+",width=100%]",s)
def anchors(s):
    out=[]; n=0
    for line in s.splitlines():
        if re.match(r"^={2,4}\s+\S",line):
            title=re.sub(r"^=+\s+","",line).strip()
            if title.casefold()!="references":
                n+=1; slug=re.sub(r"[^a-z0-9]+","_",title.casefold()).strip("_")[:44]
                out.append(f"[#unique_section_ch4_{n:03d}_{slug}]")
        out.append(line)
    return "\n".join(out).strip()+"\n"
def main():
    missing=[x for x in ORDER if not (SRC/x).is_file()]
    if missing: raise SystemExit("Missing section sources: "+", ".join(missing))
    parts=[]
    for name in ORDER:
        md=(SRC/name).read_text(encoding="utf-8")
        if not re.search(r"^#\s+\d+(?:\.\d+)*\.?\s+\S",md,re.M):
            raise SystemExit(f"{name}: missing numbered top-level heading")
        parts.append(convert(clean_headings(md)))
    refs=(SRC/"references.md").read_text(encoding="utf-8").split("## Source-review record and access limitation",1)[0]
    ref_lines=[x for x in refs.splitlines() if not x.startswith("# ")]
    refs="\n".join("**"+x[3:].strip()+"**" if x.startswith("## ") else x for x in ref_lines)
    refs=convert(refs)
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

* Spatial and spatiotemporal JEPA prediction, from I-JEPA to V-JEPA
* Collapse, predictive sufficiency, and the statistical geometry of embeddings
* LeJEPA/SIGReg and VISReg as distributional regularizers
* Action-conditioned prediction, planning, memory, hierarchy, and predictive-agent design

"""
    result=front+"\n\n".join(parts)+"\n\n== References\n\n"+refs+"\n"
    OUT.write_text(anchors(result),encoding="utf-8")
    print(f"Assembled {len(ORDER)} sections into {OUT} ({OUT.stat().st_size:,} bytes)")
if __name__=="__main__": main()
