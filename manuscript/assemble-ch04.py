#!/usr/bin/env python3
"""Assemble the canonical root chapter04 section sources."""
from __future__ import annotations
import re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"chapter04"
OUT=Path(__file__).resolve().parent/"CH04_Dadwal_World_Models_in_Action.adoc"
ORDER=["01_compatibility_and_energy.md","02_the_prediction_problem.md","03_ijepa_spatial_prediction.md","04_vjepa_spatiotemporal_prediction.md","05_degeneracy_and_collapse.md","06_predictive_sufficiency_and_geometry.md","07_linear_downstream_risk.md","08_nonlinear_risk_and_fisher_information.md","09_sigreg_and_distribution_sketching.md","10_visreg_scale_and_shape.md","11_vl_jepa_cross_modal_prediction.md","12_vjepa2_action_conditioning_and_dense_features.md","13_action_conditioned_prediction.md","14_planning_and_model_error.md","15_hierarchy_memory_and_agent_architecture.md","16_reading_a_jepa_training_step.md","17_synthesis_and_exercises.md"]
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
def convert(md):
    result = subprocess.run(
        ["pandoc", "--from=markdown+tex_math_dollars+tex_math_single_backslash", "--to=asciidoc", "--wrap=none"],
        input=md,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    output = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "[latexmath]" and i + 1 < len(lines) and lines[i + 1].strip() == "++++":
            j = i + 2
            while j < len(lines) and lines[j].strip() != "++++":
                j += 1
            if j == len(lines):
                raise SystemExit("unclosed Pandoc latexmath block")
            body = "\n".join(lines[i + 2:j]).strip()
            # Strip delimiters only when they enclose the entire TeX block.
            if body.startswith("$$"):
                body = body[2:].lstrip()
            if body.endswith("$$"):
                body = body[:-2].rstrip()
            elif body.startswith("$") and body.endswith("$"):
                body = body[1:-1].strip()
            output.extend(["[latexmath]", "++++", body, "++++"])
            i = j + 1
            continue

        line = lines[i]
        # Pandoc normally emits inline math without dollar wrappers. Handle
        # the wrapped form too, then clear any remaining display delimiters.
        line = re.sub(
            r"latexmath:\[\$+(.*?)\$+\]",
            lambda m: "latexmath:[" + m.group(1) + "]",
            line,
        )
        line = line.replace("$$", "")
        output.append(line)
        i += 1

    asciidoc = "\n".join(output).replace(
        "image::../images/chapter04/",
        "image::chapter04/",
    )
    return re.sub(
        r"(image::chapter04/[^\[]+)\[([^\]]*)\]",
        lambda m: m.group(1) + "[" + m.group(2) + ",width=100%]",
        asciidoc,
    )


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
