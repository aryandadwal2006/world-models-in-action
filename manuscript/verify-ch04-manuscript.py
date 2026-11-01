#!/usr/bin/env python3
"""Verify Chapter 4 sources, equations, figure embeds and SVG files."""
from __future__ import annotations
import re,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"chapter04"; IMG=ROOT/"images"/"chapter04"
DOC=Path(__file__).resolve().parent/"CH04_Dadwal_World_Models_in_Action.adoc"
ORDER=["01_compatibility_and_energy.md","02_the_prediction_problem.md","03_ijepa_spatial_prediction.md","04_vjepa_spatiotemporal_prediction.md","05_degeneracy_and_collapse.md","06_predictive_sufficiency_and_geometry.md","07_linear_downstream_risk.md","08_nonlinear_risk_and_fisher_information.md","09_sigreg_and_distribution_sketching.md","10_visreg_scale_and_shape.md","11_vl_jepa_cross_modal_prediction.md","12_vjepa2_action_conditioning_and_dense_features.md","13_action_conditioned_prediction.md","14_planning_and_model_error.md","15_hierarchy_memory_and_agent_architecture.md","16_reading_a_jepa_training_step.md","17_synthesis_and_exercises.md"]
REQ=["I-JEPA","VL-JEPA","V-JEPA","V-JEPA 2","V-JEPA 2-AC","V-JEPA 2.1","LeJEPA","SIGReg","VISReg","Cramér–Wold","Epps–Pulley","sliced-Wasserstein","A Path Towards Autonomous Machine Intelligence","Perception","World Model","Cost","Actor","Short-Term Memory","Configurator","Intrinsic Cost","Trainable Critic","Mode 1","Mode 2","intervention","J(p)"]
def main():
    err=[]; missing=[x for x in ORDER if not (SRC/x).is_file()]
    err += ["Missing section source: "+x for x in missing]
    paths=[SRC/x for x in ORDER if (SRC/x).is_file()]
    source="\n".join(x.read_text(encoding="utf-8") for x in paths)
    for term in REQ:
        if term.casefold() not in source.casefold(): err.append("Missing required content: "+term)
    if len(paths)!=17: err.append(f"Expected 17 section files, found {len(paths)}")
    if not DOC.exists(): err.append("Chapter 4 AsciiDoc has not been assembled."); doc=""
    else: doc=DOC.read_text(encoding="utf-8")
    if doc and not doc.startswith("= World Models in Action\n"): err.append("Document title must be first")
    blocks=[]; lines=doc.splitlines(); i=0
    while i<len(lines):
        if lines[i].strip()=="[latexmath]":
            if i+1>=len(lines) or lines[i+1].strip()!="++++": err.append(f"Malformed math block at {i+1}"); i+=1; continue
            j=i+2
            while j<len(lines) and lines[j].strip()!="++++": j+=1
            if j==len(lines): err.append(f"Unclosed math block at {i+1}"); break
            body="\n".join(lines[i+2:j]).strip()
            if not body: err.append(f"Empty math block at {i+1}")
            blocks.append(body); i=j+1
        else: i+=1
    if len(blocks)<35: err.append(f"Expected >=35 display equations, got {len(blocks)}")
    if doc.count("latexmath:[")<25: err.append("Too few inline STEM nodes")
    if re.search(r"latexmath:\[\$[^\]]*\$\]",doc): err.append("Markdown dollars remain inside inline STEM")
    if "$$" in doc: err.append("Markdown double-dollar delimiters remain in AsciiDoc STEM")
    if "![" in doc: err.append("Markdown image syntax remains")
    found=set(re.findall(r"Figure 4\.\d+",doc)); expected={f"Figure 4.{n}" for n in range(1,35)}
    err += ["Missing figure caption: "+x for x in sorted(expected-found)]
    err += ["Unexpected figure caption: "+x for x in sorted(found-expected)]
    images=re.findall(r"image::([^\[]+)\[",doc)
    if len(images)!=34: err.append(f"Expected 34 embedded figures, found {len(images)}")
    for rel in images:
        path=(ROOT/"images"/rel).resolve()
        if not path.is_file(): err.append("Missing image asset: "+rel)
    svgs=list(IMG.glob("*.svg"))
    if len(svgs)!=34: err.append(f"Expected 34 chapter SVGs, found {len(svgs)}")
    for p in svgs:
        try: ET.parse(p)
        except (ET.ParseError,OSError) as ex: err.append(f"Invalid SVG {p.name}: {ex}")
    words=len(re.findall(r"\b[\w’'-]+\b",source,flags=re.UNICODE))
    if words<25000: err.append(f"Chapter appears too short: {words:,} words")
    for p in paths:
        t=p.read_text(encoding="utf-8")
        for n,line in enumerate(t.splitlines(),1):
            if line.count(r"\(")!=line.count(r"\)"): err.append(f"{p.name}:{n}: unbalanced inline math")
        if "Figure description." in t or "Artwork placeholder:" in t: err.append(f"{p.name}: obsolete figure placeholder")
    for bad in ["gaussian is optimal for all downstream tasks","guarantees exact gaussianity","the gradient is always nonzero for every parameter"]:
        if bad.casefold() in doc.casefold(): err.append("Overgeneralized claim: "+bad)
    if err: print("\n".join("ERROR: "+x for x in err),file=sys.stderr); raise SystemExit(1)
    print(f"OK: {len(paths)} sections, {words:,} words, {len(blocks)} display equations and {len(images)} embedded figures verified.")
if __name__=="__main__": main()
