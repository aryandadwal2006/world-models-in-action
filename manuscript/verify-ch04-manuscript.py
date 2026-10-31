#!/usr/bin/env python3
"""Verify Chapter 4 topics, 15 section files and 21 figure-description placeholders."""
from pathlib import Path
import re,sys
here=Path(__file__).resolve().parent
draft=here.parent/"rough-draft"/"chapter4"/"draft"
p=here/"CH04_Dadwal_World_Models_in_Action.adoc"
if not p.exists(): raise SystemExit("Chapter 4 AsciiDoc has not been assembled.")
s=p.read_text(encoding="utf-8"); errors=[]
required=["I-JEPA","V-JEPA","V-JEPA 2-AC","V-JEPA 2.1","LeJEPA","SIGReg","VISReg",
"Cramér-Wold","Epps-Pulley","sliced-Wasserstein","A Path Towards Autonomous Machine Intelligence",
"Perception","World Model","Cost","Actor","Short-Term Memory","Configurator",
"Intrinsic Cost","Trainable Critic","Mode 1","Mode 2","What Drives Success in Physical Planning"]
for x in required:
    if x.casefold() not in s.casefold(): errors.append("Missing: "+x)
if not s.startswith("= World Models in Action\n"): errors.append("Document title must be first.")
if "image::" in s: errors.append("Unexpected artwork; descriptions only.")
if "Artwork placeholder:" in s or "@@figure_" in s: errors.append("Obsolete placeholder.")
expected={f"Figure 4.{i}" for i in range(1,22)}
found=set(re.findall(r"Figure 4\.\d+",s))
for x in sorted(expected-found): errors.append("Missing figure description: "+x)
for x in sorted(found-expected): errors.append("Unexpected figure number: "+x)
for f in draft.glob("*.md"):
    if f.name in {"README.md","references.md"}: continue
    text=f.read_text(encoding="utf-8")
    for m in re.finditer(r"\*\*Figure 4\.\d+[^*]*\*\*",text):
        if "Figure description." not in text[m.end():m.end()+700]:
            errors.append(f"{f.name}: figure heading lacks nearby description.")
for bad in ["gaussian is optimal for all downstream tasks","guarantees exact gaussianity",
"the gradient is always nonzero for every parameter"]:
    if bad.casefold() in s.casefold(): errors.append("Overgeneralized claim: "+bad)
n=len([f for f in draft.glob("*.md") if f.name not in {"README.md","references.md"}])
if n!=15: errors.append(f"Expected 15 section sources, found {n}")
if errors:
    print("\n".join("ERROR: "+x for x in errors),file=sys.stderr); raise SystemExit(1)
print("OK: topics, 15 sections and 21 figure descriptions verified.")
