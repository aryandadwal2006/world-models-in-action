#!/usr/bin/env python3
"""Verify Chapter 4's section sources, mathematics and text-only figure briefs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "chapter04"
IMG = ROOT / "images" / "chapter04"
DOC = Path(__file__).resolve().parent / "CH04_Dadwal_World_Models_in_Action.adoc"

ORDER = [
    "01_compatibility_and_energy.md",
    "02_the_prediction_problem.md",
    "03_ijepa_spatial_prediction.md",
    "04_vjepa_spatiotemporal_prediction.md",
    "05_degeneracy_and_collapse.md",
    "06_predictive_sufficiency_and_geometry.md",
    "07_linear_downstream_risk.md",
    "08_nonlinear_risk_and_fisher_information.md",
    "09_sigreg_and_distribution_sketching.md",
    "10_visreg_scale_and_shape.md",
    "11_vl_jepa_cross_modal_prediction.md",
    "12_vjepa2_action_conditioning_and_dense_features.md",
    "13_action_conditioned_prediction.md",
    "14_planning_and_model_error.md",
    "15_hierarchy_memory_and_agent_architecture.md",
    "16_reading_a_jepa_training_step.md",
    "17_synthesis_and_exercises.md",
]

REQUIRED = [
    "I-JEPA", "V-JEPA", "V-JEPA 2", "V-JEPA 2-AC", "V-JEPA 2.1",
    "VL-JEPA", "LeJEPA", "SIGReg", "VISReg", "Cramér–Wold",
    "Epps–Pulley", "sliced Wasserstein", "Fisher-information",
    "predictive sufficiency", "intervention", "Mode 1", "Mode 2",
    "Perception", "World Model", "Cost", "Actor", "Short-Term Memory",
    "Configurator", "Intrinsic Cost", "Trainable Critic", "J(p)",
]

FIGURE_RE = re.compile(r"^\*\*Figure 4\.(\d+)\.\s+.+\*\*$")
EXPECTED_FIGURES = set(range(1, 30))


def main() -> None:
    errors: list[str] = []
    missing = [name for name in ORDER if not (SRC / name).is_file()]
    errors.extend("Missing section source: " + name for name in missing)
    paths = [SRC / name for name in ORDER if (SRC / name).is_file()]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    if len(paths) != 17:
        errors.append(f"Expected 17 section files, found {len(paths)}")

    for term in REQUIRED:
        if term.casefold() not in source.casefold():
            errors.append("Missing required content: " + term)

    figures: list[int] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        expected_heading = ORDER.index(path.name) + 1
        if not re.match(rf"^#\s+{expected_heading}\.\s+\S", lines[0] if lines else ""):
            errors.append(f"{path.name}: missing or incorrect numbered top-level heading")
        if re.search(r"!\[[^\]]*\]\(", text) or "image::" in text:
            errors.append(f"{path.name}: image embeds are forbidden")
        if "Figure description." in text or "Artwork placeholder:" in text:
            errors.append(f"{path.name}: obsolete figure placeholder")

        for i, line in enumerate(lines):
            match = FIGURE_RE.match(line)
            if match:
                number = int(match.group(1))
                figures.append(number)
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines) or not lines[j].startswith("Description:"):
                    errors.append(f"{path.name}:{i+1}: figure caption lacks a Description paragraph")
                elif len(lines[j].split()) > 60:
                    errors.append(f"{path.name}:{j+1}: figure description exceeds 60 words")
            if line.count(r"\(") != line.count(r"\)"):
                errors.append(f"{path.name}:{i+1}: unbalanced inline-math delimiters")

    if sorted(figures) != sorted(EXPECTED_FIGURES):
        errors.append(
            "Figure briefs must be exactly 4.1 through 4.29; found "
            + ", ".join(f"4.{n}" for n in sorted(figures))
        )
    if len(figures) != len(set(figures)):
        errors.append("Duplicate figure-brief number")

    words = len(re.findall(r"\b[\w’'-]+\b", source, flags=re.UNICODE))
    if words < 20_000:
        errors.append(f"Chapter appears too short: {words:,} words")

    if IMG.exists():
        assets = [p for p in IMG.rglob("*") if p.is_file()]
        if assets:
            errors.append("Chapter 4 image assets are forbidden: " + ", ".join(p.name for p in assets))
    else:
        assets = []

    if not DOC.exists():
        errors.append("Chapter 4 AsciiDoc has not been assembled")
        doc = ""
    else:
        doc = DOC.read_text(encoding="utf-8")

    if doc and not doc.startswith("= World Models in Action\n"):
        errors.append("Document title must be first")
    if "image::" in doc or "![" in doc:
        errors.append("Assembled manuscript contains an image embed")

    blocks: list[str] = []
    lines = doc.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "[latexmath]":
            if i + 1 >= len(lines) or lines[i+1].strip() != "++++":
                errors.append(f"Malformed STEM block at AsciiDoc line {i+1}")
                i += 1
                continue
            j = i + 2
            while j < len(lines) and lines[j].strip() != "++++":
                j += 1
            if j == len(lines):
                errors.append(f"Unclosed STEM block at AsciiDoc line {i+1}")
                break
            body = "\n".join(lines[i+2:j]).strip()
            if not body:
                errors.append(f"Empty STEM block at AsciiDoc line {i+1}")
            if body.startswith(r"\[") or body.endswith(r"\]"):
                errors.append(f"Markdown display delimiters remain in STEM block at line {i+1}")
            blocks.append(body)
            i = j + 1
        else:
            i += 1

    if len(blocks) < 120:
        errors.append(f"Expected at least 120 display equations, found {len(blocks)}")
    inline_count = doc.count("latexmath:[")
    if inline_count < 180:
        errors.append(f"Expected at least 180 inline STEM nodes, found {inline_count}")
    if "$$" in doc:
        errors.append("Markdown double-dollar delimiters remain in AsciiDoc")
    if re.search(r"latexmath:\[\$[^\]]*\$\]", doc):
        errors.append("Markdown dollar delimiters remain inside inline STEM")

    if errors:
        print("\n".join("ERROR: " + error for error in errors), file=sys.stderr)
        raise SystemExit(1)

    print(
        f"OK: {len(paths)} sections, {words:,} words, {len(blocks)} display equations, "
        f"{inline_count} inline STEM nodes, {len(figures)} text-only figure briefs, "
        f"{len(assets)} image assets and 0 image embeds."
    )


if __name__ == "__main__":
    main()
