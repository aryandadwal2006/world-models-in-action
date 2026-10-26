# World Models in Action

[![Chapter 3 regression tests](https://github.com/aryandadwal2006/world-models-in-action/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/aryandadwal2006/world-models-in-action/actions/workflows/tests.yml)

Code companion for *World Models in Action*. This branch reorganizes Chapter 3 so the implementation worth studying line by line is separate from full experiment runners and audit artifacts.

## Source of truth

- **Chapter-oriented pedagogical implementations:** `worldmodels/`. These are compact, readable implementations for the book, not copies of upstream repositories.
- **Experiment runners:** `ch03/`. These preserve data protocols, configs, multi-seed runs, evaluation, plotting and metric export.
- **Original upstream method repositories:** `implementations/`. This directory contains Git submodules pinned to original author/team repositories or shared upstream research codebases. CMC, AMDIM and the FAIR/VISSL method suite are included alongside the method-specific repositories. No replacement method code is authored or modified there. The submodule pointers are not expanded on GitHub's web view; run `bash implementations/clone-all.sh` after cloning to populate the local source folders.
- **Verification:** `tests/`.
- **Evidence:** `ch03/figures/`, `ch03/results/`, `checkpoints/` and `ch03/publication_readiness_audit.md`.
- **Research notes:** `extra/papers-and-derivations.md`.

The chapter runners call canonical modules under `worldmodels/`. These upstream source folders are references, not runtime dependencies for chapter tests.

## Setup

Use Python 3.11 or newer. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To clone the official upstream repositories under `implementations/` at their pinned commits, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The full collection makes the checkout large; initialize one repository at a time if preferred. Every upstream keeps its own license and dependencies. Model weights and datasets are not bundled. The MoCo source is pinned to the last code-bearing upstream revision because a later upstream commit deleted the repository tree; its README and LICENSE text conflict, so inspect both before any redistribution.

The repository includes cached DMC train/validation datasets for the chapter experiments. Regenerating them requires the DeepMind Control Suite and MuJoCo rendering setup; downstream experiments can run from the caches. Experiment claims and reproduction instructions are recorded in `ch03/README.md` and `ch03/publication_readiness_audit.md`.

## Reading order

Start with the code map in [Chapter 3 README](ch03/README.md). For original method source, use the [upstream implementation manifest](implementations/README.md) and verify the method details against the primary paper before treating a repository as a faithful reproduction. The collection includes commit-pinned CPC audio, Deep InfoMax, DDPM, score-SDE and guided-diffusion references; initialize them individually or with `bash implementations/clone-all.sh`.

## Re-run an experiment

Run individual scripts from the repository root. For example:

```powershell
python ch03/03_ntxent_from_scratch.py
python ch03/04_train_contrastive.py --seed 0
python ch03/08_vicreg.py
python ch03/09_masked_autoencoder.py
python ch03/11_temporal_cpc.py
```

The training scripts may overwrite their standard checkpoints, metrics, or figures, so use isolated output paths for new runs. CPC's primary evaluation is held-out future-latent retrieval using the trained InfoNCE scores; the historical recursive gap-rollout diagnostic is not the canonical CPC result. The primary matched-length result is in `ch03/results/table_03_05_cpc_metrics.json`. For exact settings, provenance, and interpretation limits, read [the Chapter 3 code map](ch03/README.md) and [the publication-readiness audit](ch03/publication_readiness_audit.md) before interpreting stored scores as general claims.
