# World Models in Action

[![Chapter 3 regression tests](https://github.com/aryandadwal2006/world-models-in-action/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/aryandadwal2006/world-models-in-action/actions/workflows/tests.yml)

Code companion and Chapter 3 authoring repository for *World Models in Action*.

## Branch roles

- **`main`** is the canonical code-companion/reproducibility release: core implementations, experiment runners, regression tests, and empirical artifacts.
- **`manuscript-theory`** contains the current composed Chapter 3 manuscript, its figure briefs, the author-side PDF preview, and the manuscript build/verification workflow. This is the current source branch for the theory-first chapter.
- **Current chapter source:** [`manuscript/CH03_Dadwal_World_Models_in_Action.adoc`](manuscript/CH03_Dadwal_World_Models_in_Action.adoc).
- **Current author-side PDF:** [`manuscript/CH03_Dadwal_World_Models_in_Action.pdf`](manuscript/CH03_Dadwal_World_Models_in_Action.pdf). This is a preview, not Manning's final production typesetting.
- **Canonical printed figure register:** [`images/README.md`](images/README.md). The current chapter has 17 inserted figures. Artwork files `images/CH03_F01_Dadwal.png` through `images/CH03_F17_Dadwal.png` follow printed order; the Markdown briefs retain legacy `F##` identifiers.
- **Figure/table/listing crosswalk:** [`chapter3/figure-and-table-production.md`](chapter3/figure-and-table-production.md).
- The `chapter3/` section-by-section drafts and `chapter_03_design.md` are historical planning material, not current manuscript sources.

## Code source of truth

- **Core implementations:** `worldmodels/`. Models, losses, sequence sampling, data handling, and evaluation utilities live here.
- **Experiment runners:** `ch03/`. These preserve the data protocol, training configuration, multi-seed runs, evaluations, plotting, and metric export needed for the chapter.
- **Verification:** `tests/`. Tests exercise the canonical modules and important data/evaluation contracts.
- **Evidence:** `ch03/figures/`, `ch03/results/`, `checkpoints/`, and `ch03/publication_readiness_audit.md` hold generated outputs, checkpoints, and limitations.

The chapter runners call the canonical implementations. The legacy `worldmodels.losses.contrastive` and MAE exports from `worldmodels.models.encoders` remain as import-compatible facades; they do not duplicate those implementations.

## Setup

Use Python 3.11 or newer. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The repository includes cached DMC train/validation datasets for the chapter experiments. Regenerating them requires the DeepMind Control Suite and MuJoCo rendering setup; downstream experiments can run from the caches. Experiment claims and reproduction instructions are recorded in [`ch03/README.md`](ch03/README.md) and [`ch03/publication_readiness_audit.md`](ch03/publication_readiness_audit.md).

## Reading order

Start with the code map in [Chapter 3 README](ch03/README.md). It maps conceptual listings to canonical source files and empirical artifacts. For current prose, figure numbering, the PDF preview, and build steps, use the [manuscript README](manuscript/README.md) and [figure register](images/README.md).

## Re-run an experiment

Run individual scripts from the repository root. For example:

```powershell
python ch03/03_ntxent_from_scratch.py
python ch03/04_train_contrastive.py --seed 0
python ch03/08_vicreg.py
python ch03/09_masked_autoencoder.py
python ch03/11_temporal_cpc.py
```

The training scripts may overwrite their standard checkpoints, metrics, or figures, so use isolated output paths for new runs. CPC's primary evaluation is held-out future-latent retrieval using the trained InfoNCE scores; it is not physical-state prediction, action-conditioned transition prediction, recursive rollout, or planning. For exact settings, provenance, and interpretation limits, read the [Chapter 3 code map](ch03/README.md) and [publication-readiness audit](ch03/publication_readiness_audit.md) before interpreting stored scores as general claims.