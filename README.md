# World Models in Action

Code companion for *World Models in Action*. This branch reorganizes Chapter 3 so the implementation worth studying line by line is separate from full experiment runners and audit artifacts.

## Source of truth

- **Core implementations:** `worldmodels/`. Models, losses, sequence sampling, data handling, and evaluation utilities live here.
- **Experiment runners:** `ch03/`. These preserve the full data protocol, training configuration, multi-seed runs, evaluations, plotting, and metric export needed for the chapter.
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

The repository includes cached DMC train/validation datasets for the chapter experiments. Regenerating them requires the DeepMind Control Suite and MuJoCo rendering setup; downstream experiments can run from the caches.

## Reading order

Start with the code map in [Chapter 3 README](ch03/README.md). It maps the conceptual listings to the canonical source files and maps each chapter figure/table to its supplementary experiment runner.

## Re-run an experiment

Run individual scripts from the repository root. For example:

```powershell
python ch03/03_ntxent_from_scratch.py
python ch03/04_train_contrastive.py --seed 0
python ch03/08_vicreg.py
python ch03/09_masked_autoencoder.py
python ch03/11_temporal_cpc.py
```

The training scripts may overwrite their standard checkpoints, metrics, or figures. The CPC checkpoint-only gap-evaluation mode has separate safeguards documented in the script and audit report. For exact experiment settings and interpretation limits, read [the audit](ch03/publication_readiness_audit.md) before interpreting stored scores as general claims.
