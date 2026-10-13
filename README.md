# World Models in Action

A compact educational companion for learning the core ideas behind visual representations and predictive world models.

## Start with Chapter 3

Read [the short examples](ch03/minimal/README.md) in order:

1. Contrastive learning with NT-Xent
2. VICReg without negative pairs
3. Masked-pixel prediction
4. Temporal CPC with a causal GRU

The scripts are intentionally short. They focus on the model, objective, and training loop rather than experiment orchestration.

## Run

Use Python 3.11 or newer with PyTorch and NumPy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ch03/minimal/01_contrastive.py
```

Run the other examples by changing the script name. They use the included Cartpole Balance training cache at `data/dmc_cartpole_balance_train.npz`; no data collection or checkpoint download is required.

## Read the findings

[Chapter 3 results and interpretation](ch03/RESULTS.md) summarizes what the reference experiments support and where they failed. The short scripts are **teaching implementations**, not bit-for-bit reproductions of the reported experiments.

The full research/audit implementation, experiment metadata, checkpoints, tests, and figure-generation scripts remain on the separate [`audit/chapter3-publication-readiness` branch](https://github.com/aryandadwal2006/world-models-in-action/tree/audit/chapter3-publication-readiness). This branch is the book-facing code, not the full audit workspace.
