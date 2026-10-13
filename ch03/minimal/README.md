# Chapter 3: Minimal code

This branch is deliberately a **book-sized teaching tree**, not the full research workspace. It contains one cached training split and four short implementations. Run commands from the repository root.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python ch03/minimal/01_contrastive.py
python ch03/minimal/02_vicreg.py
python ch03/minimal/03_masked_prediction.py
python ch03/minimal/04_temporal_cpc.py
```

Each script accepts `--epochs` and `--seed`. The defaults are small demonstrations, not the budgets used for the reference experiments. Checkpoints are written under `ch03/minimal/` and ignored by Git.

| Script | Learn this |
|---|---|
| `01_contrastive.py` | Positive views, batch negatives, NT-Xent |
| `02_vicreg.py` | Invariance, variance floor, decorrelation |
| `03_masked_prediction.py` | Mask patches and score reconstruction only on hidden pixels |
| `04_temporal_cpc.py` | Episode-safe sequences, causal GRU context, future-latent classification |

`common.py` contains the shared encoder, data loading, augmentation, and objectives. Read each training script first, then open only the helper it uses.

The masked-prediction example is a compact convolutional illustration of the objective, not the full Transformer MAE from the reference experiment. The temporal example demonstrates the CPC learning rule; it does not implement the audited missing-frame evaluation.

## Keep claims tied to the evidence

See [`../RESULTS.md`](../RESULTS.md) for the measured conclusions and caveats. These short scripts do **not** reproduce the published checkpoints or metrics. The full experiment code, tests, data splits, figures, and evaluation artifacts remain on the separate [audit branch](https://github.com/aryandadwal2006/world-models-in-action/tree/audit/chapter3-publication-readiness).
