# Chapter 3: Minimal code

These scripts isolate one idea at a time. Run them from the repository root.

| Script | Idea | Core mechanism |
|---|---|---|
| `01_contrastive.py` | Contrastive learning | Two views, projection head, NT-Xent |
| `02_vicreg.py` | Non-contrastive learning | Invariance, variance floor, covariance penalty |
| `03_masked_prediction.py` | Masked reconstruction | Hide patches; score only hidden pixels |
| `04_temporal_cpc.py` | Temporal prediction | Causal GRU context predicts future latents |

`common.py` contains the shared encoder, cached datasets, augmentations, and loss functions. Read it after the script you are studying, rather than treating it as a framework.

## Commands

```powershell
python ch03/minimal/01_contrastive.py
python ch03/minimal/02_vicreg.py
python ch03/minimal/03_masked_prediction.py
python ch03/minimal/04_temporal_cpc.py
```

Each accepts `--epochs` and `--seed`. Defaults are intentionally small for a first run. Checkpoints are written under `ch03/minimal/` and ignored by Git.

The masked-prediction script is a small convolutional illustration of the objective, not the Transformer MAE used in the reference experiment. None of these teaching runs should be used as the source of the book's reported metrics.
