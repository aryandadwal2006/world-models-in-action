# Chapter 3: Minimal World-Model Learning Code

This is the **book-facing implementation**: short scripts that expose the learning rule, model, and training loop without the experiment bookkeeping.

The longer scripts in `ch03/` remain the reference implementation for the recorded figures, checkpoints, and metrics. These minimal scripts are teaching versions, not drop-in replacements for those experiments; do not cite their fresh runs as the values in the book.

## Setup

Run from the repository root with the existing environment:

```powershell
..venv\Scripts\Activate.ps1
python ch03/minimal/01_contrastive.py
python ch03/minimal/02_vicreg.py
python ch03/minimal/03_masked_prediction.py
python ch03/minimal/04_temporal_cpc.py
```

The scripts use the cached Cartpole Balance datasets in `data/`. They do not collect data, download weights, or overwrite the published result files. Each script has a small default budget; increase `--epochs` to experiment.

## The four ideas

1. **Contrastive learning:** two augmented views of one frame are positives; other frames in the batch are negatives. The model learns an encoder and projection head with NT-Xent.
2. **VICReg:** match the two views, keep each feature dimension variable, and discourage correlated dimensions. There are no negative examples.
3. **Masked prediction:** hide image patches and reconstruct only the hidden pixels. This compact convolutional example teaches the objective, not the full Transformer MAE architecture used in the reference experiment.
4. **Temporal CPC:** encode episode-safe frame sequences, summarize the past with a causal GRU, and identify future latents among batch negatives.

## What the measured experiments say

Use the committed research artifacts for quantitative claims, not the quick teaching runs:

- The contrastive, VICReg, and masked-reconstruction objectives are useful comparisons, but they optimize different properties; no single scalar loss establishes that a representation is a good world model.
- More latent dimensions did not establish a universal saturation point in the tested sweep.
- A stacked-frame input did not reliably recover velocity information in the tested setup.
- The expanded temporal gap test is a **negative result**: CPC mean gap (R^2) was (-2.853), (-0.128), and (-0.443) for seeds 0, 1, and 2. Across-seed mean ± population SD was (-1.141 pm 1.217). The result is seed-sensitive.
- The state-history baseline scored (R^2=0.837), but receives true physics-state history while CPC receives images. It is privileged and not a like-for-like comparison.

Full evaluation details: `ch03/results/table_03_05_cpc_metrics_200_windows.json` and `ch03/publication_readiness_audit.md`.

## Design rule

Keep the lesson small and the claims bounded. These scripts favor visible mechanisms over flags, reports, plotting, migrations, and publication checks. The reference scripts retain those safeguards for reproducibility.
