# Chapter 3 Implementation Plan & Progress

Based on *Chapter 3 Design — Representation Learning Before Generation (v3)*.

## 1. Environment & Setup
- [x] Clone `dm_control` into `dm_control_repo/` for local reference
- [x] Setup virtual environment `.venv` with Python 3.11
- [x] Install pinned requirements (`torch`, `numpy`, `matplotlib`, `dm_control`, `mujoco`)
- [x] Verify offscreen rendering on Windows
- [x] Initialize git repository and track progress

## 2. Shared `worldmodels/` Core Library
- [x] `worldmodels/__init__.py`
- [x] `worldmodels/train.py`: Seeding utilities (`set_seed`), device selection, training helpers
- [x] `worldmodels/data/augment.py`: View generation pipeline (crop shift, intensity jitter)
- [x] `worldmodels/data/dmc_data.py`: DMC environment interaction, episode collection, `.npz` caching, and dataset splits
- [x] `worldmodels/models/encoders.py`: `ConvEncoder`, `ProjectionHead`, `SimpleMAE`, `MAEEncoder`, and `Encoder` interface (`encode(obs)`, `update(state, obs)`)
- [x] `worldmodels/losses/contrastive.py`:
  - NT-Xent / InfoNCE loss (Eq 3.5, 3.6, 3.7)
  - SimSiam negative cosine similarity loss with stop-gradient (Eq 3.9, 3.10)
  - BYOL EMA updater (Eq 3.11)
  - Barlow Twins loss (Eq 3.12, 3.13)
  - VICReg variance, invariance, and covariance regularization loss (Eq 3.14 - 3.17)
- [x] `worldmodels/losses/reconstruction.py`: Masked Autoencoder loss on masked patches (Eq 3.18)
- [x] `worldmodels/eval/probes.py`:
  - Linear probe ($R^2$ score on ground truth physics state variables)
  - Forward-prediction test
  - Aliasing diagnostic (single-frame vs stacked 2-frame)

## 3. Unit Test Suite (`tests/`)
- [x] `tests/test_augment.py`: Spatial shift and photometric jitter constraints
- [x] `tests/test_encoders.py`: ConvEncoder, ProjectionHead, SimpleMAE, and unified Agent contract
- [x] `tests/test_losses.py`: NT-Xent, SimSiam, BYOL EMA, Barlow Twins, VICReg, Masked MSE
- [x] `tests/test_probes.py`: Closed-form Ridge regression linear probe and $R^2$ evaluation
- [x] `tests/test_dmc_data.py`: DMC dataset loading, multi-frame stacking, episode boundary safety
- [x] Test suite execution: 25/25 passing tests in `python -m unittest discover tests`

## 4. Chapter 3 Scripts (`ch03/`)
- [x] `ch03/01_collect_dmc_data.py`: Collect episodes for `cartpole_balance`, `finger_spin`, `cheetah_run`; cache to `.npz`; generate Figure 3.2
- [x] `ch03/02_views_and_pairs.py`: Positive/negative view augmentations and pair visualization (Figure 3.4)
- [x] `ch03/03_ntxent_from_scratch.py`: Standalone mathematical verification of NT-Xent / InfoNCE against equations (Eq 3.3-3.7)
- [x] `ch03/04_train_contrastive.py`: Contrastive learning training loop across seeds 0, 1, 2; loss curve (Figure 3.6), latent 2D projection (Figure 3.7)
- [x] `ch03/05_aliasing_diagnostic.py`: Measure position vs velocity probe $R^2$ on 1-frame vs 2-frame inputs (Table 3.3)
- [x] `ch03/06_collapse_demo.py`: Demonstrate collapse without negatives; std across dimensions (Figure 3.9)
- [x] `ch03/07_three_mechanisms.py`: Compare stop-gradient (SimSiam), EMA (BYOL), redundancy reduction (Barlow Twins)
- [x] `ch03/08_vicreg.py`: Train with VICReg across seeds 0, 1, 2; verify against contrastive probe results (Table 3.4)
- [x] `ch03/09_masked_autoencoder.py`: Implement patch-based MAE; train, reconstruct (Figure 3.12), evaluate with linear probe (Table 3.4)
- [x] `ch03/10_bottleneck_sweep.py`: Latent dimension sweep $d \in \{2, 4, 8, 16, 64, 256\}$ on `cartpole_balance` and `cheetah_run` (Figure 3.14)
- [x] `ch03/11_temporal_cpc.py`: Temporal Contrastive Predictive Coding with GRU context; unobserved intervals (Figure 3.16) and forward prediction (Table 3.5)
- [x] `ch03/12_assemble_encoder.py`: Final encoder assembly and integration into Chapter 1 agent skeleton; synthesis summary (Table 3.6)
- [x] `ch03/table_03_04_comparison.py`: Comparative evaluation across Supervised, NT-Xent, VICReg, and MAE paradigms across two tasks (Table 3.4)

## 5. Execution & Artifact Generation
- [x] Run scripts across seeds 0, 1, 2
- [x] Save all figures to `ch03/figures/` (B/W-safe, no color-only encoding)
- [x] Save all machine-readable tables/results to `ch03/results/`
- [x] Review discrepancies between design document and mathematical/experimental reality (`ch03/discrepancies_and_findings.md`)
- [ ] Git commit and record commit hash
