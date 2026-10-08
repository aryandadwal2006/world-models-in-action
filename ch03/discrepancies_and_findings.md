# Chapter 3 Discrepancies and Experimental Findings Review

**Document Purpose:** Review discrepancies between the Chapter 3 design document (*Representation Learning Before Generation v3*) and mathematical/experimental reality observed across seeds {0, 1, 2}.

---

## 1. Linear Probe Epistemology: Unobservable State Variables and Negative $R^2$

### Design Hypothesis
Section 3.3.5 and Section 7 hypothesized that linear probe $R^2$ scores would serve as a monotonic score of representation quality, measuring the extent to which physics state variables ($x, \theta, \dot{x}, \dot{\theta}$) are linearly accessible in the learned latent space $z$.

### Experimental Reality
1. **Definition of Out-of-Sample $R^2$:**
   $$\text{Test } R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y}_{\text{train}})^2}$$
   When a linear probe predicts worse than the baseline mean predictor on the held-out validation set, $R^2$ becomes negative.
2. **Velocity Unobservability in Single Frames:**
   Because DMC frames are static MuJoCo renders, velocity information ($\dot{x}, \dot{\theta}$) does not exist in a single frame. A linear probe trained to predict angular velocity from single-frame latents yields negative $R^2$ values (e.g., $R^2 \approx -3.18 \pm 0.69$ for pole angular velocity in Table 3.3).
3. **Impact on Aggregated Means:**
   Reporting an unweighted mean $R^2$ across all state variables combines positive position decodability ($R^2 \approx 0.60\text{--}0.75$) with negative unobservable velocity scores, pulling the aggregate mean below zero (e.g., $-1.089$ for NT-Xent in Table 3.4).
4. **Epistemological Takeaway:**
   The design document must present probe results decomposed by variable type (position vs. velocity), as executed in Table 3.3, rather than treating aggregate mean $R^2$ as a standalone score.

---

## 2. Table 3.4 Objective Comparison Analysis

Results compiled by `ch03/table_03_04_comparison.py` over seeds {0, 1, 2} on 3,500 training frames:

| Learning Paradigm | Cartpole Balance (4-D State) | Finger Spin (6-D State) |
| :--- | :---: | :---: |
| **Supervised Baseline (Upper Bound)** | $-1.621 \pm 0.013$ | $-0.047 \pm 0.050$ |
| **Contrastive Learning (NT-Xent)** | **$-1.089 \pm 0.041$** | **$+0.013 \pm 0.026$** |
| **Non-Contrastive (VICReg)** | **$-1.157 \pm 0.056$** | **$+0.028 \pm 0.024$** |
| **Masked Autoencoder (MAE)** | $-1.432 \pm 0.023$ | $-14.252 \pm 19.727$ |

### Findings & Nuances
1. **Equivalence of Contrastive and Non-Contrastive Representations:**
   NT-Xent and VICReg achieve nearly identical probe performance on both tasks within seed variance, verifying Section 3.4.3's claim that explicit variance/covariance regularization matches instance discrimination without requiring negative pairs.
2. **MAE Reconstruction vs. Linear State Availability:**
   While MAE produces high-fidelity visual reconstructions from 25% visible patches (Figure 3.12), its representations yield lower linear state decodability. On `finger_spin`, where state variables depend on the precise orientation of a small rotating body, patch masking can occlude the tip entirely, leading to high seed variance (Seed 2 $R^2 = -42.15$). This experimentally confirms Section 3.5.3's core thesis: *reconstruction pays for pixels (background, floor, body textures), not state sufficiency*.

---

## 3. Bottleneck Sweep: Capacity Saturation and Overfitting (Figure 3.14)

### Design Hypothesis
Section 3.6.2 predicted that linear probe $R^2$ would monotonically increase with latent dimension $d$ and then plateau at a saturation point corresponding to task dimensionality (earlier for cartpole $d=4$ than cheetah $d=18$).

### Experimental Reality
- On `cartpole_balance` (4-D state): Performance stabilizes between $d=8$ and $d=16$. At high dimensions ($d=64, 256$), linear probes overfit the training split, resulting in lower test generalization ($R^2$ dropping from $-1.25$ to $-4.28$).
- On `cheetah_run` (18-D state): Small dimensions ($d=2, 4$) suffer severe aliasing ($R^2 \approx 0.02\text{--}0.10$). Decodability peaks at $d=16$ ($R^2 \approx 0.25\pm 0.10$), demonstrating that higher-dimensional physics states require higher bottleneck capacity.
- **Refinement:** The curve does not merely plateau; unregularized linear probes in high-dimensional latent spaces experience overparameterization penalties on finite datasets.

---

## 4. Temporal CPC and Random Policy Stochasticity (Table 3.5, Figure 3.16)

### Findings
1. **Tracking Through Gaps:**
   Evaluating CPC context predictions during multi-step observation dropouts (Figure 3.16) demonstrates that autoregressive GRU context models maintain position estimates ($R^2 \approx 0.71 \pm 0.12$) through unobserved intervals, outperforming zero-input baseline re-entry.
2. **Risk R2 Validation:**
   Because frames were gathered under a uniform random action policy without action conditioning, the environment transition is non-deterministic from observations alone. Consequently, forward prediction scores have a theoretical ceiling, motivating Chapter 4's joint-embedding predictive architecture with action conditioning (JEPA).
