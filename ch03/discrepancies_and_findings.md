# Chapter 3 Discrepancies, Code Audit, and Experimental Findings Review

**Document Purpose:** Post-experimental audit and scientific review of Chapter 3 (*Representation Learning Before Generation*). This document reconciles original design hypotheses with empirical reality across seeds {0, 1, 2}, corrects earlier mathematical and numerical documentation errors, clarifies methodology, and establishes scientifically rigorous, defensible claims for the manuscript.

---

## 1. Linear Probe Epistemology: Unobservable State Variables and Negative $R^2$

### 1.1 Out-of-Sample $R^2$ Formulation
The out-of-sample coefficient of determination ($R^2$) for each physical state variable $k$ is computed as:

$$\text{Eval } R^2_k = 1 - \frac{\sum_{i=1}^N (y_{i, k} - \hat{y}_{i, k})^2}{\sum_{i=1}^N (y_{i, k} - \bar{y}_{k, \text{eval}})^2}$$

where $\bar{y}_{k, \text{eval}} = \frac{1}{N}\sum_{i=1}^N y_{i, k}$ is the sample mean of target variable $k$ on the held-out evaluation (validation) split, exactly as implemented in `worldmodels/eval/probes.py::compute_r2_score()`.

> [!NOTE]
> **Correction from Earlier Draft:** An earlier draft of this review incorrectly wrote $\bar{y}_{\text{train}}$ in the denominator. The codebase uses the ordinary evaluation-set mean $\bar{y}_{k, \text{eval}}$.

When a fitted linear probe yields a mean squared error greater than the sample variance of the validation target (i.e., predicting worse out-of-sample than the constant evaluation-split mean), $R^2$ is negative. A negative $R^2$ is standard for out-of-sample evaluation when an estimator fails to outperform the zero-information marginal mean baseline.

### 1.2 Separation of Physics/Rendering Claim from Probe Readout
There are two distinct claims regarding velocity unobservability:

1. **Physical / Rendering Claim:** A static MuJoCo visual frame is rendered strictly as a function of generalized configuration coordinates $q \in \mathbb{R}^{n_q}$. Without temporal motion blur, optical flow, or consecutive frame differencing, generalized velocities $\dot{q} \in \mathbb{R}^{n_v}$ are physically absent from an instantaneous static image.
2. **Experimental Linear Probe Claim:** Linear Ridge probes trained on single-frame latent vectors $z$ achieve negative $R^2$ on velocities ($R^2 \approx -3.18 \pm 0.69$ for pole angular velocity in Table 3.3).

The negative probe score does not itself theoretically prove unobservability; it merely demonstrates that linear decoding from the learned latent space fails relative to the mean predictor. The rendering mechanics explain *why* velocity is absent from a single frame.

### 1.3 Decomposed Reporting vs. Aggregate Means
Averaging $R^2$ across all state variables combines high positive position decodability ($R^2 \approx 0.60\text{--}0.75$) with negative unobservable velocity scores, pulling aggregate means below zero (e.g., $-1.089$ for NT-Xent in Table 3.4). All manuscript tables and discussions must report decomposed position vs. velocity metrics rather than treating aggregate mean $R^2$ as a standalone representation quality score.

---

## 2. Temporal Aliasing Diagnostic: Empirical Negative Result (Table 3.3)

### 2.1 Design Hypothesis
Section 3.3.5 hypothesized that while a single frame suffers from temporal aliasing (poor velocity decodability), stacking two immediately consecutive frames $(x_{t-1}, x_t)$ would provide sufficient temporal context to linearly decode velocity.

### 2.2 Empirical Reality
Table 3.3 compiled by `ch03/05_aliasing_diagnostic.py` across seeds {0, 1, 2}:

| State Variable | Single-Frame (1-frame) | Stacked-Frame (2-frame) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Cart position** ($x$) | $+0.132 \pm 0.040$ | $\mathbf{+0.264 \pm 0.045}$ | $+0.132$ (Improved) |
| **Pole angle** ($\theta$) | $-0.942 \pm 0.065$ | $-0.852 \pm 0.079$ | $+0.090$ (Slight improvement) |
| **Cart velocity** ($\dot{x}$) | $+0.049 \pm 0.004$ | $+0.041 \pm 0.060$ | $-0.008$ (Flat / poor) |
| **Pole angular velocity** ($\dot{\theta}$) | $\mathbf{-3.179 \pm 0.690}$ | $\mathbf{-3.661 \pm 0.045}$ | $-0.482$ (Worsened) |

### 2.3 Scientific Finding
**Stacking two immediately consecutive frames did NOT recover velocity.** In fact, pole angular velocity probe decodability worsened.

**Mechanism & Explanation:**
1. Under default DMC cartpole physics ($dt = 0.01\text{ s}$), the pole rotates by an average of only $\sim 0.07\text{ rad}$ per step, corresponding to subpixel displacement ($<1\text{ px}$) at $64\times 64$ resolution.
2. During contrastive pre-training, the view pipeline applies spatial translations of up to $\pm 3\text{ pixels}$ (`random_crop_shift`).
3. Because the spatial jitter is larger than the inter-frame motion, and the NT-Xent objective optimizes representation invariance across crops of the 6-channel stacked input, the encoder treats subtle inter-frame differences as noise and discards them.
4. Consequently, two immediately adjacent $64\times 64$ frames under this contrastive setup are insufficient for the learned representation to linearly expose velocity.

This is a valuable negative result: resolving temporal aliasing cannot be achieved by naive 2-frame stacking under spatial-crop contrastive learning; it requires multi-step recurrent context (CPC with GRU) or action-conditioned transition modeling.

---

## 3. Objective Comparison Analysis (Table 3.4)

Results compiled by `ch03/table_03_04_comparison.py` over seeds {0, 1, 2} on 3,500 training frames:

| Learning Paradigm | Cartpole Balance (4-D State) | Finger Spin (6-D State) |
| :--- | :---: | :---: |
| **Supervised State-Prediction Reference** | $-1.621 \pm 0.013$ | $-0.047 \pm 0.050$ |
| **Contrastive Learning (NT-Xent)** | **$-1.089 \pm 0.041$** | **$+0.013 \pm 0.026$** |
| **Non-Contrastive (VICReg)** | **$-1.157 \pm 0.056$** | **$+0.028 \pm 0.024$** |
| **Masked Autoencoder (MAE)** | $-1.432 \pm 0.023$ | $-14.252 \pm 19.727$ |

### 3.1 Terminology: "Supervised Reference" vs. "Upper Bound"
The supervised model is trained for 12 epochs with a linear readout probe on privileged states. Calling this an "upper bound" is unjustified; it represents a **supervised state-prediction reference** under this particular training budget and architecture.

### 3.2 NT-Xent vs. VICReg Equivalence Scope
NT-Xent and VICReg achieve very similar linear probe performance under this specific evaluation protocol (DMC frames, ConvEncoder, 12 epochs, Ridge probe). The manuscript should not claim a universal theoretical equivalence between variance/covariance regularization and instance discrimination, but rather report that *under this specific setup, both paradigms learn representations with comparable linear state decodability*.

### 3.3 Masked Autoencoder (MAE) Architecture & Dynamics
The MAE baseline uses a lightweight patch-based architecture with 75% patch masking, where the encoder outputs a pooled $16$-D representation $z$ for probing, and the decoder reconstructs masked patches by un-shuffling visible tokens and learnable mask tokens using `ids_restore`.

1. **Reconstruction vs. State Sufficiency:** Visual reconstructions (Figure 3.12) demonstrate high-fidelity pixel restoration of predictable backgrounds and geometric bodies. However, pixel reconstruction pays for high-frequency visual elements rather than abstract state variables.
2. **Sensitivity on `finger_spin`:** In `finger_spin`, state variables depend on the precise angular position and velocity of a small revolving tip. When patches containing the tip are masked, the pooled representation cannot reliably retain tip orientation, resulting in high seed variance (Seed 2 $R^2 = -42.15$).

---

## 4. Latent Bottleneck Capacity Sweep (Figure 3.14)

### 4.1 Design Hypothesis vs. Empirical Curve
Section 3.6.2 predicted that linear probe $R^2$ would monotonically increase with latent dimension $d$ and then plateau at task dimensionality.

### 4.2 Empirical Reality
Across latent dimensions $d \in \{2, 4, 8, 16, 64, 256\}$:
- **Cartpole Balance (4-D state):** Performance stabilizes between $d=8$ and $d=16$ ($R^2 \approx -1.30$). At higher dimensions, probe generalization degrades significantly: $d=64 \implies R^2 \approx -1.87$; $d=256 \implies R^2 \approx -4.29$.
- **Cheetah Run (18-D state):** Small dimensions ($d=2, 4$) exhibit severe aliasing ($R^2 \approx 0.02\text{--}0.11$). Decodability peaks at $d=16$ ($R^2 \approx 0.25 \pm 0.10$). Dimensions $d=64$ and $d=256$ show instability and degradation (Seed 1 at $d=64$ drops to $-52.05$).

### 4.3 Scientific Nuance: Finite-Sample Probe Overparameterization
The curve does not simply plateau. In finite-data regimes ($N=3,000$ training frames), fitting a linear probe on large latent representations ($d=64, 256$) causes overparameterization and ill-conditioned Ridge regression normal equations, penalizing out-of-sample generalization. The empirical curve is concave, demonstrating both minimal capacity requirements and probe overparameterization penalties.

---

## 5. Temporal CPC, Latent Forward Rollout, and Action Conditioning (Table 3.5 & Figure 3.16)

### 5.1 Corrected CPC Metrics (Table 3.5)
Metrics recorded across seeds {0, 1, 2} in `ch03/results/table_03_05_cpc_metrics.json`:

| Representation Type | Position $R^2$ (Mean ± Std) | Velocity $R^2$ (Mean ± Std) |
| :--- | :---: | :---: |
| **Static Single-Frame ($z_t$)** | $+0.711 \pm 0.024$ | $+0.004 \pm 0.011$ |
| **Temporal Context GRU ($c_t$)** | $+0.706 \pm 0.097$ | $-0.005 \pm 0.055$ |
| **Next-State Forward Pred ($c_t \to s_{t+1}$)** | \multicolumn{2}{c|}{$-1.065 \pm 0.055$} |

> [!NOTE]
> **Numerical Correction:** An earlier summary incorrectly reported context position $R^2$ as $0.71 \pm 0.12$. The exact measured values are static position $0.711 \pm 0.024$ and context position $0.706 \pm 0.097$ (individual seeds: $0.733, 0.809, 0.577$).

### 5.2 Methodological Fix for Unobserved Interval Tracking (Figure 3.16)
Standard CPC optimizes future latent compatibility via a bilinear InfoNCE score:

$$f(z_{t+k}, c_t) = \exp(z_{t+k}^T W_k c_t)$$

InfoNCE optimizes ranking compatibility / mutual information, **not** Euclidean latent reconstruction ($W_0 c_t \not\approx z_{t+1}$). Treating $z_{\text{pred}} = c_t W_0^T$ directly as a simulated latent would be methodologically invalid.

To execute valid trajectory tracking across dropped-frame intervals (Figure 3.16), `ch03/11_temporal_cpc.py` implements an explicit forward latent predictor:

$$P(c_t) \approx z_{t+1}$$

trained with mean squared error alongside InfoNCE. This explicitly demonstrates the bridge to Chapter 4's Joint-Embedding Predictive Architecture (JEPA).

### 5.3 Nature of Transition Uncertainty: Hidden Actions vs. Stochastic Physics
The environment transition is ambiguous not because MuJoCo physics are stochastic—the underlying rigid-body dynamics are strictly deterministic given state $s_t$ and action $a_t$. Rather, the uncertainty arises from **hidden actions** sampled stochastically under the random exploration policy. Without action conditioning, any forward model faces an irreducible uncertainty ceiling, motivating Chapter 4's action-conditioned transition predictor $T(z_t, a_t)$.

---

## 6. Synthesis Summary for Manuscript Drafting

When drafting Chapter 3, adhere to these guidelines:
1. **Formula:** State the out-of-sample $R^2$ using the evaluation-set mean $\bar{y}_{\text{eval}}$.
2. **Velocity:** Distinguish physical rendering limitations from empirical probe failure.
3. **Aliasing:** Present 2-frame stacking as an empirical negative result explaining the role of subpixel displacement and spatial augmentations.
4. **Baseline:** Use "Supervised State-Prediction Reference" rather than "Upper Bound".
5. **Comparison:** Characterize NT-Xent and VICReg as having similar performance under this specific evaluation.
6. **Bottlenecks:** Document the concave capacity curve and finite-sample probe overparameterization penalty.
7. **Predictive Modeling:** Clarify that InfoNCE provides compatibility discrimination, while explicit latent forward prediction $P(c_t) \approx z_{t+1}$ bridges CPC to Chapter 4 JEPA under hidden-action ambiguity.
