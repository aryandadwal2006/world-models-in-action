# 01 — The JEPA prediction problem: architecture and mathematics

## 1. The generic problem

Let an observation be \(x\). A masking or selection process chooses context positions \(C\) and target positions \(Y\). For an image, positions may be spatial patches. For a video, a position includes time and space. In action-conditioned settings, a future target may also depend on a supplied action sequence.

A generic architecture has three functional components:

1. A context encoder \(f_\theta\) maps the allowed context to features.
2. A target encoder \(f_\xi\) computes the feature targets at selected target positions.
3. A predictor \(g_\phi\) estimates those target features from context features, positions and any explicit conditioning variables.

A compact notation is:
\[
z_C=f_\theta(x_C,C), \qquad
h_Y=f_\xi(x)_Y,
\]
\[
\hat h_Y=g_\phi(z_C,C,Y).
\]

A simple feature regression objective is:
\[
\mathcal L_{\mathrm{pred}}
=\frac{1}{|Y|d}\sum_{j\in Y}
\left\|\hat h_j-\operatorname{sg}(h_j)\right\|_2^2.
\]

Here \(d\) is feature dimension and \(\operatorname{sg}\) denotes a stop-gradient operation. This is a generic template, not a claim that every paper uses identical normalization, targets, mask sampling, predictor or parameter updates. The paper-specific recipe must be stated.

The name “joint embedding” refers to comparing context-derived and target-derived outputs in a learned feature space. “Predictive” refers to estimating the target feature from the context. It does not imply that the model predicts a complete physical state or a realistic pixel sequence.

## 2. Why two encoders can appear in the diagram

The target branch must provide a target representation that the context branch cannot simply copy from the input. In masked architectures the context branch sees only the context tokens, while the target branch can process the full or less-restricted observation and select target positions afterwards.

These branches need not have separate, independently trained parameters in every method. Common design choices include:
- an online encoder and a target encoder updated by an exponential moving average;
- a shared or symmetric encoder with stop-gradient on one path;
- branches with different parameterization or modalities;
- a predictor that makes the context branch solve a non-trivial conditional task.

Never explain the generic diagram as if EMA were part of the definition of JEPA. EMA is a particular target-update mechanism.

## 3. The squared-error predictor and conditional means

For a random target feature \(H\) and context \(C=c\), a deterministic predictor that minimizes expected squared error solves:
\[
a^*(c)=\arg\min_a
\mathbb E[\|H-a\|_2^2\mid C=c].
\]

Expand the square around the conditional mean:
\[
\mathbb E[\|H-a\|^2\mid c]
=
\mathbb E[\|H-\mu(c)\|^2\mid c]
+\|a-\mu(c)\|^2,
\]
where \(\mu(c)=\mathbb E[H\mid C=c]\). The cross term vanishes because
\(\mathbb E[H-\mu(c)\mid c]=0\). Therefore:
\[
a^*(c)=\mathbb E[H\mid C=c].
\]

This has an important implication. If one context is compatible with several futures, the deterministic MSE-optimal predictor returns their conditional mean in feature space. That mean need not correspond to any single physically realizable future. Whether this is a problem depends on the target geometry and task. For planning under genuine multimodality, a distribution over futures or a latent stochastic variable may be more appropriate.

Do not oversell this derivation: it explains a property of squared-error regression, not a failure unique to JEPAs.

## 4. The prediction target is a modelling decision

The target encoder defines the feature space in which errors are measured. Its invariances and biases become part of the predictive task. If it ignores illumination, the predictor need not reproduce illumination differences. If it discards object position, the predictor may not be trained to preserve position. If it is insensitive to small contact events, a controller may not get the signal needed for manipulation.

The central contrast with pixel reconstruction is not that pixels are always useless. It is that a feature-space loss weights discrepancies according to the target encoder's representation rather than raw observation distance. The target must therefore be evaluated against the downstream task.

## 5. The mask defines conditional information

A context mask does more than remove data. It changes the conditional distribution that the predictor must model.

- Nearby spatial targets may be inferable from local texture and object continuity.
- Distant image targets require larger-scale context.
- Future video targets require temporal information.
- Action-conditioned targets require the model to account for the selected intervention.
- A target that is not inferable from context contributes irreducible uncertainty under the training distribution.

The manuscript should make clear what each mask permits the model to observe. A mask diagram is not merely an implementation detail; it defines the learning problem.

## 6. EMA target updates

For an online parameter vector \(\theta_t\), an EMA target update is:
\[
\xi_{t+1}=m_t\xi_t+(1-m_t)\theta_{t+1}.
\]

If \(m_t=m\) is constant, repeated substitution gives an exponentially weighted history:
\[
\xi_t=(1-m)\sum_{k=0}^{t-1}m^k\theta_{t-k}
+m^t\xi_0.
\]

Older online parameters receive geometrically smaller weights. This creates a slowly moving target. It does not constitute a general proof against collapse, and it is not required by all JEPA variants.

## 7. What low latent prediction loss tells us

A low value of \(\mathcal L_{\mathrm{pred}}\) means the predictor fits the chosen target under the chosen context, mask, sampling distribution and feature normalization. It does not show by itself that:
- the target features contain all task-relevant state;
- a hidden physical variable is recoverable;
- future predictions remain accurate under recursive rollout;
- action effects are identified;
- uncertainty is calibrated;
- the model generalizes to new environments;
- a planner can find successful or safe actions.

This distinction should be repeated at the end of each major architecture section, using specific examples rather than repeating a generic warning.

## 8. Mathematical objects to keep separate

| Object | What it measures | What it does not measure |
|---|---|---|
| Prediction loss | Feature-space target fit | Physical correctness in an unobserved environment |
| Variance/covariance loss | Selected batch statistics | Semantic usefulness |
| SIGReg statistic | Approximate projected distributional match | Exact Gaussianity from finite sketches |
| VISReg shape loss | Projected sample transport discrepancy | Task sufficiency or controllability |
| Linear/attentive probe | Readout accessibility under a protocol | All information potentially present |
| Planning success | Performance of a model–planner–environment system | Universal reliability outside that evaluation |

## 9. Main-text teaching sequence

The final manuscript should:
1. introduce the three components with Figure 4.2;
2. derive the generic objective;
3. derive the conditional mean in a short worked box;
4. explain the target update without implying all variants use it;
5. close by showing how mask selection and target design shape the prediction problem.

This file is the generic mathematical foundation. Architecture-specific details belong in the following method notes.
