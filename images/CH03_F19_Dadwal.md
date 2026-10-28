# Figure 3.13: VICReg, SIGReg and VISReg constrain different aspects of feature distributions

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout in the artwork.

**Purpose:** Compare the mathematical object constrained by three distributional regularizers. VICReg constrains second-order feature statistics through a standard-deviation floor and off-diagonal covariance penalty. SIGReg compares empirical characteristic functions of random one-dimensional projections with the standard-normal characteristic function. VISReg separately constrains scale, projected distribution shape through sorted-sample Gaussian-quantile matching, and per-view minibatch center. Show that these approaches examine different statistical properties and that finite batches and finite projections only approximate distributional constraints.

**Format and print legibility:** Use a wide landscape canvas approximately 3600 × 2600 px. White background, dark charcoal text and outlines, and #3B6FB6 as the only restrained accent. Use a carefully typeset, flat 2D academic figure style suitable for a Manning technical textbook. At original resolution, target approximately 64–76 px for principal headings and labels and 52–60 px for equations and secondary notes. Mathematical formulas are dense, so allocate generous space; do not miniaturize them to preserve a decorative layout. If necessary, simplify secondary prose rather than shrinking formulas. All essential content must be legible at printed book-column width.

## Overall composition

Create three aligned panels with matching hierarchy:
1. **VICReg — second-order moments**
2. **SIGReg — characteristic functions of random projections**
3. **VISReg — scale, projected shape, and center**

Each panel should have one main conceptual visual and a compact mathematical signature. Use shared simple abstract feature points as visual aids, not as data or evidence. Do not make a ranking, and do not imply the regularizers are interchangeable.

Use a lower strip for two brief context notes:
- “Finite batches and finite projections approximate the distributional constraints; they do not certify exact Gaussianity.”
- “LeJEPA combines a predictive objective with SIGReg; it is not a fourth regularizer here.”

Keep both notes secondary but readable. Do not suggest LeJEPA is equivalent to VISReg.

## Panel 1: VICReg — second-order moments

Show a small batch of learned feature vectors, with two clearly separated statistical cues:

- **Variance floor:** one conceptual feature coordinate has values tightly concentrated around a constant, while another has adequate spread. Label the per-dimension standard-deviation margin \(\gamma\). The low-variance dimension receives a positive hinge penalty; a dimension at or above the margin contributes zero to this term.
- **Off-diagonal covariance:** show a small within-view covariance matrix. De-emphasize diagonal cells and highlight off-diagonal entries as the penalized terms. Do not use a cross-view correlation matrix here.

Use the concise labels:
“Standard-deviation floor”
“Penalize off-diagonal within-view covariance”

Show a compact formula block:
\[
v(Z)=\frac1d\sum_{j=1}^{d}
\max\!\left(0,\gamma-\sqrt{\mathrm{Var}(Z_j)+\epsilon}\right),
\qquad
c(Z)=\frac1d\sum_{i\ne j}\mathrm{Cov}(Z)_{ij}^{2}.
\]

Label this panel “Second-order statistics.” Make clear that VICReg regularizes the standard deviations and covariance entries rather than directly matching an entire distribution to a Gaussian. A small note may state: “Adequate variance and low linear covariance do not guarantee semantic information.”

Do not show Barlow Twins' cross-view correlation matrix as though it were VICReg's covariance matrix. Do not suggest zero covariance implies independence.

## Panel 2: SIGReg — characteristic-function discrepancy

Show a set of feature points being projected along a few randomly chosen unit directions \(w_1,\ldots,w_K\). Each projection turns multidimensional samples into a one-dimensional sample distribution. Use several simple direction arrows or small projection lines, but avoid a busy scatter plot or fabricated empirical histogram.

Show that for a projected sample \(u_b=w_k^\top z_b\), SIGReg compares the empirical characteristic function with the standard-normal characteristic function:
\[
\widehat\varphi_B(t)=\frac1B\sum_{b=1}^{B}e^{itw_k^\top z_b},
\qquad
\varphi_{\mathcal N(0,1)}(t)=e^{-t^2/2}.
\]

Below this, show a compact discrepancy signature:
\[
\mathcal L_{\mathrm{SIGReg}}(Z)
\propto \frac1K\sum_{k=1}^{K}\int_{\mathbb R}
\left|\widehat\varphi_{B,k}(t)-e^{-t^2/2}\right|^2\omega(t)\,dt.
\]

The proportional sign indicates a schematic structural summary; label it “Schematic discrepancy; implementation specifies quadrature and weights.” If room permits the full integral with its exact averaging factors, use the source form:
\[
\mathcal L_{\mathrm{SIGReg}}(Z)
=\frac1K\sum_{k=1}^{K}\int_{\mathbb R}
\left|
\frac1B\sum_{b=1}^{B}e^{itw_k^\top z_b}
-e^{-t^2/2}
\right|^2\omega(t)\,dt.
\]

Do not show both long versions if this makes the panel cluttered. Prioritize the empirical characteristic function, the standard-normal reference \(e^{-t^2/2}\), and the comparison over random projections.

Label this panel:
“Compare projected characteristic functions with the standard-normal reference.”

A small conceptual pair of curves may be used, but it must be labeled schematic and must not contain invented measurements. Avoid detailed axes or numerical scales unless they are mathematically meaningful.

Make clear that the Cramér–Wold motivation concerns all one-dimensional projections, whereas a finite number of random projections and finite-sample/quadrature approximations do not prove exact multivariate Gaussianity.

## Panel 3: VISReg — scale, shape, and minibatch center

Give this panel a little more vertical space than the others, because VISReg has three distinct terms. Show a shared batch of view-wise feature vectors \(Z_v\), with three separate branches or rows labeled:

1. **Scale:** each view's per-coordinate standard deviations \(\sigma_{v,j}\) are pulled toward one.
2. **Shape:** after centering and scale normalization with a detached scale vector, project along random unit directions, sort each projected batch, and match the sorted values to standard-normal quantiles.
3. **Minibatch center:** penalize the mean \(\mu_v\) of each view's minibatch.

### Scale term

Show the formula:
\[
\sigma_{v,j}=
\sqrt{\frac1B\sum_{b=1}^{B}\widehat z_{v,b,j}^{\,2}}+\epsilon,
\qquad
\mathcal L_{\mathrm{scale}}
=\frac1{Vd}\sum_{v=1}^{V}\sum_{j=1}^{d}(\sigma_{v,j}-1)^2,
\]
where \(\widehat z_{v,b,j}=z_{v,b,j}-\mu_{v,j}\).

Explain briefly:
“Match per-view feature scales to one.”

### Shape term

Show a clear small process diagram:
**Normalize by detached scale → project along random directions → sort projected samples → match Gaussian quantiles**.

Use:
\[
\widetilde z_{v,b,j}
=\frac{\widehat z_{v,b,j}}{\mathrm{sg}(\sigma_{v,j})}.
\]

Then display the shape objective:
\[
\mathcal L_{\mathrm{shape}}
=\frac1{VKB}\sum_{v=1}^{V}\sum_{k=1}^{K}
\left\|\mathrm{sort}(\widetilde Z_v w_k)-q_{1:B}\right\|_2^2,
\qquad
q_b=\Phi^{-1}\!\left(\frac{b}{B+1}\right).
\]

Keep the ordering step and quantile-matching target visually distinct from SIGReg's characteristic-function comparison. The two methods both use random projections but compare different objects: SIGReg compares characteristic functions; VISReg compares sorted projected sample values with Gaussian quantiles.

### Minibatch-center term

Show the per-view minibatch mean:
\[
\mu_{v,j}=\frac1B\sum_{b=1}^{B}z_{v,b,j},
\qquad
\mathcal L_{\mathrm{center}}
=\frac1{Vd}\sum_{v=1}^{V}\|\mu_v\|_2^2.
\]

Explain briefly:
“Penalize the per-view minibatch mean.”

### Complete VISReg regularizer

At the bottom of this panel or in a connected narrow equation area, show:
\[
\mathcal L_{\mathrm{Reg}}
=\lambda_{\mathrm{scale}}\mathcal L_{\mathrm{scale}}
+\lambda_{\mathrm{shape}}\mathcal L_{\mathrm{shape}}
+\lambda_{\mathrm{center}}\mathcal L_{\mathrm{center}}.
\]

If the panel becomes crowded, do not shrink any equation. Keep the full weighted regularizer and the key sorted-projection procedure, and reduce optional secondary formulas first.

### Crucial distinction: two different centers

Add a clear and short note, with a visual separation between two distinct concepts:

- **VISReg regularizer center:** per-view minibatch mean \(\mu_v\), penalized by \(\mathcal L_{\mathrm{center}}\).
- **View-invariance target:** same-image mean of global-view projections \(c_b\), used to match global and local views in the full training objective.

These are not the same operation. If a small side note is needed, show:
\[
c_b=\frac1{V_g}\sum_{v\in\mathcal V_g}z_{v,b}.
\]
Label it “Same-image global-view center for prediction/invariance, not minibatch centering.” Keep this clearly separate from \(\mu_v\) and \(\mathcal L_{\mathrm{center}}\).

Make sure the image does not suggest VISReg's minibatch-centering penalty uses the same-image global-view target.

## Shared lower notes

Across the bottom margin or in a short footer strip, include:
- “Finite batches and finite projections approximate the distributional constraints; they do not certify exact Gaussianity.”
- “LeJEPA combines prediction with SIGReg. Its predictive objective is not a fourth regularizer and is not equivalent to VISReg.”

Keep these notes legible without crowding the main equations. If space is tight, retain the first note and shorten the second to:
“LeJEPA = prediction + SIGReg; not a fourth regularizer.”

## Required conceptual accuracy

1. VICReg is shown as a second-order constraint: a standard-deviation floor and off-diagonal within-view covariance penalty. Do not suggest it matches a full distribution to a Gaussian.
2. SIGReg compares an empirical characteristic function to the standard-normal characteristic function over randomly sampled one-dimensional projections.
3. The Gaussian characteristic function is \(e^{-t^2/2}\).
4. VISReg matches sorted projected, scale-normalized samples to standard-normal quantiles and includes separate scale and minibatch-center penalties.
5. Keep the role of detached scale \(\mathrm{sg}(\sigma_{v,j})\) clear; this stop-gradient decouples scale and shape gradients and is not a BYOL-style teacher branch.
6. Distinguish SIGReg's characteristic-function discrepancy from VISReg's sorted-sample/quantile discrepancy.
7. Keep per-view minibatch mean \(\mu_v\) separate from same-image global-view mean \(c_b\) used for view invariance.
8. The same-image mean is built from global views of one source image, not averaged across different images.
9. LeJEPA combines a predictive objective with SIGReg; do not represent it as a fourth regularizer or as equivalent to VISReg.
10. Finite projections, finite batches, quadrature, and normalization approximate the distributional constraints; they do not prove exact multivariate Gaussianity.
11. Do not rank the methods or claim any regularizer guarantees semantic sufficiency or optimal downstream performance.
12. Do not invent numerical measurements, fitted distributions, benchmark outcomes, or experimental plots.

## Visual style and restrictions

Use a white background, grayscale-safe dark line work, and #3B6FB6 as the only accent. Flat 2D geometry, simple projected point sketches, clear process arrows, well-spaced equations, and direct labels.

Avoid crowded matrix textures, tiny formulas, dense paragraphs, decorative watermarks, gradients, shadows, 3D effects, Mermaid, dashboards, simulated page layouts, extra colors, icons suggesting semantic categories, and invented data. Do not draw a fitted Gaussian curve as if it were an empirical result; any reference curve must be labeled theoretical or schematic.

## Acceptance checks

1. VICReg is visually identified with variance and off-diagonal covariance rather than full distribution matching.
2. SIGReg clearly shows random one-dimensional projections and characteristic-function comparison with \(e^{-t^2/2}\).
3. VISReg clearly shows scale normalization, projection, sorting, Gaussian-quantile matching, and the per-view minibatch-center penalty.
4. SIGReg and VISReg's discrepancy mechanisms are visually distinct.
5. The per-view minibatch mean used by the center penalty is not conflated with the same-image global-view center used for invariance.
6. LeJEPA's relationship to SIGReg is a small note, not a fourth panel or fourth regularizer.
7. Finite projections and batches are described as approximations, not a proof of exact Gaussianity.
8. All essential formulas and labels remain legible at printed book-column width.
9. The figure contains no figure number, title, caption, invented empirical result, or performance ranking.

## Caption

Figure 3.13. VICReg, SIGReg and VISReg constrain different aspects of feature distributions.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.