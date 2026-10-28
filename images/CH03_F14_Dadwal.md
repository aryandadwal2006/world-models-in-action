# Figure 3.9: BYOL and SimSiam: similar predictors, different target updates

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, panel letters, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout inside the artwork.

**Purpose:** Compare the asymmetric prediction objectives of BYOL and SimSiam without conflating their target mechanisms. Both methods predict a representation of one view from another view and use stop-gradient on the target in each loss term. BYOL has a distinct target encoder and projector updated by an exponential moving average (EMA); SimSiam shares its encoder and projector between both views and uses no EMA target network. The diagrams must show the computational paths and their gradient rules, not suggest a universal proof against collapse.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2200 px, white background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Aim for a rigorous, flat 2D academic schematic suitable for a Manning technical textbook. At original resolution, use approximately 64–76 px for main labels and 52–60 px for equations and supporting notes. Arrows and stop-gradient symbols must remain visible at print size. Use aligned components, consistent typography and line weights, and generous whitespace. Do not shrink equations or crowd branch labels.

## Composition

Create two balanced side-by-side panels titled **BYOL** and **SimSiam**. Keep the view inputs, encoder / projector, predictor, target representation, stop-gradient marker, prediction direction and loss relation easy to follow. Use similar geometry across panels so their difference is immediate.

Both methods use two augmented views \(x_1\) and \(x_2\). Their representations are matched across views, and the loss is symmetric across the two directions. Use blue emphasis sparingly for the prediction target or target comparison; distinguish gradient and stop-gradient with explicit labels and arrow styles as well as color.

### Left panel: BYOL

Draw two separate but coordinated branches.

**Online branch:** Show view \(x_1\) passing through an **Online encoder** \(f_\theta\), a **Projection head** \(g_\phi\), and a **Predictor** \(q_\psi\). Label the intermediate representation \(z_1=g_\phi(f_\theta(x_1))\) and prediction \(p_1=q_\psi(z_1)\).

Show the symmetric online path for \(x_2\), producing \(z_2\) and \(p_2\). Both online branches receive gradients from their corresponding prediction losses.

**Target branch:** Beneath or alongside the online path, show a separate **Target encoder + projector** with parameters \(\bar\theta,\bar\phi\). It processes the other view to produce a target representation \(\bar z_2=g_{\bar\phi}(f_{\bar\theta}(x_2))\) for the first direction. Include the symmetric target \(\bar z_1\) for the second direction. The target representations must be produced by a separate target network, not the online network.

Put a visible **stop-gradient** marker on the target representation immediately before its comparison with the prediction. Do not draw a backpropagation arrow into the target encoder or target projector.

Show the two matching directions:
\[
p_1\longrightarrow \mathrm{sg}(\bar z_2),
\qquad
p_2\longrightarrow \mathrm{sg}(\bar z_1).
\]
The arrows indicate which prediction is compared with which detached target, not data generation.

Show the BYOL loss compactly:
\[
L_{\mathrm{BYOL}}
=D(p_1,\mathrm{sg}(\bar z_2))
+D(p_2,\mathrm{sg}(\bar z_1)),
\]
with one directional normalized regression term available as a secondary formula:
\[
D(p,z)=2-2\frac{p^\top z}{\|p\|_2\|z\|_2}.
\]
If space is limited, prioritize the full symmetric loss and include a short note that \(D\) is a normalized similarity-based regression loss.

Separate from the forward and gradient pathways, show the EMA parameter updates from online parameters to the target parameters:
\[
\bar\theta\leftarrow m\bar\theta+(1-m)\theta,\qquad
\bar\phi\leftarrow m\bar\phi+(1-m)\phi.
\]
Label the arrow **EMA parameter update after the online gradient step**. Make clear that the target's parameters are updated by EMA, not by backpropagation. The EMA arrow must not be confused with the prediction arrows or queue operations from other algorithms.

Add a short label:
“Separate target network; EMA update.”

### Right panel: SimSiam

Show two augmented views \(x_1\) and \(x_2\) flowing through a **shared encoder + projector** \(g\circ f\). Clearly indicate that the same shared network processes both views. The resulting representations are \(z_1\) and \(z_2\).

Show a **Predictor** \(q\) on each direction:
\[
p_1=q(z_1),\qquad p_2=q(z_2).
\]
The two predictors may be drawn as duplicate instances of the same shared predictor \(q\), or one predictor block reused in a visually clear way.

Show the symmetric target comparisons:
\[
p_1\longrightarrow\mathrm{sg}(z_2),
\qquad
p_2\longrightarrow\mathrm{sg}(z_1).
\]
Place a stop-gradient marker on the target-side representation in each loss term. Do not show a separate EMA target encoder. In SimSiam, the forward value of a target representation is used, but gradient propagation through that target side of the corresponding term is stopped. The shared view branch still receives gradients when it serves as the predictor-side branch in the opposite term.

Show the SimSiam loss:
\[
L_{\mathrm{SimSiam}}
=D_{\mathrm{SimSiam}}(p_1,\mathrm{sg}(z_2))
+D_{\mathrm{SimSiam}}(p_2,\mathrm{sg}(z_1)),
\]
with
\[
D_{\mathrm{SimSiam}}(p,z)
=-\frac{p^\top z}{\|p\|_2\|z\|_2}.
\]
If space is tight, prioritize the symmetric loss and state that \(D_{\mathrm{SimSiam}}\) is negative cosine similarity.

Add a short label:
“Shared encoder; stop-gradient targets; no EMA target network.”

Do not depict a separate target encoder or an EMA parameter update in this panel.

## Cross-panel comparison

Use a small, restrained comparison strip at the bottom, with two short lines:
- **BYOL:** separate target encoder/projector updated by EMA.
- **SimSiam:** shared encoder/projector; no EMA target network.

Include a brief note if space permits:
“Stop-gradient defines a gradient path; it is not by itself a theorem against collapse.”

This note should not suggest that the methods are identical or that their success has been explained by a single component. Both methods include architectural and optimization choices beyond the stop-gradient marker.

## Gradient-path conventions

Keep the notation consistent across both panels:
- Solid dark arrows indicate forward computation.
- Clearly labeled backward / gradient arrows indicate the online branches that receive gradients.
- A stop-gradient marker (for example, \(\mathrm{sg}(\cdot)\) or a small barred gradient symbol) appears on target arguments of each individual loss term.
- A separately styled arrow labeled EMA parameter update appears only in BYOL.
- Use blue emphasis for selected matching paths or targets, but retain labels and line-style distinctions so the drawing reads in grayscale.

Do not draw backpropagation into a target branch for a loss term where that target is stop-gradient. In SimSiam, do not imply a whole view branch is permanently gradient-free: the same shared branch acts as the predictor-side representation in the opposite direction.

## Required conceptual accuracy

1. BYOL uses an online encoder, projector and predictor, alongside a separate target encoder and projector.
2. Only the online BYOL network receives loss gradients; target parameters are updated by EMA.
3. BYOL uses a symmetric objective across the two views, matching \(p_1\) to \(\mathrm{sg}(\bar z_2)\) and \(p_2\) to \(\mathrm{sg}(\bar z_1)\).
4. SimSiam uses shared encoder and projector parameters for both views, with a predictor on each branch, and has no EMA target network.
5. SimSiam's symmetric loss matches \(p_1\) to \(\mathrm{sg}(z_2)\) and \(p_2\) to \(\mathrm{sg}(z_1)\).
6. Stop-gradient removes the target-side gradient in a given term without changing the target's forward numerical value. In the symmetric SimSiam objective, each view representation also receives gradients when acting as the predictor-side branch in the opposite term.
7. Do not draw an EMA update for SimSiam or draw direct loss gradients into BYOL's target network.
8. Stop-gradient is not a proof that collapse is impossible. Do not claim that either schematic by itself establishes a universal anti-collapse guarantee.
9. Do not invent empirical values, training curves, performance comparisons, or claims of superiority.

## Visual style and restrictions

Use a white background, dark grayscale-safe line work, and #3B6FB6 as the only restrained accent. Flat 2D boxes, clean arrows, aligned mathematical notation, and consistent visual grammar across the two panels.

Do not use gradients, shadows, 3D effects, Mermaid, decorative watermarks, dashboards, simulated page layouts, unnecessary icons, additional color palettes, tiny footnotes, or dense prose blocks. Keep the diagrams academically precise and readable at printed book-column width.

## Acceptance checks

1. BYOL includes online encoder/projector/predictor and separate target encoder/projector.
2. BYOL clearly shows stop-gradient on target representations and EMA parameter updates for the target network.
3. SimSiam clearly shows shared encoder/projector, predictors on both views, stop-gradient target sides, and no EMA target network.
4. Both objectives show the correct symmetric prediction directions.
5. Gradient flow is distinguished from forward computation and EMA updating.
6. The figure does not imply that an entire SimSiam view branch is always detached.
7. The figure does not claim that stop-gradient alone guarantees absence of collapse.
8. All essential labels and equations remain legible at printed book-column width.
9. No figure number, title, or caption appears inside the artwork.

## Caption

Figure 3.9. BYOL and SimSiam: similar predictors, different target updates.

## Production note

Create original explanatory artwork, not a reproduction of a paper figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.