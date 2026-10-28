# Figure 3.7: CPC: causal context predicts future latent candidates

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, panel letters, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout in the artwork.

**Purpose:** Explain Contrastive Predictive Coding (CPC) as causal sequence encoding followed by candidate classification of future latent representations. Past and current observations produce a context vector \(c_t\); separate horizon-specific scoring heads use that context to score the observed future latent against sampled candidates. This is a contrastive retrieval objective, not a recursive latent simulator or action-conditioned planner.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px. White background, dark charcoal text and line work, and #3B6FB6 as the only restrained accent. Style should resemble a carefully typeset figure from a strong machine-learning paper adapted for a Manning textbook: clean temporal layout, precise arrows, flat 2D components, consistent line weights, and generous whitespace. At the original resolution target approximately 64–76 px for principal labels and 52–60 px for equations and secondary notes. All time subscripts, horizon indices, superscripts, and vector labels must remain legible at printed book-column width.

## Composition

Create one left-to-right temporal diagram with three distinct regions:
1. Past and current sequence observations and their latent encodings.
2. A causal context network that aggregates only information available through time \(t\).
3. Horizon-specific candidate scoring for future latent targets.

Use a visible vertical or dashed temporal boundary immediately after \(t\). This boundary must make it unmistakable that future targets are outside the context computation.

### 1. Observation sequence and latent representations

Along the left side, draw a horizontal timeline of observations:
\[
x_1,\;x_2,\;\ldots,\;x_{t-1},\;x_t,\;x_{t+1},\;x_{t+2},\ldots
\]

Use simple consistent tiles, not detailed scenes. Draw past and current observations to the left of a clear boundary at \(t\), and future observations to the right. Highlight \(x_t\) or the boundary with #3B6FB6.

Show a shared encoder \(g_{\mathrm{enc}}\) mapping observations to latent vectors:
\[
z_i=g_{\mathrm{enc}}(x_i).
\]
If displaying the encoder on each time step would clutter the diagram, show one encoder block with repeated arrows to the latent sequence rather than duplicating many blocks.

Label the latent sequence:
\[
z_1,\;z_2,\;\ldots,\;z_t \quad\vert\quad z_{t+1},\;z_{t+2},\ldots
\]

Use a clear, subtle divider between \(z_t\), which is available to the context, and the future targets, which are not supplied to it.

### 2. Causal context

Send only \(z_1,\ldots,z_t\) into a box labeled **Causal sequence encoder** or **Autoregressive context model**. The block outputs a context vector \(c_t\).

Show arrows from the available latents into the context block, and an arrow from that block to \(c_t\). The graph must have no arrows from \(z_{t+1}\), \(z_{t+2}\), or any future latent target back into the context model.

Add a concise label:
“Context uses observations available through time \(t\).”

Do not prescribe a particular recurrent architecture unless it can be shown as an optional generic causal sequence model. Do not turn the context block into a detailed transformer architecture tutorial.

### 3. Horizon-specific future-latent scoring

From \(c_t\), draw separate branches to a small set of horizon-specific scoring heads, such as \(W_1\), \(W_2\), and \(W_3\). Label each branch with its prediction horizon \(k\).

For each horizon \(k\), show the context being compared against the observed target \(z_{t+k}\) and alternative sampled latent candidates. Keep the correct future latent clearly marked as the **positive target** for that horizon; show the alternative candidates in dark grayscale-safe styling and label them **sampled negatives** or **sampled candidates**.

For a compact representative score, display:
\[
s_k(c_t,z)=z^\top W_k c_t.
\]
Make clear that the score is evaluated for each candidate latent \(z\), and the observed future latent \(z_{t+k}\) is the positive candidate. The same \(W_k\) scores the positive and the negatives at that horizon.

A small example candidate row for horizon \(k=1\) is sufficient if the diagram becomes crowded. It should show \(z_{t+1}\) as the positive and other sampled latent vectors as alternatives. Then use a bracket or aligned smaller branches to indicate the same process at \(k=2\) and \(k=3\), each with its own scoring head \(W_k\). Do not add made-up scores or probabilities.

Include one concise label near the candidate comparison:
“Identify the observed future latent among sampled candidates.”

If space permits, include an optional high-level loss cue without a full derivation:
“Contrastive classification is repeated across prediction horizons.”
Do not add empirical numbers or claim this objective guarantees successful rollout.

## Temporal and causal layout

Use a visible timeline arrow from earlier time to later time. Keep the causal sequence and the candidate classification visually ordered.

The context path must terminate at \(c_t\), and all horizon-specific scoring heads must branch from \(c_t\), not from the previous predicted latent. Future latent targets must be shown as labels / comparison candidates only, outside the context-building pathway.

Do not draw arrows such as \(z_{t+1}\to z_{t+2}\) as if the model recursively generates one future latent from the preceding predicted latent. Do not show a decoder from latents to physical observations. Do not include an action input, reward signal, policy, planning tree, or control loop.

## Required conceptual accuracy

1. Past and current latent representations \(z_1,\ldots,z_t\) form the causal context; future targets \(z_{t+k}\) are held out from that context.
2. The context vector \(c_t\) summarizes information available through time \(t\).
3. For each horizon \(k\), a horizon-specific scoring head \(W_k\) scores candidate latents using \(s_k(c_t,z)=z^\top W_kc_t\).
4. The observed future latent \(z_{t+k}\) is the positive target, and sampled alternatives supply the contrastive candidates.
5. The figure describes candidate identification / retrieval under a contrastive objective. It does not establish accurate multi-step generation, stable recursive rollout, predictive sufficiency, or planning competence.
6. This is an explanatory sketch of CPC, not a universal architecture for all predictive representation methods.
7. Do not invent numerical scores, probability values, benchmark results, or empirical claims.

## Visual style and restrictions

Use white background, dark grayscale-safe line work, and #3B6FB6 as the only accent. Flat 2D shapes, clear time direction, direct labels, aligned latent-vector tiles, and spacious horizon branches.

Use a dashed or clearly styled temporal boundary after \(t\) so that past/current context and future targets are unmistakable even in grayscale. Keep the future target candidates visually separate from the context network.

Do not use gradients, shadows, 3D effects, Mermaid, decorative watermarks, dashboards, realistic control-room illustrations, simulated page layouts, extra color palettes, tiny footnotes, or unnecessary iconography.

## Acceptance checks

1. The timeline clearly distinguishes \(z_1,\ldots,z_t\) from future targets \(z_{t+k}\).
2. Only past and current latents enter the context model.
3. The context produces \(c_t\), which feeds horizon-specific scoring heads \(W_k\).
4. At each shown horizon, the observed future latent is the positive candidate and alternatives are sampled candidates.
5. The score \(z^\top W_kc_t\) is typeset correctly and belongs to the horizon-specific candidate comparison.
6. No future target enters the context pathway, and no recursive latent rollout, decoder, action input, or planner is shown.
7. All important text and equations remain readable at printed book-column width.
8. The image contains no figure number, title, or caption.

## Caption

Figure 3.7. CPC: causal context predicts future latent candidates.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter's explanation and cited sources; do not invent empirical data.