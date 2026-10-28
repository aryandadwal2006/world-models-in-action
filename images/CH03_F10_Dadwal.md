# Figure 3.6: Momentum contrast: query encoder, key encoder and queue

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, panel letters, and descriptive prose belong in the manuscript, not inside the image. Do not render a figure number, title, caption, page number, or simulated book layout in the artwork.

**Purpose:** Explain the MoCo v1/v2 queue-based mechanism: a gradient-trained query encoder compares its representation with a matching key produced by a momentum encoder and with stored keys from earlier minibatches. Make the stop-gradient key branch, EMA parameter update, and FIFO queue operations visually distinct. The figure must teach the mechanism without implying that queued keys are perfectly current or that every negative is semantically unrelated.

**Format and print legibility:** Use a landscape canvas approximately 3400 × 2100 px. Use a white background, dark charcoal outlines and text, with #3B6FB6 as the only restrained accent. Style it like a carefully typeset machine-learning paper figure adapted for a Manning technical textbook: flat 2D, precise alignment, consistent arrow conventions, clear grouping, and generous whitespace. At original resolution, use about 64–76 px for principal labels and 52–60 px for equations and secondary notes. Keep arrows and connector lines print-safe. Do not squeeze labels into the queue; give the queue enough width and height. All essential content must remain legible at printed book-column width.

## Composition

Build one integrated schematic with three visually distinct regions: the paired input and encoder paths across the upper half, the query-to-candidate comparison across the lower or right half, and the parameter-update / queue-maintenance mechanism in a clearly connected strip. Use arrows and grouping to make the data flow unambiguous.

### 1. Two views and the encoder branches

Start with two augmented views of the same source observation:
- query view \(x_q\)
- key view \(x_k\)

Show the query view flowing into a box labeled **Query encoder** with parameters \(\theta_q\), producing a normalized query vector \(q\).

Show the key view flowing into a separate box labeled **Momentum key encoder** with parameters \(\theta_k\), producing a normalized positive key \(k^+\).

Mark the query encoder path as **gradient-trained** and show a backpropagation arrow from the contrastive objective to the query encoder. Mark the key branch with an unmistakable **stop-gradient** label or stop-gradient symbol. Do not draw a loss-gradient arrow into the key encoder.

The two branches must be visually distinct, with labels directly attached to the correct path. Do not suggest that the query and key views are the same augmented tensor.

### 2. Contrast query against positive key and queued negatives

Show the normalized query \(q\) scoring:
- the matching positive key \(k^+\)
- a small queue \(Q\) of previously encoded keys, shown as \(k_1, k_2, \ldots, k_M\) or a similar compact sequence

Label the queue entries **queued negative keys**. Make clear that the positive key and queue entries form the candidate set compared with the query. The queue must not contain the anchor or be drawn as a queue of raw input images; it stores encoded key vectors.

Use a simple score / candidate arrangement rather than trying to draw every matrix operation. Add a compact loss label:
\[
\mathcal L_{\mathrm{MoCo}}
=-\log
\frac{\exp(q^\top k^+/\tau)}
{\exp(q^\top k^+/\tau)+\sum_{k_j\in Q}\exp(q^\top k_j/\tau)}.
\]
Keep the expression mathematically readable. The denominator includes the positive key and the queued negatives. It does not include the query itself.

If a full equation makes the overall layout too dense, prioritize the positive-versus-queue candidate diagram and provide a clear labeled loss block below it. Do not remove or alter the positive term or queue-sum structure.

### 3. EMA parameter update: separate from data flow

Show a distinct parameter-update connection from the query encoder parameters \(\theta_q\) to the key encoder parameters \(\theta_k\), explicitly labeled **EMA parameter update**. Display:
\[
\theta_k \leftarrow m\theta_k+(1-m)\theta_q.
\]

Add a short explanatory label:
“The key encoder tracks a moving average of query parameters.”

Use an update arrow that clearly indicates the new key parameters depend on the current query parameters and the previous key parameters. Do not draw this arrow as if it were a forward activation path, and do not confuse it with queue insertion or with backpropagation.

Make visually clear that the momentum key encoder is not updated by the contrastive loss gradient. Its parameters are updated through the EMA rule.

### 4. FIFO queue maintenance

Show a bounded queue containing key vectors from earlier minibatches. Use an ordered row or stack of small labeled vector tiles, with older entries visibly at the front / exit and new entries entering at the other end. Label the operations clearly:

- **Enqueue:** add current minibatch keys.
- **Dequeue:** remove the oldest keys to keep the queue length fixed.

Connect the current minibatch keys produced by the momentum encoder to the enqueue operation, but keep this insertion path visually separate from the EMA parameter update.

Label queue contents:
“Features encoded using earlier parameter values.”

Add a compact caution next to the queue:
“Historical keys can be stale.”

Explain the relationship succinctly:
“The momentum encoder reduces abrupt drift; it does not make old keys identical to current features.”

These notes must be readable without crowding the core architecture. They should make clear that the queue holds historical key representations, not current features recomputed after every parameter update.

## Visual hierarchy and arrow conventions

Use a consistent, explicit visual grammar:
- Solid dark arrows for the forward computation of inputs and representations.
- A blue accent for the positive candidate link or query-to-key comparison.
- A clearly labeled gradient/backpropagation arrow only along the query-encoder training path.
- A differently styled, labeled arrow for the EMA parameter update.
- Separate queue-operation arrows for enqueue and dequeue.

Do not rely on line color alone: every nonstandard arrow must have a direct label or a clearly attached notation. Avoid crossed connector lines where possible.

## Required conceptual accuracy

1. The figure represents the queue-based MoCo v1/v2 mechanism, not MoCo v3.
2. The query encoder is trained by backpropagation through the contrastive loss.
3. The matching key is produced by the momentum key encoder, and the key branch is stop-gradient.
4. Key-encoder parameters follow the EMA update:
   \(\theta_k \leftarrow m\theta_k+(1-m)\theta_q\).
5. The positive key and queued keys make up the candidate set compared with the query.
6. The queue stores key features produced by earlier parameter values. These features may become stale.
7. EMA updating mitigates abrupt representational drift but does not eliminate historical staleness.
8. Enqueue/dequeue operations are distinct from the EMA parameter update.
9. Queued negatives are negatives under the sampling procedure. Do not claim all of them are semantically unrelated.
10. Do not draw MoCo v3 as a second architecture or suggest it retains this persistent FIFO queue. MoCo v3 is explicitly distinguished in the accompanying prose.

## Visual style and restrictions

White background, grayscale-safe dark line work, and #3B6FB6 as the only accent color. Flat 2D geometry, crisp vector-like edges, aligned labels, generous whitespace, and no ornamental elements.

Do not use gradients, shadows, 3D effects, Mermaid diagrams, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, or unnecessary icons. Avoid an oversized queue that overwhelms the two encoder branches.

## Acceptance checks

1. The two augmented views, query encoder, momentum key encoder, query vector, positive key, and queue are independently identifiable.
2. A gradient path is shown for the query encoder only; the key encoder is marked stop-gradient.
3. The EMA parameter update is clearly separate from the forward path and from queue maintenance.
4. Enqueue and dequeue operations are both labeled, and queue contents are identified as historical key features.
5. The candidate loss includes the positive key and queued negatives, but not the query itself.
6. The note about historical staleness is accurate and visible.
7. The architecture is clearly MoCo v1/v2 rather than MoCo v3.
8. All essential labels and mathematical notation are legible at printed book-column width.
9. The image contains no figure number, title, or caption.

## Caption

Figure 3.6. Momentum contrast: query encoder, key encoder and queue.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical results.