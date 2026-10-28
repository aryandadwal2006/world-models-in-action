# Figure 3.10: DINO teacher-student cross-view distillation

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Figure number, title, caption, and descriptive prose belong in the manuscript, not inside the image. Do not render the figure number, figure title, caption, panel letters, page number, or simulated book layout in the artwork.

**Purpose:** Explain the original DINO teacher-student distribution-matching mechanism. A student predicts the teacher's output distribution for a different view of the same source. Both networks produce logits over learned prototypes; the teacher logits are centered and the teacher distribution sharpened before becoming a detached target. Cross-entropy trains the student, while the teacher parameters follow an EMA of the student parameters. Distinguish centering, sharpening, stop-gradient, and EMA as separate operations.

**Format and print legibility:** Use a landscape canvas approximately 3500 × 2200 px. White background, dark charcoal line work, and #3B6FB6 as the only restrained accent. Use the style of a carefully typeset machine-learning conference-paper figure adapted for a Manning technical textbook: crisp flat 2D components, precise arrows, consistent line weights, generous whitespace, and restrained visual hierarchy. At source resolution, target approximately 64–76 px for principal labels and 52–60 px for equations and secondary annotations. Use print-safe strokes and do not shrink equations or temperatures to fit. All essential content must remain readable at printed book-column width.

## Composition

Create one integrated left-to-right schematic with two parallel branches, **Student** and **Teacher**, followed by their prototype distributions and the cross-view distillation loss. Add a separate parameter-update connection from the student network to the teacher network.

The figure illustrates the original DINO recipe and its multi-crop setup, not later DINOv2 or DINOv3 additions.

### 1. Cross-view crops

At the left, show one source observation producing several crops with two visually labeled scales:
- **Global crops**: larger field of view.
- **Local crops**: smaller field of view.

Show the teacher receiving global crops. Show the student receiving global crops and local crops. Make the cross-view target relation clear: student predictions for selected crops are trained against teacher distributions from selected different views of the same source.

Do not imply every possible student-teacher crop pairing is used. Use a small bracket or compact label such as “selected cross-view pairs” instead of drawing an exhaustive grid of all pairings.

Use simple crop-window diagrams or plain image tiles. Do not use detailed or realistic imagery that distracts from the training mechanism.

### 2. Student branch

Show student crops \(x_s\) entering a box labeled **Student network**, with trainable parameters \(\theta_s\). The network produces prototype logits \(a_s(x_s)\).

Display the student distribution:
\[
p_s(x_s)=\operatorname{softmax}\!\left(\frac{a_s(x_s)}{T_s}\right).
\]

Label \(T_s\) as the **student temperature**. Keep it distinct from the teacher temperature.

Show a clear gradient/backpropagation arrow from the cross-entropy loss into the student network. Label the path **student receives gradients**.

The output coordinates correspond to learned prototypes. Label the logits or distribution as “learned prototype outputs”, not ground-truth semantic classes.

### 3. Teacher branch and target construction

Show teacher global crops \(x_t\) entering a separate **Teacher network**, with parameters \(\theta_t\). It produces prototype logits \(a_t(x_t)\).

Show the teacher output going through two separate, sequential operations:

**Center teacher logits** by subtracting a running center \(c\):
\[
a_t(x_t)-c.
\]

**Sharpen teacher distribution** by applying a lower teacher temperature \(T_t\) before softmax:
\[
q_t(x_t)
=\operatorname{softmax}\!\left(\frac{a_t(x_t)-c}{T_t}\right).
\]

Label \(T_t\) as **teacher temperature**. Make the sequence explicit: center the teacher logits, divide by the teacher temperature, then apply softmax.

Add compact notes:
- “Centering counteracts persistent dominance by output coordinates.”
- “Sharpening makes the teacher distribution more concentrated.”

These operations must be shown as different transformations, not interchangeable names or one combined mysterious box. Do not imply centering is Sinkhorn balancing. Do not show a DINOv2 patch branch or any other later-recipe addition.

Mark the teacher distribution as **stop-gradient target**. The target has a forward numerical value used by the loss, but no backpropagation gradient passes through the teacher branch from the cross-entropy. Do not draw the loss-gradient arrow into the teacher network.

### 4. Cross-view cross-entropy

Bring together the student distribution \(p_s\) and the detached teacher target \(q_t\) in a clearly labeled **Cross-view cross-entropy** block.

Display the loss:
\[
H(q_t,p_s)=-\sum_{k=1}^{K}q_{t,k}\log p_{s,k}.
\]

Keep the notation legible and make clear that \(k\) indexes learned prototype outputs. Use a labeled connector from the teacher distribution to the target input of the loss and from the student distribution to the prediction input.

Show the cross-view relation: selected student crops are matched against teacher targets from different views of the same source. Do not draw a negative-sampling or InfoNCE candidate set; DINO matches distributions and does not use explicit negative samples in this objective.

A compact optional annotation may show:
\[
H(q_t,p_s)=H(q_t)+D_{\mathrm{KL}}(q_t\Vert p_s).
\]
If it crowds the main flow, omit this optional identity; the primary cross-entropy formula and the target/prediction roles take priority.

### 5. EMA teacher update

Below or beside the main data flow, show the separate parameter update from student parameters \(\theta_s\) to teacher parameters \(\theta_t\):
\[
\theta_t\leftarrow m\theta_t+(1-m)\theta_s.
\]

Label the arrow **EMA parameter update** and explain briefly:
“The teacher tracks a moving average of student parameters.”

This arrow is a parameter update, not a forward activation path and not gradient backpropagation. Keep it visually separate from the student loss-gradient arrow. The teacher has no direct loss-gradient path.

## Visual hierarchy and arrow conventions

Use a consistent notation:
- Solid dark arrows: forward crop, network, logits, and distribution flow.
- A clearly labeled gradient arrow: cross-entropy gradient into the student only.
- A stop-gradient marker and label on the teacher distribution before its use as a target.
- A separately styled arrow labeled EMA parameter update from student parameters to teacher parameters.
- Blue accent to emphasize the student-target matching relation or teacher target, without relying on color alone.

Make the difference between centering and sharpening visible through separate labeled steps in the teacher output transformation. Do not collapse the running center, teacher temperature, softmax, stop-gradient, and EMA update into a single operation.

## Required conceptual accuracy

1. Both student and teacher produce logits over learned prototype outputs; the prototype indices are not automatically human-defined categories.
2. The student distribution is \(p_s(x)=\operatorname{softmax}(a_s(x)/T_s)\).
3. The teacher distribution is \(q_t(x)=\operatorname{softmax}((a_t(x)-c)/T_t)\), with teacher centering applied before temperature scaling and softmax.
4. Student temperature \(T_s\) and teacher temperature \(T_t\) must be labeled separately.
5. Centering and sharpening are different mechanisms: centering subtracts a running teacher-output center; sharpening uses teacher temperature to make the target distribution more concentrated.
6. The student is trained with cross-entropy against the detached teacher distribution.
7. The teacher branch is stop-gradient and its parameters are updated by EMA, not direct loss backpropagation.
8. Original DINO uses larger global crops for the teacher; the student also receives local crops under the multi-crop recipe.
9. DINO's objective does not use explicit negative samples in the way InfoNCE does.
10. Do not imply prototype outputs are ground-truth categories.
11. Do not add DINOv2 or DINOv3-specific mechanisms such as iBOT masked patch distillation, KoLeo regularization, Sinkhorn-balanced patch assignments, or Gram anchoring.
12. These mechanisms help the original DINO recipe avoid undesirable output distributions in practice, but the diagram must not claim a universal theorem that centering and sharpening alone guarantee useful representations.

## Visual style and restrictions

Use white background, dark grayscale-safe line work, and #3B6FB6 as the only accent. Use flat 2D boxes, clear branches, aligned notation, direct labels, and generous whitespace.

Do not use gradients, drop shadows, 3D effects, Mermaid, decorative watermarks, dashboards, simulated page layouts, extra color palettes, tiny footnotes, unnecessary class icons, or decorative labels. Avoid crossed arrows and crowded distribution diagrams.

## Acceptance checks

1. Student and teacher crop inputs are distinguished; the teacher receives global crops, while the student also receives local crops.
2. Both branches produce logits over learned prototype outputs.
3. Student softmax with \(T_s\) and teacher centering plus softmax with \(T_t\) are separately and correctly shown.
4. Centering and sharpening are separate, labeled operations.
5. The detached teacher distribution is the cross-entropy target; student probabilities are the prediction.
6. Backpropagation is shown through the student only.
7. EMA is clearly shown as a parameter update to the teacher, not as a forward path or gradient.
8. No negatives or DINOv2/DINOv3 add-ons appear.
9. All equations and labels remain legible at printed book-column width.
10. The image contains no figure number, title, or caption.

## Caption

Figure 3.10. DINO teacher-student cross-view distillation.

## Production note

Create original explanatory artwork, not a reproduction of a published figure or screenshot of code. Keep every visual claim consistent with the chapter explanation and cited sources; do not invent empirical data.