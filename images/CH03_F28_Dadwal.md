# Figure 3.16: Aliasing is a failure of predictive sufficiency

**Status:** Detailed production brief only. No artwork is included in this branch.

**Artwork boundary:** Do not render a figure number, title, caption, citation, or surrounding manuscript prose inside the artwork. The book layout supplies the figure metadata separately.

## Purpose

Show a concrete, target-relative failure of predictive sufficiency. Two different observation histories encode to the same representation \(z\), even though they imply different distributions over the next state under the same action. Any predictor that receives only \((z,a)\) must return the same predictive distribution for both histories, so it cannot represent both true conditional futures.

The diagram is a constructed counterexample, not a claim that similar images always imply different futures.

## Mathematical statement

Use a consistent notation throughout:

- \(H_1,H_2\): two distinct observation histories.
- \(f\): representation encoder.
- \(z\): the shared encoded representation.
- \(a\): the same action supplied in both cases.
- \(S_{t+1}\): next state or target future.
- \(q(\cdot\mid z,a)\): generic predictor whose only relevant inputs are the encoded representation and the action.

The central aliasing condition is:

\[
f(H_1)=f(H_2)=z
\]

Despite this equality, the true action-conditioned next-state distributions differ:

\[
P(S_{t+1}\mid H_1,a)
\ne
P(S_{t+1}\mid H_2,a)
\]

A predictor that receives only the same pair \((z,a)\) cannot distinguish the histories. It must return the same distribution for both:

\[
\widehat P_1=\widehat P_2=q(\cdot\mid z,a)
\]

Show this repeated prediction as one shared output or two visibly identical copies, not as two separately learned predictions.

For the relevant target and horizon, predictive sufficiency requires:

\[
P(S_{t+1}\mid H,a)=P(S_{t+1}\mid z,a)
\]

Make this condition a compact footer or side note. The main visual proof should remain the two different histories, the shared \(z\), the same action, and the different true futures.

## Composition

Use a landscape canvas with a clean left-to-right logical flow and two aligned history rows. The reader should be able to compare the two cases without following tangled connectors.

### 1. Two distinct histories

On the left, draw two small, vertically aligned sequences of three simple frame-like panels. Keep their final object location and apparent current image position the same, but show different recent motion:

- **History \(H_1\):** an object approaches the same final location while moving to the right. Indicate the recent motion with a short directional arrow in the preceding frames.
- **History \(H_2\):** the object reaches that same final location while moving to the left. Indicate the opposite recent motion in its preceding frames.

The frames are a schematic example, not a photorealistic scene. Use simple boxes, a restrained outline for the object, and small arrows that communicate the direction inferred from history. Make the final frames look essentially identical at a glance; the distinction lies in the trajectory and latent velocity that the history implies.

Label the rows \(H_1\) and \(H_2\). Do not use a detailed real-world environment or additional objects that introduce irrelevant distinctions.

### 2. A representation that aliases the histories

To the right of the history rows, connect both to the same encoder symbol \(f\), or show two encoder paths converging on the same representation node.

Label the converged representation clearly:

\[
f(H_1)=f(H_2)=z
\]

Highlight the single shared \(z\) with the restrained #3B6FB6 accent. The equality is central and must be impossible to miss.

Do not draw two different latent states or two merely similar latent vectors. The point is that the representation is exactly the same for the two histories.

### 3. Apply the same action

Show one action input \(a\) entering both counterfactual cases. Use a single shared action box or two identical action labels that are clearly the same action.

Do not introduce action differences between the rows. The future discrepancy must arise because the histories imply different hidden motion or state despite sharing \(z\), not because the action differs.

Keep action notation generic. Do not use a robot, a named control policy, or a specific control-system architecture.

### 4. Show the true future distributions differ

On the right, show the two true, action-conditioned next-state distributions separately and align each with its corresponding history row:

\[
P(S_{t+1}\mid H_1,a)
\qquad\text{and}\qquad
P(S_{t+1}\mid H_2,a)
\]

Use simple schematic probability shapes or small discrete outcome bars, not fabricated numerical probabilities. The distributions should be clearly different in position or shape to signify different predicted next states. For example, their mass can be centered on opposite sides of the same current position. Do not add numeric axes, percentages, or empirical values.

Label this region **“True futures differ”**. Ensure the illustration communicates a difference in the conditional distributions, rather than suggesting the future is deterministic in either case.

### 5. Contrast the information available to the predictor

Below or adjacent to the true-future rows, include a compact predictor pathway receiving only \((z,a)\):

\[
(z,a)\longrightarrow q\longrightarrow \widehat{P}(S_{t+1}\mid z,a)
\]

Because the representation and action are the same in both cases, show that the predictor can produce only one shared conditional distribution for the two histories. It cannot select one distinct output for \(H_1\) and another for \(H_2\) unless some distinguishing information is supplied.

Make the contrast readable:

- **True conditional futures:** different for \(H_1\) and \(H_2\).
- **Predictor input:** the same \((z,a)\).
- **Prediction from that input:** the same distribution for both.

Use a small annotation such as **“The discarded distinction cannot be recovered downstream.”** Keep this explanatory phrase readable and secondary to the equations.

### 6. Sufficiency condition

At the bottom, place the condition in a clear, uncluttered mathematical footer:

\[
P(S_{t+1}\mid H,a)=P(S_{t+1}\mid z,a)
\]

Pair it with a short label:

**“Predictive sufficiency for this target and horizon”**

The equality is the criterion; in this counterexample it fails because \(H_1\) and \(H_2\) map to the same \(z\) but have different action-conditioned futures. Keep the target and horizon scope explicit.

## Visual hierarchy and typography

- Landscape canvas, approximately 3200 × 2000 px.
- White background with dark charcoal text, frame outlines, and connectors.
- Use #3B6FB6 only for the shared latent representation \(z\), key equality annotation, and a few connectors highlighting the aliasing point. All essential distinctions must still work in grayscale.
- Main labels and central equations should be about 64–76 px at source size. Secondary equations and annotations should be about 52–60 px. No microtype.
- Use simple flat 2D frame panels, consistent line weights, clear directional arrows, and unobstructed connectors.
- Make the rows align vertically so the reader can compare \(H_1\) and \(H_2\), the shared latent, and their different true futures.
- Use a consistent mathematical typeface. Keep equations on one line when feasible.
- Leave enough whitespace around the two future distributions and the shared predictor input; this is a conceptual argument, not a dense process chart.

## Exclusions and acceptance checks

- Do not render the figure number, title, caption, chapter number, citation, watermark, or surrounding manuscript prose.
- Do not claim that similar or identical final frames always have different futures. This is one constructed counterexample involving different histories.
- Do not let the two histories map to separate latent representations. Both must map to the same \(z\).
- Do not let the two histories receive different actions. The action \(a\) must be the same.
- Do not make the true future distributions identical. Their inequality is central.
- Do not show the predictor returning different distributions from the exact same input \((z,a)\).
- Do not use fabricated probability values, benchmark statistics, or empirical results.
- Do not imply that increasing predictor capacity can recover distinctions absent from its inputs.
- Do not imply that the representation is universally insufficient. The failure is relative to the specified next-state target, action, available history, and horizon.
- Do not add a robotics illustration, named algorithm, planning outcome, training loss, or unrelated representation method.
- At printed book-column width, the equality \(f(H_1)=f(H_2)=z\), the unequal true distributions, and the sufficiency condition must remain legible.

## Caption

Figure 3.16. Aliasing is a failure of predictive sufficiency.

## Production note

Create original explanatory artwork for a technical machine-learning textbook. The visual proof must be understood at a glance: different histories, one shared representation, the same action, different true future distributions, and one indistinguishable predictor input. Prioritize mathematical faithfulness and print readability over decoration.