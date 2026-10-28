# Figure 3.17: A training loss and a probe answer different questions

**Status:** Revised production brief for original artwork.

**Artwork boundary:** The figure title, figure number and caption are manuscript metadata. Do not render them inside the image. The takeaway about predictive sufficiency belongs in the caption, not as a slogan or banner in the artwork.

## Purpose

Make one restrained paper-style diagram showing that training loss, representation diagnostics and target-specific prediction test different claims. Avoid a presentation-style three-stage infographic. The artwork should look like a single connected schematic that could sit naturally in a machine-learning paper.

## Composition

Use one horizontal encoder-and-representation pathway, with thin branching connectors to a few compact evaluation paths. Do not create large top headings, section banners, numbered stages, outlined cards, colorful panels, decorative icons, score gauges or a separate conclusion box.

At the left, show a small input label such as “training views / masked input” entering an encoder \(f_\theta\), producing the representation \(z\). From this pathway, show the pretraining objective with its target relation and label it \(\mathcal{L}_{\mathrm{pre}}\). A small inline annotation can say “chosen training target”; do not label the representation universally good or sufficient.

From the representation, use quiet, well-spaced branches for three familiar diagnostics:

- **Linear probe:** \(\hat y=Wz+b\). Inline note: “linear accessibility”.
- **Retrieval / feature geometry:** show a query point and a few candidates under a selected distance or similarity measure. Inline note: “selected neighbourhoods”.
- **Reconstruction:** \(\hat x=d(z)\). Inline note: “recovery of a chosen target”.

These are parallel tests, not steps in a sequence. Use plain labels beside the small equations or glyphs; do not place them inside matching rounded cards.

Add a fourth, slightly more consequential branch for a direct downstream prediction. Show the available history \(H\) entering the encoder to produce \(z=f(H)\), then a generic predictor \(q\) producing \(\hat Y\), compared against the selected target \(Y\). Label the endpoint only with compact, scoped text such as “target \(Y\), horizon \(k\), held-out protocol”. Do not turn this branch into an oversized final panel.

Use the same visual language throughout: charcoal strokes and labels, subtle whitespace, and #3B6FB6 sparingly to emphasize the representation and the path to target-specific prediction. Do not use one color per branch. Do not use green checks, red crosses, badges, or any ranking.

## Mathematical and conceptual accuracy

Use consistent notation:

\[
z=f_\theta(x), \qquad \hat y=Wz+b, \qquad \hat x=d(z)
\]

For the direct prediction branch, show:

\[
H\longrightarrow f\longrightarrow z\longrightarrow q\longrightarrow \hat Y
\]

Beside the prediction, show \(Y\) as the comparison target. The history, target, horizon and held-out readout protocol determine the scope of the test.

The figure should imply the following distinctions through the branches themselves, not through a wall of explanatory text:

- The pretraining loss measures fit to the relation or target selected for training.
- A linear probe tests whether a chosen property is accessible to a specified linear readout. Failure does not prove that the information is absent.
- Retrieval or geometry diagnostics test neighbourhood relationships under a selected metric.
- Reconstruction tests recovery of the selected reconstruction target.
- Direct prediction provides evidence about the particular target and protocol being tested, not a universal guarantee.

A successful pretraining objective is not by itself evidence that the representation preserves every distinction needed for downstream prediction. Likewise, none of the displayed diagnostics should be presented as a universal representation-quality score.

## Visual style

- Landscape canvas, approximately 3200 × 2000 px, white background.
- Grayscale-safe dark charcoal text, strokes and arrows.
- #3B6FB6 as the only accent, used sparingly.
- Flat 2D scientific line art, consistent thin strokes and small, precise arrowheads.
- Print-sized typography: main labels about 64–76 px and equations or secondary labels about 52–60 px at source size.
- Keep labels short and physically close to the element they explain.
- Align the branches neatly but avoid rigidly identical panel geometry. Let the equations and connectors define the layout.
- Use only the few marks needed to explain a query and its neighbours. No decorative point clouds or ornamental diagrams.
- No oversized top headings, numbered steps, cards, panels, banner text, badges, conclusion callouts, illustrations, gradients, shadows, 3D effects, fake plots, invented numbers, watermarks or simulated page furniture.
- Do not render a figure number, title, caption, citation, chapter number, or the sentence “No single metric certifies predictive sufficiency” inside the art.

## Acceptance checks

At a glance, the reader should see one learned representation being assessed in different ways, with no diagnostic presented as a universal score. The direct prediction path must visibly include its target and evaluation scope. Linear-probe failure must not imply absent information. There must be no in-art slogan or presentation-style heading.

## Caption

Figure 3.17. A training loss and a probe answer different questions. No single metric certifies predictive sufficiency.

## Production note

Create original explanatory artwork with the restraint of a research-paper schematic. Prefer a coherent connector diagram, compact mathematical labels and generous whitespace over a polished infographic template.