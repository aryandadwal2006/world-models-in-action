# 03 — V-JEPA, V-JEPA 2 and V-JEPA 2.1

## 1. From image patches to space-time tokens

A video clip is not merely a set of unrelated images. It contains observations indexed by time, and the relationships between frames provide evidence about motion, interaction and temporal persistence. V-JEPA extends latent target prediction to spatiotemporal tokens.

Represent a video clip as \(X=(x_1,\ldots,x_T)\). A token location can be indexed as \(r=(t,h,w)\), where \(t\) is temporal position and \(h,w\) are spatial coordinates. A context mask selects visible context tokens; a target mask selects space-time positions to predict.

A generic form is:
\[
\mathcal L_{\mathrm{V\text{-}JEPA}}
=\frac1{|Y|d}\sum_{r\in Y}
\|\hat h_r-h_r\|_2^2.
\]
The target is the feature at a selected space-time position, not a rendered future clip. The real paper recipe determines the temporal sampling, target encoder update, feature layer, mask geometry, loss normalization and data pipeline.

The original V-JEPA paper is Bardes et al., “Revisiting Feature Prediction for Learning Visual Representations from Video” (2024). It demonstrates that video feature prediction can learn useful frozen visual representations without relying on pretrained image encoders, text supervision, negative examples or pixel reconstruction.

## 2. What time adds

Temporal target prediction can reward features that preserve motion and interactions when those distinctions affect the chosen future target. But time does not guarantee that the representation encodes velocity, force, contact, object permanence or causal state.

The target distribution matters:
- a target close in time may be predictable from local continuity;
- a far-future target may be ambiguous;
- a target selected from the same time but distant in space tests scene context;
- a target selected from a later frame tests temporal prediction;
- a target selected after a particular action tests action-conditioned prediction only when the action is an explicit conditioning variable and is aligned with the data.

Avoid an oversimplified claim that “video teaches physics.” Report which variables and tasks were measured.

## 3. V-JEPA 2: scale plus a separate action-conditioned stage

V-JEPA 2 (Assran et al., 2025) is especially important for this book because the work makes an explicit bridge between large-scale passive video learning and robot interaction.

The paper and official repository describe two conceptually separate stages:

**Stage A: V-JEPA 2 pretraining**
- an action-free video representation model is pretrained on large-scale image/video data;
- it is evaluated on motion understanding, human action anticipation and video question-answering-related tasks;
- the representation encoder is not itself a complete robot controller.

**Stage B: V-JEPA 2-AC post-training**
- the pretrained model is adapted into an action-conditioned latent world model using robot trajectory data;
- the predictor receives action information and predicts future latent features;
- planning can use this learned predictor with image goals;
- the paper reports zero-shot deployment on Franka arms in new environments without collecting environment-specific data there or task-specific training/calibration.

The article reports using less than 62 hours of unlabeled robot videos from the DROID dataset for the action-conditioned post-training. Phrase this precisely and keep the claim tied to the paper's experimental setup.

The critical pedagogical point is not only that the representation is trained at scale. It is that the physical planning demonstration depends on a later action-conditioned predictive model and a planner. Do not attribute the planning result to the action-free encoder alone.

## 4. V-JEPA 2.1: dense feature learning

V-JEPA 2.1 (Mur-Labadia et al., 2026) is a later representation-learning recipe that focuses on dense, temporally consistent features. The official repository describes four central elements:
1. dense predictive loss, in which both visible/context and masked tokens contribute to the self-supervised objective;
2. deep self-supervision at multiple intermediate encoder layers;
3. multimodal tokenizers for images and videos;
4. scaling of model capacity and training data.

This is not simply another name for V-JEPA 2-AC. The distinction is:
- V-JEPA 2-AC adds action conditioning for latent world modelling and planning;
- V-JEPA 2.1 improves the spatially dense representation objective and features.

Dense representations are relevant where global pooled features are insufficient: object correspondence, localisation, grasp regions, interaction anticipation and spatially structured prediction. But better dense features still do not certify action-conditioned dynamics or planning success.

When reporting the paper's results, identify the benchmark and metric. Do not transfer an anticipation, depth, grasping or navigation result to a general claim of autonomous intelligence.

## 5. A method comparison table

| Line | Input | Prediction target | Action input? | Main role |
|---|---|---|---|---|
| I-JEPA | Image | Spatial target features | No | Image representation learning |
| V-JEPA | Video | Space-time target features | No | Video representation learning |
| V-JEPA 2 | Large-scale image/video | Video feature prediction | No in pretraining stage | General-purpose video representation |
| V-JEPA 2-AC | Robot/video trajectories | Future latent features conditioned on actions | Yes | Latent action-conditioned world model |
| V-JEPA 2.1 | Images and videos | Dense token features | Not by definition | Dense visual/spatiotemporal representation |

## 6. Experimental claims: three different levels

Separate:
- frozen representation tests (classification, retrieval, depth or anticipation);
- feature-prediction tests (held-out target features at a specified temporal horizon);
- control tests (task success, physical execution, generalization, failures and safety).

A model may perform well on the first and poorly on the third. This is not a contradiction; the tasks measure different properties.

## 7. Recommended figures

**Figure 4.5 — A video mask is a space-time selection.** Draw a short clip as a grid of frame patches. Highlight context tokens and a target tube spanning time. The figure should show the temporal dimension explicitly rather than use a stack of independent images.

**Figure 4.6 — V-JEPA 2 is a two-stage story.** One lane shows action-free pretraining from large-scale video; the other shows post-training with robot trajectories, an action-conditioned predictor and planning. Label the stages separately.

**Figure 4.7 — V-JEPA 2.1 makes target supervision dense.** Contrast a global/pooled target with supervision over visible and masked tokens, plus intermediate layers. Avoid implying that the model is action-conditioned.

## 8. Primary sources and official code

- Bardes et al. (2024), “Revisiting Feature Prediction for Learning Visual Representations from Video,” arXiv:2404.08471.
- Assran et al. (2025), “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning,” arXiv:2506.09985.
- Mur-Labadia et al. (2026), “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning,” arXiv:2603.14482.
- Official implementation: https://github.com/facebookresearch/vjepa2

## 9. Transition

V-JEPA's video results make the importance of target selection clear, but they also raise a new question: if the predictor can match target features, what prevents it from learning an uninformative representation? Chapter 3 has already introduced this problem and VICReg. The next section should move beyond re-teaching collapse and explain why newer JEPA recipes study the full distribution of embeddings.
