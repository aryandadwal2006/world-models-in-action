# 02  -  I-JEPA: predicting image representations from spatial context

## 1. Why I-JEPA is the first concrete architecture

I-JEPA is the clearest first example because it makes the JEPA structure visible without introducing video time, actions or a planning objective. It takes an image, forms patch tokens, selects a context region and several target blocks, and trains a predictor to estimate the target encoder's features at those locations.

The primary paper is Assran et al., “Self-supervised learning from images with a joint-embedding predictive architecture” (2023; CVPR 2023). The official implementation is the Meta FAIR repository facebookresearch/ijepa. Use the paper and repository together: the paper motivates the learning design, while the code and configuration specify practical mask sampling, target layers and training details.

## 2. Architecture

Let an image be represented as a grid of patch tokens. A large context mask \(C\) chooses visible context patches. Target masks \(Y_1,\ldots,Y_M\) choose several target blocks. The context encoder receives the context tokens and their positional information. The target encoder computes features for the full image or a less-restricted view; the target features are selected at the target locations. A predictor receives context features and the positions of the selected targets.

A schematic loss is:
\[
\mathcal L_{\mathrm{I\text{-}JEPA}}
=\frac1M\sum_{m=1}^{M}
\frac1{|Y_m|d}
\sum_{j\in Y_m}
\|\hat h_j-h_j\|_2^2.
\]

The exact normalization, target feature normalization, mask sampler, target layer and predictor details must follow the chosen paper version and code. The equation above expresses the high-level operation, not every implementation detail.

## 3. Why multiple target blocks matter

Predicting several target blocks makes the task depend on spatial context rather than a single local completion. The target geometry controls how much of the image the model must infer from the context. A small nearby block may be easy to infer from texture continuity. A distant or semantically distinct target may require a broader scene representation.

There is no mask policy that is universally correct. A mask that makes target features easy to infer may encourage local shortcuts; a mask that makes them too ambiguous can produce noisy gradients. The selection distribution is part of the method.

## 4. Why not predict RGB patches?

The predictor is trained against target encoder features rather than raw pixel values. This allows the model to ignore variations that the target encoder itself treats as unimportant. It avoids making pixel reconstruction the explicit optimization target.

However, do not claim that I-JEPA simply discards “all unpredictable details.” The model's learned feature geometry, data distribution and masking procedure determine what is retained. A task that needs small texture differences, exact spatial boundaries or contact cues may require a different target representation or a complementary objective.

## 5. Target encoder update

The I-JEPA recipe uses a target encoder that evolves as a moving average of the context encoder. In the generic EMA equation:
\[
\xi_{t+1}=m_t\xi_t+(1-m_t)\theta_{t+1}.
\]
The target feature provides a relatively stable regression target while the context encoder and predictor are optimized. Explain the update rule as part of I-JEPA's recipe, not as a universal JEPA requirement.

## 6. What should be reported

When describing the paper's experiments, keep the task and evaluation protocol visible:
- dataset and pretraining scale;
- encoder architecture and image resolution;
- mask geometry and target block count;
- feature layer and pooling/readout;
- whether the encoder is frozen;
- evaluation benchmark and metric;
- what the result says about representation quality and what it does not say about action-conditioned world modelling.

Avoid generalizing natural-image classification results to robot control without a separate transition and planning evaluation.

## 7. Recommended explanation and figures

**Figure 4.3  -  Context and target blocks in I-JEPA.** Show one image grid, a large context region, multiple target blocks, target-encoder features and predictor outputs. Distinguish visible context from target locations. Do not render the paper's original figure.

**Figure 4.4  -  Target geometry changes the prediction problem.** Use two examples: a nearby region and a distant target block. The goal is to show how mask geometry changes what information is needed, not to imply one is always better.

**Sidebar  -  What the loss does not see.** A JEPA loss compares vectors in a learned target space. If a cue is absent from that space, the predictor has no direct incentive to reproduce it.

## 8. Implementation notes

The official I-JEPA repository is the sole code source for method-faithful snippets. Candidate excerpts:
- the mask-collation logic that generates context and target block indices;
- how target tokens are selected;
- the target encoder EMA update;
- how target positions are passed to the predictor.

Use one or two short excerpts only. The mask generator is likely the most pedagogically valuable because it makes the learning problem concrete. Do not reproduce the training framework or model architecture at length.

## 9. Limitations and transition

I-JEPA establishes the spatial version of the idea. It does not yet ask a model to understand temporal continuity or to predict consequences of actions. The next step is to extend context and target positions across time, which leads to V-JEPA.

**Primary sources**
- Assran et al. (2023), arXiv:2301.08243.
- Official code: https://github.com/facebookresearch/ijepa
