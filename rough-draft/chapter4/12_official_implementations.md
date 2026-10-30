# 12 — Official implementations and reproducibility plan

## 1. General rule

The printed chapter should contain only short implementation excerpts that explain the method. Full model files, training infrastructure, distributed launchers, data pipelines and checkpoints stay in official repositories. A code excerpt is not a reproduction. The book must label what it has run, inspected or merely quoted.

## 2. Primary repositories

| Method | Official source | Intended inspection |
|---|---|---|
| I-JEPA | https://github.com/facebookresearch/ijepa | Mask collator, target selection, target encoder and predictor |
| V-JEPA 2 / 2-AC / 2.1 | https://github.com/facebookresearch/vjepa2 | Model definitions, configs for video pretraining and action-conditioned post-training, inference helpers |
| EB-JEPA examples | https://github.com/facebookresearch/eb_jepa | Image/video/action-conditioned examples and energy/planning notebooks |
| JEPA-WMs | https://github.com/facebookresearch/jepa-wms | Action-conditioned predictor, planning objective, model configurations, evaluation and environment interfaces |
| VISReg | https://github.com/HaiyuWu/visreg | Exact scale, sketching and gradient paths |
| LeJEPA | https://github.com/rbalestr-lab/lejepa | Exact SIGReg statistic and training objective |

Repository names and ownership were checked at the time of planning. Before publication, pin a commit or release for every code excerpt and verify the current license file.

## 3. What to inspect

### I-JEPA
- mask generation and mask collation;
- input token selection for the context branch;
- target encoder and selected target layer;
- predictor inputs and positional embeddings;
- EMA update implementation.

### V-JEPA 2
- video frame sampling and preprocessing;
- spatiotemporal masks;
- predictor and target feature layers;
- training configuration for the action-free stage;
- action-conditioned predictor configuration and input format;
- frozen inference/loading interface;
- distinct source tree for V-JEPA 2.1.

### LeJEPA / SIGReg
- direction sampler;
- characteristic-function computation;
- numerical quadrature and weighting;
- exact batch normalization and loss scaling;
- any distributed aggregation required by the official recipe.

### VISReg
- feature centering;
- scale normalization;
- stop-gradient locations;
- random projection matrix construction;
- sorted projected samples and Gaussian quantiles;
- exact loss weighting and distributed behaviour.

### JEPA-WMs
- how observations and actions are aligned;
- target construction and prediction horizon;
- planner's objective and action parameterization;
- replanning schedule;
- environment-specific evaluation configuration.

## 4. Reproduction boundary

The chapter's miniature example should fit a modest educational compute budget and teach the algorithmic structure. It should not claim to reproduce:
- internet-scale video pretraining;
- a billion-parameter V-JEPA checkpoint;
- the published V-JEPA 2-AC robot deployment;
- an ImageNet-1K VISReg training run;
- JEPA-WMs robot benchmark results.

A reader can load released checkpoints and inspect inference, but checkpoint use is not reproduction of the paper's training.

## 5. Code quotation and license

Check the exact repository license and file header for each code fragment. The V-JEPA 2 repository describes most of the project as MIT-licensed with some components under separate Apache 2.0 terms. The JEPA-WMs repository indicates CC-BY-NC 4.0; do not assume commercial book reproduction of its code is permitted. The VISReg and LeJEPA repositories need a separate license check at the selected commit. Never infer code licensing from the arXiv paper's license.

Use a line-level attribution and identify the commit. If the license is unclear or incompatible with reproduction, present pseudocode authored for the book and cite the upstream implementation without copying code.

## 6. Suggested implementation pages

1. One high-level pseudocode algorithm for context/target/predictor flow.
2. One official mask-collator excerpt.
3. One regularizer excerpt (SIGReg or VISReg, not both unless both add pedagogical value).
4. One frozen feature extraction example.
5. One latent planning call, if the official source is short enough and the action interface is clear.

The implementation text should focus on questions like “where are target indices applied?” and “which parameters are updated?” rather than API tutorials.

## 7. Reproducibility metadata table

For every excerpt, maintain:
- repository URL;
- exact path;
- commit SHA or tag;
- license and source header;
- lines copied or adapted;
- paper version associated with the code;
- whether the code was executed by the author;
- test/config used;
- limitations and environment requirements.

## 8. Current limitations of inspection

The GitHub connector exposed the source repositories and their metadata, but not all files have yet been inspected line-by-line. This file is an implementation audit plan, not a claim that the official code has been reproduced. The final printed listing must only be selected after direct source inspection.
