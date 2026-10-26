# Upstream code-source audit: exact paths, pinned revisions and limitations

**Purpose.** The implementation manifest identifies repositories; this note identifies the primary files and experiment/configuration entry points that help verify a method claim. Paths below were resolved against the pinned upstream Git trees. A code path is evidence of where an implementation lives, not proof that the paper has been reproduced or that it still runs with current dependencies.

The parent project keeps upstream Git submodules on the **main-extra** branch under **implementations/**. Fetch their checkouts locally with `git submodule update --init --recursive` from the repository root. No source is copied into these notes and no replacement implementation is authored here.

## 1. Global/local visual methods

### DenseCL

- Pinned upstream: https://github.com/WXinlong/DenseCL at commit `b64bbb6e267f729d16ca558233b0ef3028857381`.
- Core model: https://github.com/WXinlong/DenseCL/blob/b64bbb6e267f729d16ca558233b0ef3028857381/openselfsup/models/densecl.py
- Paper configs: https://github.com/WXinlong/DenseCL/tree/b64bbb6e267f729d16ca558233b0ef3028857381/configs/selfsup/densecl
- Audit: the pinned model exposes a queue length and loss-mixture coefficient and maintains separate global/local queues. Check local-positive correspondence, key queues, local projection head, temperature and the global/dense loss combination. Do not present its local loss as a pixel-level term independent of candidate memory.
- Primary paper: Wang et al. (2021), https://arxiv.org/abs/2011.09157.

### PixPro

- Pinned upstream: https://github.com/zdaxie/PixPro at commit `e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc`.
- Core model and local regression loss: https://github.com/zdaxie/PixPro/blob/e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc/contrast/models/PixPro.py
- Pretraining entry point: https://github.com/zdaxie/PixPro/blob/e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc/main_pretrain.py
- Audit: inspect pixel propagation, corresponding coordinates between views, asymmetric normal/propagated branches, regression consistency and optional instance-level loss weight. PixPro is not merely “pixel InfoNCE”: its signature mechanism is consistency with a propagated local target.
- Primary paper: Xie et al. (2021), https://arxiv.org/abs/2011.10043.

### VICRegL

- Pinned upstream: https://github.com/facebookresearch/VICRegL at commit `803ae4c8cd1649a820f03afb4793763e95317620`.
- Main training/loss module: https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py
- Audit: inspect the VICReg loss function; invariance MSE, variance-floor and covariance components; local correspondence selection; feature positions before final pooling; global/local coefficients. The paper's “close in feature space OR geometrically consistent” local pairing rule is method-specific. Replacing it with arbitrary matching must be recorded as a deviation.
- Status caveat: the upstream repository is archived; read its pinned LICENSE before adapting or redistributing code.
- Primary paper: Bardes, Ponce and LeCun (2022), https://arxiv.org/abs/2210.01571.

### Leopart

- Pinned upstream: https://github.com/MkUuWaUjinga/leopart at commit `7835701338d477c70d581c0a10e594257dcac198`.
- Core method: https://github.com/MkUuWaUjinga/leopart/blob/7835701338d477c70d581c0a10e594257dcac198/src/leopart.py
- Fine-tuning entry point: https://github.com/MkUuWaUjinga/leopart/blob/7835701338d477c70d581c0a10e594257dcac198/experiments/finetune_with_leopart.py
- Audit: DINO initialization and teacher flag; Sinkhorn parameters; prototype count; foreground/all/background loss mask; crop-overlap/ROI alignment; feature queue. Balanced assignment and cross-crop alignment are core training mechanics, not postprocessing details.
- Primary paper: Ziegler and Asano (2022), https://arxiv.org/abs/2204.13101.

### SelfPatch

- Pinned upstream: https://github.com/alinlab/SelfPatch at commit `81aef149fa984a32bdbb2cf5a98aa0e072591535`.
- Training entry point: https://github.com/alinlab/SelfPatch/blob/81aef149fa984a32bdbb2cf5a98aa0e072591535/main_selfpatch.py
- ViT and head: https://github.com/alinlab/SelfPatch/blob/81aef149fa984a32bdbb2cf5a98aa0e072591535/selfpatch_vision_transformer.py
- Audit: number of confident neighboring patches selected; patch output dimension; teacher/student flow; combination with the base image-level objective. The neighbor prior does not mean every adjacent patch belongs to the same semantic region.
- Primary paper: Yun et al. (2022), https://arxiv.org/abs/2206.07990.

## 2. Speech methods in fairseq and UniLM

### wav2vec 2.0

- Pinned upstream: https://github.com/facebookresearch/fairseq at commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`.
- Model: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/models/wav2vec/wav2vec2.py
- Criterion: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/criterions/wav2vec_criterion.py
- Recipe: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/wav2vec/README.md
- Audit: latent-span masking, quantizer, negative sampling, contrastive term and codebook diversity. Distinguish unlabeled pretraining from transcript-based ASR fine-tuning.

### HuBERT

- Pinned upstream: the same fairseq commit above.
- Model: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/models/hubert/hubert.py
- Official recipe and label-rate instructions: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/README.md
- Clustering and pseudo-label tools: https://github.com/facebookresearch/fairseq/tree/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert/simple_kmeans
- Audit: frame-rate alignment; source/version of cluster labels; masked prediction positions; pseudo-label iteration; separation between pretraining and CTC fine-tuning.

### data2vec

- Pinned upstream: fairseq, the same commit above.
- Model family and configs: https://github.com/facebookresearch/fairseq/tree/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/data2vec
- Guide: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/data2vec/README.md
- Audit: teacher target construction and layer aggregation; normalization; masked-token selection; modality-specific wrapper; distinction between data2vec and data2vec 2.0. The upstream README describes v2 as an efficiency-oriented update, so the config version must match the paper being discussed.

### WavLM

- Pinned upstream: https://github.com/microsoft/unilm at commit `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`.
- Core model: https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/wavlm/WavLM.py
- Transformer modules: https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/wavlm/modules.py
- Official instructions/models: https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/wavlm/README.md
- Audit: preserve the published corruptions/utterance mixing and gated relative position bias when discussing the full method. The paper's denoising objective should not be reduced to “add an arbitrary loss” without checking training configuration.

## 3. Image-text and large-scale teacher systems

### CLIP

- Pinned upstream: https://github.com/openai/CLIP at commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
- Model definitions: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py
- Model/tokenizer interface: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py
- **Limit:** this official public repo provides model loading/inference and feature extraction, not a complete reproduction of the original large-scale CLIP pretraining pipeline. Do not cite it as if it exposed the original full training recipe.

### SigLIP and SigLIP 2

- Pinned upstream: https://github.com/google-research/big_vision at commit `0127fb6b337ee2a27bf4e54dea79cff176527356`.
- SigLIP trainer/loss: https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py
- Project/configuration docs: https://github.com/google-research/big_vision/tree/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/configs/proj/image_text
- Audit: pair labels, logit scaling/bias, pair construction across devices and loss reduction. The trainer implements pairwise sigmoid log-likelihood, not CLIP's softmax loss. SigLIP 2 is a broader training recipe rather than just a loss substitution.

### DINOv3

- Pinned upstream: https://github.com/facebookresearch/dinov3 at commit `6876159a11b4df116f30f667f8c9888617df0751`.
- Training architecture: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/train/ssl_meta_arch.py
- Loss modules: https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss
- Gram loss: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py
- Gram-anchoring configuration: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml
- Audit: student/teacher losses versus Gram anchoring; whether similarity is computed per image or across the batch; exact data and schedule. The presence of a GramLoss class alone does not isolate the contribution of Gram anchoring from scale, data or training duration.

## 4. LeJEPA / SIGReg and VISReg source limits

### LeJEPA / SIGReg

- Pinned upstream: https://github.com/galilai-group/lejepa at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`.
- Minimal reference and caveats: https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md
- Epps–Pulley test implementation: https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py
- Scripts: https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052/scripts
- Limit: this repository includes SIGReg/statistical-test code and minimal examples; it should not automatically be treated as a full-scale JEPA benchmark reproduction. The minimal guide explicitly notes differences from the paper implementation. Compare target, model, loss, and config to the paper release.

### VISReg

The public pointer at https://github.com/HaiyuWu/visreg contains only a README directing readers to https://github.com/altos-labs/visreg. That destination was not accessible as a public repository through the GitHub API during this audit. Accordingly, **main-extra** no longer presents the pointer README as a cloned implementation. The authors' project page and paper remain available at https://haiyuwu.github.io/visreg/ and https://arxiv.org/abs/2606.02572. Do not substitute LeJEPA code or community code and call it the authors' official VISReg implementation.

## 5. Evidence hierarchy for code-to-paper claims

Distinguish:
1. **Paper loss definition:** the target and objective in the cited primary paper.
2. **Official source location:** a function/file realizing that target at a pinned revision.
3. **Paper configuration:** the official config/command matching the reported experiment, including data, batch, masking/negative sampling and optimizer.
4. **Executed reproduction:** a run on identified data/hardware compared using the paper's evaluation protocol.
5. **Independent reproduction:** results from a separate implementation/team.

A source checkout establishes item 2. A config can support item 3. Neither establishes items 4 or 5.

## 6. Verification scope

The paths above were resolved from Git trees at the pinned commits. This audit covers the most opaque local, speech, cross-modal, and modern regularization implementations. It intentionally avoids copying upstream code. For further work, inspect the immutable source links, compare with the exact paper version, and document any mismatch before writing a runnable teaching example.


## 7. Historical contrastive sources added on main-extra

The following pinned source paths were added after the earlier local/speech/multimodal audit.

### CMC: author-published implementation

- Pinned source: [HobbitLong/CMC](https://github.com/HobbitLong/CMC/tree/7b227be0b10ef4e526c72af07664f5079ed9ee09), stored as implementations/cmc/.
- Data and view construction: [dataset.py](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/dataset.py).
- Training orchestration: [train_CMC.py](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/train_CMC.py).
- NCE machinery: [NCE/NCEAverage.py](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/NCE/NCEAverage.py) and [NCE/NCECriterion.py](https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/NCE/NCECriterion.py).
- Scope: this is the authors' CMC implementation. The same repository labels its MoCo/InsDis implementation as unofficial.

### AMDIM: author-published implementation

- Pinned source: [Philip-Bachman/amdim-public](https://github.com/Philip-Bachman/amdim-public/tree/8754ae149ed28da8066f696f95ba4ca0e3ffebd8), stored as implementations/amdim/.
- Architecture: [model.py](https://github.com/Philip-Bachman/amdim-public/blob/8754ae149ed28da8066f696f95ba4ca0e3ffebd8/model.py).
- Objective/critic costs: [costs.py](https://github.com/Philip-Bachman/amdim-public/blob/8754ae149ed28da8066f696f95ba4ca0e3ffebd8/costs.py).
- Training-task definition and orchestration: [task_self_supervised.py](https://github.com/Philip-Bachman/amdim-public/blob/8754ae149ed28da8066f696f95ba4ca0e3ffebd8/task_self_supervised.py) and [train.py](https://github.com/Philip-Bachman/amdim-public/blob/8754ae149ed28da8066f696f95ba4ca0e3ffebd8/train.py).
- Scope: the README describes a historical multi-GPU setup and an Apex dependency for mixed precision. A pinned checkout is a source reference, not evidence of an unmodified modern reproduction.

### VISSL: shared FAIR research library

- Pinned source: [facebookresearch/vissl](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45), stored as implementations/vissl/.
- Method configurations: [npid](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/pretrain/npid), [pirl](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/pretrain/pirl), [jigsaw](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/pretrain/jigsaw), [simclr](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/pretrain/simclr) and [swav](https://github.com/facebookresearch/vissl/tree/09270ed25a6c2cf71263d955b64cbe076d34ac45/configs/config/pretrain/swav).
- Representative loss implementations: [nce_loss.py](https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/losses/nce_loss.py), [simclr_info_nce_loss.py](https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/losses/simclr_info_nce_loss.py), [swav_loss.py](https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/losses/swav_loss.py) and [barlow_twins_loss.py](https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/losses/barlow_twins_loss.py).
- Scope: VISSL is a shared research library, not a dedicated repository for every method it contains. Match the config, code path and paper version before claiming a reproduction.
