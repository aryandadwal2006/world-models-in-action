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


## 8. Additional exact source paths checked

These immutable URLs locate upstream source; the book repository does not copy or rewrite those implementations.

- **SimCLR**, revision 383d4143fd8cf7879ae10f1046a9baeb753ff438:
  - Objective: https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/objective.py
  - Pretraining call site: https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/model.py
  - Verified: two-view split, optional normalization, self-logit mask, cross-view positives, two directional losses and replica gathering.

- **MoCo**, revision 132aea9da9bb995a411c915c0420be3db62052f3:
  - Builder: https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/moco/builder.py
  - Trainer: https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/main_moco.py
  - Upstream successor 8976944da9c6b94cbd9158d7ebe50912aef807ef deletes the source tree. The gitlink is pinned to the last code-bearing parent. README and LICENSE conflict on license terms.

- **BYOL**, revision f5de0ede8430809180254ee957abf36ed62579ef:
  - Loss/update loop: https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/byol_experiment.py
  - Schedules: https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/utils/schedules.py
  - Verified: symmetric target-stop-gradient regression, online update before parameter EMA, and separate mutable target state.

- **SimSiam**, revision a7bc1772896d0dad0806c51f0bb6f3b16d290468:
  - Builder: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py
  - Trainer: https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py
  - Verified: projector/predictor BatchNorm structure, detached targets and two-way negative cosine objective.

- **Barlow Twins**, revision 8e8d284ca0bc02f88b92328e53f9b901e86b4a3c:
  - Loss: https://github.com/facebookresearch/barlowtwins/blob/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c/main.py
  - Verified: affine-free standardization, cross-device correlation reduction and sum reductions.

- **VICReg**, revision 4e12602fd495af83efd1631fbe82523e6db092e0:
  - Loss: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py
  - Verified: local representation MSE, gathered centered statistics, variance epsilon, covariance denominator and feature-dimension reduction.

These checks do not imply that the methods executed in this repository or that published results were reproduced.


## 9. DINOv2/DINOv3 immutable implementation paths

### DINOv2 at revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8

- DINO class-token target and optional Sinkhorn-Knopp assignment: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/dino_clstoken_loss.py
- iBOT patch loss and masked-patch reduction: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/ibot_patch_loss.py
- KoLeo nearest-neighbour regularizer: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/loss/koleo_loss.py
- Training orchestration: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/train/ssl_meta_arch.py
- Default config: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/configs/ssl_default_config.yaml

The pinned default config sets DINO/iBOT weights to 1.0 and KoLeo to 0.1; centering is selected by default while Sinkhorn-Knopp is also supported. It uses 65,536 prototypes for each head, shared head configuration, mask-sampling probability 0.5, mask-ratio range 0.1–0.5 and eight local crops. These are config defaults rather than constants of every DINOv2 experiment.

### DINOv3 at revision 6876159a11b4df116f30f667f8c9888617df0751

- Class-token loss: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/dino_clstoken_loss.py
- Patch loss: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/ibot_patch_loss.py
- Gram loss: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py
- Standard pretraining config: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_pretrain.yaml
- Gram-anchor config: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml

The Gram implementation forms similarity matrices from target and output features and applies MSE. Configuration controls normalization, image-level versus cross-batch matrices, optional removal of negative similarities, teacher crop construction and loss-weight scheduling. The standard config disables Gram loss while the Gram-anchor config enables it; the latter also changes duration and recipe settings. DINOv3 has a dedicated license agreement.


## 10. JEPA-family implementation paths

- I-JEPA: repository pinned at facebookresearch/ijepa revision 52c1ae95d05f743e000e8f10a1f3a79b10cff048; paper https://arxiv.org/abs/2301.08243. Audit target/context block selection and target normalization from its specific config.
- V-JEPA: paper https://arxiv.org/abs/2404.08471; code family is in facebookresearch/jepa revision 51c59d518fc63c08464af6de585f78ac0c7ed4d5.
- V-JEPA 2 / 2.1: repository pinned at facebookresearch/vjepa2 revision 204698b45b3712590f06245fbfba32d3be539812.
  - V-JEPA 2.1 training objective: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/train.py
  - Predictor and mask-position construction: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/predictor.py
  - Example config: https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/train_2_1/vitg16/pretrain-256px-16f.yaml
  - Verified: masked-target latent prediction, dense context-target loss, distance-weighted context loss, LayerNorm of target features, exponent-1 elementwise absolute-error reduction, and target EMA after the online update. The example YAML has placeholder dataset paths and requests multi-node/high-memory hardware; no run is claimed.
- EB-JEPA: repository pinned at facebookresearch/eb_jepa revision 966e61e9285b3a876f49b9774e9720d9a99a7925.
  - Prediction/unroll: https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/eb_jepa/jepa.py
  - Regularizers: https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/eb_jepa/losses.py
  - Action-conditioned training and planning: https://github.com/facebookresearch/eb_jepa/tree/966e61e9285b3a876f49b9774e9720d9a99a7925/examples/ac_video_jepa
  - Verified: observation encoder plus action encoder plus predictor; total training objective equals prediction cost plus a configurable regularizer; regularizer terms include variance hinge, off-diagonal covariance, temporal similarity and inverse dynamics. Code can perform parallel teacher-forced prediction or autoregressive rollout. The example environment is Two Rooms, distinct from large-scale V-JEPA video pretraining.


### DINO teacher-temperature warm-up parser note

Pinned file: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py

The argument warmup_teacher_temp_epochs is initialized to 0 in the parser, although its help string says “Default: 30”. The parser value is the executable default. The help-text mismatch is documented here because it materially affects a faithful recipe reconstruction; inspect the command/config for the reported run.


## CPC, Deep InfoMax and diffusion official-source audit

- **CPC audio implementation:** facebookresearch/CPC_audio at revision b98a1bdf1fe9ea219816db7a6c28115d404a3510, an archived MIT-licensed implementation for audio and the transfer setting in the 2020 paper. It is not a claim that this audio code is the only or exact original 2018 CPC training setup.
- **Deep InfoMax:** rdevon/DIM at revision bac4765a8126746675f517c7bfa1b04b88044d51. The README identifies JSD, NCE, DV and related objectives and warns that PyTorch-version changes affected results. This is dependence-estimation representation learning; do not treat it as automatic evidence of a normalized generative EBM.
- **DDPM:** hojonathanho/diffusion at revision 1e0dceb3b3495bbe19116a5e1b3596cd0706c543. The official README documents a TensorFlow 1.15/Python 3.5/TPU-GCS-oriented historical recipe.
- **Score-based SDE:** yang-song/score_sde_pytorch at revision cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44.
- **Guided diffusion:** openai/guided-diffusion at revision 22e0df8183507e13a7813f8d38d51b072ca1e67c.

These source revisions were resolved to their upstream GitHub commits during the audit. That confirms the reference identity, not execution, compatibility with modern dependencies or reproduction of published results.


## Updated CPC source status

An author/team-maintained audio implementation is pinned on main-extra: facebookresearch/CPC_audio at revision b98a1bdf1fe9ea219816db7a6c28115d404a3510. This covers the audio-transfer implementation, not every detail of the original CPC paper. Dedicated official code for NNCLR and ALIGN remains unverified in the collection.


## 18. MoCo v3 and SwAV: additional source-level checks

### MoCo v3

Pinned upstream: `facebookresearch/moco-v3` at `c349e6e24f40d3fedb22d973f92defa4cedf37a7`.

- Builder: https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/moco/builder.py
- Training and defaults: https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py
- README: https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/README.md
- Paper: https://arxiv.org/abs/2104.02057

The pinned implementation confirms:
1. Each view produces an online predictor output (q_1,q_2) and a target-key feature (k_1,k_2) from a momentum encoder.
2. Query and key vectors are L2-normalized; target keys are gathered across devices, and each query performs cross-entropy classification over all gathered keys.
3. The positive index includes the rank offset in the global gathered batch.
4. It sums the two cross-view directions: predictor(view 1) predicts the momentum-encoder key for view 2, and vice versa.
5. There is a momentum encoder and an online predictor, but no MoCo v1/v2-style persistent negative queue in this implementation.
6. The EMA update is called within the model's forward method before momentum-key computation; in this training loop that is before the current optimizer step. Do not copy BYOL's after-optimizer update order onto this code path.
7. Parser defaults include global batch size 4,096, output dimension 256, MLP hidden dimension 4,096, momentum 0.99, temperature 1.0, 100 epochs, and LARS as the default optimizer. A cosine option schedules momentum toward one.
8. The loss includes a factor (2T) multiplying cross entropy; this affects gradient scaling and should not be dropped when reconstructing the code's exact scalar loss.

The upstream README says this is a PyTorch/GPU reimplementation; the original paper implementation used TensorFlow/TPUs. The code is an official author/team reference but is not literally the original software stack or proof that this project's training reproduced the paper.

### SwAV

Pinned upstream: `facebookresearch/swav` at `06b1b7cbaf6ba2a792300d79c7299db98b93b7f9`.

- Training, assignments, queue and loss: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py
- Prototype/model head: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/src/resnet50.py
- Method/config examples: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/README.md
- Paper: https://arxiv.org/abs/2006.09882

The source confirms:
1. A projection head maps image features to an embedding space with a prototype linear layer. Prototype weights are L2-normalized during training.
2. For selected assignment crops, logits are detached and passed through a distributed Sinkhorn-Knopp procedure. It exponentiates logits divided by the assignment regularization (epsilon), globally normalizes, then alternately normalizes prototype-row mass and per-sample column mass. The resulting target assignments approximately balance prototype usage over the sample set.
3. For each assignment crop, the loss predicts that assignment from every other crop using student logits divided by the training temperature and log-softmax. It excludes the same crop from the prediction terms.
4. The cross-view assignment loss is averaged over the other views and assignment crops.
5. A queue is optional in the parser (default length 0). When configured, the implementation starts using it after the selected epoch, combines stored embeddings with current assignment scores, and updates its contents with recent embeddings. Queue use changes the assignment sample set; it is not a mandatory component of every SwAV run.
6. Parser defaults include temperature 0.1, three Sinkhorn iterations, 3,000 prototypes, per-GPU batch size 64, queue length 0 and a prototype-gradient freeze period of 313 iterations. Published configurations commonly override these defaults, including enabling a queue.

The README includes multiple training configurations and reported results, but those headline numbers should be attributed to their stated ImageNet/batch/multicrop/epoch settings rather than generalized to the loss alone. Code inspection does not establish that this companion project reran them.


## 19. DeepCluster, BEiT, SimMIM and MaskFeat source checks

### DeepCluster

Pinned upstream: `facebookresearch/deepcluster` revision `2d1927e8e3dd272329e879e510fbbdf1b1d02d17`. Training loop: https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py. Clustering: https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/clustering.py. Paper: https://arxiv.org/abs/1807.05520.

The epoch-level loop extracts dataset features, clusters them with configured K-means or PIC, maps cluster memberships to pseudo-labels, then trains a classifier head with cross-entropy. Parser defaults are 10,000 clusters and reassignment every epoch. This is offline re-clustering, not SwAV's online balanced-prototype objective. The checked-in LICENSE says CC BY-NC 4.0 although GitHub metadata has no SPDX assertion; do not describe the source as commercially unrestricted.

### BEiT

Official code is within `microsoft/unilm/beit/`, pinned at `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`; the source tree's license is MIT. Paper: https://arxiv.org/abs/2106.08254. Model: https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/beit/modeling_pretrain.py. Training loss: https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/beit/engine_for_pretraining.py.

The DALL-E discrete VAE tokenizer maps image patches to codebook IDs. The model replaces masked patch embeddings with a learned mask token, predicts logits only at masked positions, and uses cross-entropy against tokenizer IDs. The target is a discrete visual token, not raw RGB. The original README's base pretraining example uses ImageNet-scale data and 16 V100 GPUs, so this is not a lightweight reproduction.

### SimMIM

Official upstream: `microsoft/SimMIM` at `d3e29bcac950b83edc34ca33fe4404f38309052c`; MIT-licensed and archived September 2026. Paper: https://arxiv.org/abs/2111.09886. Model/loss: https://github.com/microsoft/SimMIM/blob/d3e29bcac950b83edc34ca33fe4404f38309052c/models/simmim.py.

The encoder replaces masked patch embeddings with a learned token. A decoder regresses pixels; the implementation applies elementwise L1, multiplies by the upsampled mask and normalizes by masked-pixel count and channels. The pinned Swin-Base config uses 192-pixel input, mask patch size 32, mask ratio 0.6 and 100 epochs. These values belong to that config, not every SimMIM run.

### MaskFeat

Pinned upstream: `facebookresearch/SlowFast` at `287ec0076846560f44a9327e931a5a2360240533`; repository metadata reports Apache-2.0. Paper: https://arxiv.org/abs/2112.09133. Model: https://github.com/facebookresearch/SlowFast/blob/287ec0076846560f44a9327e931a5a2360240533/slowfast/models/masked.py. Example config: https://github.com/facebookresearch/SlowFast/blob/287ec0076846560f44a9327e931a5a2360240533/configs/masked_ssl/in1k_VIT_B_MaskFeat_PT.yaml.

The cited ImageNet ViT-B config sets `PRED_HOG: True`, mask ratio 0.4, batch size 256 and 300 epochs. It predicts HOG-style local gradient descriptors at masked positions; the configured criterion is multi-MSE. This codebase also supports pixel targets and video objectives, so source descriptions must name the config rather than speak of one universal MaskFeat recipe.
