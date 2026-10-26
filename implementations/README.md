# Official implementation collection

This top-level directory uses **Git submodules**, not copied source trees. Each submodule is pinned to the exact upstream commit recorded in the parent Git tree. This keeps original licenses and histories intact and prevents this book repository from silently forking or modifying upstream code.

Initialize after cloning this branch:

```bash
git submodule update --init --recursive
```

Update one submodule deliberately, review its license and upstream changes, and commit the new gitlink only when the chapter reference is updated. Do not imply that a method is reproduced merely because its code is linked. Several upstream projects contain research code, old environment assumptions, or licenses that differ from the book repository.

## Repository manifest

| Method family | Official repository | Pinned commit | Paper / note |
|---|---|---|---|
| SimCLR / SimCLRv2 | [google-research/simclr](https://github.com/google-research/simclr) | `383d4143fd8cf7879ae10f1046a9baeb753ff438` | Chen et al. (2020), A Simple Framework for Contrastive Learning of Visual Representations; Chen et al. (2020), Big Self-Supervised Models are Strong Semi-Supervised Learners. |
| MoCo v1/v2 | [facebookresearch/moco](https://github.com/facebookresearch/moco) | `132aea9da9bb995a411c915c0420be3db62052f3` | He et al. (2020), Momentum Contrast for Unsupervised Visual Representation Learning; Chen et al. (2020), Improved Baselines with Momentum Contrastive Learning. Pin is the last code-bearing parent before upstream commit `8976944da9c6b94cbd9158d7ebe50912aef807ef` deleted the repository contents; the old README and LICENSE disagree on license terms, so check the pinned LICENSE and upstream notice before redistribution. |
| MoCo v3 | [facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3) | `c349e6e24f40d3fedb22d973f92defa4cedf37a7` | Chen et al. (2021), An Empirical Study of Training Self-Supervised Vision Transformers. |
| BYOL | [google-deepmind/deepmind-research](https://github.com/google-deepmind/deepmind-research) | `f5de0ede8430809180254ee957abf36ed62579ef` | Grill et al. (2020), Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. |
| SimSiam | [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam) | `a7bc1772896d0dad0806c51f0bb6f3b16d290468` | Chen and He (2021), Exploring Simple Siamese Representation Learning. |
| SwAV | [facebookresearch/swav](https://github.com/facebookresearch/swav) | `06b1b7cbaf6ba2a792300d79c7299db98b93b7f9` | Caron et al. (2020), Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. |
| DeepCluster | [facebookresearch/deepcluster](https://github.com/facebookresearch/deepcluster) | `2d1927e8e3dd272329e879e510fbbdf1b1d02d17` | Caron et al. (2018), Deep Clustering for Unsupervised Learning of Visual Features. The pinned LICENSE is CC BY-NC 4.0; review commercial-use permissions. |
| Barlow Twins | [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins) | `8e8d284ca0bc02f88b92328e53f9b901e86b4a3c` | Zbontar et al. (2021), Barlow Twins: Self-Supervised Learning via Redundancy Reduction. |
| VICReg | [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg) | `4e12602fd495af83efd1631fbe82523e6db092e0` | Bardes et al. (2022), VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning. |
| MAE | [facebookresearch/mae](https://github.com/facebookresearch/mae) | `efb2a8062c206524e35e47d04501ed4f544c0ae8` | He et al. (2022), Masked Autoencoders Are Scalable Vision Learners. |
| BERT | [google-research/bert](https://github.com/google-research/bert) | `eedf5716ce1268e56f0a50264a88cafad334ac61` | Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. |
| GPT-2 | [openai/gpt-2](https://github.com/openai/gpt-2) | `9b63575ef42771a015060c964af2c3da4cf7c8ab` | Radford et al. (2019), Language Models are Unsupervised Multitask Learners. |
| DINO v1 | [facebookresearch/dino](https://github.com/facebookresearch/dino) | `7c446df5b9f45747937fb0d72314eb9f7b66930a` | Caron et al. (2021), Emerging Properties in Self-Supervised Vision Transformers. |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Oquab et al. (2023), DINOv2: Learning Robust Visual Features without Supervision. |
| DINOv3 | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | `6876159a11b4df116f30f667f8c9888617df0751` | Siméoni et al. (2025), DINOv3: Self-supervised learning for vision at unprecedented scale. |
| V-JEPA | [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` | Bardes et al. (2024), Revisiting Feature Prediction for Learning Visual Representations from Video. |
| V-JEPA 2 | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `204698b45b3712590f06245fbfba32d3be539812` | Assran et al. (2025), V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. |
| CLIP | [openai/CLIP](https://github.com/openai/CLIP) | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` | Radford et al. (2021), Learning Transferable Visual Models From Natural Language Supervision. |
| SigLIP / SigLIP 2 | [google-research/big_vision](https://github.com/google-research/big_vision) | `0127fb6b337ee2a27bf4e54dea79cff176527356` | Zhai et al. (2023), Sigmoid Loss for Language Image Pre-Training; Tschannen et al. (2025), SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. |
| wav2vec 2.0 / HuBERT / data2vec | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) | `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99` | Baevski et al. (2020), wav2vec 2.0; Hsu et al. (2021), HuBERT; Baevski et al. (2022), data2vec. |
| BEiT and related pretraining work | [microsoft/unilm](https://github.com/microsoft/unilm) | `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` | Bao et al. (2022), BEiT: BERT Pre-Training of Image Transformers. |
| I-JEPA | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | Assran et al. (2023), Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. |
| ELECTRA | [google-research/electra](https://github.com/google-research/electra) | `8a46635f32083ada044d7e9ad09604742600ee7b` | Clark et al. (2020), ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. |
| T5 / span corruption | [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer) | `42909aa15d1fbd16c69333aa474a00ae8ed9fde4` | Raffel et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. |

## Additional official implementation pins

| Method family | Official repository | Pinned commit | Paper / note |
|---|---|---|---|
| W-MSE / whitening | [htdt/self-supervised](https://github.com/htdt/self-supervised) | `d9662c8d07dafd194a9045375f4f6aa09f5b03e9` | W-MSE / whitening source; `implementations/w-mse/` |
| MSN / Masked Siamese Networks | [facebookresearch/msn](https://github.com/facebookresearch/msn) | `4388dc1eadbe3042b85d3296d41b9b207656e043` | MSN / Masked Siamese Networks source; `implementations/msn/` |
| Mean Shift for SSL | [UMBCvision/MSF](https://github.com/UMBCvision/MSF) | `130191c4f9c3dbbd6355a737cfdce5fb6a7b68f0` | Mean Shift for SSL source; `implementations/mean-shift/` |
| VICRegL | [facebookresearch/VICRegL](https://github.com/facebookresearch/VICRegL) | `803ae4c8cd1649a820f03afb4793763e95317620` | VICRegL source; `implementations/vicregl/` |
| DenseCL | [WXinlong/DenseCL](https://github.com/WXinlong/DenseCL) | `b64bbb6e267f729d16ca558233b0ef3028857381` | DenseCL source; `implementations/densecl/` |
| PixPro | [zdaxie/PixPro](https://github.com/zdaxie/PixPro) | `e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc` | PixPro source; `implementations/pixpro/` |
| Leopart | [MkUuWaUjinga/leopart](https://github.com/MkUuWaUjinga/leopart) | `7835701338d477c70d581c0a10e594257dcac198` | Leopart source; `implementations/leopart/` |
| SelfPatch | [alinlab/SelfPatch](https://github.com/alinlab/SelfPatch) | `81aef149fa984a32bdbb2cf5a98aa0e072591535` | SelfPatch source; `implementations/selfpatch/` |
| LeJEPA / SIGReg | [galilai-group/lejepa](https://github.com/galilai-group/lejepa) | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | Paper-linked repository redirects here; pinned `LICENSE` is CC BY-NC 4.0. Review non-commercial and attribution terms before use or redistribution. |

| EB-JEPA | [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | `966e61e9285b3a876f49b9774e9720d9a99a7925` | EB-JEPA source; `implementations/eb-jepa/` |
| CMC | [HobbitLong/CMC](https://github.com/HobbitLong/CMC) | `7b227be0b10ef4e526c72af07664f5079ed9ee09` | Author-published implementation of Contrastive Multiview Coding; `implementations/cmc/`. The same repository also contains MoCo/InsDis code, but its README labels those implementations unofficial. |
| AMDIM | [Philip-Bachman/amdim-public](https://github.com/Philip-Bachman/amdim-public) | `8754ae149ed28da8066f696f95ba4ca0e3ffebd8` | Authors' code for Augmented Multiscale Deep InfoMax; `implementations/amdim/`. Historical dependencies and compute assumptions apply. |
| Historical method suite (PIRL, NPID, NPID++, DeepClusterV2, RotNet, Jigsaw, SwAV and others) | [facebookresearch/vissl](https://github.com/facebookresearch/vissl) | `09270ed25a6c2cf71263d955b64cbe076d34ac45` | Official FAIR/VISSL library with implementations and configs for multiple methods; `implementations/vissl/`. Verify the specific config and code path before calling it a paper reproduction. |

## Important scope / license notes

- SimCLR's Google Research repository is archived. The repository itself explicitly says it is not an official Google product; it remains useful as an author-maintained reference implementation, not an endorsement.
- DeepMind Research is a multi-project repository. BYOL lives in a subdirectory; this is pinned at a repository revision, not a tiny BYOL-only checkout.
- fairseq is a shared research codebase with wav2vec 2.0, HuBERT, and data2vec-related implementations/examples; inspect the relevant subdirectories and historical dependency versions before attempting to run it.
- The UniLM repository includes BEiT-related work, rather than being a guaranteed standalone BEiT-only checkout.
- DINOv3 is under a specific Meta license. Read its current LICENSE before redistributing weights or treating them as open-source under an OSI license.
- These are source repositories, not bundled pretrained weights or datasets. Do not check in multi-gigabyte weights, proprietary corpora, ImageNet, or audio/video datasets.

| SimMIM | [microsoft/SimMIM](https://github.com/microsoft/SimMIM) | `d3e29bcac950b83edc34ca33fe4404f38309052c` | Xie et al. (2022), SimMIM. Official repository archived in September 2026; MIT license; read-only. |
| iBOT | [bytedance/ibot](https://github.com/bytedance/ibot) | `da316d82636a7a7356835ef224b13d5f3ace0489` | Zhou et al. (2022), iBOT: Image BERT Pre-Training with Online Tokenizer. |
| MaskFeat | [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) | `287ec0076846560f44a9327e931a5a2360240533` | Wei et al. (2022), Masked Feature Prediction for Self-Supervised Visual Pre-Training; implementation is in the projects/maskfeat subdirectory. |


## Coverage notes added for the expanded collection

- Shared upstream repositories may contain multiple methods and dependencies; the method-specific source lives in the upstream subdirectory, not in a locally copied standalone implementation.
- The V-JEPA 2 source includes the V-JEPA 2.1 release. The LeJEPA paper-linked repository redirects to `galilai-group/lejepa`; the pinned source license is CC BY-NC 4.0, not a blanket permissive license. VISSL is a shared historical-method library, not a dedicated upstream for each method it implements. VISReg's author-linked GitHub pointer leads to `altos-labs/visreg`, which was not accessible as a public repository during this audit; no code submodule is included, and paper-level claims should not be described as code-verified.
- No public author/team repository was verified in this pass for ALIGN or the original NNCLR paper. CPC Audio and Deep InfoMax now have author/team source repositories pinned; the original CPC paper and the later audio-transfer implementation are distinct references. CMC and AMDIM also have published code repositories pinned, while VISSL supplies FAIR-maintained reference implementations for historical methods. These are intentionally not represented by an unofficial repository falsely labeled official.
- Initializing the whole collection clones substantial upstream history. Use `git submodule update --init implementations/<name>` to initialize one repository at a time.


## Checkout integrity audit — 2026-10-26

The current parent tree contains 45 implementation submodule pointers, and all 45 paths have matching entries in .gitmodules. Every pinned commit was queried against its upstream GitHub repository during this audit; each commit resolves and points to a non-empty Git tree. This checks source-reference integrity, not whether every upstream can still train without dependency changes, whether every upstream folder contains a dedicated implementation rather than a shared multi-method library, or whether any paper result has been reproduced.

MoCo is the documented exception to ordinary newest-commit pinning: its current upstream head points to an empty tree after a deletion commit, so the submodule stays pinned to its last code-bearing parent. The README and LICENSE in that upstream snapshot state conflicting license terms; do not redistribute MoCo source based on this audit alone.

To materialize the source checkouts into local implementation folders after cloning the parent repository, run the supplied script from the repository root:

\`\`\`bash
bash implementations/clone-all.sh
\`\`\`

The script synchronizes submodule URLs and initializes all pinned checkouts recursively. It does not copy source files into the parent repository, alter upstream histories, fetch weights or datasets, or modify any code inside the submodules. On the GitHub web UI, a submodule is displayed as a commit-pinned link rather than an expanded copy of the upstream tree.


## CPC and generative-model boundary repositories

The following are pinned originals or author/team-maintained repositories. They are included for source study and boundary comparisons; their presence does not reclassify every generative objective as representation learning.

| Family | Upstream repository | Pinned commit | Scope and warning |
|---|---|---|---|
| CPC audio | [facebookresearch/CPC_audio](https://github.com/facebookresearch/CPC_audio) | `b98a1bdf1fe9ea219816db7a6c28115d404a3510` | Archived MIT-licensed audio implementation linked to the later cross-language transfer study. Not a promise that its old audio dependencies run unchanged. |
| Deep InfoMax | [rdevon/DIM](https://github.com/rdevon/DIM) | `bac4765a8126746675f517c7bfa1b04b88044d51` | Author-maintained; BSD-3-Clause. Includes JSD, NCE, DV and local/global representation objectives. README records sensitivity to historical PyTorch versions. |
| DDPM | [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) | `1e0dceb3b3495bbe19116a5e1b3596cd0706c543` | Original authors' denoising diffusion implementation. GitHub does not expose an SPDX license for this repository; review its files and permissions before redistribution. TensorFlow 1.15/Python 3.5 and TPU/GCS assumptions are historical. |
| Score-based SDE | [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) | `cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44` | Author-team source; Apache-2.0. Implements score-based generative modeling through SDEs. |
| Guided diffusion | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | `22e0df8183507e13a7813f8d38d51b072ca1e67c` | OpenAI repository; MIT. Conditional/guided generative diffusion, not a substitute for a representation-learning benchmark. |
