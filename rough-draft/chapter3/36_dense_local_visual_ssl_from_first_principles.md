# Dense and local visual self-supervision: objectives beyond one vector per image

## 1. Why global image representations are not enough

A global image embedding compresses an image into one vector. That can be effective for image-level semantic similarity, but it discards explicit spatial correspondence. Dense prediction tasks need representations indexed by position or patch: segmentation, correspondence, localization, object parts, and geometry all depend on where information occurs.

Let an image view v produce a feature map F(v) in R^(H' x W' x d). A global method may pool this map into z = Pool(F). A dense method instead defines targets and losses over positions p, patches, regions, prototypes or local neighborhoods. It therefore adds a structural assumption: local features should preserve or predict relationships tied to image coordinates, crops or patches.

The target correspondence itself is a method choice. Under a random crop, position p in one view may not correspond to the same absolute pixel in another view. Correct cross-view matching requires tracking the crop transform, using overlap, or defining a local-neighborhood target that tolerates displacement.

## 2. DenseCL: contrastive learning at global and local scales

DenseCL extends instance-level contrastive learning with a dense feature objective. The global branch aligns image-level embeddings of two augmented views. The dense branch uses spatial feature vectors and establishes cross-view local correspondences, typically by matching each local feature to a most similar feature in the other view, then applying a contrastive objective to the matched pair.

The global term teaches image-level invariance; the dense term pressures local features to be useful across transformed views. The matching rule is an approximation to correspondence: nearest-feature matches can be wrong when repeated textures, occlusion or crop mismatch produce ambiguous candidates. The method is not equivalent to supervised pixel correspondence because the matches are inferred from features, not supplied by ground-truth geometry.

Source: [WXinlong/DenseCL](https://github.com/WXinlong/DenseCL/tree/b64bbb6e267f729d16ca558233b0ef3028857381), pinned revision b64bbb6e267f729d16ca558233b0ef3028857381. Paper: [DenseCL](https://arxiv.org/abs/2011.09157).

## 3. PixPro: pixel-to-propagation consistency

PixPro learns pixel-level representations through cross-view positive correspondence and a pixel-propagation module. The method builds local representations by propagating information among nearby spatial features, then enforces consistency between corresponding pixels across augmented views. The propagation operator is designed to incorporate local neighborhood structure rather than treating each pixel feature independently.

The inductive bias is that useful pixel representations should be spatially coherent and consistent under view transformations. This can support dense tasks, but local propagation can also smooth across boundaries or spread features through incorrect neighborhoods. Neighborhood scale, correspondence and the feature similarity used by propagation are therefore important design choices.

Source: [zdaxie/PixPro](https://github.com/zdaxie/PixPro/tree/e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc), pinned revision e390d6b60bcb017ed7ea7fd7e6647d14c5da86cc. Paper: [PixPro](https://arxiv.org/abs/2011.10043).

## 4. VICRegL: global invariance plus local feature matching

VICRegL extends variance-invariance-covariance regularization with a local feature objective. Its global branch aligns pooled representations and applies variance/covariance regularization. Its local branch selects spatial features from each view, establishes local correspondences between the two views, and applies an invariance term to matched local representations.

This combines two distinct constraints: global representation geometry and local spatial consistency. The variance/covariance terms discourage collapsed or linearly redundant features at the representation level where they are applied; they do not automatically ensure every spatial location is informative. Local matching provides a different signal and inherits errors from the correspondence procedure.

Source: [facebookresearch/VICRegL](https://github.com/facebookresearch/VICRegL/tree/803ae4c8cd1649a820f03afb4793763e95317620), pinned revision 803ae4c8cd1649a820f03afb4793763e95317620. Paper: [VICRegL](https://arxiv.org/abs/2210.01571).

## 5. iBOT: self-distillation at image and patch levels

iBOT applies a teacher-student self-distillation objective to both global image representations and patch tokens. The teacher receives a less-corrupted view and provides target distributions. The student receives masked image patches and predicts teacher outputs for masked patches, while a class-token/global objective aligns image-level outputs across views. The teacher is updated by EMA; its target distributions are sharpened/centered according to the method's training setup.

Patch-level prediction gives the model a target for local structure, not only a pooled image vector. The tokenizer-like target is learned online through the teacher's output head rather than being a fixed human label. The masking scheme, teacher/student view construction and target centering/temperature are integral to the objective.

Source: [bytedance/ibot](https://github.com/bytedance/ibot/tree/da316d82636a7a7356835ef224b13d5f3ace0489), pinned revision da316d82636a7a7356835ef224b13d5f3ace0489. Paper: [iBOT](https://arxiv.org/abs/2111.07832).

## 6. Leopart: local features and clustering

Leopart focuses on learning local patch representations through clustering and consistency across augmented views. Patch features are assigned to learned prototypes; assignments at corresponding locations are encouraged to agree across views. A balanced assignment mechanism helps prevent all patches from collapsing to one prototype. Unlike global image-level clustering, the target labels are associated with local features, making spatial structure central to the task.

The difficulty is correspondence: a patch must be related across views even though crops, scales and transformations can alter its coordinates. If the correspondence is wrong, the objective enforces consistency between unrelated content. Prototype balancing prevents one form of trivial solution but does not prove that the clusters align with semantic objects.

Source: [MkUuWaUjinga/leopart](https://github.com/MkUuWaUjinga/leopart/tree/7835701338d477c70d581c0a10e594257dcac198), pinned revision 7835701338d477c70d581c0a10e594257dcac198. Paper: [Leopart](https://arxiv.org/abs/2112.09610).

## 7. SelfPatch: learning from patch-level relations

SelfPatch changes the pretext signal from global image similarity toward relationships among image patches. The method extracts patch-level features and learns through patch relationships across views, encouraging representations to capture local content that global pooling can erase. Its implementation and exact objective should be read alongside its paper because patch sampling, correspondence, and loss construction determine which local invariances are actually trained.

The general diagnostic is to ask whether a local feature is supervised by (a) an observed spatial correspondence, (b) a feature-derived match, (c) a cluster assignment, or (d) a reconstruction target. Those targets are not interchangeable. A feature-derived match can inherit representation errors; a fixed coordinate match can fail under crop transforms; cluster assignments can be semantically arbitrary; reconstruction can overemphasize low-level detail.

Source: [alinlab/SelfPatch](https://github.com/alinlab/SelfPatch/tree/81aef149fa984a32bdbb2cf5a98aa0e072591535), pinned revision 81aef149fa984a32bdbb2cf5a98aa0e072591535. Use the repository's paper citation and configuration to identify the precise training variant.

## 8. DINOv2 and DINOv3: strong global and patch features at scale

DINOv2 develops a training recipe that combines teacher-student self-distillation with additional regularization and curated data, targeting broadly useful visual features. Its representations can support both global and dense tasks, but the training system is more than a single cross-entropy loss: data selection, model scale, teacher/student dynamics, and the complete recipe matter.

DINOv3 scales the family further and emphasizes dense feature quality, including a Gram anchoring regularizer in its published recipe. The regularizer constrains feature-map geometry relative to an anchor representation, counteracting spatial-feature degradation that can appear at scale. The important methodological distinction is that a strong global embedding does not automatically guarantee useful spatial features; dense evaluation and local-feature diagnostics are necessary.

Sources: [DINOv2 official repository](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8), [DINOv3 official repository](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751). Papers: [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv3](https://arxiv.org/abs/2508.10104). DINOv3 has a specific Meta license; check the exact license before redistribution or commercial use.

## 9. Dense objective taxonomy

| Method | Main local target | How correspondence/target is obtained | Structural pressure | Main failure mode |
|---|---|---|---|---|
| DenseCL | Local feature pair | Feature similarity matching across views | Dense contrastive alignment | False matches and repeated texture |
| PixPro | Pixel-level feature | Cross-view correspondence plus local propagation | Spatially coherent local features | Propagation across boundaries |
| VICRegL | Local feature pair | Local matching across transformed views | Local invariance plus global variance/covariance constraints | Matching error; global regularization does not guarantee local quality |
| iBOT | Teacher patch distribution | EMA teacher outputs for masked patches | Patch prediction plus global self-distillation | Teacher targets and masking can encode shortcuts |
| Leopart | Local prototype assignment | Corresponding patches across views | Balanced local clustering consistency | Incorrect spatial correspondence or arbitrary clusters |
| SelfPatch | Patch-level relations | Patch sampling and method-specific matching | Local content consistency | Depends on sampling/matching assumptions |
| DINOv2/v3 | Teacher output distributions and spatial features | Teacher-student views; large-scale recipe | Global self-distillation, with dense-feature recipe components | Global quality can conceal weak local geometry |

## 10. How to evaluate local representations

- **Linear probe at global level:** measures image-level linearly accessible information, not spatial correspondence.
- **Dense linear probe:** tests whether frozen per-location features expose segmentation or dense labels with a linear head.
- **Nearest-neighbor correspondence:** tests whether matched locations across views retrieve the same underlying region.
- **Localization and segmentation:** evaluate spatial semantics, boundary quality and small-object behavior.
- **Robustness under crop/scale/occlusion:** tests whether correspondences survive the intended transformations.
- **Feature-map geometry:** inspect spatial smoothness, rank, covariance and whether distinct objects collapse into similar features.
- **Leakage and overlap:** check duplicate images and train/test near-duplicates, especially when using large curated corpora.

No one metric is sufficient. A feature map can be smooth but semantically wrong, discriminative but spatially misaligned, or strong on common objects but weak on rare regions.

## 11. Relation to world models

Dense features can preserve object parts, spatial relationships and motion-relevant locations that a global vector may discard. However, a static image objective still does not identify temporal transitions or causal effects of actions. For a world model, spatial representation quality is a prerequisite in many settings, not a substitute for action-conditioned prediction and evaluation over future outcomes.

A useful state Z_t must preserve whatever variables change the distribution of future observations and rewards under available actions. If a dense SSL method merges two visually similar states that respond differently to an action, its invariance is harmful for control even if segmentation or retrieval metrics are strong.

## 12. Source and reproduction boundary

Commit-pinned submodule references identify the upstream source revision. They do not establish that the source was executed, that training results were reproduced, or that dependencies and pretrained weights are available. For each experiment, record the exact config, input resolution, crop/mask policy, feature layer, target-generation settings, optimizer schedule, dataset provenance, checkpoint hash and license. For methods with shared upstream repositories, record the precise subdirectory and configuration rather than naming the parent repository alone.
