# Visual self-supervision: contrastive, predictive, clustering and reconstruction objectives

This note compares major visual self-supervised learning families from their target construction and optimization mechanics. The central distinction is whether training (i) brings paired views together while separating negatives, (ii) predicts a target representation without explicit negatives, (iii) predicts cluster assignments, or (iv) reconstructs hidden input content.

## 1. Notation and shared setup

Let x be an image, t_1 and t_2 stochastic augmentations, and v_1=t_1(x), v_2=t_2(x). An encoder f maps views to representations z_i=f(v_i); a projection head g may produce h_i=g(z_i). The projection head is often used only during pretraining. The downstream representation may be z, not h, so evaluation of the correct layer matters.

Augmentation defines the invariance target. If color, crop, texture, orientation or object parts are removed or altered, the loss encourages some information to become irrelevant. This is useful only when the removed information is not needed by the downstream task. An invariance is an assumption about the task, not a universally correct property.

## 2. SimCLR: instance discrimination with in-batch negatives

For a batch of N images, two views create 2N representations. Let i and j be the two views of the same source image. The normalized temperature-scaled cross-entropy (NT-Xent) loss for anchor i is
L_i = -log [ exp(sim(h_i,h_j)/tau) / sum_{k != i} exp(sim(h_i,h_k)/tau) ].
The denominator includes the positive and all other views in the batch as negatives. The batch-averaged objective applies the loss in both directions for each positive pair.

The numerator rewards agreement between two augmented views. The denominator discourages the anchor from matching representations of other instances. Thus the representation is trained for instance discrimination, not directly for semantic class prediction. False negatives occur when different images depict the same semantic class or near-duplicate content. Larger batches increase the number of negatives and can change the learned geometry.

The temperature tau changes how sharply the softmax weights high-similarity candidates. Small tau concentrates gradient on the hardest candidates; large tau spreads it across more candidates. Augmentation strength, projection head, batch size, and temperature are part of the method, not incidental tuning.

Official code: [google-research/simclr](https://github.com/google-research/simclr/tree/383d4143fd8cf7879ae10f1046a9baeb753ff438), pinned revision 383d4143fd8cf7879ae10f1046a9baeb753ff438. Paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).

## 3. MoCo: a dictionary whose keys evolve slowly

MoCo retains a contrastive objective but changes the negative dictionary. The query encoder f_q receives gradient updates. A key encoder f_k is updated by exponential moving average:
theta_k <- m theta_k + (1-m) theta_q.
Keys from earlier batches can be stored in a queue, allowing many negatives without requiring a single huge batch. The key encoder changes slowly so representations in the queue remain more consistent than if they were produced by a rapidly changing query encoder.

For query q, positive key k+, and queued keys k_i, the loss is
L = -log [ exp(q dot k+ / tau) / (exp(q dot k+ / tau) + sum_i exp(q dot k_i / tau))].
The queue creates a temporal mismatch: old keys were encoded by earlier key parameters. Momentum updates reduce, but do not eliminate, this staleness. The queue's size, key momentum and data distribution affect the approximation.

MoCo v2 keeps the momentum-encoder/queue framework and changes design choices such as augmentation, MLP projection and cosine learning-rate scheduling. MoCo v3 explores self-supervised ViT training and adds an empirical study of training stability; it should not be described as merely the exact MoCo v1 recipe with a new backbone.

Official sources: [MoCo v1/v2](https://github.com/facebookresearch/moco/tree/132aea9da9bb995a411c915c0420be3db62052f3), [MoCo v3](https://github.com/facebookresearch/moco-v3/tree/c349e6e24f40d3fedb22d973f92defa4cedf37a7). Papers: [MoCo](https://arxiv.org/abs/1911.05722), [Improved Baselines](https://arxiv.org/abs/2003.04297), [MoCo v3](https://arxiv.org/abs/2104.02057).

## 4. BYOL: regression to a momentum target network

BYOL uses online and target networks. The online network produces a projected representation and predictor output q_theta(v_1); the target network produces a stop-gradient target z_xi(v_2). A common normalized regression loss is
L = || q_theta(v_1)/||q_theta(v_1)|| - sg(z_xi(v_2))/||z_xi(v_2)|| ||^2.
The roles are swapped across the two views. The target parameters are an EMA of online parameters, not updated by direct backpropagation.

BYOL has no explicit negative examples. It avoids the direct pairwise separation mechanism used by SimCLR and MoCo. Its asymmetry matters: online predictor, stop-gradient target and slow target-network update make the optimization different from simply minimizing the distance between two identical networks. Collapse remains a mathematical possibility for trivial representations; the architecture and optimization dynamics are intended to avoid it, not a proof that every configuration must avoid collapse.

Official source: [google-deepmind/deepmind-research BYOL](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/byol), pinned repository revision f5de0ede8430809180254ee957abf36ed62579ef. Paper: [BYOL](https://arxiv.org/abs/2006.07733).

## 5. SimSiam: stop-gradient without a momentum encoder

SimSiam uses two augmented views, a shared encoder and projection head, and a predictor on one branch. It minimizes negative cosine similarity between the predictor output of one view and the stop-gradient projection of the other, with directions swapped. There is no negative dictionary and no EMA target network.

The stop-gradient is essential to the proposed training setup: both branches are not simply optimized symmetrically through the same loss graph. The paper's ablations show that removing stop-gradient causes collapse in the studied setup. This is empirical evidence about the architecture and optimization, not a universal theorem that stop-gradient alone guarantees non-collapse for arbitrary models.

Official source: [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam/tree/a7bc1772896d0dad0806c51f0bb6f3b16d290468), pinned revision a7bc1772896d0dad0806c51f0bb6f3b16d290468. Paper: [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566).

## 6. Barlow Twins: match cross-correlation to identity

Given two views, compute batch-normalized feature matrices Z_A and Z_B. Their cross-correlation matrix is
C_ij = sum_b Z_A[b,i] Z_B[b,j] / B.
The loss is
L = sum_i (1-C_ii)^2 + lambda sum_{i != j} C_ij^2.
The diagonal term aligns corresponding dimensions across views. The off-diagonal term discourages redundancy between different feature dimensions. The loss does not require explicit negative images.

The feature normalization and batch statistics matter. The objective seeks view invariance while encouraging decorrelated coordinates, but decorrelation is not the same as statistical independence in general. A zero covariance between dimensions does not imply independence unless additional distributional assumptions hold.

Official source: [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins/tree/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c), pinned revision 8e8d284ca0bc02f88b92328e53f9b901e86b4a3c. Paper: [Barlow Twins](https://arxiv.org/abs/2103.03230).

## 7. VICReg: invariance, variance and covariance are separate constraints

VICReg uses two representations of augmented views and combines three terms:
L = lambda L_inv + mu L_var + nu L_cov.
The invariance term is mean-squared error between paired representations. The variance term penalizes each feature dimension when its batch standard deviation falls below a target gamma, for example
L_var(Z)= (1/d) sum_j max(0, gamma - sqrt(Var(Z_:,j)+epsilon)).
The covariance term penalizes off-diagonal entries of the feature covariance matrix:
L_cov(Z)= (1/d) sum_{i != j} Cov(Z)_{ij}^2.
Both views contribute variance and covariance regularization.

The terms serve distinct purposes. Invariance aligns paired views; variance discourages constant dimensions; covariance discourages linear redundancy. The variance constraint prevents the all-constant representation from trivially satisfying invariance, while covariance encourages a less redundant coordinate system. These constraints do not guarantee semantic information or downstream utility; those depend on data, architecture, augmentations and optimization.

Official source: [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg/tree/4e12602fd495af83efd1631fbe82523e6db092e0). Paper: [VICReg](https://arxiv.org/abs/2105.04906).

## 8. SwAV: predict assignments across views

SwAV maintains prototype vectors c_k and computes assignments of each view to prototypes. Rather than requiring the representation of view A to directly match the vector of view B, it predicts the assignment of one view from the representation of another. A balanced assignment matrix is typically obtained using a Sinkhorn-Knopp procedure over batch similarities. If q_A is the assignment for view A and p_B is the softmax prototype distribution for view B, a cross-view loss is
L(A->B) = -sum_k q_A,k log p_B,k,
with the opposite direction also used.

Balanced assignments help prevent trivial use of one prototype for every sample. The target is a distribution over prototypes, not a class label. Multi-crop training lets global views predict assignments for local crops, adding a spatial-scale constraint.

Official source: [facebookresearch/swav](https://github.com/facebookresearch/swav/tree/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9). Paper: [SwAV](https://arxiv.org/abs/2006.09882).

## 9. DINO: self-distillation across views

DINO trains a student network to match the output distribution of a teacher network across views. The teacher is updated by EMA and does not receive gradient from the student loss. The student distribution is softened by temperature; the teacher distribution is centered and sharpened. For teacher distribution p_t and student distribution p_s over K output dimensions,
L = -sum_k p_t,k log p_s,k.
Cross-view training asks a view's student output to match the teacher's distribution from another view.

Centering and teacher-temperature schedules address collapse dynamics in the distributional objective. The output dimensions are learned prototypes rather than predefined classes. DINO is not identical to BYOL: BYOL regresses vectors with a predictor, while DINO matches normalized output distributions and uses centering/sharpening.

Official source: [facebookresearch/dino](https://github.com/facebookresearch/dino/tree/7c446df5b9f45747937fb0d72314eb9f7b66930a). Paper: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).

## 10. MAE and SimMIM: reconstruct hidden content

MAE masks a large fraction of image patches, encodes only visible patches, and uses a decoder to reconstruct the missing pixel patches. Its loss is commonly mean-squared pixel error over masked patches. The encoder therefore avoids processing masked tokens, which reduces compute at high mask ratios. The reconstruction target is low-level input content, not a teacher feature.

SimMIM also masks image regions and reconstructs their content, but its implementation and prediction head differ from MAE. A method name should not be treated as an exact architecture specification; compare the actual mask token handling, encoder path, prediction head, target normalization and loss mask.

MAE's reconstruction objective can encourage useful representations, but pixel reconstruction may spend capacity on texture or color details that do not matter for semantics. A latent prediction objective instead defines targets in representation space, shifting the inductive bias from exact input reconstruction to feature prediction.

Official sources: [MAE](https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8), [SimMIM](https://github.com/microsoft/SimMIM/tree/d3e29bcac950b83edc34ca33fe4404f38309052c). Papers: [MAE](https://arxiv.org/abs/2111.06377), [SimMIM](https://arxiv.org/abs/2111.09886).

## 11. Objective comparison

| Family | Positive target | Negative samples? | Anti-collapse mechanism or pressure | Main target bias |
|---|---|---|---|---|
| SimCLR | Same-image augmented view | Explicit in-batch negatives | Contrastive separation | Instance discrimination and augmentation invariance |
| MoCo | Same-image momentum key | Explicit queue negatives | Contrastive separation | Instance discrimination with a slowly evolving dictionary |
| BYOL | EMA target-network vector | No explicit negatives | Online predictor, stop-gradient and EMA target | Cross-view feature agreement |
| SimSiam | Other-view projection | No explicit negatives | Predictor and stop-gradient | Cross-view feature agreement |
| Barlow Twins | Other-view feature dimensions | No explicit negatives | Identity cross-correlation target | Invariance plus decorrelation |
| VICReg | Other-view representation | No explicit negatives | Variance floor and covariance penalty | Invariance plus spread and low linear redundancy |
| SwAV | Prototype assignment | No instance negatives in the standard assignment loss | Balanced assignments | Cross-view cluster consistency |
| DINO | Teacher output distribution | No explicit instance negatives | Teacher EMA, centering and sharpening | Cross-view distributional consistency |
| MAE / SimMIM | Original masked image content | No | Reconstruction target | Recover input details from visible context |

“No explicit negatives” does not mean no competition or constraints. Prototype distributions, variance terms, covariance penalties, teacher dynamics and reconstruction all impose different structures.

## 12. Failure modes and diagnostics

- **False negatives:** contrastive methods can push apart different instances that share semantic content.
- **Augmentation mismatch:** a transformation may remove information needed by the downstream task.
- **Representation collapse:** outputs become constant or occupy a low-dimensional subspace; inspect per-dimension variance, covariance spectrum and nearest-neighbor behavior, not just training loss.
- **Target-network lag:** momentum methods use stale targets or queued keys; inspect momentum, queue age and training stability.
- **Batch-statistic dependence:** Barlow Twins and VICReg estimate feature statistics from minibatches; batch size and distributed aggregation matter.
- **Shortcut reconstruction:** pixel reconstruction can emphasize texture and color over task-relevant structure.
- **Layer mismatch:** projection-head outputs optimized by the pretext objective need not be the best downstream representation.
- **Data leakage:** duplicate images or near-duplicates across train/validation sets can inflate representation evaluations.

## 13. What this implies for world models

Visual SSL learns invariances or reconstructive features from images. A world model needs representations that preserve variables relevant to future outcomes and actions. If a crop augmentation removes an object that affects reward, or a contrastive loss treats two different physical states as equivalent, the learned invariance can erase control-relevant distinctions.

The key test is not simply whether two views are close. It is whether the learned state supports the required conditional prediction:
P(O_{t+1}, R_t | H_t, A_t) = P(O_{t+1}, R_t | Z_t, A_t)
under the intended data and action distribution. This condition is an objective for evaluation, not a guarantee supplied by any of the visual SSL losses above.

## 14. Source integrity

The repository links above are commit-pinned and point to official author/team repositories or the stated project maintainers. A submodule pointer proves which upstream revision is referenced; it does not prove that code was executed, that a published result was reproduced, or that the source remains compatible with current dependencies. For historical repositories, inspect LICENSE, dependency versions, training configs, and checkpoint/data provenance before execution or redistribution.
