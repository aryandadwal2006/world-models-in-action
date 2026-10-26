# Historical contrastive lineages: instance discrimination, pretext invariance, multiview coding and multiscale prediction

**Role in the dossier.** This note fills in the bridge between early instance-discrimination methods and the better-known SimCLR/MoCo/CPC families. It is a first-principles explanation and source map, not a claim that all four methods are variants of one identical loss. NPID/PIRL, CMC and AMDIM share a contrastive view of learning, but they define positives, negatives, candidate sets and feature granularity differently.

## 1. Start with the statistical question

Let \(X\) be an image or observation drawn from a dataset. The encoder \(f_\theta\) maps a view \(v\) to a feature \(h=f_\theta(v)\); a projection head, if present, maps it to \(z=g_\phi(h)\). A contrastive method needs more than an encoder. It must define:

1. a **query** whose representation will be scored;
2. a **positive relation** the query should recognize;
3. a **negative distribution** that supplies alternatives;
4. a **score** and normalization that turn those comparisons into a learning signal;
5. a **feature store or candidate set** that determines which comparisons are available.

Those choices are the supervision. "Unlabeled" means there is no human target class label, not that the training process lacks labels or assumptions. Instance identity, a common source image, a shared scene, a known augmentation relation and a spatial context can all induce targets.

For a query \(q\), positive \(k^+\), negative candidates \(k^-_1,\ldots,k^-_M\), similarity \(s(q,k)\) and temperature \(\tau>0\), a common softmax-form contrastive objective is

\[
\ell(q,k^+,\{k^-_j\}) =
-\log \frac{\exp(s(q,k^+)/\tau)}
{\exp(s(q,k^+)/\tau)+\sum_{j=1}^{M}\exp(s(q,k^-_j)/\tau)}.
\]

This is a useful shared mathematical reference, but it must not be retroactively treated as the exact loss for every historical paper. Some early methods used **noise-contrastive estimation (NCE)** as binary data-versus-noise classification, while later methods such as SimCLR commonly use a multiclass InfoNCE/NT-Xent softmax over a batch. The normalization, noise distribution and estimator are part of the algorithm.

For the softmax expression, define candidate logits \(a_j=s(q,k_j)/\tau\), with the positive at index \(y\). Then

\[
\ell=-a_y+\log\sum_j e^{a_j},\qquad
\frac{\partial\ell}{\partial a_j}=p_j-\mathbf 1[j=y],
\]

where \(p_j=e^{a_j}/\sum_k e^{a_k}\). Gradient descent raises the positive score relative to the normalized candidate set and lowers the scores of candidates receiving probability mass. A difficult negative with high \(p_j\) receives a larger repulsive gradient than an already easy negative. If that "negative" depicts the same semantic object or a related state, the mechanism can suppress a useful similarity. The mathematics enforces the chosen pair relation; it does not know whether that relation matches the downstream task.

## 2. Non-parametric instance discrimination (NPID)

### 2.1 Replace unavailable class labels with instance identity

In supervised classification, a classifier learns parameters associated with a finite set of semantic classes. NPID (Wu et al., CVPR 2018) asks a different question: can a feature identify the image instance that generated it, without human labels?

For a dataset with \(N\) training images, assign every image its own index \(i\in\{1,\ldots,N\}\). The index is not a semantic class. It is a proxy label constructed from the dataset's identity. Let \(v_i\) be the current feature for image \(i\), and let \(m_j\) be the stored feature vector associated with instance \(j\). A non-parametric classifier has probabilities of the form

\[
P(i\mid v)=
\frac{\exp(m_i^\top v/\tau)}
{\sum_{j=1}^{N}\exp(m_j^\top v/\tau)}.
\]

Normalizing features or using a temperature is a recipe choice that affects the geometry and sharpness of these probabilities. The conceptual point is that the stored features \(m_j\), rather than a separately learned classifier weight for every class, act as the class prototypes. The complete sum over all \(N\) instances can be expensive; NPID uses noise-contrastive estimation and a memory bank to make the task tractable.

### 2.2 Why a memory bank is required

An encoder update changes the feature for an image. Recomputing every feature after each minibatch would be expensive, so a memory bank stores a representation for each instance and refreshes it when the corresponding examples are seen. Some stored vectors are therefore stale relative to the current encoder. The method trades exact synchrony for access to a large candidate set.

The bank introduces three separate effects:

- **Coverage:** the query can be compared against many more instances than fit in the minibatch.
- **Staleness:** current and stored features may have been produced by different encoder states.
- **Memory cost:** storage scales roughly as \(O(Nd)\) for \(N\) instances and \(d\)-dimensional stored features, before any metadata or optimizer state.

These are not incidental engineering details. They determine the effective negative distribution and the stationarity of the comparison problem.

### 2.3 Where the signal comes from, and what it can fail to preserve

If two stochastic views of the same image should retain identity, the objective makes features recognize that shared instance while distinguishing it from other stored images. The network may discover texture, shape and object parts because these properties help identify images under the applied transformations. But instance identification is not semantic categorization.

If two different images both depict a dog, the instance-discrimination formulation treats them as different identities. Their features may still become similar because of learned structure, but the objective does not directly declare them equivalent. Conversely, it can reward details that distinguish two photographs of the same object even when those details do not matter for control. NPID therefore demonstrates how labels can be constructed from identity, not that identity is the ideal semantic target.

The canonical paper introduced instance discrimination as a non-parametric classification problem and used NCE to address its large number of instance classes. It is a historical ancestor of later contrastive learning, not simply "SimCLR with a smaller batch." SimCLR's view pairs and minibatch softmax changed the candidate construction and training recipe.

**Primary source:** Wu et al. (2018), [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html).

## 3. PIRL: add invariance to a pretext transformation

### 3.1 Why solving a pretext task is not the same as learning invariance

A rotation classifier or jigsaw classifier can solve its task by encoding the transformation itself. Such a feature can be **covariant** with the pretext transformation: changing the transformation systematically changes the representation. For a semantic representation, we may instead want the original image and its pretext-transformed version to retain a common identity representation.

PIRL (Misra and van der Maaten, CVPR 2020) studies this distinction using a pretext transformation such as jigsaw patch permutation. Let \(x\) be an original image and \(t(x)\) its transformed counterpart. A shared backbone \(f_\theta\) processes both; different projection heads can be used for original and transformed views:

\[
z_x=g_\phi(f_\theta(x)),\qquad
z_t=g_\psi(f_\theta(t(x))).
\]

Let \(m_x\) be the memory-bank representation for the original image and let \(\overline{\mathcal M}_x\) be negative bank entries from other image instances. PIRL combines two NCE terms:

\[
\mathcal L_{\mathrm{PIRL}}(x,t(x)) =
\lambda\,\ell_{\mathrm{NCE}}(m_x,z_t,\overline{\mathcal M}_x)
+(1-\lambda)\,\ell_{\mathrm{NCE}}(m_x,z_x,\overline{\mathcal M}_x).
\]

Here \(\lambda\in[0,1]\) weights the transformed-view term. The NCE terms distinguish a matched image/memory pair from negative images drawn from the bank. This is the paper's key design: it asks the transformed image to match the stored identity representation without requiring the encoder to predict the exact transformation label.

The PIRL paper describes NPID as a special case of its formulation at \(\lambda=0\): the transformed-image branch no longer contributes to the objective. This makes the relationship between the two methods mathematically explicit rather than merely chronological.

### 3.2 The role of the pretext transform

For jigsaw-based PIRL, the transformed image is built from patches rearranged in a random order. The transformed representation has a different input construction from a regular image view. The training objective nevertheless relates it to the same original-instance memory vector. In the released setup, the original and transformed paths use separate projection heads; the transformed path is built by processing the patches and combining their representations.

This architecture matters. It allows the network to process a transformed image through the pretext pipeline while using the original image's memory representation as the invariant target. One should not simplify the method into "predict the jigsaw permutation and match the result": PIRL's purpose is to train a representation invariant to that transformation, not to make the transformation label itself the final target.

### 3.3 Failure modes and interpretation

PIRL inherits the identity-based negative assumption and memory-bank staleness of NPID. It also inherits any misalignment between the intended invariance and the transformation. If reordering patches destroys spatial relations that matter for the downstream task, making the representation invariant to that operation may remove useful information. Conversely, if the transformation changes only properties irrelevant to the target task, invariance may improve transfer.

The coefficient \(\lambda\) controls how much the transformed branch contributes, but no coefficient can make a bad invariance assumption harmless for every downstream task. A lower pretraining loss is evidence that the constructed task is easier to solve, not that the representation is a sufficient state for a world model.

**Primary source:** Misra and van der Maaten (2020), [Self-Supervised Learning of Pretext-Invariant Representations](https://openaccess.thecvf.com/content_CVPR_2020/html/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.html).

## 4. Contrastive Multiview Coding (CMC)

### 4.1 Change the positive relation from one image to a shared source

CMC (Tian, Krishnan and Isola; paper published in the 2020 proceedings volume) starts from a multiview assumption: two observations of the same underlying scene may have different noise and view-specific information but share factors such as scene identity, objects or physical structure.

Write \(v_i^{(a)}\) and \(v_i^{(b)}\) for two views of source \(i\). The views can be sensory channels, modalities or different transformations. View-specific encoders \(f_a\) and \(f_b\) produce embeddings \(z_i^{(a)}\) and \(z_i^{(b)}\). The paired views of the same source form a positive; views from other sources provide negatives.

A useful softmax-form view-to-view objective is

\[
\ell_{a\to b}(i)=
-\log \frac{\exp(s(z_i^{(a)},z_i^{(b)})/\tau)}
{\sum_{j}\exp(s(z_i^{(a)},z_j^{(b)})/\tau)}.
\]

A symmetric formulation also asks view \(b\) to identify its matching view in \(a\), and multiple-view settings can sum or average losses across selected view pairs. This equation captures the pairing logic; the authors' released implementation uses an NCE estimator and memory machinery, so it should not be mistaken for a line-by-line transcription of every training variant.

### 4.2 Why multiview signals can help

Imagine that two views are conditionally independent given some shared latent scene factor \(S\), while each has additional view-specific nuisance \(N_a\) or \(N_b\):

\[
V^{(a)} \leftarrow (S,N_a),\qquad
V^{(b)} \leftarrow (S,N_b).
\]

A feature that helps pair the two views is rewarded for retaining information predictive of their shared source. When the only stable common structure is in \(S\), this can encourage a representation that retains \(S\) rather than view-specific noise. The conclusion depends on the data-generation assumptions; the objective cannot identify a latent factor merely because the method calls it "shared."

CMC experiments use image views including channels from different colour spaces, with separate networks for views in important configurations. More generally, the framework can be applied wherever paired views share source identity. The more views are added, the more cross-view constraints may become available, but each new view also brings its own noise, encoder, sampling and compute choices.

### 4.3 The mutual-information interpretation needs qualifications

The paper motivates contrastive multiview learning in terms of mutual information between views and compact representations. A contrastive estimate may be related to a mutual-information bound under specified sampling assumptions. It does not directly optimize an unconstrained, exactly computed \(I(Z^{(a)};Z^{(b)})\) for arbitrary finite batches, nor does it uniquely recover a semantic latent variable.

The denominator, number of negatives, score family, memory-bank update and construction of paired views determine the actual objective. If negative examples share relevant structure with the positive, the task may push apart representations that should remain similar. If the views share only low-level artifacts, the objective can exploit those instead of meaningful common causes.

CMC is thus a foundational example of **changing what counts as a positive**: from an image's stored identity to agreement between distinct observations of the same source. That idea is related to modern cross-modal training, but CMC should not be described as equivalent to CLIP, which uses image-text pairing and a batch classification objective with its own noise and data assumptions.

**Primary source:** Tian et al. (2019 preprint; 2020 publication), [Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849).

## 5. AMDIM: contrast local and global features across scales

### 5.1 From image-level identity to spatially indexed relationships

Augmented Multiscale Deep InfoMax (AMDIM; Bachman, Hjelm and Buchwalter, 2019) extends Deep InfoMax with independently augmented copies of the same image and multiple feature scales. Instead of comparing only two global vectors, the encoder can supply a global feature and spatially indexed local features from intermediate layers.

Let \(f_n(x)_{ij}\) denote a feature at location \((i,j)\) from encoder scale \(n\). Let \(f_m(x')_{kl}\) be a feature from scale \(m\) of another independently augmented view \(x'\) of the same source. A critic scores whether the local/scale-specific features are compatible as a positive pair, and scores from other examples or marginally sampled features provide negatives.

An abstract contrastive cost for one such pair is

\[
\ell_{n\to m}(u,v,\mathcal N_m)=
-\log\frac{\exp(\phi_{n,m}(u,v))}
{\exp(\phi_{n,m}(u,v))+
\sum_{\tilde v\in\mathcal N_m}\exp(\phi_{n,m}(u,\tilde v))},
\]

where \(\phi_{n,m}\) is the critic score and \(\mathcal N_m\) is a set of negative features from the relevant marginal distribution. AMDIM has a family of such costs over selected scale pairs and feature locations; the expression above is a compact schematic of the positive-versus-negative mechanism, not the full paper's notation and estimator.

### 5.2 Why multiple feature scales matter

A global vector can summarize the image but loses explicit spatial indexing. A local feature can represent a region but has a smaller receptive field and less context. Training local-to-global and other cross-scale relations creates multiple constraints:

- a local feature should be compatible with a global feature from the same source;
- features from unrelated sources should be less compatible under the critic;
- independently augmented views encourage the model to use relationships that survive those augmentations;
- multiple scale pairs prevent the objective from depending on only one global/local pairing.

The paper uses several pairings such as early-global, early-to-intermediate and intermediate-to-intermediate features. It also emphasizes receptive-field design: if a positive pair's receptive fields overlap too much, the prediction task can become trivial and transfer can suffer. The architecture and data augmentation therefore define the difficulty of the positive task.

### 5.3 What multiscale contrast does not guarantee

AMDIM does not force every local feature to correspond to a human-defined object part. A feature may use texture, position priors or other stable shortcuts. A representation that identifies image-level content need not preserve accurate local geometry, and an informative local/global relation does not prove causal or physical state recovery.

The "Deep InfoMax" name and mutual-information motivation should be read with the estimator assumptions visible. The trained finite-sample critic and negative distribution define a tractable surrogate. They are not a certificate that all relevant information has been preserved, that the latent is minimal, or that a downstream world-model transition is Markovian.

**Primary source:** Bachman et al. (2019), [Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/abs/1906.00910).

## 6. What is shared, and what is not

| Method | Positive relation | Negative/candidate construction | Main architectural or statistical device | Characteristic risk |
|---|---|---|---|---|
| NPID | Current view and stored identity for the same image | Many other image identities, accessed through a memory bank and NCE | Non-parametric instance classifier plus feature memory | Instance discrimination can oppose similarity between semantically related images |
| PIRL | Original image and its pretext-transformed version, matched to the original instance memory | Other image-instance entries in memory | Pretext-specific path/projector and a weighted combination of NCE terms | The chosen pretext transformation may destroy information needed downstream |
| CMC | Views/modalities from the same source | Views from other sources; NCE/memory machinery in the released code | Separate view-specific encoders in common configurations | The shared view relation may omit task-relevant information or contain shortcuts |
| AMDIM | Local/global or multiscale features from independently augmented copies of the same source | Feature samples from other sources or the corresponding marginal distribution | A set of cross-scale critics and spatial features | Positive pairs may be too easy or exploit non-semantic correlation |
| SimCLR | Two augmentations of the same image | Other views in the minibatch, with a symmetric NT-Xent objective | Shared encoder, nonlinear projector and large-batch softmax | Batch size, augmentation and false negatives shape the objective |
| MoCo | Query and momentum-key views of the same image | Queue/dictionary of keys from many other examples | Momentum-updated key encoder and queue | Staleness, queue dynamics and sampling determine candidate behavior |
| CPC | Context representation and a future latent from the same sequence | Other latent candidates at each prediction horizon | Context network and horizon-specific compatibility scores | Temporal retrieval is not automatically physical rollout or action-conditioned prediction |

The table is a set of comparison axes, not a claim that the methods form one linear progression or share an identical NCE estimator.

## 7. Implications for a world-model representation

The methods above ask several different questions:

- **Can the feature identify this training example?** NPID.
- **Can it retain that identity despite a constructed pretext transformation?** PIRL.
- **Can it match different views of the same source?** CMC.
- **Can spatial and global features from a common source be recognized across scales?** AMDIM.
- **Can it discriminate two augmentations among a minibatch?** SimCLR.
- **Can a context identify future latent candidates?** CPC.

None directly asks the complete world-model question: does \(z_t\), together with action \(a_t\), preserve enough information to predict the conditional distribution of future observations and rewards? A relevant sufficiency target is

\[
P(O_{t+1},R_t\mid H_t,A_t)
=
P(O_{t+1},R_t\mid Z_t,A_t),
\]

for a clearly defined history \(H_t\), latent state \(Z_t\), action distribution and target population. Instance retrieval, source matching, multiscale mutual-information estimates and linear probes can all be useful evidence, but none proves this equality by itself.

The distinction also affects experiment design. A representation audit should test variables that the proxy may omit: velocity when only static appearance is visible, contact state when it is occluded, object identity across camera changes, and different futures under different actions despite similar observations. For a claim about planning, evaluation must additionally show that a controller using the model's predictions performs reliably without exploiting model errors. These are separate tests, not implied outcomes of a contrastive loss.

## 8. Source and replication audit

The current main-extra branch pins these upstream repositories under implementations/; the checkouts remain unmodified source references.

### CMC

- Repository: [HobbitLong/CMC](https://github.com/HobbitLong/CMC), pinned at **7b227be0b10ef4e526c72af07664f5079ed9ee09**.
- Start with the training script, dataset/view-construction module, model directory, NCE average and NCE criterion modules.
- The repository README identifies CMC as the official method implementation but identifies its MoCo/InsDis variants as unofficial. Cite the repository as CMC source, not as the authoritative implementation of every other method it contains.
- The repository's license file is BSD 2-Clause. Check the pinned license and any dependency/license obligations before adapting code.

### AMDIM

- Repository: [Philip-Bachman/amdim-public](https://github.com/Philip-Bachman/amdim-public), pinned at **8754ae149ed28da8066f696f95ba4ca0e3ffebd8**.
- Start with the model, objective/critic costs, self-supervised task and training modules.
- The author's README describes an older research setup using multiple V100 GPUs and an Apex dependency for mixed precision. A source checkout is not evidence that the released recipe runs unchanged on a modern environment.
- The repository's LICENSE.txt identifies the MIT license.

### VISSL

- Repository: [facebookresearch/vissl](https://github.com/facebookresearch/vissl), pinned at **09270ed25a6c2cf71263d955b64cbe076d34ac45**.
- Useful configs include the NPID, PIRL, jigsaw, SimCLR and SwAV directories under the pretraining config tree.
- Useful loss files include NCE, SimCLR InfoNCE, SwAV, Barlow Twins, DINO and iBOT loss modules.
- VISSL is a shared research library with many methods, not a claim that each method has its own dedicated repository. Its license is MIT, but inspect NOTICE, bundled components and any external dependencies before redistribution.

The pinned upstream revision is the code locator. The paper defines the claim; the official config clarifies the recipe; only an executed run with the paper's data and evaluation protocol supports a reproduction claim. Historical code is valuable for understanding a method, but its existence alone is not a reproduction.
