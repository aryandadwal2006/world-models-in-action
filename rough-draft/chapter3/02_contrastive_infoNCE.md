# Contrastive learning from first principles

## 1. The learning problem

A contrastive objective starts from an anchor x and a set of candidates. One candidate x+ is designated as related to the anchor; the remaining candidates x− are sampled as distractors. The encoder maps each input to a feature, and a similarity function scores pairs. The training signal says: rank the designated positive above the candidates sampled for this instance.

For normalized features zi and zj, use cosine similarity s(zi,zj)=ziᵀzj. A temperature τ>0 rescales logits. Smaller τ sharpens the softmax and increases the influence of high-similarity candidates; it also changes gradient magnitude and optimization sensitivity. Temperature is not a generic “better similarity” parameter.

## 2. Deriving NT-Xent / InfoNCE

A minibatch contains N source examples. Two stochastic augmentations of every source produce 2N views and 2N feature vectors. For a chosen anchor i, let p(i) denote the other view of the same source. Every other view in the 2N batch except i competes in the denominator. The per-anchor objective is

\[
\ell_i=-\log \frac{\exp(s(z_i,z_{p(i)})/\tau)}
{\sum_{a=1}^{2N}\mathbf{1}[a\ne i]\exp(s(z_i,z_a)/\tau)}.
\]

The batch loss averages over all 2N anchors. The positive term appears in the denominator as one of the candidates; the anchor itself is excluded. Therefore the candidate count for each anchor is 2N-1, with one positive and 2N-2 negatives. Naming conventions differ across papers, so the exact candidate set should be written rather than inferred from a letter like K.

Define the candidate logit ai,j=s(zi,zj)/τ and p(i,j)=exp(ai,j)/sum_{k≠i}exp(ai,k). Then the loss is ordinary cross-entropy with correct index p(i). Its gradient with respect to a negative candidate logit is p(i,j); for the positive it is p(i,p(i))-1. Rewriting with the similarity itself adds a factor 1/τ. The gradient is not uniform across negatives: candidates already assigned high probability receive more repulsive gradient. With a dot product, gradients also depend on feature norms unless features are normalized.

## 3. Why the numerator and denominator are necessary

If the loss only maximized s(zi,zp(i)), one trivial solution could enlarge norms (for unnormalized features) or make all points poorly structured while still improving positive similarity. The candidate denominator creates a relative ranking objective: positive similarity must exceed the competitors' similarity. With normalized embeddings the geometry is bounded, but the model can still exploit unhelpful directions, augmentations or shortcut features.

Contrastive learning therefore does not ask “are two inputs semantically equal?” It asks whether the chosen positive can be distinguished among the negatives sampled by the procedure. If two semantically identical observations appear as negatives, the objective treats them as competing classes. These false negatives are not a bug in the gradient; they arise from a mismatch between the sampling label and the semantic relation one wants the feature to encode.

## 4. The mutual-information interpretation and its limits

Under a particular joint distribution p(x,y), with positive pairs sampled from that joint and negatives sampled from an appropriate product-of-marginals construction, the optimal critic for discriminating a positive pair from negatives is related to the density ratio p(x,y)/(p(x)p(y)). The InfoNCE objective gives a lower bound of the form

\[
I(X;Y)\ge \log K-\mathcal{L}_{\mathrm{NCE}},
\]

where K denotes the number of candidates in the specific convention used by the derivation. The exact formula changes notation across papers; the bound requires the specified sampling setup and does not follow just because a loss is called InfoNCE. With finite K, the bound's maximum is log K. A tight bound on mutual information is not the same as a good representation for a downstream decision problem. The encoder may retain nuisance information that increases dependence between views or discard low-mutual-information details that matter for control.

The chapter should derive the classification form first and present the information bound second. Doing so avoids the common mistake of treating “maximize mutual information” as the operational objective in every implementation.

## 5. SimCLR

SimCLR's distinctive contribution was a clear, effective recipe: stochastic augmentation creates two views; an encoder maps each view to a representation; a nonlinear projection head maps it to the space where the contrastive loss is applied; NT-Xent compares each view with the other batch views. The paper's empirical conclusions depend on the augmentation composition, batch size, training schedule, optimizer and architecture. The projection head is not a decorative layer: the paper found that discarding the projection head after pretraining can yield better downstream representations than using the loss-space projection as the final feature.

A textbook derivation should distinguish h=f(x), the encoder representation used downstream, from z=g(h), the projection used by the training objective. It should not suggest every contrastive model must use the same projection head or huge batch.

## 6. MoCo and the dictionary problem

SimCLR relies on candidates in the current minibatch. MoCo asks how to maintain a large and comparatively consistent dictionary without increasing the minibatch to the same size. It uses a query encoder and a key encoder. The key encoder parameters follow an exponential moving average:

\[
\theta_k \leftarrow m\theta_k+(1-m)\theta_q,
\]

with momentum m near one in the original recipe. Encoded keys enter a FIFO queue. The queue allows each query to compare against many keys from earlier minibatches. The moving average reduces rapid drift in the target encoder as queued keys remain present.

The trade-off is staleness: queued keys were created by older encoder parameters. Momentum makes the representation geometry change more slowly, improving consistency but not making keys identical to those that would be computed by the current model. MoCo's queue solves a dictionary-size problem, not the false-negative problem. MoCo v2 is a recipe improvement (including stronger augmentation and an MLP projection head); MoCo v3 changes the training regime for vision transformers and must be discussed in that context.

## 7. CPC as historical bridge rather than centerpiece

Contrastive Predictive Coding encodes a sequence, aggregates past latents into context c_t, and uses a compatibility score to identify a future latent among candidates. A canonical multi-horizon form scores z_{t+k} with W_k c_t and applies an InfoNCE-style classifier. The objective trains the context to rank the actual future latent above sampled distractors. It does not automatically require pixel reconstruction.

CPC matters because it made prediction in latent space a concrete objective and connects sequence learning with later predictive-representation ideas. It is not equivalent to an action-conditioned dynamics model: if the context has no action input, the score does not tell us which action produced a future. Nor does success at future-latent retrieval prove recursive rollout accuracy, a Markov state, or planning utility. The main chapter should give CPC a short historical bridge, not a long empirical section.

## 8. CLIP and SigLIP: the same relational template in another domain

CLIP has paired image and text inputs and learns their embeddings through symmetric image-to-text and text-to-image cross-entropy over a batch. The positive is the paired caption; other images or captions in the batch are distractors. This makes the semantics of “positive” dependent on the pairing and the caption. Captions can be incomplete, generic or noisy.

SigLIP uses a sigmoid-based pairwise objective rather than normalizing a global softmax distribution across the batch. The paper's objective decomposes over image-text pair labels and avoids requiring one global softmax normalization across the full batch. This can change scaling properties and training behavior, but both CLIP and SigLIP remain image-text alignment methods. Neither objective by itself establishes fine-grained grounding or physical reasoning.

## 9. When contrastive learning is a good fit

Contrastive objectives are especially natural when reliable positive pairs can be constructed and the relevant negative samples define a meaningful distinction. Advantages include a direct ranking signal and a clear gradient; costs include candidate memory, batch/queue engineering, temperature tuning, false negatives, sampling assumptions and possible overdependence on augmentation. Their value depends on the pair-generation protocol as much as on the named loss.

## 10. Canonical references

- van den Oord, A., Li, Y. and Vinyals, O. (2018), Representation Learning with Contrastive Predictive Coding, arXiv:1807.03748.
- Chen, T. et al. (2020), A Simple Framework for Contrastive Learning of Visual Representations, arXiv:2002.05709.
- He, K. et al. (2020), Momentum Contrast for Unsupervised Visual Representation Learning, arXiv:1911.05722.
- Chen, X. et al. (2021), An Empirical Study of Training Self-Supervised Vision Transformers, arXiv:2104.02057.
- Radford, A. et al. (2021), Learning Transferable Visual Models From Natural Language Supervision, arXiv:2103.00020.
- Zhai, X. et al. (2023), Sigmoid Loss for Language Image Pre-Training, arXiv:2303.15343.
