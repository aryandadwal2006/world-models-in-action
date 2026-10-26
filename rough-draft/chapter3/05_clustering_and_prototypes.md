# Clustering and prototypes: DeepCluster, SwAV, and DINO

## 1. Why create pseudo-labels?

Contrastive methods label a positive pair as matching and other candidates as nonmatching. Clustering methods construct a set of latent categories from the data itself, then train the encoder to predict or agree with those categories. This approach replaces an instance-level relationship with a prototype-level one. Multiple examples may share a cluster, which can offer a semantic abstraction but also imports every error of the clustering procedure into the learning target.

## 2. DeepCluster: alternate between feature learning and clustering

DeepCluster alternates two stages.

1. Extract features for the dataset with the current network.
2. Run k-means on those features to assign pseudo-labels.
3. Train the network to predict those cluster IDs.
4. Recompute features and repeat.

The loop is self-referential: features define labels, and labels update features. The method needs a strategy for empty or unbalanced clusters and for the fact that cluster indices are arbitrary identifiers. A permutation of all cluster IDs represents the same partition; pseudo-labels should not be read as fixed semantic categories.

This historical method is valuable because it shows a different source of supervision than paired-view losses. However, offline feature extraction and clustering impose a periodic global step. The representations change during training, so cluster assignments can be stale.

## 3. SwAV: predict another view's cluster assignment

SwAV moves the assignment step online. A set of learned prototype vectors produces scores for features. The scores define assignment probabilities for each view. The training objective predicts the assignment of one view from the representation of another view: the network should produce compatible assignments for different crops of the same image.

For a score matrix Q, SwAV uses a Sinkhorn-Knopp-style procedure to approximate a balanced transport assignment. In an idealized balanced formulation, assignment rows sum to one example's normalized assignment mass and prototype columns have prescribed totals. This discourages the trivial solution in which all examples assign to one prototype. The balancing assumption is an inductive bias: it assumes that spreading the batch across prototypes is useful. Real data clusters need not be equally populated, so balancing can also impose a mismatch.

The important distinction is that SwAV does not simply run k-means each epoch and classify every image with a fixed cluster label. It constructs assignments online and trains cross-view prediction of those assignments.

## 4. DINO shares prototype outputs but has a different center of gravity

DINO also produces distributions over output prototypes, but its standard framing is teacher-student self-distillation. It learns by matching a student distribution to a teacher distribution, with the teacher updated by an EMA and target distributions centered/sharpened. DINO's contribution should not be erased by placing it under a generic “clustering method” heading. Its prototype space supports self-distillation; the teacher update and view setup are central to the method.

## 5. When prototypes help and when they mislead

Prototype objectives can encourage invariance across views, share information among examples, and provide a compact target. But cluster IDs have no intrinsic meaning, the number of prototypes is a design choice, balanced assignment can be wrong for skewed data, and clusters can organize nuisance dimensions. A world-model representation should be evaluated on the distinctions that matter for prediction and action, not assumed to be useful because it yields crisp clusters.

## References

- Caron, M. et al. (2018), Deep Clustering for Unsupervised Learning of Visual Features, ECCV, arXiv:1807.05520.
- Caron, M. et al. (2020), Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, NeurIPS, arXiv:2006.09882.
- Caron, M. et al. (2021), Emerging Properties in Self-Supervised Vision Transformers, ICCV, arXiv:2104.14294.
