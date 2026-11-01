# 3. Why agreement alone is not enough

An objective that asks two representations to agree does not necessarily require either representation to contain information. The distinction is elementary, but it determines the design of non-contrastive self-supervised learning and joint-embedding prediction.

The collapse problem should be derived from the objective itself. It is not useful to describe it merely as an optimization instability or a nuisance that a particular architecture happens to avoid.

## 3.1 The constant solution

Let \(X\) and \(X'\) be two views of the same observation. Consider an encoder \(f_\theta\) and a simple agreement loss,

\[
\mathcal L(\theta)
=
\mathbb E\left[
\|f_\theta(X)-f_\theta(X')\|_2^2
\right].
\]

Suppose the encoder ignores its input and returns the same vector \(c\) for every possible observation:

\[
f_\theta(x)=c\qquad\text{for all }x.
\]

Then

\[
\mathcal L(\theta)
=
\mathbb E[\|c-c\|_2^2]
=0.
\]

Because a squared distance cannot be negative, zero is a global minimum. The model has satisfied the objective perfectly while discarding all information about the input.

The argument does not depend on the particular value of \(c\). It can be the zero vector or any constant vector. It also does not depend on the use of two augmented image views. Any objective that only rewards agreement between outputs can admit this solution when both outputs are allowed to become the same constant.

The same problem occurs in feature prediction. If the context and target encoders both map every input to \(c\), and the predictor outputs \(c\), then

\[
\|g_\phi(f_\theta(x),q)-f_{\bar\theta}(y)\|_2^2
=
\|c-c\|_2^2=0.
\]

The feature prediction is perfect in the narrow sense measured by the objective. It is useless as a representation.

This is an identifiability problem: the training criterion does not distinguish an informative solution from a constant one. An optimizer cannot infer a requirement that the loss does not express.

## 3.2 Agreement and information are separate properties

The constant solution exposes two logically different requirements.

The first is **consistency**. Two views or branches that should describe the same underlying content should produce compatible representations.

The second is **non-degeneracy**. A representation should vary with the input in ways that preserve distinctions needed by the learning problem.

Consistency does not imply non-degeneracy. If every input maps to the same vector, all views agree. Non-degeneracy alone does not imply consistency either. Two branches can each produce varied embeddings while assigning different meanings to corresponding samples.

A useful objective therefore needs both kinds of pressure. In an abstract two-branch formulation,

\[
\mathcal L(Z,Z')
=
\lambda\,s(Z,Z')
+
r(Z)
+
r(Z'),
\]

where \(s\) measures agreement between corresponding views and \(r\) constrains the geometry of one branch independently. The expression is a decomposition of responsibilities, not a claim that one particular choice of \(r\) is universally sufficient.

This separation is the central motivation behind variance and distributional regularization. The cross-view term says which pairs should agree. The per-branch term says which embedding distributions are admissible.

## 3.3 Complete collapse and dimensional collapse

Complete collapse occurs when every input is mapped to the same vector. If a batch of centred embeddings is represented by a matrix \(Z\in\mathbb R^{B\times D}\), then its covariance matrix is zero:

\[
\widehat\Sigma_Z
=
\frac{1}{B-1}Z^\top Z
=0
\]

when the rows of \(Z\) are centred and identical before centring.

A less obvious failure occurs when the representation varies but occupies only a low-dimensional subspace. Suppose the embedding dimension is \(D\), but the centred embeddings have rank \(r\ll D\). Then the covariance spectrum has at most \(r\) nonzero eigenvalues. The representation uses only a small fraction of its nominal dimensionality.

This is often called dimensional collapse or rank collapse. It does not mean every coordinate is constant. Several coordinates may vary, but they may be redundant linear combinations of the same few underlying directions.

A coordinate-wise variance floor can prevent individual dimensions from becoming constant, yet still allow correlated or low-rank structure. Conversely, penalizing covariance alone does not guarantee that every dimension has sufficient spread. The two requirements address different degeneracies. (Bardes et al., 2022; Zbontar et al., 2021)

## 3.4 The information-theoretic view

For a deterministic encoder \(Z=f_\theta(X)\), complete collapse implies that \(Z\) is independent of \(X\):

\[
I(X;Z)=0.
\]

Here \(I(X;Z)\) denotes mutual information. A constant representation contains no information about which input was observed. A low-rank representation may still contain information, but it can discard distinctions that matter to a downstream prediction problem.

It is tempting to say that the goal should be to maximize \(I(X;Z)\) without qualification. That is not sufficient either. A representation that preserves every nuisance detail may be unnecessarily difficult to learn or use. Some transformations of the input should be treated as irrelevant for a particular task. The real requirement is task-relative: preserve information needed for the predictions and decisions that matter, while making the chosen invariances explicit.

This is why the target encoder matters. If it maps two observations to the same target feature, the predictive loss gives no reason to distinguish them. The objective cannot recover a distinction that the target representation has deliberately removed. That removal may be desirable for one task and destructive for another.

## 3.5 Why different anti-collapse mechanisms exist

Several families of methods modify the optimization problem so that the constant solution is no longer an attractive optimum or is avoided by the training dynamics.

**Negative comparisons.** Contrastive objectives compare a positive pair against negative examples. When every sample maps to the same normalized vector, positive and negative similarities become equal, so the model cannot minimize the contrastive objective by collapsing all samples. The mechanism depends on the negative distribution and on how false negatives, batch size and sampling are handled.

**Balanced assignments.** Clustering-based approaches assign embeddings to learned prototypes. A balancing constraint discourages every sample from selecting the same prototype. This shifts the comparison from sample-to-sample similarities to sample-to-prototype assignments, and introduces its own assumptions about assignment balance.

**Asymmetric target branches.** Methods such as BYOL and I-JEPA use asymmetric training paths. A trainable online branch is matched to a target branch that does not receive the same gradient update. EMA updates can make the target change slowly. Stop-gradient changes the gradient path. A predictor can add a transformation between the online representation and the target. These choices can stabilize optimization, but one should not infer a universal theorem from their empirical success.

**Moment constraints.** Variance and covariance penalties add explicit constraints on batch statistics. A variance floor discourages constant coordinates; a covariance penalty discourages pairwise linear redundancy. They are direct properties of the batch distribution rather than implicit consequences of branch architecture.

**Distributional constraints.** Instead of matching only a few moments, a regularizer can compare the projected or full embedding distribution with a target distribution. This constrains a broader set of distributional properties, at the cost of introducing a target geometry and an approximation procedure.

These families should not be treated as interchangeable tricks. They make different mathematical objects part of the objective. Negative samples constrain relations among examples; balanced assignments constrain prototype usage; EMA changes the target dynamics; variance, covariance and distributional penalties constrain statistics of the embedding batch. (Chen et al., 2020; Caron et al., 2020; Grill et al., 2020; Assran et al., 2023; Balestriero and LeCun, 2025)

## 3.6 A moving target is not a proof of non-collapse

Consider an online parameter vector \(\theta_t\) and an EMA target parameter vector \(\xi_t\):

\[
\xi_{t+1}
=
m\xi_t+(1-m)\theta_{t+1},
\qquad 0\le m<1.
\]

Repeated substitution shows that the target is an exponentially weighted history of online parameters:

\[
\xi_t
=
m^t\xi_0
+
(1-m)\sum_{j=0}^{t-1}m^j\theta_{t-j}.
\]

The target therefore changes more slowly than the online network, depending on the momentum schedule. This can stabilize a teacher-student optimization process. It does not, by itself, remove every constant solution from the mathematical objective. Whether the training system avoids collapse depends on the full objective, network, data, normalization and optimization dynamics.

The same care is needed with stop-gradient. It specifies which computational path receives a derivative. It is not a semantic constraint and does not guarantee that the representation contains the information required by an arbitrary task.

LeJEPA provides a useful counterexample to the idea that every JEPA needs an EMA teacher and predictor. Its basic multi-view objective directly compares embeddings to the mean of global views while regularizing their distribution. Conversely, action-conditioned prediction can require a predictor because the target depends on actions or future information unavailable in the context. The presence or absence of a predictor should follow from the information structure of the problem.

## 3.7 Non-collapse is necessary, not sufficient

A representation with nonzero variance can still omit object position, velocity, contact state or another variable needed by an agent. A representation with full-rank covariance can still be arranged so that the task-relevant variable is difficult for a downstream estimator to access. A Gaussian distribution can have the desired statistics without being a sufficient state for physical prediction.

The first objective is therefore modest: rule out degenerate solutions admitted by the training criterion. The next question is more demanding. Among non-degenerate representations, which ones make relevant downstream predictions easier, and under what assumptions can that statement be proved?

The answer requires treating the embedding distribution as a geometric and statistical object rather than judging it only by whether it has collapsed.
