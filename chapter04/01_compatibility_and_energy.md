# 1. Compatibility, energy and prediction

A predictive model can be described in many ways. It can estimate a future observation, output a distribution over possible futures, or assign a score to a candidate future. These descriptions are related, but they are not interchangeable. A joint-embedding predictive architecture does not attempt to reproduce every observed detail. It learns a representation in which a particular target is compatible with the available context.

The most useful starting point is therefore not an encoder architecture. It is the notion of **compatibility**.

## 1.1 Prediction as a compatibility problem

Let \(x\) denote information available to a model and let \(y\) denote a candidate target. A model can assign a scalar score

\[
E_\theta(x,y)\in\mathbb R
\]

to the pair. We use the convention that lower values indicate greater compatibility. The function \(E_\theta\) is called an *energy* or *compatibility energy*. Given \(x\), an inference procedure can select a target by solving

\[
\hat y(x)=\arg\min_y E_\theta(x,y).
\]

This is a scoring rule over candidate configurations. It is not, by itself, a probability model. It says which candidate the model prefers, not how much probability mass that candidate should receive.

To turn an energy into a conditional probability density, one may define a Gibbs distribution,

\[
p_\theta(y\mid x)
=
\frac{\exp[-E_\theta(x,y)]}{Z_\theta(x)},
\qquad
Z_\theta(x)
=
\int_{\mathcal Y}\exp[-E_\theta(x,y')]\,d y'.
\]

The normalization term \(Z_\theta(x)\), called the partition function, depends on the conditioning input. It ensures that the density integrates to one with respect to a specified base measure on \(\mathcal Y\). For high-dimensional continuous targets, evaluating this integral can be intractable. An energy-based model may therefore use the score directly for ranking or optimization without explicitly computing a normalized density.

This distinction should remain visible whenever the word *energy* appears. A model that minimizes a feature discrepancy is not automatically learning a calibrated probability distribution over futures. A model that scores candidate trajectories is not automatically using a probabilistic objective. Probability requires a normalization convention and a measure; optimization requires only a scalar objective and a search domain.

## 1.2 Why introduce a latent configuration?

A direct compatibility function \(E_\theta(x,y)\) may not capture every reason that a target is compatible with a context. There may be hidden factors, alternative explanations or multiple plausible completions. Introduce a latent variable \(z\) that describes one such configuration:

\[
E_\theta(x,y,z).
\]

One way to score the context-target pair is to minimize over this latent configuration,

\[
F_\theta(x,y)=\inf_z E_\theta(x,y,z).
\]

This operation asks whether there exists a latent configuration that makes the pair compatible. It is an optimization over \(z\). It should not be confused with marginalization:

\[
-\log \int \exp[-E_\theta(x,y,z)]\,d z.
\]

The first expression selects a minimum-energy configuration. The second aggregates over configurations and, with appropriate assumptions, corresponds to a negative log marginal likelihood up to a normalization term. They behave differently when several latent configurations are plausible. In particular, minimizing over \(z\) can ignore how many compatible explanations exist, whereas marginalization can depend on the volume of the compatible region.

There is also a failure mode. If \(z\) is expressive enough to encode the target \(y\) itself, the model can make the energy small by transmitting the answer through the latent variable. The latent then ceases to represent a restricted or abstract explanation. A useful architecture must control what information \(z\) can carry, how it is inferred, and how it enters the compatibility function. LeCun's proposed JEPA formulation makes this explicit: a latent variable can represent alternative futures, but its information content must be constrained so that it does not simply copy the target.

## 1.3 Joint-embedding prediction

A joint-embedding predictive architecture moves the compatibility comparison from observation space to a learned representation space.

Let

\[
s_x=f_\theta(x),\qquad s_y=f_{\bar\theta}(y)
\]

be representations of a context and target. The context representation is passed to a predictor, which may also receive a query \(q\) specifying which target is requested:

\[
\hat s_y=g_\phi(s_x,q).
\]

A simple energy is then

\[
E_{\theta,\bar\theta,\phi}(x,y,q)
=
d\!\left(g_\phi(f_\theta(x),q), f_{\bar\theta}(y)\right),
\]

where \(d\) measures discrepancy in the learned feature space. For squared Euclidean distance,

\[
d(u,v)=\|u-v\|_2^2.
\]

Training lowers the energy of context-target configurations drawn from the training distribution. The target representation determines which discrepancies count. A difference ignored by \(f_{\bar\theta}\) contributes little or nothing to the energy; a difference preserved by the target encoder remains visible to the loss.

This is the essential change from pixel reconstruction. A pixel-level objective treats discrepancies in the observation as the quantities to explain. A latent prediction objective treats discrepancies in a learned representation as the quantities to explain. Neither choice is universally correct. Pixel detail can be essential for a task involving fine contact geometry, texture discrimination or precise image synthesis. It can also be irrelevant to a task that needs object identity, spatial arrangement or motion. Feature prediction changes the target of learning; it does not establish in advance that the learned features are sufficient for an agent.

The representation should not be called semantic simply because it is latent. Its invariances are determined by the data, target encoder, masking distribution, temporal sampling, architecture and objective. The feature space is a model choice. Its quality must be evaluated against the distinctions that downstream tasks require.

## 1.4 Two ways to represent multiple compatible targets

A context need not determine a unique target. A video prefix may be consistent with several plausible continuations. The JEPA formulation can represent this multiplicity without requiring the model to generate every detail of an observation.

The first mechanism is **invariance in the target encoder**. Suppose several target observations (y_1,ldots,y_m) differ in details that are irrelevant to the prediction problem. If the target encoder maps them to the same representation,

[
f_{ar	heta}(y_1)=cdots=f_{ar	heta}(y_m),
]

then the model treats them as equivalent targets. The energy can be low for all of them because the representation deliberately removes the differences. This is appropriate only when those differences are irrelevant to the task. If the encoder removes object position or contact state when those variables matter, the resulting invariance is harmful.

The second mechanism is a **latent variable that selects among alternatives**. Let

[
E_	heta(x,y,z)
=
D!left(f_{ar	heta}(y),
g_phi(f_	heta(x),z)ight).
]

For fixed context (x), different values of (z) can produce different low-energy predictions. The latent variable represents information needed to choose among compatible futures that is not determined by the context. For example, a video of a vehicle approaching a fork may be compatible with either a left or a right turn. A latent variable can index those alternatives without asking the predictor to average them into a single implausible continuation.

These mechanisms are not interchangeable. Invariance removes a distinction from the target representation. A latent variable preserves a distinction in the predicted representation but allows the model to choose among alternatives. The first says that two outcomes should count as the same for the task. The second says that they remain different, but context alone does not determine which will occur.

## 1.5 The latent variable must not become a copy of the answer

The latent variable creates a shortcut if its information capacity is unconstrained. Suppose (z) can carry a vector with the same information as the target representation (s_y), and the predictor is expressive enough to ignore the context and output (z). Then it can choose (z=s_y), making the prediction discrepancy zero without learning any relation between context and target.

The model would have low prediction energy for the wrong reason. Its energy surface would no longer express which targets are compatible with the context; the latent would transmit the target directly.

LeCun's formulation therefore includes a fourth requirement in addition to informative context and target representations and low prediction error: the latent variable should contain as little information as possible while still representing the irreducible uncertainty in the target. The four conceptual requirements are:

1. The context representation must retain information about the context.
2. The target representation must retain information about the target that matters to the task.
3. The target representation should be predictable from context and the permitted latent variable.
4. The latent variable should not carry information that the context can already explain, and must not bypass prediction by copying the target.

These are architectural and information-theoretic requirements, not a claim that the original position paper specifies one universally applicable mutual-information loss. Practical methods must choose how to limit latent capacity, for example through dimensionality, discreteness, noise or a prior. The exact choice changes the family and number of futures the model can represent.

The energy-minimization view now has a precise interpretation. Low energy indicates compatibility between context and target representations under a restricted latent configuration. The latent can encode alternative futures, but it must not make every target compatible by carrying the answer.

## 1.6 Energy, prediction loss and task cost

Three scalar quantities appear repeatedly in world-model systems:

**Prediction loss** compares a model output with a training target. For example,

\[
\mathcal L_{\mathrm{pred}}
=
\mathbb E_{(x,y,q)}
d\!\left(g_\phi(f_\theta(x),q),f_{\bar\theta}(y)\right).
\]

It is defined by the chosen representation and the training distribution. It answers whether the predictor matches the selected target features.

**Compatibility energy** scores how plausible or compatible a candidate configuration is under a learned model. It may be the same mathematical expression as a prediction discrepancy, but its interpretation depends on how it is trained and used. A compatibility energy need not be a calibrated probability.

**Task cost** evaluates a predicted state or trajectory according to an agent's objective. It may penalize distance from a goal, collision, excessive force, energy use or discomfort. It answers whether a predicted outcome is desirable for the current task.

A scalar notation does not make these quantities interchangeable. A small prediction loss does not imply that the future is desirable. A low compatibility energy does not imply safety. A task cost may be learned or hand-specified and need not equal the discrepancy used to train the representation.

This distinction becomes important when an action-conditioned model is placed inside a planner. The model predicts what may happen under candidate actions. A separate cost evaluates which predicted outcomes should be preferred. The planner searches for actions that reduce that cost. A world model is useful to an agent only when these components are connected without conflating their roles.

## 1.5 What does the target represent?

A withheld target is not necessarily a future. In image prediction, a target may be a spatial block from the same image. In video prediction, the target may be a masked spatiotemporal region. A later frame can be included, but temporal ordering alone does not make the objective action-conditioned. In cross-modal prediction, the target can be a text representation paired with an image.

The information supplied to the model changes the question:

- A hidden image region asks what can be inferred about a spatially withheld part of an observation.
- A masked video region asks what features are compatible with visible spatiotemporal context.
- A future target asks what is predictable from an earlier observation history.
- An action-conditioned target asks what is predicted when the model is additionally given a specified intervention.
- A text target asks which linguistic representation is compatible with a visual or other context.

These are different conditional prediction problems. The phrase *predictive representation* is broad enough to include all of them, but the kind of prediction must be stated precisely.

**Figure 1.1. A compatibility energy over candidate targets.** Draw a context \(x\) on the left and a space of candidate targets \(y\) on the right. Show an energy surface over candidate targets, with low-energy regions indicating compatibility. Include two low-energy regions to illustrate that more than one target may be compatible with the same context. Label the energy as a score, not as a probability density. A separate annotation should show that normalization is required before the energy defines a Gibbs distribution.

**Figure 1.2. Three different scalar objectives.** Use three panels: a predictor output compared with a target embedding; a compatibility score assigned to alternative candidate configurations; and a task cost assigned to predicted trajectories. Use separate symbols and describe the function of each scalar. The point is to prevent a training discrepancy, a model energy and an agent objective from being treated as synonyms.

The next section defines the context, target and query variables more carefully. This is necessary before discussing the optimization objective, the information it can preserve, or the degenerate solutions it admits.
