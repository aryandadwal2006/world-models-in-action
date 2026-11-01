# 16. Synthesis and exercises

## 16.1 The argument of the chapter

A joint-embedding predictive architecture learns a relation between a context and a target in a representation space. Its apparent simplicity hides several questions. What information is available to the predictor? Which target is being requested? What differences does the target encoder preserve? Which representations can minimize the loss? What statistical properties make the resulting features useful to downstream predictors? What additional model and decision machinery is required before the predictions can guide action?

The chapter developed these questions in sequence.

A compatibility energy assigns a score to a context and candidate target. A JEPA can define that compatibility through a discrepancy between a predicted representation and a target representation. The distinction between a score and a probability remains important: a normalized density requires a partition function and a base measure. A task cost is different again because it expresses what an agent should prefer.

The prediction problem is determined by the context, target, query and sampling distribution. A target may be spatially hidden, temporally withheld, action-conditioned or cross-modal. The predictor is justified by the information structure of the problem, not required by the name JEPA. A deterministic squared-error predictor returns the conditional mean of the target representation, which can average across distinct possible futures.

An unconstrained agreement objective admits a constant representation as a global minimum. Preventing this collapse is necessary, but nonzero variance does not establish that a representation preserves every task-relevant variable. A useful state is sufficient only relative to a target and an action domain:

\[
P(Y\mid X,A)=P(Y\mid Z,A).
\]

The distribution of embeddings also matters to downstream estimation. Under the assumptions of the linear-probe analysis, covariance anisotropy can increase ridge bias for a task aligned with a weak feature direction and increase OLS coefficient variance at fixed total variance. For local nonlinear probes, the score of the embedding density enters the leading bias. Under the paper's smoothness, gradient and covariance assumptions, the integrated bias depends on the Fisher-information functional

\[
J(p)=\int\|\nabla\log p(z)\|_2^2p(z)\,dz.
\]

For the estimator families and scalar covariance constraints analysed by LeJEPA, an isotropic Gaussian uniquely minimizes the relevant risk criteria. This is a conditional mathematical result, not a universal claim about every downstream task.

SIGReg turns Gaussian distribution matching into a collection of projected one-dimensional tests. The Cramér–Wold result concerns all directions; training uses a finite stochastic sketch and an approximate numerical integral. VISReg uses a different one-dimensional statistic and explicitly separates centre, scale and shape. Both methods constrain embedding geometry, but neither establishes that the learned variables are causally complete or sufficient for physical control.

Action-conditioned prediction changes the information supplied to the model. Passive video estimates what tends to happen in observed trajectories. A controlled model is trained to predict the future given a specified action and relevant state. Planning then searches over action sequences, evaluates predicted outcomes with a task cost, executes a selected action and replans from new observations. Its reliability depends on the representation, transition model, data coverage, cost, optimizer, constraints and feedback loop.

Finally, a world model operates inside a larger agent. Perception estimates state; World Model imagines futures; Cost evaluates them; Actor selects actions; Short-Term Memory retrieves experience; Configurator adapts the modules to the task. Mode 1 provides reactive behaviour, while Mode 2 reasons through imagined trajectories. Hierarchical prediction, memory and multiple futures are proposed mechanisms for extending this process across time scales and uncertainty.

The chapter's central claim is therefore limited but useful: predicting representations can supply a compact internal model of selected relations in the world. Whether those relations support action depends on what the representation preserves, what the predictor is conditioned on, and how an agent uses its predictions.

## 16.2 What has and has not been established

The main mathematical conclusions can be separated from the engineering and empirical claims.

| Result | What it establishes | What it does not establish |
|---|---|---|
| Constant embeddings minimize unconstrained agreement loss | Agreement alone does not require information | Every training run will collapse |
| Predictive sufficiency \(P(Y\mid X,A)=P(Y\mid Z,A)\) | A task-relative condition for retaining predictive information | Universal physical state sufficiency |
| Ridge and OLS covariance analysis | Anisotropy can worsen specified linear-estimator bias or variance | Every task performs worse in every anisotropic representation |
| Fisher-information analysis | Local-estimator bias is linked to the embedding density score under stated assumptions | Gaussianity is optimal for every nonlinear model or control task |
| Cramér–Wold | All one-dimensional projections identify a multivariate distribution | A finite set of projections certifies equality |
| Epps–Pulley/SIGReg | A differentiable practical statistic for projected Gaussian matching | Exact population Gaussianity from one minibatch |
| VISReg scale and shape decomposition | Separate terms can target separate batch properties | Universal gradient recovery or no-collapse guarantees |
| Action-conditioned prediction | Actions are explicit inputs to the transition estimate | Causal identification without suitable data and assumptions |
| Planning success in an experiment | A complete model-planner-environment system worked under a reported protocol | General reliability or safety in arbitrary environments |
| LeCun's architecture | A proposed organization of agent capabilities | A validated end-to-end autonomous system |

## 16.3 Exercises

### 1. Energy and probability

Let \(E(x,y)\) be a compatibility energy. Derive the conditional Gibbs density associated with this energy. Identify the partition function and explain why minimizing energy over \(y\) does not require computing it. Give an example of a task cost that should not be interpreted as a probability density.

### 2. Latent optimization versus marginalization

Compare

\[
\inf_z E(x,y,z)
\quad\text{and}\quad
-\log\int e^{-E(x,y,z)}\,dz.
\]

Construct a case with two equal low-energy latent configurations and a second case with one low-energy configuration. Explain how minimization and marginalization treat these cases differently.

### 3. A predictor or direct view agreement?

Consider two training views \(X_1,X_2\) of the same image. Propose a setting in which direct embedding agreement is appropriate. Then consider a robot observation history \(H_t\), an action \(A_t\), and a future visual target \(Y_{t+1}\). Explain why the second setting has an information asymmetry that can justify a predictor.

### 4. Conditional-mean prediction

Let \(H\mid X=x\) take the values \(-2\) and \(+2\) with equal probability. Derive the deterministic squared-error predictor and show that its value is not one of the possible target outcomes. Explain one reason why this can be harmful for a planner.

### 5. Collapse

For the objective

\[
\mathcal L=
\mathbb E\|f_\theta(X)-f_\theta(X')\|_2^2,
\]

show that a constant representation is a global minimizer. Does adding an EMA target encoder change this algebraic fact about a constant solution? Explain the difference between a property of the objective and a statement about optimization dynamics.

### 6. Predictive equivalence

Construct two histories \(x_1,x_2\) that have similar current images but lead to different future distributions under the same action. Define a representation that maps them to the same vector. Show why the predictive-sufficiency condition fails.

### 7. Ridge bias

Let \(X^\top X=Q\Lambda Q^\top\). Starting from the ridge estimator, derive its coefficient bias in the eigenbasis. Under a fixed sum of eigenvalues, construct the downstream coefficient vector that makes an anisotropic covariance more biased than an isotropic covariance.

### 8. OLS variance

Under a full-rank design and homoskedastic noise, derive the conditional covariance of the OLS estimator. Use Jensen's inequality to show that the trace of this covariance is minimized by equal eigenvalues when the total feature variance is fixed. State the assumptions required.

### 9. Local regression bias

Starting with the Nadaraya–Watson estimator, explain why the leading bias contains both the Laplacian of the target function and the dot product of the target gradient with the density score. Give an intuitive example in which the sample density is high on one side of a query point.

### 10. Fisher-information lower bound

Let \(p\) be a mean-zero density with covariance \(\Sigma\succ0\). Use integration by parts to show

\[
\mathbb E[Z\nabla\log p(Z)^\top]=-I.
\]

Then use positive semidefiniteness of an outer-product expectation to derive

\[
J(p)\ge\operatorname{tr}(\Sigma^{-1}).
\]

State the equality condition and identify the density that achieves it.

### 11. Projection matching

Construct two distributions with equal means and covariances but different shapes. Explain why a finite number of moments may fail to distinguish them. Then state precisely what Cramér–Wold requires and why a finite random projection set is a sketch.

### 12. SIGReg versus VISReg

Write the conceptual one-dimensional statistic used by SIGReg and the sorted-sample transport statistic used by VISReg. Identify the role of normalization, random directions, numerical integration and sorting. Explain which parts of each method are stochastic approximations.

### 13. Passive and controlled prediction

Suppose a camera records an object moving while a robot controller acts in the background. Give two data-generating mechanisms that produce similar passive videos but different consequences under a new robot action. Specify which variables should enter an action-conditioned model and which alignment information the dataset must retain.

### 14. Planning and model exploitation

Construct a simple example in which a planner discovers an action sequence that receives a low predicted goal cost but fails in the real environment. Explain how more candidate samples can make this problem worse. Propose two mitigations and identify their limitations.

### 15. Mode 1 and Mode 2

Describe a task for which a Mode-2 planner can generate useful action targets and a Mode-1 policy can later execute them cheaply. Write a distillation loss. Explain why the reactive policy may inherit systematic errors from the planner.

### 16. Hierarchical prediction

Design a two-level predictor for a robot moving an object to a shelf. Specify an abstract high-level target, a lower-level physical state and the interface between them. Identify a state variable that must not be discarded by the high-level representation.

### 17. Memory

Give an example in which a state that is not visible in the current observation is useful for prediction. Explain how associative retrieval differs from simply concatenating the most recent fixed number of frames.

### 18. Evidence design

Design an evaluation that separates frozen representation quality, action-conditioned prediction, multi-step rollout and physical planning. For each stage, state the metric, data split and failure mode it is intended to detect.

**Figure 16.1. The complete predictive-agent argument.** Connect representation learning, target prediction, distributional regularization, action-conditioned dynamics, task cost, planning, memory and feedback. Use separate interfaces for learning a representation, predicting under an action and selecting a desirable action; do not merge them into one loss.
