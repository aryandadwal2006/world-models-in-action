# 5. Why prediction agreement can still collapse

The JEPA objective asks a predictor to match target features. It does not automatically require those features to vary with the input. This is the central difficulty of non-contrastive feature prediction: agreement is a relation between outputs, whereas information is a property of how those outputs depend on the observations.

The distinction should be derived from the loss rather than treated as an optimizer failure.

## 5.1 The constant solution

Consider the feature-prediction loss

\[
\mathcal L
=
\mathbb E
\left[
\left\|
g_\phi(f_\theta(X_C),Q_T)
-
f_{\bar\theta}(X)_T
\right\|_2^2
\right].
\]

Suppose the context encoder and target encoder both return the same constant vector \(c\) for every input, and the predictor returns \(c\) for every query. The loss becomes

\[
\mathcal L
=
\mathbb E[\|c-c\|_2^2]
=0.
\]

Since squared distance is nonnegative, this is a global minimum of the stated objective. The model has satisfied the training criterion while discarding all information about the input.

The argument does not depend on the value of \(c\). It can be zero or any fixed vector. It also does not depend on whether the loss is squared error, absolute error or another discrepancy that is zero when its arguments are equal. If both branches are allowed to emit the same constant, agreement alone cannot distinguish this solution from an informative one.

This is an identifiability problem. The loss specifies a relation the outputs should satisfy, but it does not identify which solutions preserve useful information. Optimizing the objective cannot supply a requirement that the objective never expresses.

## 5.2 A target update changes dynamics, not the mathematical requirement

I-JEPA and the original V-JEPA use an online context encoder and an EMA target encoder. The target parameters evolve according to

\[
\bar\theta_{k+1}
=
m_k\bar\theta_k+(1-m_k)\theta_{k+1},
\qquad 0\leq m_k<1.
\]

The target is a smoothed history of the online parameters. A large momentum coefficient makes the target change slowly. Stop-gradient prevents the prediction loss from directly differentiating through the target features; the EMA update is a separate mechanism that changes target parameters.

These choices can stabilize optimization. They do not remove the constant solution from the objective. If both branches are constant and the predictor matches them, the prediction loss remains zero. Whether training enters or escapes such a solution depends on initialization, data, architecture, normalization, optimizer and the complete training dynamics.

It is therefore incorrect to say that an EMA teacher mathematically prevents collapse in every JEPA. The teacher changes the moving-target problem faced by the online network. It does not, by itself, require the representation to encode any particular variable.

## 5.3 Collapse can be partial

Complete collapse maps every input to the same feature. A less obvious form occurs when the representation varies only in a low-dimensional subspace.

Let \(Z\in\mathbb R^{N\times D}\) be a batch of centred feature vectors and define the empirical covariance

\[
\widehat\Sigma
=
\frac{1}{N-1}Z^\top Z.
\]

If the centred features span a subspace of dimension \(r<D\), then

\[
\operatorname{rank}(\widehat\Sigma)\leq r.
\]

At least \(D-r\) eigenvalues are zero. The representation uses only a subset of its nominal dimensions. This is dimensional collapse: the representation is not constant, but multiple coordinates may carry redundant combinations of the same few directions.

Marginal variance alone does not rule this out. Two coordinates can both vary while being nearly identical or exact linear combinations. A variance floor can prevent each coordinate from becoming constant, yet still permit strong correlations. A covariance penalty can reduce linear redundancy, but it need not give every dimension useful scale unless scale is constrained separately.

Nor does full rank establish semantic quality. A representation can fill every direction with nuisance variation and still discard the location, velocity or contact information a downstream model needs.

## 5.4 Agreement and representation quality are separate constraints

It is useful to separate the learning objective into two responsibilities:

\[
\mathcal L
=
\mathcal L_{\mathrm{agreement}}
+
\lambda\,\mathcal R(Z).
\]

The first term specifies which examples or views should produce compatible outputs. The second constrains the distribution or geometry of the embeddings. This decomposition does not prescribe one universal regularizer. It makes explicit that the two terms answer different questions.

Agreement asks: which outputs should be close?

A representation constraint asks: what kinds of output distributions should be allowed?

A negative-sample objective couples examples by asking a positive pair to be more similar than candidate negatives. A variance penalty asks that each coordinate retain a minimum amount of spread. A covariance penalty discourages linear redundancy. A distributional penalty compares the whole embedding distribution, or a sketch of it, with a chosen reference. These mechanisms are not interchangeable because they constrain different mathematical objects.

Chapter 3 already developed contrastive and non-contrastive mechanisms in detail. Here the relevant point is narrower: feature prediction needs an explanation for why the target features do not all become identical and why non-identical features retain the right information.

## 5.5 A predictor is not an anti-collapse proof

A predictor can make the two branches asymmetric. It can also make the optimization problem easier by learning a mapping from context features to target features. But adding a predictor does not logically eliminate every collapsed solution. If both encoders emit a constant and the predictor emits that same constant, the feature loss can still be zero.

The same caution applies to stop-gradient, normalization and architectural asymmetry. Each changes the computation or optimization path. None should be described as a universal semantic guarantee.

There are methods that rely on explicit distributional constraints instead. LeJEPA uses view agreement together with SIGReg, which regularizes each view's embedding distribution toward an isotropic Gaussian. VISReg uses an invariance term together with separate scale, shape and centring constraints. Their objectives differ from the EMA-predictor recipes of I-JEPA and V-JEPA. The reason to introduce them is not to catalogue another pair of methods. It is to examine what representation geometry is being requested and what statistical argument supports that request.

## 5.6 Non-collapse is only the first test

A representation can have nonzero variance, full-rank covariance and a roughly Gaussian marginal distribution while still failing to preserve a task-relevant distinction. A video encoder may vary strongly with appearance but ignore object velocity. A robot representation may encode gripper position but discard the relation between the gripper and a nearby object.

Non-collapse rules out one class of trivial solutions. It does not establish predictive sufficiency. The next question is more demanding: for a specified downstream target and action space, does the representation preserve all distinctions in the history that change the relevant future distribution?

**Figure 4.7. Complete collapse and dimensional collapse are different failures.**

Description: In complete collapse, all observations map to one point. In dimensional collapse, the embedding cloud varies but lies in a lower-dimensional subspace, leaving several directions unused or redundant. A nonzero variance in every coordinate does not by itself rule out the second failure.