# 1. From representation learning to prediction

Chapter 3 ended with an encoder. Given an observation \(x\), an encoder produces a feature vector or feature map \(z=f_\theta(x)\). The representation is useful only insofar as it preserves distinctions needed by a later task. That leaves a question which the encoder alone does not answer: how should a model learn which information in one part of an observation predicts information somewhere else?

A joint-embedding predictive architecture (JEPA) makes the relation explicit. It selects a context, selects a target, encodes both into a shared feature space, and trains a predictor to estimate the target features from the context features. The prediction is made in representation space rather than directly in pixel space.

This change is easy to state and easy to misunderstand. It does not mean that the model predicts an abstract state whose meaning is already known. The target encoder defines the feature being predicted. The mask defines which information is withheld. The predictor learns the relation between the two. Whether the resulting representation is a useful state is a further question, not a consequence of the name JEPA.

## 1.1 The problem left by an encoder

Suppose a camera observes a table with a cup, a hand and a robot gripper. An encoder can map the image to a grid of features. But the grid by itself does not tell us which features should be related. One training task might require identifying the cup. Another might require locating its rim. A third might ask what visual change is likely after the gripper closes.

These tasks depend on overlapping information, but they do not ask for the same target. An image-level category may ignore the cup's exact position. A localization task cannot. A prediction of the result of closing the gripper must preserve information about contact and geometry. The representation-learning objective must therefore specify which relation the encoder is being trained to support.

Pixel reconstruction is one possible choice. It asks a decoder to reproduce the withheld pixels. That objective can be useful when appearance itself matters, but it also rewards detail that may not matter to the task. A model may spend capacity reproducing texture, lighting or small image variations even when the useful prediction concerns object identity, spatial arrangement or motion.

Feature prediction changes the target. Instead of asking the decoder to reproduce a missing image region, the model asks a predictor to estimate the target encoder's features at the corresponding positions. The training signal is still derived from the observation, but the object being matched is a learned representation.

Neither target is automatically superior. Reconstructing appearance can preserve detail that a feature encoder discards. Predicting features can avoid spending the entire objective on pixel-level fidelity. The decision is about what the downstream system needs to retain.

## 1.2 Context, target and predictor

Let \(x\) be an observation, such as an image or a video clip. A sampling rule divides its information into a context and a target. Write \(x_C\) for the input available to the context branch and \(T\) for the positions whose features are to be predicted.

The context encoder produces

\[
z_C=f_\theta(x_C).
\]

A target encoder produces a feature map from the target-side observation. The target features at the selected positions are

\[
z_T=\left[f_{\bar\theta}(x)\right]_T.
\]

The notation \([\,\cdot\,]_T\) means that features are selected at target positions \(T\). It does not mean that the target encoder necessarily receives only a cropped target patch. In I-JEPA and V-JEPA, the target encoder processes the full image or clip before the selected target features are read out.

The predictor receives the context representation and information identifying the requested target positions:

\[
\widehat z_T=g_\phi(z_C,q_T).
\]

Here \(q_T\) is a query containing the target locations. For an image, these locations are spatial. For a video, they include temporal as well as spatial coordinates. In a controlled dynamics model, the condition must also include the action whose consequence is being predicted.

The learning objective compares the predicted target features with the target encoder's features:

\[
\mathcal L_{\mathrm{pred}}
=
\mathbb E_{(x,C,T)\sim\mathcal D}
\left[
\ell\!\left(
g_\phi(f_\theta(x_C),q_T),
[f_{\bar\theta}(x)]_T
\right)
\right].
\]

The expectation is over the training distribution \(\mathcal D\), including the observations and the sampled context-target configurations. The function \(\ell\) measures disagreement in the chosen feature space. Squared error, absolute error and smooth-\(L_1\) loss make different statistical choices; the architecture does not fix one universal discrepancy.

A JEPA is therefore not specified by three network boxes alone. It is specified by the input distribution, the context sampler, the target sampler, the target encoder, the query representation, the predictor and the feature loss. Changing any of these can change what the model is required to learn.

## 1.3 Why the target is a feature

Let \(r(x)\) be a representation that an ideal downstream task would use. A pixel loss compares the prediction with the observation \(x\). A feature loss compares it with \(f_{\bar\theta}(x)\). This gives the target encoder an important role: it decides which differences in the observation remain visible to the prediction objective.

If a change in illumination is removed by the target encoder, the predictor is not penalized for failing to reproduce it. If object position is retained, the predictor must estimate position-dependent features. The target encoder therefore supplies an implicit definition of which distinctions count.

That definition can be helpful or harmful. An encoder that removes background texture may make prediction easier without damaging a control task. An encoder that removes the relative position of two objects may make prediction easier while destroying information required for collision avoidance. Feature prediction does not discover the correct invariances independently of its target representation; it learns to match the invariances already expressed by that representation.

This is why low prediction error is not itself evidence of a good state. It establishes that one branch can match selected outputs of another branch on the sampled training distribution. To establish that the features are useful, they must be tested on predictions or decisions that were not built into the pretraining loss.

## 1.4 A score is not yet a probability or a task objective

It is useful to describe the feature discrepancy as a compatibility score. Define

\[
E(x_C,q_T,z_T)
=
\ell\!\left(g_\phi(f_\theta(x_C),q_T),z_T\right).
\]

A smaller value means that the candidate target feature is more compatible with the context and query under the learned model. Averaging this score over training examples gives a prediction loss.

The score is not automatically a probability. To turn an energy \(E\) into a normalized conditional density, one must choose a base measure and a partition function:

\[
p(z_T\mid x_C,q_T)
=
\frac{\exp[-E(x_C,q_T,z_T)]}
{\int \exp[-E(x_C,q_T,z)]\,d\nu(z)}.
\]

For a high-dimensional continuous feature space, the normalizing integral may be intractable. A model can therefore use an energy for ranking candidates without having a calibrated probability distribution over them.

A task cost is different again. A planner may care about reaching a goal, avoiding collision or limiting force. The prediction loss teaches the model to match observed target features; a task cost evaluates whether an imagined outcome is desirable. Conflating these quantities makes a learned model appear more capable than its objective warrants.

This distinction will become important when the chapter moves from masked prediction to robot control. A predictor can estimate a future feature. A planner needs a way to compare futures. The two operations are connected, but they are not the same.

**Figure 4.1. The prediction target is a representation, not a reconstructed patch.**

Description: A context encoder processes the visible part of an image or clip. A separate target pathway encodes the complete observation and supplies features at selected positions. The predictor receives the context features and target-location query, then estimates the selected target features.