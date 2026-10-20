# Exercises

These exercises connect the chapter's mathematical objectives to the behavior of the representations they train. For each experiment, write down a hypothesis before running the code, preserve the configuration and random seeds, and distinguish a measured result from an explanation of why it occurred.

1. Deriving the contrastive gradient

Starting from equation 3.5, derive the gradient of the NT-Xent loss with respect to the anchor representation $z_1$.

Separate the gradient into the contribution from the positive example and the contributions from the negative examples. Identify which terms pull the positive closer and which push negative candidates away. Explain how the temperature $\tau$ changes the relative weights assigned to candidates.

2. Temperature and negative-example weighting

Consider InfoNCE as the temperature approaches infinity, $\tau\to\infty$. Show what happens to the softmax probabilities when the similarities are divided by an increasingly large temperature.

Predict the effect on the gradients and on the learned representation. Then increase the temperature in the contrastive training experiment and compare the loss curve and frozen-feature state probes against a baseline. Keep the data, architecture, and training budget fixed.

3. Removing a VICReg term

VICReg combines invariance, variance, and covariance penalties. Which term prevents the trivial constant representation from satisfying the objective at zero cost? What failure would you expect if the covariance penalty were removed while the other terms remained?

State your predictions for latent spread, effective rank, and state-probe performance. Then modify the experiment in ch03/08_vicreg.py, run it with the same seeds and budget, and compare the measurements with the complete objective. Distinguish a reduction in redundancy from a genuine collapse.

4. When augmentation removes state information

The contrastive view pipeline uses crop shifts of up to three pixels. Suppose the maximum shift increases to 16 pixels, while Cartpole remains rendered at $64\times64$ resolution.

Predict how this change will affect the cart-position probe and explain your prediction using the chapter's central principle: the objective rewards information preserved by the view pipeline. Then run the experiment with the larger shift, keeping all other settings fixed. Report the cart-position score across seeds and compare it with the baseline.

5. Choosing the latent dimension

The bottleneck sweep measures state-probe performance at latent dimensions $d\in{2,4,8,16,64,256}$ for Cartpole Balance and Cheetah Run.

At what dimension does Cartpole's probe approach a plateau? Does Cheetah show the same pattern within the tested range? Relate the results to the tasks' four- and eighteen-dimensional physical states, respectively, and explain why the optimal latent dimension need not equal the number of state variables.

Use Figure 3.14 and ch03/results/bottleneck_sweep_metrics.json. Identify the smallest tested dimension that appears to approach saturation for each task, and state what further experiment would be needed if performance is still rising at the largest dimension.

6. Extending CPC's prediction horizon

Modify the CPC experiment to predict five future steps instead of three. Predict how future-latent retrieval, InfoNCE loss, and the positive-versus-hardest-negative score margin will change as the horizon increases.

Train fresh checkpoints with five horizon-specific scoring matrices. Use separate checkpoint, result, and figure paths so the canonical Table 3.5 artifacts remain unchanged. Compare the new results with the three-horizon baseline using the same evaluation protocol.

Explain the difference between identifying the correct future latent among candidates and predicting a physical state accurately. In particular, state why a strong retrieval result does not establish accurate recursive prediction, action-conditioned dynamics, or planning utility.

7. Testing temporal aliasing on Finger Spin

The existing aliasing diagnostic compares single-frame features with features formed from two consecutive Cartpole frames. Its current runner hardcodes the Cartpole datasets and four Cartpole state-variable names. Extend it to support Finger Spin before running the experiment, including the correct variable mapping and the periodicity-aware evaluation for the spinner's angle.

Predict which Finger Spin state variables will be easier to decode from a single frame and which, if any, will benefit from stacking two frames. Then compare single-frame and stacked-frame probe scores across the same three seeds.

> **NOTE: Interpreting Finger Spin probes.** Finger Spin's rendered spinner has $\pi$-periodic symmetry: a static image does not uniquely identify its absolute $2\pi$ angle. Account for this when interpreting the probe scores. Distinguish a variable that is unobservable from a single image because of the rendering symmetry from one that is merely difficult to decode under the tested model and training budget.

Summary