# CPC, energy-based learning and diffusion: objective boundaries from first principles

The terms self-supervised, contrastive, energy-based, predictive, and generative describe different aspects of a learning system. They overlap, but none implies the others. This note gives CPC the full temporal treatment it deserves, explains when a contrastive score is and is not an energy-based density model, and identifies diffusion as a denoising generative objective rather than automatically a representation-learning objective.

## 1. CPC: predict future latent variables, not future pixels

Let an observed sequence be x_1,...,x_T. An encoder produces z_t=f_theta(x_t), and a context network aggregates the past: c_t=g_phi(z_<=t). For a future offset k, CPC defines a compatibility score, commonly bilinear:
s_k(c_t,z_(t+k)) = z_(t+k)^T W_k c_t.
The matrix W_k is horizon-specific in the original formulation. Given one positive future latent and K-1 sampled candidates, a classification loss selects the positive:
L_k = - E[ log exp(s_k(c_t,z_pos)/tau) / sum_j exp(s_k(c_t,z_j)/tau) ].
The full loss averages over sampled positions and prediction horizons.

The context must make the positive future distinguishable from candidate negatives. This learns temporal dependencies useful for identifying the correct future latent. Unlike pixel prediction, it need not reconstruct every future observation detail. Unlike a static image objective, the positive relation connects context at time t to a target at time t+k.

### 1.1 CPC's assumptions and limits

InfoNCE is a candidate-classification objective first. A mutual-information lower-bound interpretation needs its sampling assumptions: the positive is sampled from the joint/contextual conditional, while negatives follow a specified noise or marginal distribution. If sampling changes, the density-ratio and bound interpretation may change. The finite candidate count K also limits the standard bound by log K, so the estimator saturates at log K even if true mutual information is larger.

CPC does not directly establish a recursively usable transition model. Held-out future-latent retrieval means asking whether the correct target ranks above other candidates. It does not by itself mean that the model predicts the next latent recursively, predicts reward, or estimates what happens under a chosen action. Those are different targets and evaluations.

The authors' public audio implementation is [facebookresearch/CPC_audio](https://github.com/facebookresearch/CPC_audio/tree/b98a1bdf1fe9ea219816db7a6c28115d404a3510), pinned at b98a1bdf1fe9ea219816db7a6c28115d404a3510. It implements CPC for audio and the evaluation setting described in [Unsupervised Pretraining Transfers Well Across Languages](https://arxiv.org/abs/2002.02848). The foundational paper is [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748). Distinguish the original CPC formulation from this later audio implementation.

## 2. Energy-based models: score states with an energy

An energy-based model defines a scalar E_theta(x) and associates lower energy with more compatible or probable inputs. A common Gibbs form is p_theta(x)=exp(-E_theta(x))/Z_theta, where Z_theta is the partition function integral exp(-E_theta(x)) dx (or a sum for a discrete space).

Maximum likelihood minimizes L(theta)=-E_data[log p_theta(x)]. Its gradient has a positive data term and a negative model-sampling term:
grad_theta L = E_data[grad_theta E_theta(x)] - E_model[grad_theta E_theta(x)].
Training pushes down energy on observed data and, through the model expectation, pushes relative energy up on states overproduced by the model. The difficult part is the expectation under p_theta, because sampling can be expensive.

### 2.1 Not every contrastive score is a normalized energy model

A learned scalar s_theta(x,y) can be an energy-like compatibility score when it ranks matches above mismatches. But putting that score in a softmax or binary classification loss does not automatically define a normalized generative density over x. The model must specify the sample space, the probability model, and its normalizer or noise distribution.

Noise-contrastive estimation (NCE) turns density estimation into data-versus-noise classification. Given one data sample and k noise samples from known density p_n(x), the optimal logit for data rather than noise represents log p_data(x)-log(k p_n(x)). This is a density ratio. With an unnormalized model, NCE can estimate model parameters without directly calculating the partition function in the same way as maximum likelihood. Its interpretation depends on the noise distribution and classifier construction.

Deep InfoMax (DIM) uses neural discriminators to encourage dependence between local and global features. Its JSD/NCE/DV losses distinguish joint pairs from mismatched pairs sampled from a product-of-marginals distribution. This is related to energy-based discrimination, but it is not automatically a generative EBM for complete images with a normalized density.

Source: author-maintained [rdevon/DIM](https://github.com/rdevon/DIM/tree/bac4765a8126746675f517c7bfa1b04b88044d51), pinned at bac4765a8126746675f517c7bfa1b04b88044d51; the repository contains JSD, NCE, DV and related objectives. Paper: [Deep InfoMax](https://arxiv.org/abs/1808.06670). Original NCE: [Noise-contrastive estimation](https://arxiv.org/abs/1006.2224).

### 2.2 Contrastive divergence is a different approximation

Some EBMs estimate the negative phase with short Markov chains initialized near data rather than samples fully equilibrated under p_theta. Contrastive divergence compares statistics at the data distribution with statistics after a finite chain. It is an approximation for normalized-energy likelihood learning, not the same as InfoNCE candidate classification. The similarly named contrastive objectives should not be treated as one algorithm.

## 3. Diffusion: learn to denoise across noise levels

Diffusion models define a forward process that gradually adds noise, then learn a reverse process that removes it. In a discrete DDPM, a common transition is q(x_t|x_(t-1))=Normal(sqrt(1-beta_t)x_(t-1), beta_t I). Let alpha_t=1-beta_t and alpha_bar_t=product_(s=1)^t alpha_s. A noised sample can be written directly as
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon, epsilon~Normal(0,I).
Training samples x_0, time t and noise epsilon, then predicts that noise:
L_simple = E[||epsilon-epsilon_theta(x_t,t)||^2].
The predicted noise parameterizes the reverse sampling process. Its target is a known corruption variable, not a positive example, prototype label, teacher latent, or future-embedding target.

### 3.1 Relation to score matching

For Gaussian corruption, grad_(x_t) log q(x_t|x_0) = -epsilon/sqrt(1-alpha_bar_t). Thus the noise predictor can be converted to a score estimate by rescaling and changing sign. Denoising score matching connects corruption-and-denoising regression to learning the score, the gradient of log density at each noise level.

This is a generative objective. Generation requires a reverse sampling procedure that repeatedly applies a learned reverse transition or related sampler. A denoising network can yield useful features, but that possibility does not make the objective identical to contrastive representation learning. To claim transfer, evaluate frozen features or downstream fine-tuning separately from sample quality and likelihood.

### 3.2 Diffusion, score-based SDEs and flow matching

In score-based SDE models, a continuous-time stochastic process perturbs data toward a tractable noise distribution. A neural network estimates the score of each perturbed marginal, and a reverse-time SDE or probability-flow ODE generates samples. The target is a score field, not the identity of a positive among negatives.

Flow matching trains a vector field to match a specified probability path's conditional velocity. It shares continuous-time generative goals with score-based methods, but its regression target is a velocity field. It is not another name for diffusion or a representation-only objective.

Pinned source repositories:
- Original DDPM implementation: [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion/tree/1e0dceb3b3495bbe19116a5e1b3596cd0706c543), pinned at 1e0dceb3b3495bbe19116a5e1b3596cd0706c543. Its README documents TensorFlow-era dependencies and TPU/GCS assumptions.
- Score-based SDE implementation: [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch/tree/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44), pinned at cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44.
- Guided diffusion implementation: [openai/guided-diffusion](https://github.com/openai/guided-diffusion/tree/22e0df8183507e13a7813f8d38d51b072ca1e67c), pinned at 22e0df8183507e13a7813f8d38d51b072ca1e67c.
- Papers: [DDPM](https://arxiv.org/abs/2006.11239), [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456), [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

These are source references, not evidence of an executed training run in this chapter's companion project. They are historical implementations with dependency, data, compute and checkpoint requirements.

## 4. Compare the learning targets

| Family | What is scored or predicted? | Candidate/corruption source | Direct training signal | What it does not imply |
|---|---|---|---|---|
| CPC | Future latent | True future versus sampled candidates | Identify temporally correct latent target | Recursive dynamics or action conditioning |
| DIM | Local-global feature compatibility | Joint pairs versus product-of-marginals mismatches | Dependence between feature levels | Normalized generative image density |
| Generative EBM | Scalar energy of a data state | Data samples versus model-generated negative phase | Density learning under specified model/training assumptions | Transfer-optimized representation |
| DDPM | Added noise or equivalent denoising/score target | Known forward corruption | Reverse denoising and generation | Latent state with Markov sufficiency |
| Score-SDE | Score of perturbed data density | Continuous noise process | Score field for reverse-time generation | Contrastive candidate discrimination |
| Flow matching | Conditional velocity on a probability path | Constructed path interpolation | Vector field for transport/generation | Semantic encoder by definition |

"Self-supervised" can broadly describe training signals derived from data, but the objectives and evaluation goals differ. A realistic sample does not prove intermediate features are optimal for transfer; a transferable representation does not by itself support high-fidelity sample generation.

## 5. Chapter 3 boundary

Chapter 3 concerns how pre-generation training signals shape representations. Its core methods are contrastive candidate classification, bootstrap prediction, feature statistics, clustering/prototypes, masked prediction and joint-embedding prediction. CPC is a useful bridge because it predicts future features rather than pixels. DIM shows how dependence-estimation critics connect to local/global representation learning.

Generative energy models and diffusion belong in the map because they are often discussed alongside SSL and can be components of world-model systems. Their full likelihood, sampling and generative dynamics should be developed in the later generation chapter. Chapter 3 should prevent the category error that all unlabelled-data training objectives are equivalent.

Terminology: EB-JEPA is a JEPA research line; EBM here means energy-based model. Do not conflate EB-JEPA's configurable latent regularizers with a normalized energy model unless a particular paper explicitly defines and trains one.

## 6. Recipe details needed for valid claims

For CPC, record future horizon, positive construction, negative source, candidate count, temperature, target normalization and whether the metric is retrieval or rollout. For DIM, name the critic, divergence/objective, local/global feature locations and negative construction. For diffusion, record noise schedule, parameterization (noise, score or velocity), weighting across timesteps, sampler and sampling steps. These details change the meaning of a reported objective or metric.
