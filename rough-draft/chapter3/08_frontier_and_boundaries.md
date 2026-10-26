# Frontier map and explicit chapter boundaries

## 1. Current landscape as of October 2026

The chapter should name methods based on verified papers/releases, but avoid forecasting adoption or declaring a universal winning objective. Two current developments merit attention:

- **DINOv3 (2025):** a large-scale self-supervised visual feature family from Meta. Its published contribution includes a scaling recipe and Gram anchoring to address dense-feature degradation during long training schedules. The official repository supplies the implementation and models but has a method-specific license.
- **V-JEPA 2 and V-JEPA 2.1:** video feature prediction intended to support visual understanding and anticipation. V-JEPA 2-AC is an action-conditioned post-training system for robotic control, which must be distinguished from passive video pretraining. The V-JEPA 2.1 pretraining improvements include dense predictive losses and deeper self-supervision at intermediate representations. Verify the paper/repository dates and versions when finalizing any current-state claim.

The wider research direction is not limited to one loss function. It includes spatially faithful features, larger and more diverse datasets, improved teacher targets, modality alignment, multiscale objectives, and downstream evaluation with local/dense or interactive tasks rather than classification alone.

## 2. JEPA belongs to the next conceptual step

I-JEPA predicts a target block's representation from a context block's representation. A predictor and target encoder define the learning interface; the objective is not pixel reconstruction and does not rely on explicit negative examples. V-JEPA extends the idea to video. Later action-conditioned variants add additional training stages/data. A reader should understand this distinction at the end of Chapter 3, but the full theory and implementation story belong to Chapter 4. Chapter 3 should not spend pages deriving JEPA and then restart the same material in the next chapter.

The handoff can be built around three questions:
1. Is matching different views sufficient, or should one view predict the latent of another?
2. Does predicting a target feature discard nuisance details without discarding future-relevant structure?
3. What changes when targets are in space-time and the prediction is conditioned on an action?

The answer to the third question is what turns a representation-learning story into a world-modeling question. A passive frame or video encoder is not automatically action-conditioned dynamics.

## 3. Do not conflate SSL with all generative modeling

Diffusion and flow matching learn to generate data through denoising or vector-field objectives. They can be trained on unlabeled observations and can be components of generative world models. Nevertheless, their central objective, sampling procedure and role in the book differ from the representation-learning objectives surveyed here. Keep their deeper derivation in the book's generation chapter. In Chapter 3, use them as an explicit boundary case: training on data without manual labels is not enough to classify algorithms as contrastive or feature-based SSL.

## 4. Keep the scope broad, but the chapter selective

The dossier can be expansive because it records the landscape. The reader-facing chapter should use a curated chain:
- transformations create the invariance problem;
- contrastive discrimination supplies an explicit competitor set;
- bootstrap/self-distillation removes the explicit negative dictionary but changes the collapse problem;
- variance/covariance regularization makes statistical anti-collapse constraints explicit;
- clustering exposes pseudo-labels/prototypes as targets;
- masking exposes target representation and context as choices;
- multimodal and speech examples show generality;
- DINOv3 and JEPA show the current frontier and the next question.

This structure is more explanatory than presenting 15 paper summaries in publication-year order. Each section should answer a question left unresolved by the section before it.

## 5. Claims requiring source verification in the final pass

- The exact method/version and training recipe associated with each DINOv3 and V-JEPA 2 result.
- DINOv3's Gram-anchoring motivation, form and stated scope from the original paper, not a blog summary alone.
- V-JEPA 2.1's publication status and implementation differences from V-JEPA 2.
- SigLIP 2's paper date/version and which claims are measured at inference versus fine-tuning.
- License labels and usage restrictions for official code and pretrained weights.
- Current adoption claims for MoCo, CPC, MAE, DINOv2/v3; frame this as lineage and contemporary examples unless supported by survey evidence.
- Whether each upstream implementation is runnable with current PyTorch/dependencies; code presence is not equivalent to reproducibility.

## References

- Siméoni et al. (2025), DINOv3, https://arxiv.org/abs/2508.10104.
- Assran et al. (2023), I-JEPA, https://arxiv.org/abs/2301.08243.
- Bardes et al. (2024), V-JEPA, https://arxiv.org/abs/2404.08471.
- Assran et al. (2025), V-JEPA 2, https://arxiv.org/abs/2506.09985.
- Mur-Labadia, L. et al. (2026), V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning, https://arxiv.org/abs/2603.14482; official implementation https://github.com/facebookresearch/vjepa2.


## Distributional regularization after VICReg

A 2025–2026 line of work makes the shape of the embedding distribution an explicit training target. LeJEPA combines a joint-embedding prediction loss with SIGReg, which compares random one-dimensional projections to a standard Gaussian using an empirical characteristic-function test. Its authors claim Gaussian embedding distributions minimize downstream predictive risk under their JEPA theoretical setup; this is not a universal theorem about every downstream task. VISReg keeps a variance/scale term, replaces covariance regularization with sliced-Wasserstein matching of projected samples to Gaussian quantiles, and adds a centering term. Its authors report stronger gradients near collapse and favorable OOD results in the settings they test. The complete derivations, assumptions and limitations are in [20_lejepa_sigreg_visreg.md](20_lejepa_sigreg_visreg.md). The book should locate the full explanation in Chapter 4, where the JEPA prediction objective is introduced, and use only a brief frontier handoff in Chapter 3.
