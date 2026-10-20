# Chapter 3 references and figure credits

This file is the working source list for converting in-text Markdown notes into Manning-style numbered footnotes. Put first-use URLs in footnotes in the composed book. References here are grouped by role; verify exact edition/title metadata in the final bibliography before production.

## Benchmark and environment

- Tunyasuvunakool, S. et al. (2020). “dm_control: Software and tasks for continuous control.” *Software Impacts*, 6, 100022. https://doi.org/10.1016/j.simpa.2020.100022
- Google DeepMind, `dm_control` source repository: https://github.com/google-deepmind/dm_control
- MuJoCo documentation: https://mujoco.org/

## Self-supervised objectives

- Gui, J. et al. (2024). “A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends.” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12), 9052–9071. https://doi.org/10.1109/TPAMI.2024.3415112. Preprint: https://arxiv.org/abs/2301.05712
- LeCun, Y. (2020). “Self-Supervised Learning.” ICLR 2020 keynote. Official session page: https://iclr.cc/virtual_2020/speaker_7.html. Public video recording: https://www.youtube.com/watch?v=8TTK-Dd0H9U.
- Chen, T. et al. (2020). “A Simple Framework for Contrastive Learning of Visual Representations.” *ICML*. https://arxiv.org/abs/2002.05709
- van den Oord, A., Li, Y., and Vinyals, O. (2018). “Representation Learning with Contrastive Predictive Coding.” https://arxiv.org/abs/1807.03748
- Chen, X. and He, K. (2021). “Exploring Simple Siamese Representation Learning.” *CVPR*. https://arxiv.org/abs/2011.10566
- Grill, J.-B. et al. (2020). “Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.” *NeurIPS*. https://arxiv.org/abs/2006.07733
- Zbontar, J. et al. (2021). “Barlow Twins: Self-Supervised Learning via Redundancy Reduction.” *ICML*. https://arxiv.org/abs/2103.03230
- Bardes, A., Ponce, J., and LeCun, Y. (2022). “VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.” *ICLR*. https://arxiv.org/abs/2105.04906
- He, K. et al. (2022). “Masked Autoencoders Are Scalable Vision Learners.” *CVPR*. https://arxiv.org/abs/2111.06377

## Implementation and reproducibility

- Canonical implementation modules: `worldmodels/` in the code companion.
- Experiment runners, outputs, and numerical provenance: `ch03/` and `ch03/results/`.
- Current figure assets: `ch03/figures/`.
- Exact runtime and rendering guidance: [`ch03/README.md`](../ch03/README.md). Hardware-dependent runtime estimates must be reported as configuration- and machine-specific, not guaranteed speedups.

## Figure credits

Figures 3.3, 3.10, 3.12's architecture panel, and 3.13 are adapted/redrawn from Gui et al. (2024), with Figure 3.12's MAE architecture also informed by He et al. (2022). The rendered DMC frames are the book's own renders of the Control Suite simulator, not third-party photographs or generated images. Figures 3.1, 3.4 schematic, 3.5, 3.11 schematic, 3.15, and 3.17 are original illustrations. Figures 3.2, 3.6, 3.7, 3.9, 3.11 curves, 3.12 reconstruction examples, 3.14, and 3.16 derive from committed experiment outputs; use the precise asset/procedure in `figure-and-table-production.md`.