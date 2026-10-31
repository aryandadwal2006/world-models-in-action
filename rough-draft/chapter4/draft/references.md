# Chapter 4 references and source map

## Primary JEPA papers

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. (2023). “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” Proceedings of CVPR 2023. arXiv:2301.08243.
2. Bardes, A. et al. (2024). “Revisiting Feature Prediction for Learning Visual Representations from Video.” arXiv:2404.08471.
3. Assran, M. et al. (2025). “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” arXiv:2506.09985. The source PDF is mirrored in agent_memory_mustread/core_papers/assran-etal-2025-vjepa2.pdf.
4. Mur-Labadia, L. et al. (2026). “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.” arXiv:2603.14482.
5. LeCun, Y. (2022). “A Path Towards Autonomous Machine Intelligence.” Version 0.9.2, 27 June 2022. OpenReview: BZ5a1r-kVsf.

## Regularization and representation geometry

6. Bardes, A., Ponce, J. and LeCun, Y. (2022). “VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.” ICLR 2022. arXiv:2105.04906.
7. Balestriero, R. and LeCun, Y. (2025). “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.” arXiv:2511.08544.
8. Wu, H., Balestriero, R. and Levine, M. (2026). “VISReg: Variance-Invariance-Sketching Regularization for JEPA Training.” arXiv:2606.02572.

## Other modalities and probabilistic formulations

9. Fei, Z., Fan, M. and Huang, J. (2023). “A-JEPA: Joint-Embedding Predictive Architecture Can Listen.” arXiv:2311.15830.
10. Tuncay, L. et al. (2025). “Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning.” arXiv:2507.02915.
11. Chen, D. et al. (2025/2026). “VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.” ICLR 2026. OpenReview paper.
12. Gögl, M. and Yau, C. (2026). “Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture.” Proceedings of ICML 2026, PMLR 306.
13. Huang, Y. (2026). “VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models.” Proceedings of ICML 2026, PMLR 306.

## Action-conditioned world models and implementation references

14. Terver, B. et al. (2026). “What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?” arXiv:2512.24497.
15. Terver, B. et al. (2026). “A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures.” arXiv:2602.03604. Official repository: facebookresearch/eb_jepa.
16. Facebook Research. “I-JEPA.” Official PyTorch repository: facebookresearch/ijepa.
17. Meta FAIR. “V-JEPA 2.” Official PyTorch repository: facebookresearch/vjepa2.
18. Meta FAIR. “JEPA-WMs.” Official research repository: facebookresearch/jepa-wms. Its stated CC BY-NC-ND 4.0 license must be respected; code and weights are not copied into this manuscript.

## Earlier papers used as Chapter 3 context

19. Chen, T., Kornblith, S., Norouzi, M. and Hinton, G. (2020). “A Simple Framework for Contrastive Learning of Visual Representations.” Proceedings of ICML 2020, PMLR 119. Source PDF mirrored in agent_memory_mustread/core_papers/chen-etal-2020-simclr.pdf.
20. Zbontar, J., Jing, L., Misra, I., LeCun, Y. and Deny, S. (2021). “Barlow Twins: Self-Supervised Learning via Redundancy Reduction.” Proceedings of ICML 2021, PMLR 139. Source PDF mirrored in agent_memory_mustread/core_papers/zbontar-etal-2021-barlow-twins.pdf.

## Source-file inventory

The original PDFs present in the repository's core-paper folder at the start of this work were V-JEPA 2, SimCLR and Barlow Twins. The latter two are cited only to establish continuity with Chapter 3: SimCLR's augmentation-defined invariance and Barlow Twins' cross-correlation objective are not retaught here. The V-JEPA 2 PDF is the only JEPA-specific original PDF in that folder. I-JEPA, original V-JEPA, LeJEPA, VISReg and the related papers are cited from their primary arXiv, proceedings or OpenReview sources.

The author's JEPA notes file is a StackEdit HTML export stored with a Markdown extension. It was consulted as the personal research notebook, but exploratory statements were cross-checked against primary publications. In particular, the chapter narrows Gaussian-optimality and gradient claims to their stated assumptions and evidence.
