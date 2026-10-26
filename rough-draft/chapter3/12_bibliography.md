# Bibliography for Chapter 3 theory dossier

This is the working citation register; the final AsciiDoc reference section should use Manning's required style consistently. Verify author names, proceedings/journal details, official version and dates against each paper before typesetting. Links below are stable arXiv URLs when available.

## Surveys and general framing
- Gui, J. et al. (2024). A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends. IEEE Transactions on Pattern Analysis and Machine Intelligence / survey version; arXiv: https://arxiv.org/abs/2301.05712.
- Uelwer, T. et al. (2025). A Survey on Self-supervised Methods for Visual Representation Learning. Machine Learning 114, 111. https://doi.org/10.1007/s10994-024-06708-7.
- Wittscher, L. (2026). A Survey on Design Choices for Self-Supervised Learning in Computer Vision. Artificial Intelligence Review. https://doi.org/10.1007/s10462-026-11506-9.

## Pretext tasks and contrastive objectives
- van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. https://arxiv.org/abs/1807.03748.
- Chen, T. et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709.
- He, K. et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. https://arxiv.org/abs/1911.05722.
- Chen, X. et al. (2020). Improved Baselines with Momentum Contrastive Learning. https://arxiv.org/abs/2003.04297.
- Chen, X. et al. (2021). An Empirical Study of Training Self-Supervised Vision Transformers. https://arxiv.org/abs/2104.02057.

## Negative-free and self-distillation
- Grill, J.-B. et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. https://arxiv.org/abs/2006.07733.
- Chen, X. and He, K. (2021). Exploring Simple Siamese Representation Learning. https://arxiv.org/abs/2011.10566.
- Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. https://arxiv.org/abs/2104.14294.
- Oquab, M. et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193.
- Siméoni, O. et al. (2025). DINOv3: Self-supervised learning for vision at unprecedented scale. https://arxiv.org/abs/2508.10104.

## Statistics and clustering
- Zbontar, J. et al. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. https://arxiv.org/abs/2103.03230.
- Bardes, A., Ponce, J., and LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning. https://arxiv.org/abs/2105.04906.
- Caron, M. et al. (2018). Deep Clustering for Unsupervised Learning of Visual Features. https://arxiv.org/abs/1807.05520.
- Caron, M. et al. (2020). Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. https://arxiv.org/abs/2006.09882.

## Masked modeling and language pretraining
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI report.
- Clark, K. et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. https://arxiv.org/abs/2003.10555.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. https://arxiv.org/abs/1910.10683.
- Bao, H. et al. (2022). BEiT: BERT Pre-Training of Image Transformers. https://arxiv.org/abs/2106.08254.
- He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners. https://arxiv.org/abs/2111.06377.
- Zhou, J. et al. (2022). iBOT: Image BERT Pre-Training with Online Tokenizer. https://arxiv.org/abs/2111.07832.
- Baevski, A. et al. (2022). data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language. https://arxiv.org/abs/2202.03555.

## Cross-modal and speech
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. https://arxiv.org/abs/2103.00020.
- Jia, C. et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. https://arxiv.org/abs/2102.05918.
- Zhai, X. et al. (2023). Sigmoid Loss for Language Image Pre-Training. https://arxiv.org/abs/2303.15343.
- Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. https://arxiv.org/abs/2006.11477.
- Hsu, W.-N. et al. (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. https://arxiv.org/abs/2106.07447.

## Joint-embedding predictive architectures
- Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. https://arxiv.org/abs/2301.08243.
- Bardes, A. et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. https://arxiv.org/abs/2404.08471.
- Assran, M. et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. https://arxiv.org/abs/2506.09985.

## Critical reading notes
- Use a paper's official loss and hyperparameter definitions before coding a demonstration.
- Paper publication year, arXiv first-posted year and code release date can differ. The references should follow the style guide's chosen convention consistently.
- “Official code” means an upstream repository identified by the paper authors/team; it does not guarantee permissive licensing, active maintenance, exact paper reproduction or current dependency compatibility.
- Papers about image-text alignment may use noisy captions and could be considered weakly supervised depending on the source of text. Avoid blanket claims that all such methods are supervision-free in every ordinary sense.


## Additional historical pretext-task papers
- Doersch, C., Gupta, A. and Efros, A. A. (2015). Unsupervised Visual Representation Learning by Context Prediction. https://arxiv.org/abs/1505.05192.
- Noroozi, M. and Favaro, P. (2016). Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. https://arxiv.org/abs/1603.09246.
- Zhang, R., Isola, P. and Efros, A. A. (2016). Colorful Image Colorization. https://arxiv.org/abs/1603.08511.
- Gidaris, S., Singh, P. and Komodakis, N. (2018). Unsupervised Representation Learning by Predicting Image Rotations. https://arxiv.org/abs/1803.07728.

## Additional mainstream masked prediction
- Xie, Z. et al. (2022). SimMIM: A Simple Framework for Masked Image Modeling. https://arxiv.org/abs/2111.09886.
- Wei, C. et al. (2022). Masked Feature Prediction for Self-Supervised Visual Pre-Training. https://arxiv.org/abs/2112.09133.
- Peng, Z. et al. (2022). BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers. https://arxiv.org/abs/2208.06366.

## Verified 2026 frontier update
- Mur-Labadia, L. et al. (2026). V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. https://arxiv.org/abs/2603.14482. The official repository lists a dense predictive loss over visible and masked tokens, deep self-supervision at intermediate layers, multimodal image/video tokenizers, and model/data scaling.


## Source articles supplied for this dossier
- Wikipedia contributors. “Self-supervised learning.” Living reference page, accessed 26 October 2026. https://en.wikipedia.org/wiki/Self-supervised_learning.
- Rajendran, S. (2025, December 25). “A Beginner’s Guide to Self-Supervised Learning.” Towards AI. https://pub.towardsai.net/a-beginners-guide-to-self-supervised-learning-27941f2bb4a9.
- Balestriero, R. et al. (2023). “A Cookbook of Self-Supervised Learning.” https://arxiv.org/abs/2304.12210.
- Gui, J. et al. (2024). “A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends.” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12), 9052–9071. https://doi.org/10.1109/TPAMI.2024.3415112. Preprint: https://arxiv.org/abs/2301.05712.

## Additional methods and theory
- Poole, B., Ozair, S., van den Oord, A., Alemi, A. and Tucker, G. (2019). “On Variational Bounds of Mutual Information.” *Proceedings of ICML*, PMLR 97, 5171–5180. https://proceedings.mlr.press/v97/poole19a.html.
- Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S. and Lucic, M. (2020). “On Mutual Information Maximization for Representation Learning.” *International Conference on Learning Representations*. https://arxiv.org/abs/1907.13625.
- Dwibedi, D. et al. (2021). “With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations.” https://arxiv.org/abs/2104.14548.
- Assran, M. et al. (2022). “Masked Siamese Networks for Label-Efficient Learning.” https://arxiv.org/abs/2204.07141.
- Ermolov, A. et al. (2021). “Whitening for Self-Supervised Representation Learning.” https://arxiv.org/abs/2007.06346.
- Wang, X. et al. (2021). “Dense Contrastive Learning for Self-Supervised Visual Pre-Training.” https://arxiv.org/abs/2011.09157.
- Xie, Z. et al. (2021). “Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning.” https://arxiv.org/abs/2011.10043.
- Bardes, A. et al. (2022). “VICRegL: Self-Supervised Learning of Local Visual Features.” https://arxiv.org/abs/2210.01571.
- Ziegler, A. and Asano, Y. M. (2022). “Self-Supervised Learning of Object Parts for Semantic Segmentation.” https://arxiv.org/abs/2112.10582.
- Yun, S. et al. (2022). “SelfPatch: Self-Supervised Path-Level Representation Learning for Local Features.” https://arxiv.org/abs/2112.01216.
- Mur-Labadia, L. et al. (2026). “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.” https://arxiv.org/abs/2603.14482.

- Koohpayegani, S. A., Tejankar, A. and Pirsiavash, H. (2021). “Mean Shift for Self-Supervised Learning.” *IEEE/CVF International Conference on Computer Vision*. https://arxiv.org/abs/2105.07269; code: https://github.com/UMBCvision/MSF.
- Yun, S., Lee, H., Kim, J. and Shin, J. (2022). “Patch-Level Representation Learning for Self-Supervised Vision Transformers.” *IEEE/CVF Conference on Computer Vision and Pattern Recognition*. https://arxiv.org/abs/2206.07990.
- Xie, Z. et al. (2022). “SimMIM: A Simple Framework for Masked Image Modeling.” https://arxiv.org/abs/2111.09886.
- Wei, C. et al. (2022). “Masked Feature Prediction for Self-Supervised Visual Pre-Training.” https://arxiv.org/abs/2112.09133.
- Peng, Z. et al. (2022). “BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers.” https://arxiv.org/abs/2208.06366.
- Poole, B. et al. (2019). “On Variational Bounds of Mutual Information.” *International Conference on Machine Learning*. https://proceedings.mlr.press/v97/poole19a.html.
- Tschannen, M. et al. (2020). “On Mutual Information Maximization for Representation Learning.” *International Conference on Learning Representations*. https://arxiv.org/abs/1907.13625.

- Balestriero, R. and LeCun, Y. (2025, version 3 dated 14 November 2025). “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.” https://arxiv.org/abs/2511.08544.
- Wu, H., Balestriero, R. and Levine, M. (2026). “VISReg: Variance-Invariance-Sketching Regularization for JEPA training.” https://arxiv.org/abs/2606.02572. Project page: https://haiyuwu.github.io/visreg/.
- Epps, T. W. and Pulley, L. B. (1983). “A test for normality based on the empirical characteristic function.” *Biometrika*.
- Cramér, H. and Wold, H. (1936). “Some theorems on distribution functions.”
- Bonneel, N. et al. (2015). “Sliced and Radon Wasserstein barycenters.”

- Epps–Pulley characteristic-function test background is covered by Epps, T. W. and Pulley, L. B. (1983), “A test for normality based on the empirical characteristic function,” *Biometrika*.
- Cramér–Wold theorem reference: Cramér, H. and Wold, H. (1936), “Some theorems on distribution functions.”

- For the formal discussion of representations as sufficient statistics and the data-processing inequality, standard references include Cover, T. M. and Thomas, J. A. (2006), *Elements of Information Theory*, 2nd ed., Wiley.

- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley. Background reference for entropy, mutual information and the data-processing inequality.

- Wang, T. and Isola, P. (2020). “Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.” https://arxiv.org/abs/2005.10242.
