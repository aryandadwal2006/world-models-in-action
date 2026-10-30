# 07 — JEPA beyond images and video: audio, language and variational formulations

## 1. Why this is a short survey, not a second core chapter

The centre of Chapter 4 should remain image/video JEPA and the transition to action-conditioned world models. Audio, vision-language and variational JEPAs are important because they test whether the latent-prediction design is tied to one modality. They should be covered in a concise, comparative section, not expanded into full independent tutorials.

The key question for each line is: what is the context, what is the target embedding, what does the predictor condition on, and how is the target branch trained?

## 2. A-JEPA

Fei, Fan and Huang's “A-JEPA: Joint-Embedding Predictive Architecture Can Listen” (2023) adapts the masked feature-prediction principle to audio spectrograms. The method treats spectrogram regions as the structured input and uses time-frequency-aware masking to select context and targets. The target encoder is an EMA of the context encoder in the described recipe.

Audio needs a different masking discussion from images:
- time and frequency axes have different meanings;
- local correlations are strong and anisotropic;
- masking a contiguous time region is not equivalent to masking a rectangular image patch;
- speech, music and environmental sound have different temporal structures.

Do not claim audio features are “the same as video tokens” without explaining the time-frequency structure.

## 3. Audio-JEPA

Tuncay et al.'s “Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning” (2025) is a distinct later audio-oriented paper. It uses a ViT-style backbone and masked mel-spectrogram feature prediction on AudioSet, with evaluation across speech, music and environmental sound tasks.

Make clear that A-JEPA and Audio-JEPA are separate works, not alternative names for one paper. Compare their target construction, masking, data and claimed results only after reading the respective primary sources and implementation.

## 4. VL-JEPA

VL-JEPA (Chen et al., 2025) predicts a target text embedding from visual input rather than generating text tokens autoregressively at every step. A text decoder can be used when linguistic output is required, while the embedding can support retrieval or discriminative tasks.

This offers a useful conceptual example:
- output representation can be language-derived;
- prediction need not occur in the vocabulary/token space;
- decoding and representation prediction can be separated;
- a continuous embedding target can support discriminative tasks without always invoking a text generator.

Do not call it an autonomous world model. Vision-language prediction is not, by itself, a controlled dynamics model.

## 5. Var-JEPA and VJEPA are different papers

There are two similarly named 2026 lines that must not be conflated.

**Var-JEPA — Gögl and Yau (ICML 2026).**  
The paper presents a variational formulation of JEPA and relates the canonical deterministic JEPA architecture to a class of coupled latent-variable models. It derives an ELBO-based formulation and studies uncertainty-aware latent variables, including a tabular instantiation. This is a connection between predictive representation learning and probabilistic latent-variable modelling.

**VJEPA — Huang (ICML 2026).**  
This paper is titled “VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models.” It studies probability distributions over future latent states and discusses predictive state representations, Bayesian filtering and a Product-of-Experts extension called BJEPA.

Although both introduce variational formulations, they are distinct papers, with different authors, methods and experiments. The book must cite and describe them separately.

## 6. How these methods fit the book

| Line | Modality/task | Prediction space | Main extra idea | Chapter role |
|---|---|---|---|---|
| A-JEPA | Audio | Spectrogram feature embeddings | Time-frequency-aware masks | Short modality extension |
| Audio-JEPA | Audio | Masked audio feature embeddings | Audio-specific pretraining/evaluation | Short modality extension |
| VL-JEPA | Vision-language | Text embedding | Optional text decoder after embedding prediction | Bridge to multimodal models |
| Var-JEPA | General/initially tabular instantiation | Probabilistic latent variables | ELBO and uncertainty | Bridge to generative/probabilistic modelling |
| VJEPA | World-model tasks | Distributions over future latents | Probabilistic prediction, filtering and expert priors | Bridge to Chapters 6–7 |

Keep the table compact. The full derivations and datasets are not required for this chapter's central argument.

## 7. Scientific limits

- A probabilistic latent objective does not guarantee calibrated uncertainty outside the training distribution.
- A text embedding is not the same target as a physical state.
- Time-frequency masking encodes inductive assumptions about audio.
- The fact that a method uses a JEPA-like encoder/predictor does not make it action-conditioned.
- Similar names are not evidence of identical architectures.

## 8. Suggested sidebar

**A useful naming warning:** Var-JEPA (Gögl & Yau) and VJEPA (Huang) are two separate ICML 2026 papers. Keep the capitalization and author names visible in all tables and prose so readers do not merge their claims.

## 9. Sources

- Fei, Fan and Huang (2023), “A-JEPA: Joint-Embedding Predictive Architecture Can Listen,” arXiv:2311.15830.
- Tuncay et al. (2025), “Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning,” arXiv:2507.02915.
- Chen et al. (2025), “VL-JEPA: Joint Embedding Predictive Architecture for Vision-language,” arXiv:2512.10942.
- Gögl and Yau (2026), “Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture,” PMLR 306.
- Huang (2026), “VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World Models,” PMLR 306.
