# Beyond image-only SSL: language, image-text and speech

## 1. Why include other modalities?

The chapter's main architecture progression is visual, because the book's focus is a world model that consumes observations. A complete landscape note should nevertheless show that the same design questions recur elsewhere: what creates positive pairs; what context is visible; which target is predicted; what makes the target stable; and which distinctions the objective values.

This document is a survey register. The chapter should include a concise cross-domain section and a reference table rather than several extra full-length derivations.

## 2. BERT versus GPT: two different pretraining factorizations

BERT uses masked language modeling. A subset of tokens is corrupted/masked, and the model predicts the original token identities using bidirectional contextual representations. The task encourages contextual representations that can use evidence from both directions, but it does not train the model to generate an entire sequence in left-to-right order in the same way as a causal language model.

GPT's causal objective maximizes the sequence log-likelihood under an autoregressive factorization. Its self-supervision is the next observed token. This objective is generative: it defines a normalized distribution over sequences and can produce samples. It also provides a general-purpose training signal for prediction. The distinction is useful because “self-supervised” is the umbrella training source, not a single architecture or a synonym for “contrastive learning.”

T5 makes text-to-text transfer and span corruption explicit. ELECTRA takes a discriminator-oriented route. These variants illustrate that the choice of corruption and target can change token efficiency and downstream transfer.

## 3. CLIP and ALIGN: naturally paired modalities

CLIP builds image and text encoders and projects them into a shared embedding space. Given B matched pairs, their similarity matrix defines B-way retrieval tasks in both directions: each image should find its paired text, and each text should find its paired image. The loss reuses contrastive cross-entropy. The crucial difference from SimCLR is the source of the positive relation. The image-text pair is supplied by naturally occurring captions, which may be noisy or underspecified.

ALIGN also demonstrates large-scale image-text contrastive learning with noisy web data. These methods are sometimes called vision-language pretraining or weakly supervised learning because text is not manually annotated in a conventional class ontology for each image. The precise term depends on the labeling arrangement; they should not be described as “no supervision at all” in the everyday sense when captions are the training signal.

## 4. SigLIP: replace batchwise softmax with pairwise sigmoid

CLIP normalizes scores over the candidate texts/images in the batch. SigLIP uses a sigmoid loss on pair labels. Abstractly, for pair logit s_ij and label y_ij in {+1,-1}, a pairwise logistic term can be written

\[
\ell_{ij}=\log(1+\exp(-y_{ij}s_{ij})).
\]

The exact published scaling, bias and sampling recipe matter. This generic expression shows the statistical form, not the complete SigLIP implementation. Removing the global softmax normalization can change how batch scaling works, but it does not remove the need for a good pairing process, negative-pair sampling, data filtering or careful evaluation.

SigLIP 2 (2025) extends the family with improvements aimed at multilinguality, semantic understanding, localization and dense features. The chapter may note this direction, but should cite the paper and not attribute every downstream improvement solely to the sigmoid loss.

## 5. wav2vec 2.0: contrastive targets from quantized speech

wav2vec 2.0 maps raw audio through a convolutional feature encoder to latent speech frames, masks spans of latent features and feeds them to a contextual transformer. A quantization module produces target representations. The objective contrasts the contextualized masked representation with the corresponding quantized target against distractor quantized candidates. Quantization creates a discrete target space; negatives create a discrimination problem. The speech signal itself supplies the training data, while transcription labels are used later for downstream fine-tuning.

## 6. HuBERT: cluster labels become speech targets

HuBERT discovers pseudo-targets by clustering acoustic representations, then trains a masked prediction model to predict these labels from context. Iterative clustering can improve the pseudo-labels, but target quality depends on the units and clustering recipe. The method's core distinction from wav2vec 2.0 is the way the target units are obtained and supervised, not merely the fact that both mask audio.

## 7. data2vec: a common teacher target across modalities

data2vec uses a masked student to predict contextualized teacher representations. In principle, the same architecture-level idea can operate on speech, vision or language by changing the input encoder and preprocessing. Its targets are latent contextual features rather than raw input values, language tokens or quantized acoustic classes. This is a strong example of a modality-general target construction, but the original recipe and reported results differ across modalities.

## 8. Practical distinctions table

| Family | Pair/target source | Main loss form | Main warning |
|---|---|---|---|
| CLIP / ALIGN | Natural image-text pair | Contrastive batch classification | Captions are incomplete/noisy; alignment does not ensure grounding |
| SigLIP | Natural image-text pair | Pairwise sigmoid/logistic | Objective change does not solve pair quality |
| wav2vec 2.0 | Context-to-quantized audio latent | Contrastive discrimination | Quantizer and negative set shape the target |
| HuBERT | Clustered speech units | Masked unit classification | Units inherit clustering errors |
| data2vec | EMA teacher contextual features | Latent regression | Teacher feature quality is itself an assumption |
| BERT | Masked words | Token cross-entropy | Not an autoregressive generation objective |
| GPT | Next observed token | Causal next-token cross-entropy | The model's target is sequence continuation, not direct physical-state supervision |

## References

- Radford, A. et al. (2021), CLIP, arXiv:2103.00020.
- Jia, C. et al. (2021), Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision, arXiv:2102.05918.
- Zhai, X. et al. (2023), Sigmoid Loss for Language Image Pre-Training, arXiv:2303.15343.
- Baevski, A. et al. (2020), wav2vec 2.0, arXiv:2006.11477.
- Hsu, W.-N. et al. (2021), HuBERT, arXiv:2106.07447.
- Baevski, A. et al. (2022), data2vec, arXiv:2202.03555.
- Raffel, C. et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.


For first-principles derivations and pinned source boundaries for language and speech SSL, see [the objective dossier](34_language_and_speech_ssl_objectives_from_first_principles.md).
