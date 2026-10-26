# Language and speech self-supervision: first-principles reference

## 1. What defines an objective?

A self-supervised objective is defined by the target, the visible context, and the positions receiving loss, not by the loss formula alone. Let x be a sequence, M a set of target positions, and p_theta a model distribution. For a categorical target y and logits a, cross-entropy ell = -log softmax(a)_y has gradient d ell / d a_j = softmax(a)_j - 1[j=y]. BERT, ELECTRA and HuBERT can all use cross-entropy while training materially different prediction relations.

Four target families organize the methods below: original symbols, generated binary/pseudo-labels, quantized latent vectors, and continuous teacher features.

## 2. BERT: masked language modelling

Let x be the original tokens and x-tilde a corrupted sequence. The loss is
L_MLM = -(1/|M|) sum_{t in M} log p_theta(x_t | x-tilde).
The encoder is bidirectional: context can come from either side of a target. In the original BERT data pipeline, about 15% of positions are selected; each selected input token is replaced by a mask symbol 80% of the time, a random token 10%, and left unchanged 10%. The target remains the original token in every case. The random and unchanged cases prevent the model from relying exclusively on a visible mask symbol.

Original BERT also adds next-sentence prediction (NSP), a binary segment-pair task. Thus the original recipe combines MLM and NSP, but NSP is not a defining requirement of all later masked language models.

Sources: [official data corruption](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py), [pretraining losses](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_pretraining.py), pinned revision eedf5716ce1268e56f0a50264a88cafad334ac61. Paper: [BERT](https://arxiv.org/abs/1810.04805).

## 3. GPT: causal next-token likelihood

Autoregressive modelling factorizes p(x_1:T) = product_t p(x_t | x_<t), minimizing L_AR = -(1/T) sum_t log p_theta(x_t | x_<t). Inputs and labels are shifted by one token, and causal attention prevents position t from seeing future tokens. Unlike MLM, no corruption is required: the next observed token is the target.

This objective rewards whatever internal information improves next-token prediction. It does not establish that a representation is minimal, causally grounded, or sufficient for control. Likelihood, generated sample quality and world-state understanding are distinct evaluation properties.

Sources: [GPT-2 causal model](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py), [sampling code](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/sample.py), pinned revision 9b63575ef42771a015060c964af2c3da4cf7c8ab. Source limitation: the public tree contains model and sampling code, not the original full pretraining loop.

## 4. ELECTRA: replaced-token detection

A generator G is trained by MLM and samples replacements for selected positions. The corrupted sequence is x-hat. A discriminator predicts r_t = 1[x-hat_t = x_t], whether each valid token matches the original. Its binary loss is
L_RTD = -mean_t [r_t log D_t + (1-r_t) log(1-D_t)].
The generator loss is L_G = -mean_{t in M} log p_G(x_t | x-tilde), and the combined objective is lambda_G L_G + lambda_D L_RTD.

The discriminator receives direct labels at many more positions than the generator. If a sampled replacement equals the original token, its label is original, not replaced. Discrete sampling does not pass discriminator gradients through the sampled token into the generator; the generator learns through its own MLM loss. The generator's quality controls the difficulty and distribution of discriminator examples.

Sources: [ELECTRA training](https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/run_pretraining.py), [configuration](https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/configure_pretraining.py), pinned revision 8a46635f32083ada044d7e9ad09604742600ee7b. Paper: [ELECTRA](https://arxiv.org/abs/2003.10555).

## 5. T5: span corruption

T5 removes contiguous spans, replaces each removed span in the encoder input with a unique sentinel token, and trains an autoregressive decoder to emit each sentinel followed by the original tokens from that span. The target concatenates the removed spans in source order. This is sequence-to-sequence span reconstruction, not independent token classification at masked input positions.

Source: [official T5 span-corruption preprocessor](https://github.com/google-research/text-to-text-transfer-transformer/blob/42909aa15d1fbd16c69333aa474a00ae8ed9fde4/t5/data/preprocessors.py), pinned revision 42909aa15d1fbd16c69333aa474a00ae8ed9fde4. The function defaults include noise density 0.15 and mean span length 3.0; both are configurable. Paper: [T5](https://arxiv.org/abs/1910.10683).

## 6. Speech SSL: predict over latent time

Speech starts as waveform samples. A convolutional feature encoder produces a shorter latent sequence z_1:S, and a contextual network produces c_1:S. The models below mask latent time steps and differ in the target: quantized vectors, offline cluster labels, or teacher features. These are not necessarily raw-waveform reconstruction objectives.

## 7. wav2vec 2.0: contrastive quantized targets

The feature encoder produces z_t. The target branch quantizes it to q_t; the contextual network receives masked input and produces c_t. For a positive target q_t, sampled negatives q^-_j, and temperature tau, define s(c,q) = cosine(c,q)/tau. Then
L_ctr = -mean_{t in M} log [exp(s(c_t,q_t)) / (exp(s(c_t,q_t)) + sum_j exp(s(c_t,q^-_j)))].
The task is to identify the correct quantized target among candidates. Negative sampling changes which distinctions the model is trained to separate.

A separate diversity regularizer discourages codebook collapse. It shapes quantizer usage, while the contrastive term trains contextual target discrimination. The implementation permits multiple target configurations, so a faithful reproduction must record whether quantization is enabled and which configuration is used.

Source: [fairseq wav2vec 2.0 model](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/models/wav2vec/wav2vec2.py), pinned revision 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99. Paper: [wav2vec 2.0](https://arxiv.org/abs/2006.11477).

## 8. HuBERT: offline cluster prediction

HuBERT first assigns acoustic frames pseudo-labels through an offline clustering stage. The current model masks latent time positions and predicts cluster ID y_t:
L_HuBERT = -mean_{t in M} log p_theta(y_t | masked sequence).
The fairseq implementation predicts using similarities to learned label embeddings and candidate logits. Unlike wav2vec 2.0, whose target quantization is an online model component, HuBERT's pseudo-labels are computed separately. Clustering features, cluster count, label version and target-position options are part of the recipe; errors in pseudo-labels can propagate.

Sources: [fairseq HuBERT model](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/models/hubert/hubert.py), [HuBERT recipes](https://github.com/facebookresearch/fairseq/tree/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/hubert), same pinned revision. Paper: [HuBERT](https://arxiv.org/abs/2106.07447).

## 9. WavLM: masked prediction under richer speech mixtures

WavLM retains a HuBERT-like masked pseudo-label prediction framework while changing the training conditions and architecture. The paper emphasizes utterance mixing, which creates overlapping/noisier speech inputs, and gated relative-position bias in the Transformer. Mixing can promote robustness when target alignment is preserved and the mixture distribution resembles downstream data.

Important source boundary: the pinned Microsoft UniLM WavLM directory contains architecture, modules and release support, not a complete paper-scale pretraining trainer. The objective description is therefore paper-grounded, not verified from architecture files alone.

Sources: [WavLM model](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/wavlm/WavLM.py), [modules](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/wavlm/modules.py), pinned revision 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd. Paper: [WavLM](https://arxiv.org/abs/2110.13900).

## 10. data2vec: regress to EMA teacher features

data2vec constructs targets from a teacher updated by exponential moving average. For selected teacher layers L, a schematic target is y_t = Norm(mean_{l in L} h_t^(l)). The student receives masked input and predicts the teacher's contextualized feature at masked positions:
L_d2v = mean_{t in M} || yhat_t - stopgrad(y_t) ||^2.
Implementations support configuration-dependent normalization and can use smooth-L1. Unlike wav2vec 2.0, this does not require sampled negative target vectors; unlike HuBERT, the target is continuous rather than an offline cluster ID.

Source: [fairseq data2vec audio](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/data2vec/models/data2vec_audio.py), pinned fairseq revision 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99. The code implements EMA teacher targets, selected-layer aggregation and configurable normalization. Paper: [data2vec](https://arxiv.org/abs/2202.03555).

## 11. Comparison

| Method | Target | Loss positions | Core loss | Main limitation |
|---|---|---|---|---|
| BERT | Original token | Selected corrupted positions | Token CE; original recipe also NSP | Bidirectional MLM is not causal generation |
| GPT | Next token | Causal sequence positions | Autoregressive CE | Likelihood does not prove causal state understanding |
| ELECTRA | Original/replaced indicator | Valid discriminator positions | Binary CE plus generator MLM | Depends on sampled corruption distribution |
| T5 | Removed spans | Decoder target tokens | Autoregressive CE | Span scheme defines the task |
| wav2vec 2.0 | Quantized latent | Masked acoustic frames | Contrastive softmax plus diversity | Quantizer and negatives matter |
| HuBERT | Offline cluster ID | Usually masked frames | Cluster prediction | Pseudo-label errors propagate |
| WavLM | HuBERT-like pseudo-label | Masked frames | Label prediction with mixed speech | Full trainer absent from pinned model folder |
| data2vec | EMA teacher feature | Masked frames | L2 or configured smooth-L1 | Teacher layers and normalization change target geometry |

This is not a ranking. Each objective rewards a different prediction relation. Low pretext loss does not establish semantic abstraction, robustness, causal understanding, or downstream sufficiency.

## 12. Relation to world models

A world model for action selection should support predictions of outcomes under actions, for example P(O_{t+1}, R_t | H_t, A_t), with assumptions stated about history and action coverage. Next-token prediction, masked acoustic prediction and teacher-feature regression do not automatically learn an action-conditioned transition law. They may yield useful representations, but action conditioning, reward relevance, intervention coverage and calibrated uncertainty require separate evidence.

## 13. Reproducibility boundaries

- BERT, revision eedf5716ce1268e56f0a50264a88cafad334ac61: corruption and MLM/NSP losses are in source.
- GPT-2, revision 9b63575ef42771a015060c964af2c3da4cf7c8ab: causal model and sampling are present; original full pretraining loop is absent.
- ELECTRA, revision 8a46635f32083ada044d7e9ad09604742600ee7b: generator MLM, replacement sampling, discriminator loss and weighting are present.
- T5, revision 42909aa15d1fbd16c69333aa474a00ae8ed9fde4: span corruption is present.
- wav2vec 2.0, HuBERT and data2vec audio, fairseq revision 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99: model target construction and losses are present.
- WavLM, revision 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd: architecture/modules are present, not a complete trainer.

Source inspection verifies code paths, not a reproduced training run. Reproduction also requires exact configs, data processing, target-generation assets, optimizer/schedule and checkpoint provenance.
