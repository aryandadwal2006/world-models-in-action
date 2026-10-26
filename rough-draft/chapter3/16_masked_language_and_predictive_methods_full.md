# Masked, autoregressive, latent and predictive objectives: full derivations

The method family often called “masked prediction” contains multiple distinct tasks. The mask defines which observations the model sees; the target definition and loss determine what it must infer. Language autoregression and JEPA-style latent prediction are included here as neighboring cases, not because they are the same objective.

## 1. A common target-prediction template

Let x be broken into positions \(1,\ldots,T\). The visible set is V and the target set is M. A context encoder processes the observed input \(x_V\), and a prediction head produces \(\hat y_i\) for each target position i. A general loss is

\[
L=\frac1{|M|}\sum_{i\in M}\ell(\hat y_i,y_i).
\]

This template does not specify:
- whether hidden positions are replaced by mask tokens, removed from the encoder, or corrupted;
- whether the context is bidirectional, causal or spatially restricted;
- whether the target is a pixel value, discrete token, handcrafted descriptor, teacher feature or predicted future latent;
- whether loss is computed only on masked positions or also on visible context;
- whether the prediction head is retained after pretraining.

These choices should not be hidden by the phrase “masked modeling.”

## 2. Historical pretext tasks: learn to predict an imposed relation

Early visual self-supervision included:
- rotation prediction (predict which geometric rotation was applied);
- jigsaw/patch-order prediction (recover relative spatial arrangement);
- context prediction (predict relationships between image regions);
- colorization (predict color channels or chrominance from grayscale input);
- denoising and inpainting (restore input content corrupted or removed).

Each task constructs a surrogate label from the observed data. It can produce useful visual features because solving it may require semantic or spatial information. But it can also admit shortcuts. Rotation labels can correlate with object orientation; jigsaw tasks can be solved using patch-edge continuity or local textures; colorization is underdetermined when a grayscale input is compatible with multiple colors; inpainting can average plausible completions.

These tasks are historically important because they introduced the central design problem: is the surrogate task asking for features that transfer to the intended objective? They should be compact in the final Chapter 3, not treated as an equally current alternative to every large-scale SSL architecture.

## 3. MAE: an asymmetric pixel-reconstruction objective

Divide an image into N patches \(x_1,\ldots,x_N\), and sample a visible subset V and masked subset M. MAE sends only visible patches through the encoder:

\[
h_V=f_\theta(x_V).
\]

The decoder receives encoded visible tokens, position information and learned mask tokens; it predicts patch pixels for the reconstructed sequence. The usual training loss is mean-squared error over target patches, often restricted to masked positions:

\[
L_{\mathrm{MAE}}=\frac1{|M|}\sum_{i\in M}\lVert \hat x_i-x_i\rVert_2^2.
\]

A high masking ratio reduces the number of tokens the expensive encoder processes. The asymmetric encoder-decoder is thus a computational design as well as an objective design.

### Why MSE leads to a conditional mean

For context C and random target Y, choose a deterministic prediction a and minimize conditional square error:

\[
R(a\mid C=c)=\mathbb E[\lVert Y-a\rVert_2^2\mid C=c].
\]

Set \(m=\mathbb E[Y\mid C=c]\) and write \(Y-a=(Y-m)+(m-a)\). Expand:

\[
R(a\mid c)=
\mathbb E[\lVert Y-m\rVert^2\mid c]
+\lVert m-a\rVert^2
+2\,\mathbb E[(Y-m)^\top(m-a)\mid c].
\]

The last term is zero because \(\mathbb E[Y-m\mid c]=0\). Therefore

\[
R(a\mid c)=
\mathbb E[\lVert Y-m\rVert^2\mid c]+\lVert m-a\rVert^2
\]

and the minimum is achieved at \(a=m\). When multiple completions are plausible, the conditional mean can average them. This explains one limitation of a pixel MSE target; it does not establish that every MAE output is blurry or that pixel reconstruction cannot learn useful features.

The encoder often becomes the transferred component, while the decoder is discarded. Pretraining loss and downstream representation quality are therefore different measurements.

## 4. SimMIM: simpler pixel regression is a meaningful branch

SimMIM uses masked image modeling with a relatively simple reconstruction target and prediction head. Its importance is conceptual: pixel-space MIM does not require every implementation to use a complex transformer decoder or a discrete visual tokenizer. The precise patch size, mask token, loss scaling and backbone recipe should be taken from the original paper and official code.

Compare MAE and SimMIM along four axes:
1. encoder input: visible tokens only versus corrupted/masked token sequence;
2. decoder/prediction head capacity;
3. target and whether only masked positions receive loss;
4. compute savings and the feature actually used downstream.

Do not infer the complete architecture from the general phrase “predict masked pixels.”

## 5. BEiT and BEiT v2: discrete visual target vocabularies

BEiT first obtains discrete visual-token IDs using a tokenizer. A masked image transformer predicts those IDs at masked positions through classification loss:

\[
L_{\mathrm{BEiT}}=-\frac1{|M|}\sum_{i\in M}\log p_\theta(t_i\mid x_V),
\]

where \(t_i\) is the tokenizer's target code. The target is a code index, not the RGB patch itself. This is analogous to BERT's target type: predict a discrete symbol from context.

The tokenizer influences the learning signal. If it groups visually similar patterns, the target can encourage structure useful for recognition. If it loses a distinction that matters downstream, the target may teach the encoder to ignore that distinction. BEiT v2 improves the tokenization/target construction with semantic visual tokenizers based on vector quantization. It should be explained as a change to target quality rather than merely a new architecture name.

## 6. MaskFeat: predict a descriptor, not the pixels

MaskFeat predicts handcrafted features such as Histogram of Oriented Gradients (HOG) in masked regions. It uses the same general mechanism of withholding input content but a different target space. This creates an illuminating comparison:
- MAE's target is raw pixel intensity;
- BEiT's target is a discrete learned visual code;
- MaskFeat's target is a designed feature descriptor;
- iBOT/data2vec predict teacher representations or distributions;
- JEPA predicts target embeddings within a joint representation space.

The target determines which variation the loss cares about. HOG emphasizes local edge orientation and discards much color information; this can be beneficial or harmful depending on the downstream objective. Handcrafted targets are not free of inductive bias; their prior is simply explicit.

## 7. iBOT: mask plus self-distillation

iBOT applies an online-tokenizer/self-distillation objective to masked image patches. The target is generated by a teacher on a less-corrupted view/input, with prototype-like distributions. It combines three concepts:
- masked input, which restricts context;
- teacher-student prediction, which defines a slowly changing target;
- prototype distribution, which defines a discrete output space.

This overlap is not a taxonomy flaw. It shows that target construction, context restriction and collapse control are independent axes.

## 8. BERT: masked language modeling

Let tokens be \(x_1,\ldots,x_T\). Corrupt or mask a subset M. The bidirectional Transformer processes the corrupted sequence and predicts original token IDs at selected positions:

\[
L_{\mathrm{MLM}}=-\sum_{i\in M}\log p_\theta(x_i\mid \tilde x_1,\ldots,\tilde x_T).
\]

The model may use context to the left and right of the masked token. The exact original corruption strategy and auxiliary objectives must be stated rather than assuming all later BERT-style models keep every component of the original recipe.

The self-supervised target comes from the observed sequence itself. The method is not simply “a transformer”; it is an encoder architecture plus a corruption process plus token cross-entropy.

## 9. GPT: autoregressive factorization

For any joint distribution over a sequence, the chain rule gives

\[
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
\]

The negative log-likelihood becomes

\[
L_{\mathrm{AR}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
\]

GPT trains a causal model to predict each observed next token from the prefix. It differs from BERT in the context restriction and target placement. GPT is generative in the sense that it defines a sequence distribution that can be sampled autoregressively. It is still self-supervised because each next token is observed in the data rather than externally annotated.

This comparison is useful for distinguishing “self-supervised” from “contrastive” and “masked.” Those terms describe different axes: source of target, loss relation and available context.

## 10. ELECTRA: identify replacements

ELECTRA constructs a corrupted sequence with a generator replacing selected tokens, then trains a discriminator to decide at token positions whether the token is original or replaced. The discriminator's target is a binary origin label, not the original token identity. If y_i indicates “original,” the discriminator objective is

\[
L_{\mathrm{RTD}}=-\sum_i
\left[y_i\log D_\theta(\tilde x)_i+(1-y_i)\log(1-D_\theta(\tilde x)_i)\right].
\]

The generator's corruption distribution determines which replacement decisions are easy or hard. The discriminator gets a supervision signal at many positions, rather than only at masked positions. This improves training-signal density in the intended setting, but it changes the task: the discriminator need not reconstruct the exact original token to know whether a replacement occurred.

## 11. T5 and span corruption

T5 formulates several NLP tasks in text-to-text form and uses span corruption: contiguous spans are replaced with sentinel tokens and the model generates the removed content in an autoregressive decoder. The target is a sequence of corrupted-out spans rather than independent masked token labels. This supports a useful broader point: one can vary the corruption granularity and the decoder's factorization while keeping self-supervision from raw sequences.

## 12. data2vec: latent contextual targets

data2vec uses an online/student network on a masked input and predicts contextualized feature targets from a teacher network, commonly an EMA teacher. The target aggregates selected teacher-layer representations under the paper's recipe. Its central design choice is to predict contextual features rather than modality-specific labels such as words, HOG descriptors, pixels or quantized speech units.

A feature target can encourage context-sensitive information and avoid direct raw-pixel reconstruction. But the target is not a neutral representation of the true world state: it inherits the teacher's inductive biases and omitted information. “Feature prediction” does not guarantee that the model learns causal variables, a Markov state or an action-conditioned transition function.

## 13. wav2vec 2.0, HuBERT and WavLM

### wav2vec 2.0

Raw waveform is processed by a convolutional feature encoder. A contextual Transformer receives masked latent spans. A quantization module supplies target representations. The objective contrasts the contextual representation at a masked time position with the true quantized target against distractor quantized representations. This combines masking, quantization and contrastive classification. Fine-tuning on speech labels is downstream, not part of the original self-supervised target.

### HuBERT

HuBERT generates pseudo-labels by clustering acoustic features, then predicts those labels at masked time positions. Later iterations may recompute units or refine targets. Its target space is discrete cluster IDs rather than the same learned quantization mechanism as wav2vec 2.0. Clustering errors and the unit inventory shape the representation.

### WavLM

WavLM extends the speech pretraining line with denoising-oriented pretraining and masked speech prediction, using mixtures of speech/noise and targets designed for broader speech tasks. For a final method-specific section, obtain its official recipe and paper equation rather than inferring details from its name.

### Lesson

The same generic skeleton—corrupt or mask context and predict a target—covers very different learning problems. The target can be phonetic cluster IDs, quantized latent vectors, word IDs, pixels, or teacher features. The loss's semantics are determined by that target.

## 14. I-JEPA, V-JEPA and action-conditioned extensions

For detailed equations, loss reductions and source audits of I-JEPA, V-JEPA, V-JEPA 2/2.1 and EB-JEPA, see [JEPA prediction and action-conditioned world models](33_jepa_prediction_and_action_conditioned_world_models.md). This note distinguishes target-block latent prediction from pixel reconstruction, and passive video SSL from action-conditioned post-training.


I-JEPA predicts a target image region's representation from a context region's representation. The target encoder and predictor define a latent prediction task; pixel-level generation is not required. This can bias the objective toward predictable high-level structure rather than reconstructing every texture detail. The method still does not prove that all target information is preserved or that its latent transition is physically sufficient.

V-JEPA extends feature prediction to video, so targets can occupy space-time regions. The temporal setting allows the training signal to capture dynamics-related structure, but passive video prediction is not automatically an action-conditioned model: without action inputs/targets, it does not identify which future follows a selected control.

V-JEPA 2 adds a distinct action-conditioned post-training system, V-JEPA 2-AC, trained with robot interaction trajectories for manipulation. The official repository announced V-JEPA 2.1 on 16 March 2026; its paper is arXiv:2603.14482. The reported approach includes dense predictive loss on both context/visible and masked tokens, deep self-supervision at intermediate encoder layers, multimodal image/video tokenizers, and model/data scaling. The exact paper version should be checked before finalizing the book, but this development is important because it shifts attention to dense and temporally consistent features, not only a single global embedding.

The chapter handoff must preserve three distinct statements:
1. learning reusable visual features from images/video;
2. predicting target latent features from context;
3. predicting consequences conditioned on an action and using them for control.

These claims are related but not interchangeable.

## 15. Diffusion and flow matching: explicit boundary case

Diffusion models learn denoising or score/vector-field objectives across corrupted noise levels; flow-matching methods learn a vector field that transports one distribution into another. They can train on unlabeled observations and can be used inside generative world models. However, their central objective is generative sampling, not primarily to make a reusable representation invariant or predictive. They belong in the book's generation section, where the sampling dynamics can be derived without making Chapter 3 a catch-all survey of all learning without human labels.

## 16. Target-space comparison

| Method | Context | Target | Loss | What the loss directly encourages |
|---|---|---|---|---|
| Rotation / jigsaw | Image, sometimes transformed | Applied rotation / spatial ordering | Classification | Features that identify a synthetic transformation |
| MAE | Visible patches | RGB pixels on masked patches | MSE | Conditional pixel reconstruction |
| SimMIM | Corrupted masked image | Raw pixel values | Reconstruction loss | Predicting target intensities from visible content |
| BEiT | Visible token sequence | Discrete visual code IDs | Cross-entropy | Recovering tokenizer symbols |
| BEiT v2 | Masked image | Semantic quantized visual tokens | Cross-entropy | Predicting target codebook entries |
| MaskFeat | Visible regions | HOG or other descriptors | Feature regression | Predicting prescribed local features |
| BERT | Bidirectional corrupted tokens | Original token IDs | Cross-entropy | Reconstructing missing words from both directions |
| GPT | Causal prefix | Next observed token | Cross-entropy | Autoregressive sequence likelihood |
| ELECTRA | Corrupted tokens | Original / replaced bit | Binary classification | Discriminating generated corruptions |
| data2vec | Masked input | Contextual teacher feature | Feature regression | Matching teacher-derived contextual targets |
| wav2vec 2.0 | Masked latent audio plus context | Quantized speech latent | Contrastive loss | Identifying the correct unit among candidates |
| HuBERT | Masked audio context | Cluster-derived unit | Cross-entropy | Recovering pseudo-unit labels |
| I-JEPA | Context region | Target region's feature | Latent prediction | Predicting target representation, not pixels |
| V-JEPA | Video context | Target space-time features | Latent prediction | Predicting masked/future video features |
| V-JEPA 2.1 | Video context plus visible tokens | Dense latent targets at multiple depths | Dense predictive loss | Improving local and temporal feature consistency |


For the target/loss derivations and source ledger for language and speech SSL, see [the objective dossier](34_language_and_speech_ssl_objectives_from_first_principles.md).
