# 11. When the target belongs to another modality

So far, the examples have used visual context and visual target features. The JEPA construction does not require the target to belong to the same modality as the input. VL-JEPA makes this point concrete: it predicts a text representation from visual evidence, with a language query specifying what should be expressed.

The important idea is not a catalogue of vision-language tasks. It is that the target space is a design choice. The visual encoder, query and target encoder play separate roles.

## 11.1 The query is not the answer

Let \(X_V\) denote an image or video, \(X_Q\) a text query and \(Y\) the target text. The visual encoder produces

\[
S_V=f_\theta(X_V),
\]

while a text encoder maps the answer to

\[
S_Y=e_\psi(Y).
\]

The predictor combines visual features and the query:

\[
\widehat S_Y=p_\phi(S_V,X_Q).
\]

The query tells the model which question is being asked; the target text is the answer against which its prediction is trained. For one video, a query might ask which object a person is about to touch. A different query could ask what action the person is performing. The visual evidence is shared, but the requested conditional prediction changes.

This is the same role played by a spatial query in I-JEPA or a spatiotemporal query in V-JEPA. The query specifies the requested target. It is not itself the target representation.

## 11.2 Why predict a text embedding?

A token-generative model predicts a sequence of discrete symbols. VL-JEPA instead estimates a continuous representation produced by a text encoder. Semantically related descriptions can be close in that target space even when their word sequences differ. The objective therefore compares meanings as represented by the chosen text encoder, rather than requiring the predictor to reproduce one exact sequence of tokens.

VL-JEPA uses a bidirectional InfoNCE objective between predicted and target text embeddings. For normalized predicted and target embeddings \(\bar{\hat s}_i\) and \(\bar s_i\), one direction has the form

\[
\mathcal L_{\hat S\rightarrow S}
=
-\frac1B\sum_{i=1}^{B}
\log
\frac{
\exp(\bar{\hat s}_i^\top\bar s_i/\tau)
}{
\sum_{j=1}^{B}
\exp(\bar{\hat s}_i^\top\bar s_j/\tau)
},
\]

where \(B\) is batch size and \(\tau\) is temperature. The reverse direction exchanges the predicted and target roles. The positive pair is encouraged to align, while other batch targets provide contrastive comparisons. This is the paper's objective; it does not use SIGReg as its collapse regularizer. (Chen et al., 2026)

The choice of text encoder defines which linguistic distinctions remain visible. If two phrasings are close in its embedding space, the predictor is not strongly penalized for treating them as similar. If two answers differ in a small but important detail and the text encoder maps them too close, the target may be inadequate for that task. Embedding prediction does not remove the need to evaluate the target geometry.

## 11.3 Prediction and language generation are different operations

The text decoder is not required to train the embedding-prediction objective. Some tasks can use the predicted representation directly. For retrieval, it can be compared with candidate text embeddings. For classification, encoded label descriptions can be ranked against it. Captioning or open-ended question answering requires a readable string, so a decoder is invoked to turn the predicted embedding into language.

The published VL-JEPA system uses a frozen V-JEPA 2 ViT-L visual encoder. Its predictor is initialized from Transformer layers of Llama 3.2, while its text target encoder is initialized from EmbeddingGemma. Projection heads map predicted and target features into a shared 1,536-dimensional space. These are the paper's implementation choices, not requirements of cross-modal prediction. Training is staged: large-scale visual-text alignment is followed by query-conditioned supervised fine-tuning for tasks such as visual question answering. (Chen et al., 2026)

Because the embedding can be updated without decoding every frame into text, the paper also evaluates selective decoding for streaming video. In its EgoExo4D protocol, decoding at 0.35 Hz matches the reported performance of uniform decoding at 1 Hz, with approximately 2.85 times fewer decoding operations. This is a result for that evaluation protocol, not a general latency guarantee. (Chen et al., 2026)

## 11.4 A language query is not a control input

A query may ask a hypothetical question, but that wording does not make the model an intervention model. The predictor estimates a text embedding that answers the query. A controlled transition model estimates the consequence of a specified action:

\[
\widehat S_Y=p_\phi(f_\theta(X_V),X_Q),
\qquad
\widehat S_{t+1}=F_\omega(S_t,A_t).
\]

The first is cross-modal semantic prediction. The second is action-conditioned dynamics. To compare candidate robot commands, the predictor must receive the candidate action and be trained on trajectories that connect actions to subsequent observations. A natural-language question about what might happen is not a substitute for that action interface.

VL-JEPA therefore extends the range of possible target modalities without collapsing the distinction between representation prediction and controlled dynamics. The target may be text, but the conditioning variable must still correspond to the quantity whose effect the model is meant to estimate.

**Figure 4.18. Cross-modal prediction separates visual context, query and text target.**

Description: Visual features and a text query enter a predictor. A separate text encoder supplies the target feature during training. The predicted embedding can be compared with candidate texts directly or decoded when a readable answer is required.