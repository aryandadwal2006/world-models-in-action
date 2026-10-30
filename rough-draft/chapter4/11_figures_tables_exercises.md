# 11 — Figure storyboard, tables and exercises

## 1. Overall visual system

Use the same academic textbook style already established in Chapter 3:
- landscape diagrams where needed, white background;
- charcoal labels and line work;
- restrained blue accent #3B6FB6;
- grayscale-safe contrast;
- no gradients, 3-D objects, glossy icons, poster layouts or decorative title text;
- one claim per figure;
- no figure number, title or caption embedded inside artwork;
- all labels readable at final print width.

Figure captions should stay within approximately 340 characters and explain the visual argument, not merely name the objects. The actual manuscript should reference each figure in the prose and keep the image brief aligned with the caption.

## 2. Suggested 20-figure register

| Proposed figure | Core teaching claim | Caption direction |
|---|---|---|
| 4.1 | Pixel reconstruction and latent prediction impose different targets | Reconstruction measures observation error; JEPA measures error in a learned target space. Neither target is universally correct for every downstream task. |
| 4.2 | Generic JEPA training graph | The context encoder sees selected context; the target encoder supplies target features; the predictor estimates those features using context and target positions. |
| 4.3 | I-JEPA spatial masks | One context region predicts several target blocks in the same image; the mask geometry defines the conditional prediction task. |
| 4.4 | Multiple target blocks | Predicting nearby and distant targets changes the spatial context needed for the task. |
| 4.5 | V-JEPA space-time mask | Context tokens and target tokens occupy spatial and temporal positions in a clip. |
| 4.6 | Conditional-mean effect under MSE | When one context has multiple possible targets, deterministic squared error predicts their conditional mean, which need not be one realized future. |
| 4.7 | Target update mechanisms | EMA, stop-gradient and symmetric branches create different training dynamics and should not be treated as synonyms. |
| 4.8 | Action-free vs action-conditioned V-JEPA 2 stages | V-JEPA 2 pretraining learns video features; V-JEPA 2-AC adds an action-conditioned predictor through robot trajectory post-training. |
| 4.9 | V-JEPA 2.1 dense feature objective | Dense loss on visible and masked tokens and deep self-supervision encourage spatially grounded representations. |
| 4.10 | VICReg, SIGReg and VISReg | Each objective constrains a different aspect of feature statistics: moments, projected distribution or separated scale and shape. |
| 4.11 | Gaussian target by projections | A high-dimensional target distribution can be tested along sampled one-dimensional directions; finite samples only approximate the all-directions condition. |
| 4.12 | Characteristic-function comparison | Empirical characteristic functions are compared with the standard-normal target over selected frequencies. |
| 4.13 | Sliced-Wasserstein sorting | Projected samples are sorted and compared with target quantiles along each sampled direction. |
| 4.14 | Scale and shape can vary independently | A cloud can be too small, anisotropic or non-Gaussian; these are distinct properties of the embedding distribution. |
| 4.15 | Action-conditioned latent prediction | The model predicts a future representation conditioned on a current latent and a specified action. |
| 4.16 | Closed-loop planning bridge | A planner proposes actions, the model predicts latent outcomes, a cost evaluates them, and the agent replans after new observations. |
| 4.17 | Model/planner/environment evidence | Planning success belongs to the complete model–planner–environment loop rather than to an encoder benchmark alone. |
| 4.18 | JEPA-WM data/model/planner components | Training trajectories, encoder, predictor, planner and environment jointly shape measured success. |
| 4.19 | LeCun's proposed agent architecture | A world model is one component alongside perception, memory, costs, actor, critic and configurator. |
| 4.20 | Chapter synthesis: representation to planning | The path from feature learning to action requires distinct evidence at representation, prediction and planning levels. |

Do not create separate figures for every acronym. Combine a small number of related concepts where one diagram can teach a comparison cleanly.

## 3. Optional figures for a 90–100 page version

- a step-by-step diagram of the exact Epps–Pulley statistic after source verification;
- a covariance-vs-distribution example with non-Gaussian variables;
- a visual of multimodal futures and the MSE mean;
- a temporal-horizon diagram showing uncertainty growth;
- an action-label alignment timeline;
- a comparison of global versus dense targets;
- a failure example where a planner exploits predictor error.

These are optional because too many diagrams can fragment the argument.

## 4. Suggested tables

**Table 4.1 — Core image/video JEPA methods.** I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2-AC and V-JEPA 2.1; columns for input, target, temporal scope, action conditioning, main contribution and evidence type.

**Table 4.2 — Target branch and predictor mechanisms.** Compare EMA, stop-gradient, symmetric branches, predictor inputs and target normalization. Cite the primary method for each row.

**Table 4.3 — VICReg, SIGReg/LeJEPA and VISReg.** Compare invariance signal, scale term, shape statistic, projection requirement, cost dependencies and caveats.

**Table 4.4 — Adjacent modality/variational lines.** A-JEPA, Audio-JEPA, VL-JEPA, Var-JEPA and VJEPA. Explicitly distinguish Var-JEPA (Gögl & Yau) from VJEPA (Huang).

**Table 4.5 — What kind of evidence does the paper provide?** Frozen representation, latent prediction, action-conditioned prediction, physical planning. This prevents mixing incomparable metrics.

Limit tables to five unless a later draft demonstrates a genuine reader need.

## 5. Code listing budget

Use 3–5 short listings at most:
1. generic objective pseudocode, labeled pseudocode rather than official implementation;
2. mask generation or target feature extraction from official I-JEPA code;
3. one short SIGReg or VISReg statistic from the relevant official repository;
4. V-JEPA 2 encoder loading or inference snippet from the official repository;
5. optionally a planner-facing prediction call from official JEPA-WMs.

All listings should answer a specific conceptual question. Do not reproduce a full training loop or large model class. Any copied code needs an exact file/line reference and a checked license. If it is not licensed for book reproduction, paraphrase the operation in pseudocode and attribute the upstream implementation.

## 6. Exercises

1. **Objective derivation.** Starting from the generic context/target setup, write a feature-prediction loss and explain which branch sees target tokens.
2. **Mask design.** Construct two context/target masks for a video clip. Explain why a nearby target may be easier to predict than a temporally distant target.
3. **Conditional mean.** A context is compatible with two target vectors. Derive the deterministic MSE predictor and explain why its result may not be a realizable target.
4. **Update mechanisms.** Contrast EMA target updates with stop-gradient. Which parameters receive prediction gradients in each mechanism?
5. **V-JEPA 2 stages.** Explain why an action-free pretrained encoder and an action-conditioned post-trained predictor are different components.
6. **Distributional regularization.** Give two distributions with identical mean and variance but different characteristic functions. Explain what this says about covariance penalties.
7. **Cramér–Wold.** Explain why all one-dimensional projections identify a distribution but a finite set of sampled projections does not prove exact equality.
8. **Sliced-Wasserstein.** For equal-size samples on the real line, explain why sorting gives the optimal 1-D matching.
9. **Action conditioning.** Give two trajectories where the same observation may lead to different next-state distributions under different actions. Write a conditional prediction equation.
10. **Planning evidence.** Propose an experiment that distinguishes good frozen features from a reliable action-conditioned world model.
11. **Claim audit.** Identify which assumptions would be needed to claim that a Gaussian embedding minimizes a downstream risk.
12. **System design.** Map a JEPA encoder/predictor into LeCun's proposed agent architecture. Identify at least three components not supplied by the representation loss.

## 7. Exercise policy

Exercises should test understanding, derivation and design rather than recall of paper acronyms. Include a mix of:
- short derivations;
- architecture diagrams;
- critical reading of an abstract or method section;
- evaluation design;
- failure analysis.

Avoid asking the reader to reproduce internet-scale pretraining as a homework exercise.
