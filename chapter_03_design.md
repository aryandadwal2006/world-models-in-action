# Chapter 3 Design — Representation Learning Before Generation (v3)

**Book:** World Models in Action (Manning) · **Status:** design document, not manuscript
**Inputs used:** v2 design, CH01 text, CH02 AsciiDoc source, Manning Manual of Style, MQR, Gui et al. "A Survey on Self-supervised Learning" (arXiv:2301.05712), dm_control official repository (github.com/google-deepmind/dm_control), world-model benchmark survey discussion

---

## 0. Changelog from v2 to v3 — the warehouse is gone

The chapter's thesis, structure (3.1–3.8), mathematics (equations 3.1–3.20), VICReg treatment, MAE comparison, bottleneck sweep, CPC→JEPA bridge, probe epistemology, and 3-seed methodology are **unchanged**. What changes is the ground everything stands on: the custom warehouse world is deleted and replaced by the DeepMind Control Suite (DMC). Substantive changes:

1. **Data source replaced.** v2's procedural warehouse generator (~40 lines of NumPy, `data/warehouse.py`) is removed entirely. Chapter 3 now uses rendered frames from `dm_control.suite`, with the simulator's physics state (`qpos`, `qvel`) as the ground truth for probes and tests. The state is used **only for evaluation** — never passed to any representation-training objective. This is stricter than v2 in one way (we no longer control the renderer) and stronger in another (the ground truth is a real physics engine's state, not our own bookkeeping).
2. **Task subset fixed and minimal.** Primary task: `cartpole_balance` (4-dimensional state; every core experiment runs here). Secondary: `finger_spin` (rotating body; used in the objective-comparison table). Hard end: `cheetah_run` (18-dimensional state; used in the bottleneck sweep to show the saturation point moves with state dimensionality). Three tasks, one difficulty gradient, nothing more. Memory Maze, Atari 100k, Crafter, Meta-World, and DROID are explicitly deferred (see §2.2) — chapter 3 does not touch them.
3. **Velocity-leak machinery replaced by a measurement.** v2 needed rendering constraints plus a pixel-equality unit test to guarantee that velocity was absent from single frames — because we built the renderer and could have leaked it. In DMC, each frame is a static MuJoCo render; motion information cannot exist in one frame by construction of the physics engine, not by our discipline. The unit test is deleted. In its place: the **aliasing diagnostic** (new script `05_aliasing_diagnostic.py`) — probe positions and velocities from single frames versus two-frame-stacked inputs. Hypothesis: positions decode linearly from one frame, velocities do not, and stacking two frames recovers them. This turns a rigging-risk we had to engineer away into a measured demonstration of chapter 2's aliasing concept. It is a better experiment than the one it replaces.
4. **Occlusion test replaced by an unobserved-interval test.** DMC has no shelves; the forklift-behind-a-shelf experiment cannot be run and is removed. Section 3.7.3's first test becomes: drop k consecutive frames at evaluation time; the CPC context model predicts latents across the gap; a probe reads the state from predicted latents and is compared against a frame-only encoder. This is honestly supported by DMC and tests the same underlying capacity — holding a state estimate through missing observations. **Consequence (flagged honestly):** chapter 3 no longer literally renders chapter 2's figure 2.3 scenario. The prose bridge is rewritten (§4, 3.7.3); a light chapter 2 touch-up is logged as follow-up work (§11, risk R5).
5. **Figure 3.2 (pretext zoo) and all warehouse-specific figures redrawn on DMC frames.** A new figure (3.2) shows the chapter's three tasks as rendered 64×64 frames. The latent-projection figure now uses marker shape for binned pole angle and grayscale for cart position. Figure register renumbered to 15 figures.
6. **Repo tree updated.** `worldmodels/data/warehouse.py` → `worldmodels/data/dmc_data.py` (collection + caching) plus `worldmodels/data/augment.py` (view pipeline). Script list grows to 12 with the aliasing diagnostic inserted; all scripts renumbered (§2.1).
7. **Dependencies relaxed, deliberately.** v2's "torch + numpy + matplotlib only for Part I" rule is amended: `dm_control` (pinned) is added, needed **only** by the data-collection script `01_collect_dmc_data.py`, which writes a cached `.npz`. Every downstream script runs from the cache with torch + numpy + matplotlib. Rationale and rendering-backend caveats (`MUJOCO_GL`) documented in §2.2 and risk R1.
8. **Data collection policy stated.** Frames are collected under a uniform random policy within each task's action spec. Consequence for CPC: the future is genuinely stochastic and the model is not action-conditioned, which bounds achievable identification scores. This is now a stated hypothesis with a measurement attached, and it becomes the *measured* motivation for chapter 4's action conditioning instead of a promissory note (§4, 3.7.2; risk R2).
9. **Bottleneck sweep strengthened.** Because DMC state dimensionality is known exactly (cartpole: 4; cheetah: 18), the sweep's plateau prediction is now quantitative: probe R² should plateau at smaller d on cartpole than on cheetah. Figure 3.13 becomes a two-task curve. This is a sharper, falsifiable version of v2's "rises then plateaus."
10. **Naming and cross-references.** Prose name is "the Control Suite" (first use: "DeepMind Control Suite, DMC"); code name `dmc_data.py`. All warehouse references in sections 3.1(f), 3.2, 3.3.1, 3.5.3, 3.7, 3.8, exercises 4–6, and the closing checklist are rewritten. Later-chapter environment placement is stated once (§2.2) and echoed in 3.8's closing paragraph: Memory Maze for the memory chapter's evaluation, DMC retained for dynamics/planning chapters, Atari 100k / Crafter for the generality chapters, DROID for the embodied-AI chapter.

---

## 1. What this chapter has to do

(Unchanged in substance from v2; only the experimental-contract sentence changes.)

Chapter 1 made three promises: observations get compressed into a state, the state gets predicted, predictions get planned over. Chapter 2 turned those into contracts — sufficiency, minimality, aliasing, the Markov property — and explicitly said "the way we train the encoder determines the negotiation, which is what we will focus on in chapters 3 and 4."

Chapter 3 pays that debt for the static half: **how a state representation is learned from data with no labels, and how to check — honestly, with named limits on what each check can show — whether the learned representation carries what the chapter 2 contract demands.** Chapter 4 then adds prediction *inside* that representation space (JEPA). The division of labor:

- **Chapter 3 owns:** self-supervision as the training signal; contrastive learning in full mathematical depth; collapse and its fixes; masked prediction; bottlenecks and capacity; a working encoder module in the repo; and the measurement toolkit (linear probe, forward-prediction test) with explicit statements of what each tool can and cannot establish.
- **Chapter 4 owns:** joint-embedding prediction with a *predictor* and *target encoder* as first-class components (I-JEPA, V-JEPA, V-JEPA 2, LeJEPA), and action-conditioning.
- **The handoff:** ch3 ends by replacing the question "are these two views the same scene?" with "given the past, which future is this?" — contrastive predictive coding. Ch4 opens by removing the negatives from exactly that setup. No overlap, no gap.

**The chapter's thesis sentence** (stated in 3.1, echoed at every section head, cashed out in a closing synthesis table):

> A representation-learning objective is a specification of what information the representation is paid to keep.

Contrastive learning pays for what survives the view transformation. Masked reconstruction pays for what helps rebuild missing input. Temporal prediction pays for what predicts the future. Chapter 2 said a useful state keeps what matters and drops the rest; chapter 3 shows that "what matters" is decided by the loss, before any data is seen.

**Reader (per MQR):** basic Python, basic PyTorch (loss/optimizer/training loop), basic neural nets, basic transformers. No SSL background assumed. Every equation is derived from definitions the reader already has: dot products, softmax, cross-entropy, expectations.

**Register:** plain English, short declarative sentences, no analogies. Full derivations, every symbol defined at first use, every empirical claim framed as hypothesis → measurement, with the configuration named.

**Why the environment change serves the thesis:** the thesis is about objectives, not worlds. DMC is the environment the world-model literature itself uses (Dreamer, TD-MPC2, and successors), so the reader learns the mechanisms on the same ground where they will meet them again — and the encoder trained in chapter 3 is still alive in the dynamics and planning chapters, which keep DMC as their teaching environment. The chapter's experiments measure what representations retain from *real benchmark observations* instead of from a world we invented to make the measurements come out right.

---

## 2. The six design decisions

### 2.1 Repo: one growing library + numbered chapter scripts
Same merged Raschka/Spring-in-Action pattern as v2. Updated tree:

```
world-models-in-action/
├── worldmodels/                  # the growing library (survives all chapters)
│   ├── data/dmc_data.py          # ch3: DMC collection, caching, episode splits
│   ├── data/augment.py           # ch3: view pipeline (crop shift, intensity jitter)
│   ├── models/encoders.py        # ch3: ConvEncoder; ch4: +JEPA encoders; ch5: +SSM backbone
│   ├── losses/contrastive.py     # ch3: NT-Xent, SimSiam, VICReg; ch4: +JEPA losses
│   ├── losses/reconstruction.py  # ch3: masked MSE
│   ├── eval/probes.py            # ch3: linear probe, forward-prediction test; reused ch4–ch12
│   └── train.py                  # shared loop utilities (seeded runners)
├── ch03/
│   ├── 01_collect_dmc_data.py    # the only script needing dm_control
│   ├── 02_views_and_pairs.py
│   ├── 03_ntxent_from_scratch.py
│   ├── 04_train_contrastive.py
│   ├── 05_aliasing_diagnostic.py # single-frame vs stacked-frame probes
│   ├── 06_collapse_demo.py
│   ├── 07_three_mechanisms.py    # stop-grad, EMA, redundancy reduction (short cases)
│   ├── 08_vicreg.py
│   ├── 09_masked_autoencoder.py
│   ├── 10_bottleneck_sweep.py
│   ├── 11_temporal_cpc.py
│   └── 12_assemble_encoder.py
├── ch04/ … (empty until ch4)
├── notebooks/                    # optional, mirror the scripts
└── requirements.txt              # torch, numpy, matplotlib; dm_control pinned (script 01 only)
```

Rules: every numbered listing is an excerpt of a named file; short teaching examples should be CPU-accessible; the full three-seed objective comparison, latent-capacity sweep, and CPC sequence experiment can take substantially longer than ten minutes on CPU. Data collection is cached, experiment outputs are recorded, and resumable experiments must skip completed compatible runs. Every quantitative result reports mean ± standard deviation over seeds 0, 1, 2.

### 2.2 Data: the DeepMind Control Suite
Decision: **use the benchmark the field uses; do not build a world.** Chapter 3 renders frames from `dm_control.suite` (MuJoCo physics; Tunyasuvunakool et al. 2020, cited in a footnote at first use) and treats the simulator's physics state as privileged evaluation-only ground truth.

**Task subset, with reasons:**

| Task | State dim | Role in ch3 |
|------|-----------|-------------|
| `cartpole_balance` | 4 | Primary. Every core experiment (contrastive training, collapse, VICReg, MAE, aliasing, CPC) runs here. Small state → probe tables stay readable; renders are cheap. |
| `finger_spin` | 6 (3 generalized positions + 3 generalized velocities) | Secondary task in objective-comparison table 3.4. Its spinner hinge angle is not uniquely identifiable from a single rendered frame because the spinner geometry is symmetric under a half-turn; report that variable separately and exclude it from aggregate position scores. |
| `cheetah_run` | 18 | Hard end of the bottleneck sweep only: the plateau should move to larger d. Also the honest "this is harder than it looks" frame in figure 3.2. |

**Data specification:** 64×64 RGB frames; ~30k training frames and ~5k validation frames per task, collected as episodes under a uniform random policy within the action spec. Each frame is stored with raw MuJoCo generalized positions followed by generalized velocities (`qpos` then `qvel`) and an episode ID. The state is collected directly from simulator physics, not from the task's observation dictionary. `01_collect_dmc_data.py` writes one `.npz` per split; all other scripts read the cache. Collection is seeded and simulator versions are recorded.

**Sampling contract:** when an experiment caps the number of frames used for training or evaluation, it must choose indices across episode IDs and spread the chosen frames within each episode. Taking the first N frames is not permitted for capped evaluation sets: a 1,000-frame prefix can come from a single episode and have a narrower target distribution than the held-out split. For temporally stacked frames, sampling is performed over valid frame centres in the original arrays so stacks remain contiguous and never cross episode boundaries. For CPC, sample valid sequence starts across episodes while retaining every frame within each sequence. Each result records this sampling policy.

**The evaluation-only ground-truth rule, stated in the text:** the physics state is never an input to any representation-learning objective. It appears in exactly three places: the linear probe's targets, the forward-prediction test's targets, and the supervised baseline of table 3.4. A sidebar in 3.1 says why this mirrors the real setting — a robot streams pixels, not `qpos` — and why it is fairer than v2's generator, where we controlled both sides of the comparison.

**Rejected alternatives, updated:**

- **Custom warehouse generator (v2's choice):** removed. Its ground truth was ours, its geometry was ours, and every measured "phenomenon" risked being an artifact of both. Also: it made the book's evidence chain rest on a world no reader will meet again.
- **Memory Maze:** genuinely needed only where long-term memory is the subject. Reserved for the memory chapter's evaluation. Chapter 3 does not build a synthetic memory task to fake the need.
- **Atari 100k / Crafter:** real world-model difficulty (discrete actions, sparse rewards, exploration) belongs to the later generality chapters. Loading them into ch3 would bury the representation-learning story under environment engineering.
- **DROID / real robotics data:** the embodied chapter's case study (with V-JEPA 2 as the roadmap reference). Far too heavy for CPU-bound ch3.
- **Moving MNIST / dSprites:** v2's rejections stand (recognition-centric or no dynamics), with one update: dSprites' known-factors argument is now moot — DMC gives us known factors *and* physics.
- **Staying purely offline (ship pre-rendered frames):** rejected as the primary path because it hides the environment from the reader; kept as a contingency (risk R1) for readers whose rendering setup fails.

**Rendering reality (must be in the manuscript's setup sidebar):** `dm_control` renders offscreen via MuJoCo; on laptops this works out of the box, on headless Linux the reader sets `MUJOCO_GL=osmesa` (documented in the official repo). Script 01 sets and checks this; a TIP callout gives the Debian/Ubuntu one-liner (`libgl1-mesa-glx libosmesa6`). `dm_control` is pip-installed and pinned; the repo README's editable-mode and Homebrew notes go into a troubleshooting sidebar, not the prose.

### 2.3 Scope: temporal prediction stays, but only as the bridge — and as prediction, not invariance
(Unchanged from v2.) §3.7 introduces CPC with a GRU context. It does **not** build JEPA, add actions, or do rollouts. It exists so ch3 touches the book's thesis (prediction in latent space) and ch4 doesn't open cold. If page pressure hits, §3.7 is the designated cut. One scope note added: the random-policy data makes the action-conditioning gap *visible in a measurement* — that is a feature; ch4 inherits a demonstrated problem, not an asserted one.

### 2.4 Math: numbered display equations, derived inline
(Unchanged from v2.) Equations (3.1)–(3.20), latexmath, notation fixed in table 3.2, long proofs to Appendix A.

### 2.5 Manning mechanics, wired in from the start
(Unchanged from v2; checklist in §8.) One addition: DMC is a third-party artifact — first-use footnote cites the tech report; figure 3.2's rendered frames are our own renders of their simulator, captioned as such.

### 2.6 Environment roadmap across the book (new, one paragraph in 3.1 and echoed in 3.8)
The book uses the environment that best exposes the concept being taught: **DMC for representation, dynamics, and planning (ch3–ch7); Memory Maze where long-term memory is genuinely tested (memory chapter); Atari 100k / Crafter where generality and exploration are the subject; DROID where real embodiment is the subject.** Chapter 3 states this roadmap once so the reader knows the suite is a companion, not a detour — the same encoder architecture and probe toolkit survive every transition.

---

## 3. Updated chapter TOC (with page budget)

Structure and page budget unchanged from v2 (~50 pp). The 3.7.3 retitle is the only heading change.

```
3  Representation learning before generation                     ~50 pp
   This chapter covers (≤5 bullets, ≤45 chars/line)
   Chapter intro (no heading, ~1.5 pp)                              2
3.1 Learning a state without labels                                 4
3.2 Pretext tasks: supervision from structure                       4
3.3 Contrastive learning from first principles                     10
    3.3.1 Views, positives, and negatives
    3.3.2 From similarity to a loss
    3.3.3 What InfoNCE actually optimizes
    3.3.4 Training a contrastive encoder
    3.3.5 Reading the state back with a linear probe
3.4 The collapse problem and how it is avoided                      6
    3.4.1 Demonstrating collapse
    3.4.2 Three mechanisms in brief
    3.4.3 VICReg: making the anti-collapse force explicit
3.5 Masked prediction                                               6
    3.5.1 From masked words to masked patches
    3.5.2 Building a masked autoencoder
    3.5.3 Reconstruction versus representation
3.6 Latent bottlenecks and compact state                            6
    3.6.1 The information bottleneck as a lens
    3.6.2 Choosing the latent dimension
3.7 Predicting over time: from invariance to prediction             7
    3.7.1 Why the future is not just another view
    3.7.2 Contrastive predictive coding
    3.7.3 Two tests: unobserved intervals and forward prediction
3.8 Assembling the encoder                                          3
Exercises                                                           1
Summary                                                             1
```

---

## 4. Section-by-section blueprint

Notation: each subsection lists **Goal** · **Beats** · **Equations** · **Listings** · **Figures/Tables** · **Callouts/Sidebars**.

### Chapter opener + intro (2 pp)
**Beats:** Recall ch1's loop and ch2's contract in four sentences. State the problem: the contract says *what* the state must be; nothing says *how to get it* without labels. Thesis sentence. Preview the arc. Roadmap sentence naming the repo, the Control Suite, and that everything runs on CPU. Methodology sentence: "Unless stated otherwise, quantitative results are mean ± standard deviation over three seeds."
**"This chapter covers" bullets (draft):**
- Learning representations without labels
- Contrastive learning, derived and trained
- Why encoders collapse, and how it is prevented
- Masked prediction and latent bottlenecks
- Measuring what a representation contains

### 3.1 Learning a state without labels (4 pp)
**Goal:** Motivate SSL as *a* scalable way to attack ch2's contract; fix notation; state the thesis; introduce the environment honestly.
**Beats:** (a) The contract as an information statement. (b) Labels fall short three ways (scarce; task-shaped; unavailable to a robot streaming frames); "self-supervision is a scalable alternative," one sentence acknowledging simulator labels, synthetic supervision, and demonstrations as other routes — and noting this chapter *uses* simulator labels for evaluation only, which is exactly the honest version of that route. (c) Self-supervision defined precisely; supervised/unsupervised/self-supervised one paragraph each (survey framing). (d) **Thesis sentence, boxed.** (e) Notation table 3.2. (f) The chapter's experimental contract: every claim ends in a run; ground truth comes from the Control Suite's physics state and is used only for evaluation; seeds and configs reported; the data is rendered by `01_collect_dmc_data.py` from the actual benchmark, not generated by the book. (g) One paragraph: the book's environment roadmap (§2.6) and why ch3's three tasks are cartpole, finger, cheetah.
**Equations:** (3.1) z = f_θ(x); informal sufficiency statement, formalized as a lens in 3.6.1.
**Figures:** 3.1 The encoder's contract as an information diagram (original; mirrors ch2's visual language) · 3.2 The chapter's three tasks: one rendered 64×64 frame each from cartpole balance, finger spin, cheetah run (from run 01; captioned as our renders of the DMC simulator).
**Callouts:** DEFINITION sidebar "Self-supervised learning" · NOTE: LeCun's ICLR 2020 framing · SIDEBAR: the thesis sentence · SIDEBAR: "Why a benchmark and not a toy world" (half page: the evaluation-only ground-truth rule; the field uses this suite; the encoder survives into ch6–7).

### 3.2 Pretext tasks: supervision from structure (4 pp)
**Goal:** unchanged.
**Beats:** (a) Rotation prediction worked end-to-end on one cartpole frame: derive the 4-way classification objective (compressed from the survey). (b) Jigsaw and colorization, one paragraph each. (c) The limitation through the thesis lens: the pretext defines the payment schedule and pays for the wrong things — a rotation classifier can discard the pole's angle; a colorizer can discard geometry. (d) Pivot: we want invariance to nuisance (sensor noise, crop jitter) and sensitivity to state (positions, velocities). That requirement has a name — instance discrimination — and a machinery — contrastive learning.
**Equations:** (3.2) rotation-prediction cross-entropy.
**Figures:** 3.3 Pretext task zoo — rotation, jigsaw, colorization on one cartpole frame (redrawn in the book's style; **adapted from Gui et al. — credit line required**).
**Callouts:** NOTE crediting the survey.

### 3.3 Contrastive learning from first principles (10 pp) — the chapter's center of mass
**3.3.1 Views, positives, and negatives.** A "view" of a rendered frame: the same frame re-augmented with a small random crop shift and intensity jitter. Stated immediately: view generation is a *design decision with consequences* — it defines what the encoder learns to ignore. Positives = two augmentations of one frame; negatives = augmentations of other frames (with an honest NOTE: frame-level negatives on trajectory data can include near-identical frames from the same episode — the batch treats them as negatives anyway; this "false negative" wrinkle is real on video-like data, we name it rather than hide it, and the measurements still stand). Listing: the pair pipeline. Figure 3.4: pair-construction schematic.
**3.3.2 From similarity to a loss.** Unchanged build: cosine similarity (3.3); identification as softmax (3.4); cross-entropy = NT-Xent/InfoNCE (3.5); τ as sharpness.
**3.3.3 What InfoNCE actually optimizes.** Unchanged two results: the bound I(z₁; z₂) ≥ log(K+1) − L_InfoNCE (3.6) with the log(K+1) convention; then the honest interpretation — the objective pays the encoder to keep exactly what the view pipeline preserves. The design question lands on DMC concretely: an augmentation that jitters brightness is harmless; a crop aggressive enough to cut the pole out of frame would pay the encoder to throw the state away — and exercise 4 makes the reader measure exactly that. Gradient decomposition (3.7) in three lines.
**3.3.4 Training a contrastive encoder.** Listings: CNN encoder skeleton; NT-Xent in ~15 lines; training loop. Run on cartpole frames: loss curve (3 seeds).
**3.3.5 Reading the state back with a linear probe.** Probe defined (3.8): frozen encoder, linear map from z to physics-state variables, R². Epistemology stated once for the whole book: **a probe measures what the representation makes explicitly, linearly available — no more.** Then the chapter's first genuinely temporal measurement, the aliasing diagnostic: probe *positions* (cart position, pole angle via cos/sin) and *velocities* (cart velocity, angular velocity) from (i) single-frame latents and (ii) latents of two stacked frames. Hypothesis: positions decode from one frame; velocities sit near chance from one frame and recover with two — chapter 2's aliasing, measured, and the setup for §3.7's claim that time must enter the objective, not just the input.
**Equations:** (3.3)–(3.8) as registered.
**Listings:** 3.1 data collection excerpt (task loading, random policy, state capture — including the comment marking states as evaluation-only) · 3.2 pair pipeline · 3.3 ConvEncoder · 3.4 NT-Xent loss · 3.5 training loop (excerpt) · 3.6 linear probe.
**Figures:** 3.5 InfoNCE as classification over a batch · 3.6 training curve, 3-seed band · 3.7 latent space before vs. after training — 2-D projection, marker shape = binned pole angle, grayscale = cart position, no color.
**Tables:** 3.3 probe R² per state variable, single-frame vs stacked-frame, mean ± std over 3 seeds.
**Callouts:** WARNING on τ · TIP: batch size = negatives · SIDEBAR "Where InfoNCE comes from" · WARNING: what a probe can and cannot tell you · NOTE: the false-negative wrinkle on trajectory batches.

### 3.4 The collapse problem and how it is avoided (6 pp)
Unchanged in every load-bearing respect — collapse is data-agnostic, and that is worth one sentence in the text: the trivial minimum exists for any joint-embedding objective on any data, which is why this section's experiments are the book's cheapest.
**3.4.1 Demonstrating collapse.** Remove negatives, keep the attraction term, watch latents converge to a point while the loss hits zero (3 seeds, cartpole frames). Figure 3.9: per-dimension latent std over training, with vs. without negatives.
**3.4.2 Three mechanisms in brief.** SimSiam (3.9, 3.10); BYOL EMA (3.11); Barlow Twins (3.12, 3.13). One mechanism + one equation + one paragraph each.
**3.4.3 VICReg.** Deep treatment unchanged: variance hinge with named ε floor (3.14), invariance (3.15), covariance (3.16), total (3.17). Run `08_vicreg.py` as a standalone illustration of the three forces. Do not claim it should match Table 3.3 within seed noise: it uses a different training budget and sample cap, while Table 3.4 separately compares method families under a matched budget.
**Figures:** 3.9 collapse curves · 3.10 Siamese family pipelines (redrawn, **credit: adapted from Gui et al.**) · 3.11 VICReg's three forces (original).
**Tables:** 3.1 family comparison (rebuilt from the survey, **credit line**).
**Callouts:** SIDEBAR "Is stop-gradient enough?" · WARNING: symmetric losses double compute.

### 3.5 Masked prediction (6 pp)
**3.5.1** BERT in three sentences; MAE's move; why images tolerate 75% masking. One added sentence of honesty: DMC frames are far simpler than natural video, so the section's comparison is config-bound — the hypothesis is about *this* architecture, budget, and data, and the survey supplies the general picture.
**3.5.2** Listings: patchify + random mask; masked MSE on masked patches only (3.18). Run on cartpole frames; show reconstructions (pole and cart legibly rebuilt from 25% of patches — a genuinely striking panel at this scale).
**3.5.3 Reconstruction versus representation.** Through the thesis lens: reconstruction pays for every pixel, including the background texture and floor that carry no state. Hypothesis → measurement: compare state probes and reconstructions under the stated budgets, then report what the two tasks actually show. Table 3.4 is a method-family comparison, not a controlled loss-only ablation: MAE uses a patch-transformer encoder/decoder, while the other learned representations use a convolutional backbone (and the supervised reference is trained end-to-end on privileged labels). Attribute the findings to these complete configurations, not to the objective alone. Punchline unchanged: keep masked prediction's strengths without paying for pixels — predict in latent space; §3.7 and ch4.
**Listings:** 3.10 masking + MAE forward pass (excerpt).
**Figures:** 3.12 MAE pipeline with real cartpole reconstructions (redrawn schematic + run outputs; **credit for the schematic: adapted from Gui et al.**) · 3.13 joint embedding vs. reconstruction (redrawn, **credit**).
**Tables:** 3.4 method-family comparison: contrastive, VICReg, MAE, and a supervised state-prediction reference, with stated state-probe metrics across two tasks. Include variable-level results and disclose the architecture differences and the symmetry-unobservable Finger Spin hinge coordinate; do not treat this as a loss-only ablation. The simulator supplies privileged labels for the supervised reference, and those labels must remain excluded from the self-supervised objectives.
**Callouts:** NOTE: VAEs/GANs deferred to the generative chapters.

### 3.6 Latent bottlenecks and compact state (6 pp)
**3.6.1 The information bottleneck as a lens.** Unchanged: L = I(z; x) − β I(z; y) (3.19), β as exchange rate, both disclaimers stated plainly (a deterministic autoencoder is a capacity bottleneck, not an information-theoretic one; VICReg does not optimize the IB objective).
**3.6.2 Choosing the latent dimension.** The experiment is now quantitatively falsifiable. Hypothesis: on cartpole balance (4 state variables) probe R² rises with d and plateaus early; d = 2 aliases exactly as ch2 predicts; on cheetah run (18 state variables) the plateau sits at larger d. Run the 3.3 encoder at d ∈ {2, 4, 8, 16, 64, 256} on both tasks, 3 seeds each. Figure 3.14: R² vs. d, two curves with seed bands. Rule of thumb box: smallest d whose probe saturates — minimality as small as sufficiency allows, with a measurement attached. One honest sentence: the plateau is not guaranteed to sit exactly at the state dimension (linear readout of angles needs sin/cos capacity; the measurement, not the arithmetic, decides) — this is hypothesis-first discipline, not waffling.
**Listings:** 3.11 the sweep driver (excerpt).
**Figures:** 3.14 sweep curves (from run).
**Callouts:** TIP: the sweep costs an afternoon and settles a vibe argument · SIDEBAR "The bottleneck in one paragraph."

### 3.7 Predicting over time: from invariance to prediction (7 pp) — the bridge
**3.7.1 Why the future is not just another view.** Unchanged trap statement: pairing frames at t and t+Δ as contrastive positives pays the encoder to treat state change as nuisance. Now reinforced by a result the reader has already seen: table 3.3 showed velocity is *not* in a single frame — so a frame-level invariance objective has no way to pay for it, and a frame-pair invariance objective would pay to destroy it. The question changes: not "do these two frames agree?" but "**given the past, which future is this?**"
**3.7.2 Contrastive predictive coding.** Unchanged machinery: GRU context c_t = g(z₁…z_t) (WARNING: context never sees the future); compatibility f(z_{t+k}, c_t) = exp(z_{t+k}ᵀ W_k c_t) (3.20); identification reuses equation 3.5's softmax, presented in parallel columns. One new honest paragraph: the data comes from a random policy, so the future is genuinely stochastic and the model sees no actions — the achievable identification score has a ceiling neither we nor the method can remove without action conditioning. Stated as hypothesis, measured in the run, and handed to ch4 as a demonstrated motivation: ch4 deletes the distractors *and* adds the actions, and the reader will know why both matter.
**3.7.3 Two tests: unobserved intervals and forward prediction.** *Test one (unobserved interval):* at evaluation, drop k consecutive frames; the CPC context predicts latents across the gap; a probe reads cart position and pole angle from the predicted latents, compared against a frame-only encoder re-entering after the gap and against linear extrapolation of the state. Hypothesis: the context model's estimates degrade gracefully across short gaps and beat both baselines; measured, 3 seeds. The prose makes the connection to ch2's occlusion discussion explicit *as an analogy, not a re-enactment*: the suite has no occluders, and the chapter says so — what is tested is the same capacity (holding a state estimate through missing observations) in the form DMC genuinely supports. *Test two (forward prediction):* unchanged from v2 — frozen encoder, small predictor from current latent to next physics state, baseline sees a short state history directly; remaining gap to planning utility stated honestly and deferred to the planning chapter.
**Listings:** 3.12 CPC context model + compatibility scoring (excerpt).
**Figures:** 3.15 CPC schematic (pair-designed with ch4's JEPA diagram) · 3.16 tracking through the gap — probe-read cart position over time, dashed band marking the dropped-frame interval, context-predicted trace vs. frame-only re-entry (from run; grayscale).
**Tables:** 3.5 static single-frame vs. stacked-frame vs. CPC-context models on {position, velocity, gap-interval state, next-state prediction}, mean ± std over 3 seeds.
**Callouts:** NOTE naming CPC (van den Oord et al., 2018) and the lineage toward JEPA · WARNING: causality · WARNING: a GRU context is a placeholder — the memory question reopens properly in the memory chapter, whose evaluation environment (Memory Maze) is built for exactly that question.

### 3.8 Assembling the encoder (3 pp)
**Beats:** Collect the chapter into `worldmodels/encoders.py`: one `Encoder` exposing `encode(obs)` and `update(state, obs)` — the signature ch1's skeleton calls; slot it into the ch1 agent (dynamics and reward still placeholders). **Synthesis table 3.6** unchanged: objective → paid to keep → may discard → invited failure. Checklist mapping each ch2 contract clause to the section and its measured evidence, with residual gaps named. Final paragraph, updated: what ch3 still cannot do — predict without negatives, condition on actions (now a *measured* gap, table 3.5), remember long horizons — as ch4's and the memory chapter's agenda; and one sentence of environment continuity: this encoder, these probes, and this suite return when dynamics enter.
**Listings:** 3.13 the assembled Encoder module.
**Figures:** 3.17 the ch1 loop with the encoder slot filled (styled after fig 1.1).
**Tables:** 3.6 the synthesis table.

### Exercises (1 p) — before Summary
1. Derive the NT-Xent gradient w.r.t. z₁; identify pull/push terms (sketches in appendix).
2. Show that as τ → ∞ InfoNCE weights all negatives nearly equally; predict the effect on the learned space; verify with one run.
3. In VICReg, which single term prevents collapse, and what failure appears if the covariance term is removed? Predict, then test.
4. Your views use a ±3-pixel crop shift. What happens to the cart-position probe if the shift grows to ±16 pixels on cartpole frames — where the cart's range is only tens of pixels wide? Answer from the thesis sentence first, then run it.
5. At what latent dimension does your probe saturate on cartpole balance, and how does the answer move on cheetah run? Relate both to the tasks' state dimensions, then run the sweep.
6. Modify the CPC target from one step ahead to five. What happens to the loss and the forward-prediction test — and which of the two effects comes from the log(K+1) bound and which from the random policy's stochasticity? Explain, then run it.
7. (New, replaces nothing — the deleted warehouse exercise slot) Rerun the aliasing diagnostic on finger spin. Which state variables behave like cartpole's, and which break the pattern? Form the hypothesis before looking at the table.

### Summary (1 p)
Six to eight bullets, one per section, each carrying a derived or measured result; final bullet states the thesis sentence. Nothing after the list.

---

## 5. Equation register (fixed order, numbered 3.1–3.20)

Unchanged from v2 — the mathematics never touched the warehouse.

| # | Content | Section |
|---|---------|---------|
| 3.1 | z = f_θ(x); encoder definition | 3.1 |
| 3.2 | Rotation-prediction cross-entropy | 3.2 |
| 3.3 | Cosine similarity | 3.3.2 |
| 3.4 | Softmax identification of the positive | 3.3.2 |
| 3.5 | NT-Xent / InfoNCE loss | 3.3.2 |
| 3.6 | MI lower bound: I(z₁;z₂) ≥ log(K+1) − L, K negatives | 3.3.3 |
| 3.7 | InfoNCE gradient decomposition (pull/push) | 3.3.3 |
| 3.8 | Linear probe objective | 3.3.5 |
| 3.9 | SimSiam negative cosine similarity | 3.4.2 |
| 3.10 | SimSiam symmetric loss | 3.4.2 |
| 3.11 | BYOL EMA update | 3.4.2 |
| 3.12 | Barlow Twins cross-correlation matrix | 3.4.2 |
| 3.13 | Barlow Twins loss | 3.4.2 |
| 3.14 | VICReg variance hinge (ε floor named) | 3.4.3 |
| 3.15 | VICReg invariance term | 3.4.3 |
| 3.16 | VICReg covariance term | 3.4.3 |
| 3.17 | VICReg total loss | 3.4.3 |
| 3.18 | Masked MSE (masked patches only) | 3.5.2 |
| 3.19 | Information-bottleneck Lagrangian (lens, with disclaimers) | 3.6.1 |
| 3.20 | CPC compatibility function f(z_{t+k}, c_t) = exp(z_{t+k}ᵀ W_k c_t) | 3.7.2 |

CPC's temporal InfoNCE stays "equation 3.5 re-aimed" (parallel columns), not a new number. Appendix A absorbs the (3.6) proof and (3.7) gradient algebra.

## 6. Figure register (17 figures — renumbered; all B/W-safe)

| Fig | Content | Source |
|-----|---------|--------|
| 3.1 | Encoder contract as information flow | Original |
| 3.2 | The chapter's three tasks (cartpole balance, finger spin, cheetah run), 64×64 rendered frames | From run (01_collect_dmc_data.py; our renders of the DMC simulator) |
| 3.3 | Pretext task zoo on one cartpole frame | Redrawn, adapted from Gui et al. — credit |
| 3.4 | Positive/negative pair pipeline | Original |
| 3.5 | InfoNCE as batch classification | Original |
| 3.6 | Contrastive training curve, 3-seed band | From run (04) |
| 3.7 | Latent space before/after training (marker = binned pole angle, grayscale = cart position) | From run (04) |
| 3.8 | (Reserved slot — merged into 3.6/3.7 if page-tight) | — |
| 3.9 | Collapse: latent std with/without negatives | From run (06) |
| 3.10 | Siamese family pipelines | Redrawn, adapted from Gui et al. — credit |
| 3.11 | VICReg's three terms as three forces | Original |
| 3.12 | MAE pipeline + real cartpole reconstructions | Redrawn schematic (credit) + run outputs (09) |
| 3.13 | Joint embedding vs. reconstruction | Redrawn, adapted from Gui et al. — credit |
| 3.14 | Probe R² vs. latent dimension, cartpole vs. cheetah, seed bands | From run (10) |
| 3.15 | CPC schematic (pair-designed with ch4's JEPA figure) | Original |
| 3.16 | Tracking through the unobserved interval | From run (11) |
| 3.17 | Ch1 loop with learned encoder installed | Original, styled after fig 1.1 |

Credit-line format per style manual §22.2. Redraw everything in the ch2 figure style. Never encode information in color alone. DMC frame panels need no credit beyond the caption note and the tech-report footnote (Apache-2.0 code; our own renders), but the manuscript should still state the source — that is what the benchmark is for.

Tables: 3.1 SSL family comparison (credit, rebuilt) · 3.2 notation · 3.3 probe results incl. aliasing diagnostic · 3.4 objective comparison probes (two tasks) · 3.5 static vs. temporal tests · 3.6 synthesis table.

## 7. Experiments that produce the empirical figures

| Run | Script | Output artifact |
|-----|--------|-----------------|
| E0 | 01_collect_dmc_data.py | cached .npz per task; fig 3.2 |
| E1 | 04_train_contrastive.py | figs 3.6, 3.7 |
| E2 | 05_aliasing_diagnostic.py | table 3.3 (single-frame vs. stacked-frame probes) |
| E3 | 06_collapse_demo.py | fig 3.9 |
| E4 | 07_three_mechanisms.py / 08_vicreg.py | table 3.1 (verification column); VICReg probe row for table 3.4 |
| E5 | 09_masked_autoencoder.py | fig 3.12 reconstructions, table 3.4 |
| E6 | 10_bottleneck_sweep.py | fig 3.14 |
| E7 | 11_temporal_cpc.py | fig 3.16, table 3.5 |

Methodology (stated once in the chapter): all quantitative results mean ± std over seeds {0, 1, 2}; every figure caption states the run configuration (task, frames, architecture, budget). Feasibility: a 3-conv encoder (16-d latent) on 64×64 DMC frames trains NT-Xent in CPU-minutes at small data scale; OSMesa rendering of ~35k frames per task is a one-time cost of minutes-to-tens-of-minutes on a laptop CPU and is cached thereafter; three seeds add minutes, not hours. **Every expected outcome in this design is a hypothesis to be measured at code-lock time; nothing above asserts a result.** If a measurement contradicts a hypothesis at drafting time, the text reports the measurement — the book's epistemology, stated in 3.1.

## 8. Manning compliance checklist (chapter-specific)

- [ ] "This chapter covers": 3–5 bullets, ≤45 chars/line, ≤8 total lines, no colon after "covers"
- [ ] First heading within two pages; no figures/tables in the unheaded intro
- [ ] Sentence-case headings; colon (not em dash) in two-part headings; no abbreviations defined in headings; no character styles in headings
- [ ] No stacked headings; ≥1 sentence of body after every heading; never a lone level-2/3 heading under its parent
- [ ] Listings: ≤76 chars/line, ≤55 with annotations; cueball every discussed line; captions ≤65 chars, no period, not a full sentence; lead-in colon rules (§4.3)
- [ ] Figures: numbered callouts "(figure 3.16)", lowercase refs, caption after the referring paragraph; no color references; credit lines where marked
- [ ] Equations: numbered, "equation 3.6"; latexmath stem; variables italicized in text
- [ ] Callouts only from {NOTE, TIP, WARNING, DEFINITION}; sidebars for anything longer
- [ ] Forbidden words scan: impact, issue, leverage, simplistic, remediate, verbosity; "variance" only in the statistical sense (3.4.3 qualifies)
- [ ] American English; series commas; curly quotes in text, straight in code; active voice; no one-sentence paragraphs
- [ ] Numbers: spell out under 10 (except units, code values, percentages)
- [ ] Exercises before Summary; Summary last, bullets only
- [ ] Cross-refs lowercase; level-1/2 refs by number only
- [ ] URLs as footnotes; no Wikipedia; **DMC tech report (Tunyasuvunakool et al. 2020) footnoted at first use**

## 9. Open decisions for the author

1. **Cut priority under page pressure:** §3.7 first, then 3.4.2 compresses into a table; never 3.3 or 3.6.2. (Unchanged.)
2. **Probe epistemology with named limits** — linear probes for explicitness, forward-prediction test for future-sufficiency, planning utility deferred. k-NN and fine-tuning stay out; one sidebar says why. (Unchanged.)
3. **Supervised baseline in table 3.4:** physics-state labels are free and honest. Keep. (Unchanged, new label source.)
4. **Naming:** "the DeepMind Control Suite" at first use, "the suite" or "DMC" thereafter; `dmc_data.py` in code. The suite's own task names (`cartpole_balance`) appear in code font.
5. **ch4 co-design:** figs 3.15 (CPC) and ch4's JEPA diagram drawn as a before/after pair by the same hand — flag to the illustrator now. (Unchanged.)
6. **False-negative policy in 3.3.1:** keep the simple in-batch rule and the honest NOTE, or filter same-episode pairs? Default: keep + NOTE (simpler code, teaches a real wrinkle). Revisit if the measurement looks pathological at code-lock.
7. **Cache distribution fallback:** decide at code-lock whether the repo offers an optional downloadable cache for readers who cannot render (risk R1). Default: no — the troubleshooting sidebar is expected to suffice, and generating the data is part of the lesson.

## 10. Durability guardrails

- **Mechanisms in prose, models in tables.** The prose spine teaches: objectives as payment schedules; invariance vs. prediction; collapse as the universal joint-embedding failure; bottlenecks as budgets; probes and their limits. Named systems (SimCLR, BYOL, SimSiam, Barlow Twins, VICReg, MAE, CPC) are *instances*.
- **Every empirical claim is config-bound.** "Under this architecture, budget, task, and view policy" is the standing qualifier. Nothing asserts a universal ranking of methods.
- **Mathematical claims stated with their conventions** (log(K+1), ε floor, EMA direction).
- **Vocabulary comes from ch2, not the literature cycle.** Sufficiency, minimality, aliasing, collapse, invariance, prediction.
- **Environments are cited, not worshipped.** DMC is introduced as "a standard benchmark the field measures itself against," with the roadmap sentence (§2.6) making clear the book chose it for continuity and honesty — so if the field's favorite suite changes by 2028, the chapter's logic stands.

---

## 11. Remaining risks (scientific and implementation)

**R1 — Reader-side rendering friction (implementation, medium).** `dm_control` needs an OpenGL backend; headless Linux needs OSMesa system packages; macOS has Homebrew caveats; the package cannot be pip-installed editable. Mitigations: pinned version; script 01 detects and instructs; TIP/troubleshooting sidebar; cached data makes this a one-time cost. Residual risk: a minority of readers on locked-down machines cannot run script 01. Contingency (open decision 7): ship a small downloadable cache.

**R2 — CPC on random-policy data may underwhelm (scientific, medium).** With a uniform random policy and no action input, the one-step future is genuinely stochastic; identification accuracy and gap-tracking may be visibly bounded. This is partly by design (it motivates ch4's action conditioning with a measurement). But if the effect is *too* weak — context model barely beats baselines — 3.7's punchline lands soft. Mitigation at code-lock: consider a slightly smoothed random policy (correlated actions) for the CPC data only, stated in the config; worst case, the section reports the weak result honestly and leans harder on the forward-prediction test.

**R3 — MAE may look too good on simple renders (scientific, low-medium).** DMC frames are visually simple; masked reconstruction may produce decent state probes, narrowing the contrastive-vs-MAE gap the narrative expects. Hypothesis-first framing absorbs this: if MAE probes are strong, table 3.4 says so and the prose explains *why simple renders flatter reconstruction* (few state-free pixels, high spatial redundancy) — which is itself the thesis sentence at work. The claim survives either outcome; only the emphasis changes.

**R4 — Probe confounds on correlated state variables (scientific, low).** Positions and velocities are not independent under the physics (e.g., angular velocity correlates with angle near the swing limits), so a "velocity from one frame" probe may read slightly above chance. Handle in the text: the aliasing diagnostic's claim is comparative (single-frame ≪ stacked-frame), not absolute; the WARNING on probe limits already in 3.3.5 covers the interpretation.

**R5 — Chapter 2 cross-reference debt (editorial, certain but small).** Ch2's figure 2.3 (forklift occlusion) and any warehouse running-example prose are outside this document's scope but now dangle: ch3 no longer re-enacts them. Needed follow-up (not done here): a light ch2 edit either genericizing the example or adding one bridging sentence ("chapter 3 tests this capacity in the form the benchmark supports — unobserved intervals"). Ch1 is unaffected (its loop and listing are environment-free).

**R6 — Version drift in the benchmark (implementation, low).** `dm_control`/MuJoCo updates can alter renders and physics subtly, breaking byte-level cache reproducibility. Mitigation: pinned requirement, recorded version in script 01's output metadata, seeds on both the policy and the renderer. Numerical results in the book are seed-banded anyway; exact-pixel reproducibility is not claimed.

**R7 — CPU budget at code-lock (implementation, low).** The sweep (6 dimensions × 2 tasks × 3 seeds) is the largest run. If it exceeds the "afternoon" promise, the fix is shrinking frames-per-task or moving cheetah to d ∈ {8, 32, 128} — not cutting the two-task design, which carries the section's falsifiable claim.

**R8 — Scope creep from the benchmark roadmap (process, low).** Naming Memory Maze / Atari / Crafter / DROID in ch3 invites the reader (and the author) to expect them soon. The roadmap paragraph is deliberately one paragraph; later-chapter designs must honor it or the sentence gets cut at those chapters' design time.
