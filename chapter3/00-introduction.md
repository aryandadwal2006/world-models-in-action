# Representation learning before generation

## This chapter covers

- Learning representations without labels
- Contrastive learning, derived and trained
- Collapse and its remedies
- Masked prediction and latent bottlenecks
- Measuring and assembling a state representation

Chapter 1 introduced the three components of a world model: an encoder that compresses observations into a state, a dynamics model that predicts how the state changes, and a planner that evaluates possible futures. Chapter 2 examined what makes a useful state. It must retain the information needed to predict what happens next, avoid carrying unnecessary detail, and distinguish situations that lead to different futures. But how do we train an encoder to meet these requirements when its input is a stream of pixels and nobody has labeled which details matter?

This chapter develops one answer: self-supervised learning, in which a model learns from training signals derived from the data itself rather than from human-provided labels. An image contains more structure than its pixel values alone suggest. Different views may depict the same underlying scene. A missing region may be predictable from its surroundings. An earlier observation may contain clues about what comes next. Each relationship provides a possible learning signal, and each encourages the encoder to retain a different kind of information.

That difference is the central idea of this chapter: a learning objective specifies what information a representation is encouraged to retain. A contrastive objective rewards features that remain consistent across selected views. Masked prediction rewards features that help reconstruct missing parts of an observation. Temporal prediction rewards features that help anticipate later observations. None of these objectives automatically guarantees a good state for a world model. The important question is not simply whether the training loss decreases, but whether the resulting representation preserves the information that prediction and planning require.

We will build and test these ideas using rendered observations from the DeepMind Control Suite (DMC)[^dmc]. The experiments use tasks such as balancing a cart-pole and spinning a finger-controlled body. For the self-supervised experiments, the encoder receives images without the simulator's physics-state labels. We then use the simulator's state variables to evaluate what the learned representations retain. A separate supervised reference provides a comparison later in the chapter. These tests let us examine the representations directly instead of assuming that a successful training objective must have learned the right state.

> **TIP: Hardware, GPU selection, and rendering**
> Training runners use `worldmodels.train.get_device()`. It selects CUDA automatically when `torch.cuda.is_available()` is true and otherwise selects the CPU. No extra command-line switch is required. The selection applies to PyTorch training, not to MuJoCo/OpenGL rendering. Cached observations let readers skip repeated rendering after the initial collection. The check below confirms the selected device; if it prints `cpu`, use a CUDA-enabled PyTorch build compatible with the installed driver rather than assuming the code has selected a GPU. The runners do not select Apple MPS.
>
```python
import torch
from worldmodels.train import get_device
device = get_device()
print(f"Selected device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(device))
```
>
> For the supported Python version, environment creation, experiment commands, output-path cautions, and headless rendering help, follow [`ch03/README.md`](../ch03/README.md). On headless Linux, the DMC renderer may require `MUJOCO_GL=osmesa` and OSMesa system libraries. On Debian/Ubuntu, the design calls for `sudo apt-get install libgl1-mesa-glx libosmesa6`; check the repository README for platform-specific notes.

Unless stated otherwise, each number is the mean ± standard deviation over the actual trained seeds recorded for that experiment. The collapse demonstration and Table 3.4 use three seeds; the expanded SimSiam/BYOL/Barlow Twins study and primary CPC retrieval use five. Standard deviations are across trained model seeds, not independent episodes or trajectories.

We begin with simple pretext tasks, which reveal both the appeal and the limitations of deriving supervision from data. We then develop contrastive learning step by step, from constructing pairs of views to deriving the loss function. Next, we examine representation collapse and the different mechanisms used to avoid it. Masked prediction offers a different route: rather than comparing views, the model reconstructs parts of its input that were hidden from it. We then investigate how the size of the latent representation affects what can be recovered from it, before moving from individual observations to temporal prediction.

The chapter ends by assembling a reusable visual encoder. The static encoder maps a frame, or a fixed stack of frames, to a latent representation; it does not carry state from one call to the next. An optional recurrent wrapper can carry a hidden state across observations. Neither component alone is a complete world model: the latter also needs a learned transition model that predicts how the state changes under actions, along with components for evaluating or planning over possible futures. Chapter 4 develops predictive representations and action-conditioning further. For now, we study how learning signals shape representations and test what they retain.

> **NOTE: Three components, three different jobs**
>
> A **static visual encoder** maps the current image, or a fixed stack of images, to a latent vector $z_t$. It has no memory across calls. A **recurrent wrapper** is optional: it takes the current visual latent and a previous hidden state $h_{t-1}$, then returns an updated hidden state $h_t$. That hidden state is a learned model state, not the simulator's physical state. A **full world model** additionally needs a transition model that predicts future state or observations conditioned on actions, and components that evaluate or plan over possible futures. The GRU wrapper and CPC experiment in this chapter do not, by themselves, implement that full system.

## Running the chapter code

The repository requires Python 3.11 or newer. From the repository root, create an environment, install the pinned dependencies, run the regression tests, and start with the single-seed contrastive experiment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python ch03/04_train_contrastive.py --seed 0 --checkpoints-dir checkpoints/quickstart --figures-dir ch03/figures/quickstart --results-dir ch03/results/quickstart
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The full multi-seed comparison, latent-dimension sweep, and CPC sequence experiment can take hours on CPU, depending on hardware and configuration. This chapter does not claim a universal runtime or measured GPU speed-up. The device selection applies to PyTorch training only; rendering the Control Suite is a separate dependency, and cached observations mean rendering is normally a one-time data-collection step.

The book's listings are short excerpts from the canonical `worldmodels/` implementations; the `ch03/` runners add the full data protocol, seed management, evaluation, plotting, and saved results. Start with the [Chapter 3 code map](../ch03/README.md), which documents experiment-specific settings and commands. Full multi-seed comparisons, the latent-dimension sweep, and CPC can take hours on CPU, depending on hardware and configuration. The runners select CUDA automatically when PyTorch detects it; the TIP above shows how to verify the selected device. Data collection is a separate rendering step, and downstream experiments read cached `.npz` files. When changing or reproducing an experiment, use isolated output paths so that canonical result JSONs, checkpoints, and figures are not overwritten.

[^dmc]: Tunyasuvunakool, S. et al. (2020). “dm_control: Software and tasks for continuous control.” *Software Impacts*, 6, 100022. [https://doi.org/10.1016/j.simpa.2020.100022](https://doi.org/10.1016/j.simpa.2020.100022).