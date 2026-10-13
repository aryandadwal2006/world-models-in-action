# Chapter 3: What the experiments support

These are results from the separate audited implementation, not outputs of the short teaching scripts.

## Temporal CPC: missing five frames

The checkpoint-only evaluation used 200 non-overlapping, episode-safe validation windows per model seed: 40 windows from each of five episodes, selected from 4,880 eligible windows. Existing checkpoints were evaluated; no retraining was performed.

| Model seed | CPC gap mean (R^2) |
|---:|---:|
| 0 | -2.853 |
| 1 | -0.128 |
| 2 | -0.443 |
| Mean ± population SD | -1.141 ± 1.217 |

Per-variable across-seed mean ± population SD:

| State variable | Gap (R^2) |
|---|---:|
| Cart position | -3.331 ± 4.267 |
| Pole angle | -0.382 ± 0.111 |
| Cart velocity | -0.401 ± 0.504 |
| Pole angular velocity | -0.451 ± 0.178 |

This is a negative, seed-sensitive result, especially for cart position. The windows are clustered within only five episodes; they are not 200 independent trajectories. The privileged state-history baseline scored (R^2=0.837), but it receives true physics-state history while CPC receives rendered frames. Do not present that comparison as like-for-like.

## Other conclusions

- Temporal context improved current-state velocity decoding over a single-frame representation in this setup, although the learned CPC latent rollout still performed poorly through the missing interval.
- The tested frame-stacking setup did not reliably recover velocity information.
- The latent-dimension sweep did not establish a universal saturation point.
- Contrastive, VICReg, and masked-reconstruction objectives optimize different properties. Loss values alone do not establish that a representation supports planning or control.

These findings are bounded to the tasks, data, seeds, and training budgets that were evaluated. They do not establish a general ranking of representation-learning methods.

For the complete metrics, methodology, caveats, and reproducibility safeguards, use the [audit branch](https://github.com/aryandadwal2006/world-models-in-action/tree/audit/chapter3-publication-readiness).
