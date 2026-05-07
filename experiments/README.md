# DreamGrid Experiments

This directory is the local experiment workspace. Most files here are generated and intentionally ignored by git:

- datasets: `*.npz`
- candidate checkpoints: `*.pt`
- benchmark exports: `*.json`, `*.csv`
- training logs: `*.log`
- contact sheets and replay media under subdirectories

The exception is the published default checkpoint:

```text
experiments/world_model_v3_50ep.pt
```

That file is tracked with Git LFS and is the checkpoint used by learned rollout endpoints and `learned_mpc` by default.

The promoted source checkpoint is also tracked for reproducibility:

```text
experiments/world_model_candidate_weighted_30ep_best.pt
```

## Current Local Artifacts

The May 2026 planner-optimized checkpoint came from:

```text
dataset_mixed_3000.npz
world_model_candidate_weighted_30ep_best.pt
heldout_candidate_weighted_30ep_full.json
heldout_default_50ep_full.json
```

The durable write-up lives in:

```text
docs/EXPERIMENTS.md
```

Keep large generated artifacts out of git unless they are intentionally promoted through Git LFS or copied as small illustrative assets into `docs/assets/`.
