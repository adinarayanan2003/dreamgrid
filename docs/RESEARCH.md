# DreamGrid Research Report

## Abstract

DreamGrid is a compact model-based planning system for studying learned visual dynamics. It generates episodes in a seeded 2D rescue world, trains a latent CNN dynamics model from transition tuples, and exposes planners plus diagnostics through a dashboard.

## Problem Setting

The task is a visual navigation problem. An agent starts near the upper-left of a grid, tries to reach a rescue target near the lower-right, and must avoid moving hazards and walls. Each step produces an RGB observation, a scalar reward, and a terminal flag.

This setup is small enough to inspect, but rich enough to expose core model-based planning problems:

- learning transition dynamics from pixels
- predicting terminal events and rewards
- measuring compounding rollout error
- comparing learned prediction against classical planning
- separating simulator planning from learned-model planning

## Method

### Environment

`GridRescueEnv` creates deterministic seeded maps with:

- static walls
- moving hazards
- an agent
- a goal
- five discrete actions: up, down, left, right, stay
- terminal events: goal, collision, timeout

The environment emits both a symbolic state for debugging and an RGB frame for visual model training.

### Dataset

The dataset generator records transition tuples:

```text
obs_t, action_t, obs_t+1, reward_t, done_t, episode_id, step, seed
```

The first full local dataset contains:

| Metric | Value |
| --- | ---: |
| Episodes | 1,000 |
| Transitions | 38,881 |
| Observation shape | 64 x 64 x 3 |
| Reward range | -1.0 to 1.0 |
| Terminal transitions | 1,000 |

### World Model

The model predicts the next observation, reward, and terminal state:

```text
image -> CNN encoder -> latent state
latent state + action -> dynamics model -> next latent state
next latent state -> decoder -> predicted next image
next latent state -> reward head
next latent state -> done head
```

This is deliberately a simple latent dynamics baseline. It is not a Dreamer/RSSM implementation yet.

### Planners

Implemented planners:

- **Random:** lower-bound baseline.
- **A\*:** symbolic shortest-path baseline using true grid state.
- **Random shooting MPC:** samples action sequences, simulates them, and executes the first action from the best sequence.
- **CEM planner:** iteratively biases action-sequence sampling toward high-scoring candidates.

Classical planners use simulator rollouts. `learned_mpc` uses recursive world-model predictions plus visual/reward/done scoring, making it the main test of whether learned dynamics can support action selection.

## Results

The default checkpoint is now the best-validation checkpoint from a 30-epoch mixed-scenario run. This run was selected because it improves learned MPC behavior on held-out validation/test seeds, not because it has the best decoded-frame fidelity.

| Metric | Value |
| --- | ---: |
| Dataset episodes | 3,000 |
| Dataset transitions | 104,958 |
| Loss weights | frame 1.0, reward 0.5, done 0.2 |
| Best validation loss | 0.04116 at epoch 12 |
| Best frame MSE | 0.01647 |
| Best reward MAE | 0.02715 |
| Best done accuracy | 0.9815 |

On the 25-episode validation/test held-out benchmark across nominal, moving-hazard, and dense-wall scenarios, the promoted checkpoint improved learned MPC planner outcomes:

| Metric | Previous default | Promoted checkpoint |
| --- | ---: | ---: |
| Average learned MPC success rate | 0.393 | 0.513 |
| Average learned MPC collision rate | 0.220 | 0.133 |
| Average learned MPC reward | -0.837 | -0.602 |
| Average horizon-5 frame MSE | 0.0744 | 0.0807 |
| Average horizon-5 reward MAE | 0.0680 | 0.2082 |
| Average horizon-5 done accuracy | 0.972 | 0.792 |

The promoted checkpoint is therefore planner-optimized: it reduces collisions and improves success, while regressing visual rollout and scalar-head fidelity at longer horizons.

| Checkpoint | Epochs | Frame MSE |
| --- | ---: | ---: |
| `world_model_v2.pt` | 2 | 0.05650 |
| previous `world_model_v3_50ep.pt` | 50 | 0.00393 |
| promoted `world_model_v3_50ep.pt` | 30 | 0.01647 |

## Qualitative Evidence

Prediction contact sheets are exported during training. Each row shows:

```text
current frame | target next frame | predicted next frame | error heatmap
```

These artifacts are ignored by git because they are generated experiment outputs.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the current experiment journal, selected contact sheet, replay artifact, and full held-out benchmark summary.

The dashboard screenshots below show the interactive planning interface.

![DreamGrid desktop dashboard](assets/dreamgrid-dashboard-desktop.png)

![DreamGrid mobile dashboard](assets/dreamgrid-dashboard-mobile.png)

## Limitations

- Learned rollout inspection and aggregate oracle-vs-learned metrics are API-backed.
- Held-out validation/test seed splits are API-backed, and the CLI can export JSON/CSV tables plus replayable planner failure-case traces and GIFs.
- The training script saves both final and best-validation checkpoints; default checkpoint promotion remains a manual review step.
- The promoted checkpoint was trained locally with stronger reward/done loss weights to improve learned MPC behavior.
- The current model is a one-step latent CNN dynamics model, not a recurrent state-space model.
- Classical planners still exploit simulator truth; learned MPC is available as an explicit separate planner.

## Next Milestones

1. Add multi-objective checkpoint selection so planner score and rollout fidelity can be tracked separately.
2. Train a checkpoint that preserves the promoted planner gains while recovering visual rollout fidelity.
3. Add periodic checkpoint writes to make long CPU/MPS training runs easier to resume.
