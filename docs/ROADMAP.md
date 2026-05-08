# DreamGrid Roadmap

## Current State

DreamGrid currently has a complete simulator/planner/dashboard MVP plus a trained latent world-model checkpoint. Learned rollout inspection, aggregate oracle-vs-learned metrics, and held-out split evaluation are available through the API.

Moving hazards are implemented and are part of the default stress-testing story. The remaining gap is stronger time-aware planning and hazard-prediction diagnostics, not basic hazard support.

## Phase 1: Working Lab

- [x] Implement deterministic `GridRescueEnv` with seeded map generation.
- [x] Add random, A*, random-shooting, and CEM-style planners.
- [x] Expose episode and planner APIs.
- [x] Build an interactive dashboard with real state, imagined candidates, and metrics.

## Phase 2: Learned Dynamics

- [x] Generate mixed-policy transition datasets.
- [x] Train a latent CNN dynamics model on image observations.
- [x] Track one-step prediction metrics.
- [x] Save best-validation checkpoint during training.
- [x] Track multi-step prediction error.
- [x] Add model checkpoints to the API planner path.

## Phase 3: Evaluation

- [x] Compare random, A*, random-shooting, and CEM planners.
- [x] Add oracle-vs-learned-model rollout comparison.
- [x] Evaluate on held-out map splits and moving hazard scenarios.
- [x] Export metrics tables and failure cases.
- [x] Replay held-out failure cases.
- [x] Export rollout GIFs.
- [x] Add learned MPC first-action gating from the current rendered board.

## Phase 4: Frontier Extensions

- [ ] Add multi-objective checkpoint selection for planner score and rollout fidelity.
- [ ] Add periodic checkpointing/resume support for long local training runs.
- [ ] Add time-expanded A* or another hazard-aware oracle baseline.
- [ ] Track hazard-motion prediction accuracy separately from aggregate frame MSE.
- [ ] Ensemble world models for uncertainty-aware planning.
- [ ] Partial observability with recurrent latent dynamics.
- [ ] Domain randomization across visual themes.
- [ ] Model-free PPO/DQN baseline for comparison.
