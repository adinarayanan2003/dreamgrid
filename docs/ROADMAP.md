# DreamGrid Roadmap

## Current State

DreamGrid currently has a complete simulator/planner/dashboard MVP plus a trained latent world-model checkpoint. The next milestone is to move from simulator rollouts to learned-model rollouts in the API and dashboard.

## Phase 1: Working Lab

- [x] Implement deterministic `GridRescueEnv` with seeded map generation.
- [x] Add random, A*, random-shooting, and CEM-style planners.
- [x] Expose episode and planner APIs.
- [x] Build an interactive dashboard with real state, imagined candidates, and metrics.

## Phase 2: Learned Dynamics

- [x] Generate mixed-policy transition datasets.
- [x] Train a latent CNN dynamics model on image observations.
- [x] Track one-step prediction metrics.
- [ ] Save best-validation checkpoint during training.
- [ ] Track multi-step prediction error.
- [ ] Add model checkpoints to the API planner path.

## Phase 3: Evaluation

- [x] Compare random, A*, random-shooting, and CEM planners.
- [ ] Add oracle-vs-learned-model rollout comparison.
- [ ] Evaluate on held-out map splits and moving hazard scenarios.
- [ ] Export rollout GIFs, metrics tables, and failure cases.

## Phase 4: Frontier Extensions

- [ ] Ensemble world models for uncertainty-aware planning.
- [ ] Partial observability with recurrent latent dynamics.
- [ ] Domain randomization across visual themes.
- [ ] Model-free PPO/DQN baseline for comparison.
