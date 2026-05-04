# DreamGrid Roadmap

## Phase 1: Working Lab

- Implement deterministic `GridRescueEnv` with seeded map generation.
- Add random, A*, random-shooting, and CEM-style planners.
- Expose episode and planner APIs.
- Build an interactive dashboard with real state, imagined candidates, and metrics.

## Phase 2: Learned Dynamics

- Generate mixed-policy transition datasets.
- Train a latent CNN dynamics model on image observations.
- Track one-step and multi-step prediction error.
- Add model checkpoints to the API planner path.

## Phase 3: Evaluation

- Compare random, A*, oracle, random-shooting, and CEM planners.
- Evaluate on unseen maps and moving hazards.
- Export rollout GIFs, metrics tables, and failure cases.

## Phase 4: Frontier Extensions

- Ensemble world models for uncertainty-aware planning.
- Partial observability with recurrent latent dynamics.
- Domain randomization across visual themes.
- Model-free PPO/DQN baseline for comparison.

