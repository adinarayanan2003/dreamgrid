# DreamGrid

DreamGrid is a visual model-based planning project. It provides a 2D rescue environment, planner baselines, transition dataset generation, latent dynamics training, and a dashboard for inspecting episodes and planner behavior.

The codebase is structured as a reproducible research/engineering system: simulator, planners, dataset pipeline, model training, API, UI, tests, and documentation.

![DreamGrid dashboard](docs/assets/dreamgrid-dashboard-desktop.png)

The repository contains:

- `backend/`: GridRescue environment, planners, dataset generation, model code, FastAPI service, and tests.
- `frontend/`: React dashboard for live episodes, imagined rollouts, and planner comparisons.
- `docs/`: research report, architecture notes, roadmap, and screenshots.

## Objective

Evaluate whether a compact visual dynamics model can learn useful transition predictions from a grid-rescue simulator and support rollout-based planning workflows.

The project is built around a deliberately small but inspectable setup:

- **Environment:** a seeded 2D rescue grid with walls, moving hazards, a goal, and sparse terminal rewards.
- **Dataset:** transition tuples of `(observation, action, next observation, reward, done)`.
- **Model:** a latent CNN dynamics model that predicts the next frame, reward, and terminal state.
- **Baselines:** random policy, A*, random-shooting MPC, and CEM-style planning.
- **Interface:** a dashboard for live state, planner controls, candidate rollouts, and evaluation metrics.

## Implementation Status

Implemented:

- Deterministic `GridRescueEnv` with RGB rendering and symbolic state.
- Random, A*, random-shooting, and CEM planners.
- Dataset generation and planner evaluation CLIs.
- FastAPI backend for episodes, planner calls, rollouts, and evaluation.
- React/Vite dashboard for the live grid and planner metrics.
- PyTorch latent world-model architecture and training script.
- Validation metrics and prediction contact-sheet export.

Limitations:

- Classical planners still use the true simulator for rollout scoring.
- Learned rollouts and learned MPC require PyTorch plus a local checkpoint.
- Held-out learned rollout and planner metrics are available through the API and CLI.

## Results Snapshot

Generated training artifacts are ignored by git except for the published Git LFS checkpoint. The current default checkpoint is planner-optimized: it improves learned MPC success and collision rates, while trading off visual rollout fidelity.

| Artifact | Value |
| --- | --- |
| Dataset | `dataset_mixed_3000.npz` |
| Episodes | `3,000` |
| Transitions | `104,958` |
| Best trained checkpoint | `world_model_v3_50ep.pt` |
| Source run | `world_model_candidate_weighted_30ep_best.pt` |
| Best validation loss epoch | `12` |
| Best validation loss | `0.04116` |
| Best frame MSE | `0.01647` |
| Best reward MAE | `0.02715` |
| Best done accuracy | `0.9815` |

On the 25-episode validation/test held-out benchmark across nominal, moving-hazard, and dense-wall scenarios, the promoted checkpoint improved learned MPC average success from `0.393` to `0.513` and reduced average collision rate from `0.220` to `0.133`. Horizon-5 frame MSE regressed from `0.0744` to `0.0807`, so it should be treated as a planner-optimized checkpoint rather than a visually superior rollout model.

The trained checkpoint is tracked with Git LFS at `experiments/world_model_v3_50ep.pt`. See the [experiment journal](docs/EXPERIMENTS.md) for the full benchmark table, contact sheet, replay artifact, and reproduction commands.

See [docs/RESEARCH.md](docs/RESEARCH.md) for methodology, results, and limitations.

## Model Inspection

The dashboard includes a multi-step drift view for comparing simulator frames, model predictions, and per-step error across imagined action sequences.

![DreamGrid multi-step drift](docs/assets/dreamgrid-multistep-drift.png)

## Quick Start

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn dreamgrid.api:app --reload --port 8000
```

Add the `train` extra when you want to train the PyTorch latent world model.

Learned rollout endpoints and `learned_mpc` use `experiments/world_model_v3_50ep.pt` by default. Override it with:

```bash
export DREAMGRID_MODEL_PATH=/path/to/world_model.pt
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL and use the dashboard to generate an episode, step planners, and inspect imagined futures.

## Core Commands

Generate a transition dataset:

```bash
cd backend
python -m dreamgrid.dataset \
  --episodes 3000 \
  --scenarios nominal moving_hazards \
  --out ../experiments/dataset_mixed_3000.npz
```

Train a latent world model:

```bash
cd backend
python -m dreamgrid.train \
  --dataset ../experiments/dataset_mixed_3000.npz \
  --epochs 30 \
  --reward-loss-weight 0.5 \
  --done-loss-weight 0.2 \
  --out ../experiments/world_model_candidate_weighted_30ep.pt \
  --best-out ../experiments/world_model_candidate_weighted_30ep_best.pt \
  --metrics-out ../experiments/world_model_candidate_weighted_30ep_metrics.json \
  --sample-dir ../experiments/world_model_candidate_weighted_30ep_samples
```

Training writes both final and best-validation checkpoints. Candidate checkpoints stay opt-in until their held-out rollout and planner metrics beat the published default.

Evaluate planners:

```bash
cd backend
python -m dreamgrid.evaluate --episodes 25 --grid-size 16
```

Evaluate learned rollout drift:

```bash
cd backend
python -m dreamgrid.evaluate --learned-rollouts --episodes 20 --rollout-horizons 1 3 5 10
```

Evaluate held-out splits and stress scenarios:

```bash
cd backend
python -m dreamgrid.evaluate --heldout \
  --planners cem learned_mpc \
  --episodes-per-split 25 \
  --splits validation test \
  --scenarios nominal moving_hazards dense_walls \
  --rollout-horizons 1 3 5 10 \
  --model-path ../experiments/world_model_candidate_weighted_30ep_best.pt \
  --out-json ../experiments/heldout_candidate_weighted_30ep_full.json \
  --out-csv ../experiments/heldout_candidate_weighted_30ep_full.csv
```

Replay one held-out failure case:

```bash
cd backend
python -m dreamgrid.heldout --replay \
  --split validation \
  --scenario moving_hazards \
  --seed 10000 \
  --planner astar \
  --out-gif ../experiments/replays/astar-moving-hazards-10000.gif
```

Run API/backend tests:

```bash
cd backend
pytest
```

Build frontend:

```bash
cd frontend
npm run build
```

## Project Docs

- [Research report](docs/RESEARCH.md)
- [Experiment journal](docs/EXPERIMENTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Experiment artifact index](experiments/README.md)
- [Contributing](CONTRIBUTING.md)

Experiment datasets and sample images are ignored by git. The published checkpoint is stored with Git LFS; run `git lfs pull` if your clone does not download it automatically.
