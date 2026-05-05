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

- The dashboard/planners still use the true simulator for rollouts.
- The trained checkpoint is not yet wired into API-driven learned rollouts or learned MPC.

## Results Snapshot

Generated training artifacts are ignored by git. The latest recorded run produced:

| Artifact | Value |
| --- | --- |
| Dataset | `dataset_1000.npz` |
| Episodes | `1,000` |
| Transitions | `38,881` |
| Best trained checkpoint | `world_model_v3_50ep.pt` |
| Final frame MSE | `0.00393` |
| Final reward MAE | `0.01667` |
| Final done accuracy | `0.986` |
| Best validation loss epoch | `17` |

See [docs/RESEARCH.md](docs/RESEARCH.md) for methodology, results, and limitations.

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
python -m dreamgrid.dataset --episodes 200 --out ../experiments/dataset_200.npz
```

Train a latent world model:

```bash
cd backend
python -m dreamgrid.train \
  --dataset ../experiments/dataset_200.npz \
  --epochs 10 \
  --out ../experiments/world_model.pt \
  --sample-dir ../experiments/world_model_samples
```

Evaluate planners:

```bash
cd backend
python -m dreamgrid.evaluate --episodes 25 --grid-size 16
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
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [TODO](TODO.md)

Experiment datasets, checkpoints, and sample images are ignored by git. Regenerate them with the commands above, or publish them separately as release artifacts if needed.
