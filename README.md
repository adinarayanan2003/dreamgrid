# DreamGrid

DreamGrid is a non-LLM world-model lab: a visual 2D rescue environment where an agent learns how the world changes, then plans by rolling out imagined futures.

The repository contains:

- `backend/`: GridRescue environment, planners, dataset generation, model code, FastAPI service, and tests.
- `frontend/`: React dashboard for live episodes, imagined rollouts, and planner comparisons.
- `docs/`: implementation roadmap and experiment notes.

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

Train the first latent world model:

```bash
cd backend
python -m dreamgrid.train --dataset ../experiments/dataset_200.npz --epochs 10 --out ../experiments/world_model.pt
```

Run API/backend tests:

```bash
cd backend
pytest
```

## Current Progress

- Generated the first real dataset locally: `experiments/dataset_1000.npz` with 1,000 episodes and 38,881 transitions.
- Trained `world_model_v3_50ep.pt` for 50 CPU epochs; final frame MSE was `0.00393`.
- Exported prediction contact sheets under `experiments/world_model_v3_50ep_samples/`.

Experiment datasets, checkpoints, and sample images are intentionally ignored by git. Regenerate them with the commands above, or publish them separately as release artifacts if needed.
