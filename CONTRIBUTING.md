# Contributing

## Setup

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Training environment:

```bash
cd backend
python3.13 -m venv .venv-train
source .venv-train/bin/activate
pip install -e ".[dev,train]"
```

Frontend:

```bash
cd frontend
npm install
```

## Local Development

Start the API:

```bash
cd backend
source .venv/bin/activate
uvicorn dreamgrid.api:app --reload --port 8000
```

Start the dashboard:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

## Verification

Backend tests:

```bash
cd backend
pytest
```

Backend lint:

```bash
cd backend
ruff check .
```

Frontend build:

```bash
cd frontend
npm run build
```

## Experiment Workflow

Generate a small dataset:

```bash
cd backend
python -m dreamgrid.dataset --episodes 200 --grid-size 16 --out ../experiments/dataset_200.npz
```

Train a checkpoint:

```bash
cd backend
python -m dreamgrid.train \
  --dataset ../experiments/dataset_200.npz \
  --epochs 10 \
  --batch-size 128 \
  --out ../experiments/world_model.pt \
  --sample-dir ../experiments/world_model_samples
```

Evaluate planners:

```bash
cd backend
python -m dreamgrid.evaluate --episodes 25 --grid-size 16
```

## Git Hygiene

Generated artifacts are ignored:

- `experiments/*.npz`
- `experiments/*.pt`
- `experiments/**/*.png`
- virtualenvs
- frontend build outputs

Keep source, docs, and reproducible commands in git. Publish large datasets/checkpoints separately if needed.

