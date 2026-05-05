# DreamGrid Architecture

## System Overview

```text
React Dashboard
  -> FastAPI service
    -> GridRescueEnv
    -> planners
    -> evaluation runner
    -> optional world-model checkpoint inference

Dataset CLI
  -> GridRescueEnv rollouts
  -> transition .npz files

Training CLI
  -> transition .npz files
  -> latent CNN world model
  -> checkpoint + metrics + sample sheets
```

## Runtime Components

### Environment

`GridRescueEnv` owns the simulation:

- seeded map generation
- wall and hazard placement
- action transitions
- rewards and terminal events
- RGB rendering
- symbolic state export/import

The symbolic state is used by planners and the dashboard. The RGB frame is used by the world-model training pipeline.

### Planners

Planner implementations share a `plan(env)` interface and return:

- selected action
- action name
- score
- candidate action sequences
- candidate paths/events

The current CEM and random-shooting planners use cloned simulator state for rollout scoring. This keeps the dashboard useful while learned-model planning is still under development.

### World Model

The world model is implemented in PyTorch:

```text
encoder: RGB frame -> latent vector
action embedding: discrete action -> action vector
dynamics: latent + action -> next latent
decoder: next latent -> predicted RGB frame
reward head: next latent -> scalar reward
done head: next latent -> terminal logit
```

The checkpoint stores:

- model config
- state dict
- dataset path
- training history
- final metrics

### API

FastAPI endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | health check |
| `POST /api/episodes/generate` | create a seeded episode |
| `GET /api/episodes/{episode_id}` | inspect episode state |
| `POST /api/episodes/{episode_id}/step` | apply an action or planner step |
| `POST /api/planners/plan` | score planner candidates |
| `POST /api/models/rollout` | rollout actions through the simulator clone |
| `POST /api/models/predict-next` | compare one learned next-frame prediction with simulator truth |
| `POST /api/models/predict-rollout` | compare multi-step learned rollout drift with simulator truth |
| `POST /api/eval/run` | compare planners across episodes |
| `POST /api/eval/learned-rollouts` | aggregate oracle-vs-learned rollout metrics by horizon |
| `POST /api/eval/heldout` | evaluate planners and optional learned drift on held-out seed splits |
| `POST /api/eval/heldout/replay` | replay one held-out split/scenario/seed/planner trace |

Learned-model endpoints and `learned_mpc` resolve checkpoints in this order:

1. request-level `model_path`, where supported
2. `DREAMGRID_MODEL_PATH`
3. `experiments/world_model_v3_50ep.pt`

Missing checkpoints return an unavailable response instead of falling back to simulator rollouts.

Held-out evaluation uses fixed seed ranges for `validation` and `test` splits, plus scenario configs for nominal maps, denser walls, and higher moving-hazard pressure. This keeps training-time smoke metrics separate from generalization checks. The CLI can export the full JSON payload, a flat CSV summary, and capped failure-case traces for collided or timed-out planner episodes. A replay endpoint reconstructs any split/scenario/seed/planner case as a step-by-step symbolic trace.

### Dashboard

The dashboard contains:

- planner controls
- live grid visualization
- candidate rollout list
- planner evaluation metrics
- learned rollout drift metrics
- responsive desktop/mobile layout

Current dashboard screenshots are stored under `docs/assets/`.

## Data Flow

### Interactive Planning

```text
user selects planner
  -> dashboard calls /api/planners/plan
  -> backend clones current env
  -> planner scores candidate futures
  -> dashboard renders selected candidate summaries
```

### Training

```text
dataset CLI generates transitions
  -> training CLI loads .npz arrays
  -> train/validation split is created
  -> model predicts next frame, reward, and done
  -> metrics and sample sheets are exported
  -> checkpoint is saved
```

## Design Constraints

- Keep the environment small enough for CPU smoke tests.
- Keep generated datasets outside git and publish reusable checkpoints through Git LFS or release artifacts.
- Preserve a clear boundary between simulator planning and learned-model planning.
- Use visual diagnostics because world-model quality cannot be judged by scalar loss alone.
