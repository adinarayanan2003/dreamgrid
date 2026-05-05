# DreamGrid TODO

## Completed

- [x] Generate initial real training dataset: `experiments/dataset_1000.npz` with 1,000 episodes and 38,881 transitions.
- [x] Verify PyTorch training dependency in Python 3.13 `.venv-train` with `torch 2.11.0`.
- [x] Train first world-model checkpoint: `experiments/world_model_v1.pt` for 2 CPU epochs on `dataset_1000.npz`; loss decreased from `0.07666` to `0.06477`.
- [x] Add validation metrics and sample prediction export to `dreamgrid.train`.
- [x] Train validation-aware checkpoint: `experiments/world_model_v2.pt` with final `val_loss=0.06416`, `frame_mse=0.05650`, `reward_mae=0.05136`, `done_acc=0.987`.
- [x] Export prediction contact sheets to `experiments/world_model_v2_samples/`.
- [x] Train longer checkpoint: `experiments/world_model_v3_50ep.pt` for 50 CPU epochs; final `frame_mse=0.00393`, `reward_mae=0.01667`, `done_acc=0.986`.
- [x] Capture dashboard screenshots for desktop and mobile documentation.
- [x] Add research-style documentation: research report, architecture, contributing guide, and README links.

## Immediate Engineering Tasks

- [ ] Verify GPU/MPS training acceleration on the target machine; current local PyTorch reports `mps False`.
- [ ] Run a real world-model training job on a larger dataset, starting with 250k transitions.
- [ ] Add model checkpoint loading to the API so dashboard planning can use learned dynamics, not only simulator rollouts.
- [ ] Add saved experiment metadata for dataset version, model config, metrics, and checkpoint path.
- [ ] Add one-command project startup script for backend + frontend.
- [ ] Add frontend error states for backend offline, slow planner calls, and invalid episode sessions.
- [ ] Add screenshots/GIF exports for portfolio demos.

## World Model Tasks

- [ ] Save a separate best-validation checkpoint during training.
- [ ] Add multi-step rollout evaluation for horizons 1, 3, 5, and 10.
- [ ] Export side-by-side real vs predicted rollout images.
- [ ] Upgrade dynamics from MLP to GRU if multi-step drift is high.
- [ ] Add uncertainty estimates through a small ensemble of world models.

## Planning Tasks

- [ ] Implement learned-model rollout scoring for MPC.
- [ ] Compare learned MPC against oracle simulator MPC.
- [ ] Add planner latency metrics.
- [ ] Add safer scoring for moving hazards and predicted collisions.
- [ ] Add configurable CEM settings in the dashboard.
- [ ] Add failure-case capture when planner chooses a bad action.

## Environment Tasks

- [ ] Add train/validation/test map split files.
- [ ] Add more rescue scenarios: blocked corridors, moving hazard patrols, narrow exits, and decoy paths.
- [ ] Add partial observability mode with limited local vision.
- [ ] Add visual theme randomization for domain-randomized training.
- [ ] Add map difficulty labels.

## Dashboard Tasks

- [ ] Show predicted frames from the learned model once checkpoint inference is wired in.
- [ ] Add real-vs-imagined comparison view.
- [ ] Add prediction error heatmap.
- [ ] Add experiment browser backed by saved metadata.
- [ ] Add benchmark charts for success rate, collision rate, reward, steps, and latency.
- [ ] Add mobile layout QA and browser screenshot checks.

## Documentation Tasks

- [x] Write a technical report explaining model-based RL, learned simulators, and failure modes.
- [x] Add architecture diagram.
- [ ] Add cloud GPU training instructions.
- [x] Add portfolio README section with demo story, metrics, and screenshots.
- [x] Document what is intentionally non-LLM about the project.
