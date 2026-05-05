from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from dreamgrid.env import GridRescueEnv
from dreamgrid.evaluate import evaluate
from dreamgrid.heldout import DEFAULT_HELDOUT_SCENARIOS, DEFAULT_HELDOUT_SPLITS, evaluate_heldout
from dreamgrid.model import (
    ModelCheckpointUnavailableError,
    TorchUnavailableError,
    load_model,
    require_model_path,
    require_torch,
)
from dreamgrid.planners import PlanCandidate, make_planner
from dreamgrid.rollout_metrics import DEFAULT_ROLLOUT_HORIZONS, evaluate_learned_rollouts
from dreamgrid.types import ACTION_NAMES


class EpisodeRequest(BaseModel):
    grid_size: int = Field(default=16, ge=8, le=32)
    seed: int = 7
    max_steps: int | None = Field(default=None, ge=10, le=500)
    hazard_count: int = Field(default=3, ge=0, le=12)
    wall_density: float = Field(default=0.16, ge=0.0, le=0.4)


class StepRequest(BaseModel):
    action: int | None = Field(default=None, ge=0, le=4)
    planner: Literal["random", "astar", "random_shooting", "cem", "learned_mpc"] | None = None


class PlanRequest(BaseModel):
    episode_id: str
    planner: Literal["random", "astar", "random_shooting", "cem", "learned_mpc"] = "cem"
    horizon: int = Field(default=12, ge=1, le=32)
    num_candidates: int = Field(default=256, ge=8, le=2048)


class RolloutRequest(BaseModel):
    episode_id: str
    actions: list[int] = Field(default_factory=list, max_length=64)


class PredictNextRequest(BaseModel):
    episode_id: str
    action: int = Field(default=3, ge=0, le=4)
    model_path: str | None = None


class PredictRolloutRequest(BaseModel):
    episode_id: str
    actions: list[int] = Field(default_factory=list, min_length=1, max_length=16)
    model_path: str | None = None


class EvalRequest(BaseModel):
    planners: list[Literal["random", "astar", "random_shooting", "cem", "learned_mpc"]] = Field(
        default_factory=lambda: ["random", "astar", "random_shooting", "cem", "learned_mpc"]
    )
    episodes: int = Field(default=6, ge=1, le=250)
    grid_size: int = Field(default=16, ge=8, le=32)
    seed: int = 100
    horizon: int = Field(default=6, ge=1, le=32)
    num_candidates: int = Field(default=48, ge=8, le=2048)


class RolloutMetricsRequest(BaseModel):
    episodes: int = Field(default=12, ge=1, le=100)
    grid_size: int = Field(default=16, ge=8, le=32)
    seed: int = 200
    horizons: list[int] = Field(
        default_factory=lambda: list(DEFAULT_ROLLOUT_HORIZONS),
        min_length=1,
        max_length=8,
    )
    model_path: str | None = None


class HeldoutEvalRequest(BaseModel):
    planners: list[Literal["random", "astar", "random_shooting", "cem", "learned_mpc"]] = Field(
        default_factory=lambda: ["astar", "cem", "learned_mpc"]
    )
    episodes_per_split: int = Field(default=3, ge=1, le=25)
    grid_size: int = Field(default=16, ge=8, le=32)
    splits: list[Literal["validation", "test"]] = Field(
        default_factory=lambda: list(DEFAULT_HELDOUT_SPLITS)
    )
    scenarios: list[Literal["nominal", "dense_walls", "moving_hazards"]] = Field(
        default_factory=lambda: list(DEFAULT_HELDOUT_SCENARIOS)
    )
    horizon: int = Field(default=6, ge=1, le=32)
    num_candidates: int = Field(default=48, ge=8, le=2048)
    rollout_horizons: list[int] = Field(
        default_factory=lambda: list(DEFAULT_ROLLOUT_HORIZONS),
        min_length=1,
        max_length=8,
    )
    include_learned_rollouts: bool = True
    model_path: str | None = None


app = FastAPI(title="DreamGrid API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, GridRescueEnv] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/episodes/generate")
def generate_episode(request: EpisodeRequest) -> dict:
    env = GridRescueEnv(
        grid_size=request.grid_size,
        max_steps=request.max_steps,
        hazard_count=request.hazard_count,
        wall_density=request.wall_density,
    )
    env.reset(seed=request.seed)
    episode_id = str(uuid4())
    SESSIONS[episode_id] = env
    return _episode_payload(episode_id, env, event="reset", reward=0.0, done=False)


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str) -> dict:
    env = _session(episode_id)
    return _episode_payload(episode_id, env, event="state", reward=0.0, done=False)


@app.post("/api/episodes/{episode_id}/step")
def step_episode(episode_id: str, request: StepRequest) -> dict:
    env = _session(episode_id)
    if request.action is None:
        planner_name = request.planner or "cem"
        plan = _make_planner(planner_name, seed=env.seed + env.step_count).plan(env)
        action = plan.selected_action
    else:
        action = request.action
    result = env.step(action)
    return _episode_payload(
        episode_id,
        env,
        event=result.info["event"],
        reward=result.reward,
        done=result.done,
        action=action,
    )


@app.post("/api/planners/plan")
def plan(request: PlanRequest) -> dict:
    env = _session(request.episode_id)
    planner = _make_planner(
        request.planner,
        seed=env.seed + env.step_count,
        horizon=request.horizon,
        num_candidates=request.num_candidates,
    )
    result = planner.plan(env)
    return {
        "selected_action": result.selected_action,
        "selected_action_name": result.selected_action_name,
        "score": result.score,
        "candidates": [_candidate_payload(candidate) for candidate in result.candidates],
    }


@app.post("/api/models/rollout")
def rollout(request: RolloutRequest) -> dict:
    env = _session(request.episode_id).clone()
    frames = [{"state": env.symbolic_state(), "reward": 0.0, "event": "start", "done": False}]
    done = False
    for action in request.actions:
        if action not in ACTION_NAMES:
            raise HTTPException(status_code=400, detail=f"invalid action {action}")
        if done:
            break
        result = env.step(action)
        done = result.done
        frames.append(
            {
                "state": env.symbolic_state(),
                "reward": result.reward,
                "event": result.info["event"],
                "done": done,
                "action": action,
                "action_name": ACTION_NAMES[action],
            }
        )
    return {"frames": frames}


@app.post("/api/models/predict-next")
def predict_next(request: PredictNextRequest) -> dict:
    env = _session(request.episode_id)
    model_path = _model_path_or_404(request.model_path)

    current_obs = env.render()
    actual_env = env.clone()
    actual_result = actual_env.step(request.action)
    actual_next = actual_result.obs

    try:
        torch = require_torch()
        model = _cached_model(str(model_path.resolve()))
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    obs_tensor = _obs_tensor(torch, current_obs)
    action_tensor = torch.tensor([request.action]).long()
    with torch.no_grad():
        pred = model(obs_tensor, action_tensor)
        pred_next = _tensor_to_uint8(pred["next_obs"][0])
        predicted_reward = float(pred["reward"][0].detach().cpu())
        predicted_done_probability = float(torch.sigmoid(pred["done_logit"][0]).detach().cpu())

    error = _error_image(pred_next, actual_next)
    return {
        "model_id": model_path.name,
        "action": request.action,
        "action_name": ACTION_NAMES[request.action],
        "current_image": _image_data_url(current_obs),
        "actual_next_image": _image_data_url(actual_next),
        "predicted_next_image": _image_data_url(pred_next),
        "error_image": _image_data_url(error),
        "actual_reward": actual_result.reward,
        "actual_done": actual_result.done,
        "actual_event": actual_result.info["event"],
        "predicted_reward": predicted_reward,
        "predicted_done_probability": predicted_done_probability,
        "predicted_done": predicted_done_probability >= 0.5,
    }


@app.post("/api/models/predict-rollout")
def predict_rollout(request: PredictRolloutRequest) -> dict:
    env = _session(request.episode_id)
    model_path = _model_path_or_404(request.model_path)
    invalid_actions = [action for action in request.actions if action not in ACTION_NAMES]
    if invalid_actions:
        raise HTTPException(status_code=400, detail=f"invalid actions: {invalid_actions}")

    try:
        torch = require_torch()
        model = _cached_model(str(model_path.resolve()))
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    actual_env = env.clone()
    current_obs = env.render()
    predicted_obs_tensor = _obs_tensor(torch, current_obs)
    frames = []
    cumulative_mse = 0.0
    stopped = False

    with torch.no_grad():
        for step_idx, action in enumerate(request.actions, start=1):
            if stopped:
                break
            actual_result = actual_env.step(action)
            action_tensor = torch.tensor([action]).long()
            pred = model(predicted_obs_tensor, action_tensor)
            predicted_next = _tensor_to_uint8(pred["next_obs"][0])
            error = _error_image(predicted_next, actual_result.obs)
            frame_mse = _frame_mse(predicted_next, actual_result.obs)
            cumulative_mse += frame_mse

            frames.append(
                {
                    "step": step_idx,
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "actual_image": _image_data_url(actual_result.obs),
                    "predicted_image": _image_data_url(predicted_next),
                    "error_image": _image_data_url(error),
                    "frame_mse": frame_mse,
                    "actual_reward": actual_result.reward,
                    "actual_done": actual_result.done,
                    "actual_event": actual_result.info["event"],
                    "predicted_reward": float(pred["reward"][0].detach().cpu()),
                    "predicted_done_probability": float(
                        torch.sigmoid(pred["done_logit"][0]).detach().cpu()
                    ),
                }
            )
            predicted_obs_tensor = _obs_tensor(torch, predicted_next)
            stopped = actual_result.done

    return {
        "model_id": model_path.name,
        "actions": request.actions,
        "action_names": [ACTION_NAMES[action] for action in request.actions],
        "frames": frames,
        "avg_frame_mse": cumulative_mse / max(1, len(frames)),
    }


@app.post("/api/eval/run")
def run_eval(request: EvalRequest) -> dict:
    try:
        metrics = evaluate(
            planners=list(request.planners),
            episodes=request.episodes,
            grid_size=request.grid_size,
            seed=request.seed,
            horizon=request.horizon,
            num_candidates=request.num_candidates,
        )
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelCheckpointUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"metrics": metrics}


@app.post("/api/eval/learned-rollouts")
def run_learned_rollout_eval(request: RolloutMetricsRequest) -> dict:
    try:
        metrics = evaluate_learned_rollouts(
            episodes=request.episodes,
            horizons=request.horizons,
            grid_size=request.grid_size,
            seed=request.seed,
            model_path=request.model_path,
        )
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelCheckpointUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"metrics": metrics}


@app.post("/api/eval/heldout")
def run_heldout_eval(request: HeldoutEvalRequest) -> dict:
    try:
        metrics = evaluate_heldout(
            planners=list(request.planners),
            episodes_per_split=request.episodes_per_split,
            grid_size=request.grid_size,
            splits=list(request.splits),
            scenarios=list(request.scenarios),
            horizon=request.horizon,
            num_candidates=request.num_candidates,
            rollout_horizons=request.rollout_horizons,
            include_learned_rollouts=request.include_learned_rollouts,
            model_path=request.model_path,
        )
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelCheckpointUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"metrics": metrics}


def _session(episode_id: str) -> GridRescueEnv:
    env = SESSIONS.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"episode not found: {episode_id}")
    return env


def _make_planner(name: str, **kwargs):
    try:
        return make_planner(name, **kwargs)
    except TorchUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelCheckpointUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _model_path_or_404(path: str | None = None) -> Path:
    try:
        return require_model_path(path)
    except ModelCheckpointUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _episode_payload(
    episode_id: str,
    env: GridRescueEnv,
    event: str,
    reward: float,
    done: bool,
    action: int | None = None,
) -> dict:
    payload = {
        "episode_id": episode_id,
        "state": env.symbolic_state(),
        "event": event,
        "reward": reward,
        "done": done,
        "actions": ACTION_NAMES,
    }
    if action is not None:
        payload["action"] = action
        payload["action_name"] = ACTION_NAMES[action]
    return payload


def _candidate_payload(candidate: PlanCandidate) -> dict:
    return {
        "actions": candidate.actions,
        "action_names": [ACTION_NAMES[action] for action in candidate.actions],
        "score": candidate.score,
        "path": candidate.path,
        "event": candidate.event,
    }


@lru_cache(maxsize=4)
def _cached_model(path: str):
    return load_model(Path(path))


def _tensor_to_uint8(tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def _obs_tensor(torch_module, image: np.ndarray):
    return torch_module.tensor(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _frame_mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    diff = predicted.astype(np.float32) / 255.0 - actual.astype(np.float32) / 255.0
    return float(np.mean(diff**2))


def _error_image(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    error = np.abs(predicted.astype(np.float32) - actual.astype(np.float32)).mean(axis=2) / 255.0
    scaled = np.clip(error * 8.0, 0.0, 1.0)
    image = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    image[..., 0] = (scaled * 255).astype(np.uint8)
    image[..., 1] = ((1.0 - scaled) * 70).astype(np.uint8)
    image[..., 2] = ((1.0 - scaled) * 120).astype(np.uint8)
    return image


def _image_data_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
