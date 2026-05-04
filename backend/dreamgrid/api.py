from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dreamgrid.env import GridRescueEnv
from dreamgrid.evaluate import evaluate
from dreamgrid.planners import PlanCandidate, make_planner
from dreamgrid.types import ACTION_NAMES


class EpisodeRequest(BaseModel):
    grid_size: int = Field(default=16, ge=8, le=32)
    seed: int = 7
    max_steps: int | None = Field(default=None, ge=10, le=500)
    hazard_count: int = Field(default=3, ge=0, le=12)
    wall_density: float = Field(default=0.16, ge=0.0, le=0.4)


class StepRequest(BaseModel):
    action: int | None = Field(default=None, ge=0, le=4)
    planner: Literal["random", "astar", "random_shooting", "cem"] | None = None


class PlanRequest(BaseModel):
    episode_id: str
    planner: Literal["random", "astar", "random_shooting", "cem"] = "cem"
    horizon: int = Field(default=12, ge=1, le=32)
    num_candidates: int = Field(default=256, ge=8, le=2048)


class RolloutRequest(BaseModel):
    episode_id: str
    actions: list[int] = Field(default_factory=list, max_length=64)


class EvalRequest(BaseModel):
    planners: list[Literal["random", "astar", "random_shooting", "cem"]] = Field(
        default_factory=lambda: ["random", "astar", "random_shooting", "cem"]
    )
    episodes: int = Field(default=6, ge=1, le=250)
    grid_size: int = Field(default=16, ge=8, le=32)
    seed: int = 100
    horizon: int = Field(default=6, ge=1, le=32)
    num_candidates: int = Field(default=48, ge=8, le=2048)


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
        plan = make_planner(planner_name, seed=env.seed + env.step_count).plan(env)
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
    planner = make_planner(
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


@app.post("/api/eval/run")
def run_eval(request: EvalRequest) -> dict:
    return {
        "metrics": evaluate(
            planners=list(request.planners),
            episodes=request.episodes,
            grid_size=request.grid_size,
            seed=request.seed,
            horizon=request.horizon,
            num_candidates=request.num_candidates,
        )
    }


def _session(episode_id: str) -> GridRescueEnv:
    env = SESSIONS.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"episode not found: {episode_id}")
    return env


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
