from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from dreamgrid.env import GridRescueEnv
from dreamgrid.model import load_model, require_model_path, require_torch
from dreamgrid.types import ACTION_NAMES, ACTION_TO_DELTA, Position


@dataclass
class PlanCandidate:
    actions: list[int]
    score: float
    path: list[dict[str, int]]
    event: str


@dataclass
class PlanResult:
    selected_action: int
    selected_action_name: str
    score: float
    candidates: list[PlanCandidate]


class Planner(Protocol):
    def plan(self, env: GridRescueEnv) -> PlanResult:
        ...


class RandomPlanner:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def plan(self, env: GridRescueEnv) -> PlanResult:
        action = int(self.rng.integers(0, len(ACTION_NAMES)))
        return PlanResult(
            selected_action=action,
            selected_action_name=ACTION_NAMES[action],
            score=0.0,
            candidates=[PlanCandidate([action], 0.0, [_pos(env.agent)], "random")],
        )


class AStarPlanner:
    def plan(self, env: GridRescueEnv) -> PlanResult:
        path = _shortest_path(env)
        if len(path) < 2:
            action = 4
        else:
            action = _action_between(path[0], path[1])
        path_dict = [_pos(pos) for pos in path]
        score = 1.0 / max(1, len(path))
        return PlanResult(action, ACTION_NAMES[action], score, [PlanCandidate([action], score, path_dict, "astar")])


class RandomShootingPlanner:
    def __init__(self, horizon: int = 12, num_candidates: int = 256, seed: int = 0) -> None:
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.rng = np.random.default_rng(seed)

    def plan(self, env: GridRescueEnv) -> PlanResult:
        candidates = [
            self._score_sequence(env, self.rng.integers(0, len(ACTION_NAMES), self.horizon).tolist())
            for _ in range(self.num_candidates)
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        best = candidates[0]
        action = best.actions[0] if best.actions else 4
        return PlanResult(action, ACTION_NAMES[action], best.score, candidates[:8])

    @staticmethod
    def _score_sequence(env: GridRescueEnv, actions: list[int]) -> PlanCandidate:
        sim = env.clone()
        path = [_pos(sim.agent)]
        total = 0.0
        event = "horizon"
        for idx, action in enumerate(actions):
            result = sim.step(action)
            total += result.reward * (0.96**idx)
            path.append(_pos(sim.agent))
            event = result.info["event"]
            if result.done:
                break
        total -= 0.015 * sim.shortest_path_distance()
        return PlanCandidate(actions=actions, score=float(total), path=path, event=event)


class CEMPlanner:
    def __init__(
        self,
        horizon: int = 12,
        num_candidates: int = 256,
        iterations: int = 4,
        elite_fraction: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.iterations = iterations
        self.elite_fraction = elite_fraction
        self.rng = np.random.default_rng(seed)

    def plan(self, env: GridRescueEnv) -> PlanResult:
        action_count = len(ACTION_NAMES)
        probs = np.full((self.horizon, action_count), 1.0 / action_count)
        scored: list[PlanCandidate] = []

        for _ in range(self.iterations):
            sequences = np.array(
                [
                    [self.rng.choice(action_count, p=probs[t]) for t in range(self.horizon)]
                    for _ in range(self.num_candidates)
                ],
                dtype=np.int64,
            )
            scored = [RandomShootingPlanner._score_sequence(env, seq.tolist()) for seq in sequences]
            scored.sort(key=lambda candidate: candidate.score, reverse=True)
            elite_count = max(4, int(self.num_candidates * self.elite_fraction))
            elites = np.array([candidate.actions for candidate in scored[:elite_count]], dtype=np.int64)
            probs = np.full_like(probs, 0.025)
            for t in range(self.horizon):
                counts = np.bincount(elites[:, t], minlength=action_count).astype(float)
                probs[t] += counts / max(1.0, counts.sum())
                probs[t] /= probs[t].sum()

        best = scored[0]
        action = best.actions[0] if best.actions else 4
        return PlanResult(action, ACTION_NAMES[action], best.score, scored[:8])


class LearnedMPCPlanner:
    def __init__(
        self,
        horizon: int = 4,
        num_candidates: int = 128,
        seed: int = 0,
        model_path: Path | None = None,
    ) -> None:
        self.horizon = min(horizon, 6)
        self.num_candidates = num_candidates
        self.rng = np.random.default_rng(seed)
        self.model_path = require_model_path(model_path)
        self.torch = require_torch()
        self.model = load_model(self.model_path)

    def plan(self, env: GridRescueEnv) -> PlanResult:
        current_obs = env.render()
        expected_size = self.model.config.image_size
        if current_obs.shape[:2] != (expected_size, expected_size):
            raise ValueError(
                f"LearnedMPCPlanner expects {expected_size}x{expected_size} observations; "
                f"got {current_obs.shape[0]}x{current_obs.shape[1]}"
            )
        sequences = self.rng.integers(0, len(ACTION_NAMES), size=(self.num_candidates, self.horizon))
        candidates = [self._score_sequence(current_obs, seq.tolist()) for seq in sequences]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        best = candidates[0]
        action = best.actions[0] if best.actions else 4
        return PlanResult(action, ACTION_NAMES[action], best.score, candidates[:8])

    def _score_sequence(self, current_obs: np.ndarray, actions: list[int]) -> PlanCandidate:
        obs_tensor = self._obs_tensor(current_obs)
        total = 0.0
        done_probability = 0.0
        with self.torch.no_grad():
            for idx, action in enumerate(actions):
                action_tensor = self.torch.tensor([action]).long()
                pred = self.model(obs_tensor, action_tensor)
                reward = float(pred["reward"][0].detach().cpu())
                done_probability = float(self.torch.sigmoid(pred["done_logit"][0]).detach().cpu())
                total += (reward - 0.35 * done_probability) * (0.96**idx)
                obs_tensor = pred["next_obs"].detach()
                if done_probability >= 0.75:
                    break
        event = "learned_done" if done_probability >= 0.5 else "learned_horizon"
        return PlanCandidate(actions=actions, score=float(total), path=[], event=event)

    def _obs_tensor(self, image: np.ndarray):
        return self.torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def make_planner(name: str, seed: int = 0, horizon: int = 12, num_candidates: int = 256) -> Planner:
    if name == "random":
        return RandomPlanner(seed=seed)
    if name == "astar":
        return AStarPlanner()
    if name == "random_shooting":
        return RandomShootingPlanner(horizon=horizon, num_candidates=num_candidates, seed=seed)
    if name == "cem":
        return CEMPlanner(horizon=horizon, num_candidates=num_candidates, seed=seed)
    if name == "learned_mpc":
        return LearnedMPCPlanner(horizon=horizon, num_candidates=num_candidates, seed=seed)
    raise ValueError(f"Unknown planner: {name}")


def _shortest_path(env: GridRescueEnv) -> list[Position]:
    hazards = {hazard.pos for hazard in env.hazards}
    queue: deque[Position] = deque([env.agent])
    came_from: dict[Position, Position | None] = {env.agent: None}

    while queue:
        current = queue.popleft()
        if current == env.goal:
            break
        for delta in ACTION_TO_DELTA.values():
            nxt = current.move(delta.row, delta.col)
            if nxt in came_from or nxt in hazards:
                continue
            if nxt.row < 0 or nxt.col < 0 or nxt.row >= env.grid_size or nxt.col >= env.grid_size:
                continue
            if env.walls[nxt.row, nxt.col]:
                continue
            came_from[nxt] = current
            queue.append(nxt)

    if env.goal not in came_from:
        return [env.agent]

    path = []
    current: Position | None = env.goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))


def _action_between(a: Position, b: Position) -> int:
    dr = b.row - a.row
    dc = b.col - a.col
    for action, delta in ACTION_TO_DELTA.items():
        if delta.row == dr and delta.col == dc:
            return action
    return 4


def _pos(pos: Position) -> dict[str, int]:
    return {"row": int(pos.row), "col": int(pos.col)}
