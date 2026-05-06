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


@dataclass(frozen=True)
class _VisualFeatures:
    agent_center: tuple[float, float] | None
    goal_center: tuple[float, float] | None
    agent_confidence: float
    goal_confidence: float
    hazard_risk: float
    goal_distance: float | None
    action_scores: tuple[float, ...] | None


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
    _PALETTE = {
        "agent": np.array([32, 104, 212], dtype=np.float32) / 255.0,
        "goal": np.array([20, 148, 92], dtype=np.float32) / 255.0,
        "hazard": np.array([220, 72, 72], dtype=np.float32) / 255.0,
        "wall": np.array([36, 42, 54], dtype=np.float32) / 255.0,
    }
    _COLOR_TOLERANCE = 0.38
    _MASK_THRESHOLD = 0.15
    _DISCOUNT = 0.96
    _REWARD_WEIGHT = 1.0
    _PROGRESS_WEIGHT = 2.0
    _HAZARD_WEIGHT = 0.45
    _ACTION_PRIOR_WEIGHT = 2.5
    _TARGET_HAZARD_PRIOR_WEIGHT = 0.22
    _SOFT_DONE_PENALTY = 0.08
    _BAD_TERMINAL_PENALTY = 0.85
    _STAY_PENALTY = 0.015
    _DONE_EVENT_THRESHOLD = 0.5
    _DONE_BREAK_THRESHOLD = 0.75

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
        previous_features = self._visual_features(current_obs)
        total = 0.0
        done_probability = 0.0
        last_reward = 0.0
        with self.torch.no_grad():
            for idx, action in enumerate(actions):
                action_tensor = self.torch.tensor([action]).long()
                pred = self.model(obs_tensor, action_tensor)
                reward = float(pred["reward"][0].detach().cpu())
                last_reward = reward
                done_probability = float(self.torch.sigmoid(pred["done_logit"][0]).detach().cpu())
                next_features = self._visual_features(pred["next_obs"])
                step_score = self._score_prediction_step(
                    reward=reward,
                    done_probability=done_probability,
                    action=action,
                    previous_features=previous_features,
                    next_features=next_features,
                )
                total += step_score * (self._DISCOUNT**idx)
                previous_features = next_features
                obs_tensor = pred["next_obs"].detach()
                if done_probability >= self._DONE_BREAK_THRESHOLD:
                    break
        if done_probability >= self._DONE_EVENT_THRESHOLD:
            event = "learned_goal_like" if last_reward > 0 else "learned_collision_like"
        else:
            event = "learned_horizon"
        return PlanCandidate(actions=actions, score=float(total), path=[], event=event)

    def _obs_tensor(self, image: np.ndarray):
        return self.torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    @classmethod
    def _score_prediction_step(
        cls,
        *,
        reward: float,
        done_probability: float,
        action: int,
        previous_features: _VisualFeatures,
        next_features: _VisualFeatures,
    ) -> float:
        score = cls._REWARD_WEIGHT * reward
        if previous_features.goal_distance is not None and next_features.goal_distance is not None:
            progress_confidence = min(
                previous_features.agent_confidence,
                previous_features.goal_confidence,
                next_features.agent_confidence,
                next_features.goal_confidence,
            )
            score += cls._PROGRESS_WEIGHT * (
                previous_features.goal_distance - next_features.goal_distance
            ) * progress_confidence
        if previous_features.action_scores is not None and 0 <= action < len(
            previous_features.action_scores
        ):
            score += cls._ACTION_PRIOR_WEIGHT * previous_features.action_scores[action]
        score -= cls._HAZARD_WEIGHT * next_features.hazard_risk
        if action == 4:
            score -= cls._STAY_PENALTY
        if done_probability >= cls._DONE_EVENT_THRESHOLD:
            if reward <= 0:
                score -= cls._BAD_TERMINAL_PENALTY * done_probability
        else:
            score -= cls._SOFT_DONE_PENALTY * done_probability
        return float(score)

    @classmethod
    def _visual_features(cls, frame: object) -> _VisualFeatures:
        image = cls._frame_to_rgb(frame)
        agent_mask = cls._palette_mask(image, cls._PALETTE["agent"])
        goal_mask = cls._palette_mask(image, cls._PALETTE["goal"])
        hazard_mask = cls._palette_mask(image, cls._PALETTE["hazard"])
        wall_mask = cls._palette_mask(image, cls._PALETTE["wall"])

        agent_center, agent_confidence = cls._weighted_centroid(agent_mask)
        goal_center, goal_confidence = cls._weighted_centroid(goal_mask)
        goal_distance = cls._normalized_distance(image.shape, agent_center, goal_center)
        hazard_risk = cls._hazard_risk(image.shape, agent_center, hazard_mask)
        action_scores = cls._action_scores(
            image.shape,
            agent_center,
            goal_center,
            agent_confidence,
            goal_confidence,
            agent_mask,
            wall_mask,
            hazard_mask,
        )
        return _VisualFeatures(
            agent_center=agent_center,
            goal_center=goal_center,
            agent_confidence=agent_confidence,
            goal_confidence=goal_confidence,
            hazard_risk=hazard_risk,
            goal_distance=goal_distance,
            action_scores=action_scores,
        )

    @staticmethod
    def _frame_to_rgb(frame: object) -> np.ndarray:
        if hasattr(frame, "detach"):
            array = frame.detach().cpu().numpy()
        else:
            array = np.asarray(frame)

        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"expected RGB frame, got shape {array.shape}")

        image = array.astype(np.float32, copy=False)
        if image.size and float(np.max(image)) > 1.5:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0)

    @classmethod
    def _palette_mask(cls, image: np.ndarray, color: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(image - color, axis=-1)
        return np.clip(1.0 - (distance / cls._COLOR_TOLERANCE), 0.0, 1.0)

    @staticmethod
    def _weighted_centroid(mask: np.ndarray) -> tuple[tuple[float, float] | None, float]:
        mass = float(mask.sum())
        if mass < 0.5:
            return None, 0.0
        rows, cols = np.indices(mask.shape, dtype=np.float32)
        center = (
            float((rows * mask).sum() / mass),
            float((cols * mask).sum() / mass),
        )
        expected_tile_mass = max(1.0, mask.size / 256.0)
        confidence = min(1.0, mass / expected_tile_mass)
        return center, confidence

    @staticmethod
    def _normalized_distance(
        shape: tuple[int, ...],
        first: tuple[float, float] | None,
        second: tuple[float, float] | None,
    ) -> float | None:
        if first is None or second is None:
            return None
        height, width = shape[:2]
        diagonal = max(1.0, float(np.hypot(height - 1, width - 1)))
        return float(np.hypot(first[0] - second[0], first[1] - second[1]) / diagonal)

    @classmethod
    def _hazard_risk(
        cls,
        shape: tuple[int, ...],
        agent_center: tuple[float, float] | None,
        hazard_mask: np.ndarray,
    ) -> float:
        if agent_center is None:
            return 0.0
        hazard_pixels = np.argwhere(hazard_mask >= cls._MASK_THRESHOLD)
        if hazard_pixels.size == 0:
            return 0.0

        center = np.array(agent_center, dtype=np.float32)
        distances = np.linalg.norm(hazard_pixels.astype(np.float32) - center, axis=1)
        nearest_distance = float(np.min(distances))
        height, width = shape[:2]
        diagonal = max(1.0, float(np.hypot(height - 1, width - 1)))
        danger_radius = max(4.0, diagonal * 0.18)
        proximity = max(0.0, 1.0 - (nearest_distance / danger_radius))

        hazard_mass = float(hazard_mask[hazard_pixels[:, 0], hazard_pixels[:, 1]].sum())
        expected_tile_mass = max(1.0, hazard_mask.size / 256.0)
        hazard_confidence = min(1.0, hazard_mass / expected_tile_mass)
        return float(proximity * hazard_confidence)

    @classmethod
    def _action_scores(
        cls,
        shape: tuple[int, ...],
        agent_center: tuple[float, float] | None,
        goal_center: tuple[float, float] | None,
        agent_confidence: float,
        goal_confidence: float,
        agent_mask: np.ndarray,
        wall_mask: np.ndarray,
        hazard_mask: np.ndarray,
    ) -> tuple[float, ...] | None:
        if agent_center is None or goal_center is None:
            return None
        if min(agent_confidence, goal_confidence) < 0.75:
            return None

        agent_pixels = int(np.count_nonzero(agent_mask >= cls._MASK_THRESHOLD))
        tile_step = max(1.0, float(np.sqrt(max(1, agent_pixels))))
        height, width = shape[:2]
        diagonal = max(1.0, float(np.hypot(height - 1, width - 1)))
        current_distance = float(
            np.hypot(agent_center[0] - goal_center[0], agent_center[1] - goal_center[1])
        )

        scores: list[float] = []
        for action in range(len(ACTION_NAMES)):
            delta = ACTION_TO_DELTA[action]
            target = (
                agent_center[0] + delta.row * tile_step,
                agent_center[1] + delta.col * tile_step,
            )
            target_distance = float(np.hypot(target[0] - goal_center[0], target[1] - goal_center[1]))
            score = (current_distance - target_distance) / diagonal

            if action == 4:
                score -= 0.02
            if target[0] < 0 or target[1] < 0 or target[0] >= height or target[1] >= width:
                score -= 0.25
            else:
                wall_strength = cls._patch_mean(wall_mask, target, tile_step)
                hazard_strength = cls._patch_mean(hazard_mask, target, tile_step)
                target_hazard_risk = cls._hazard_risk(shape, target, hazard_mask)
                if wall_strength > 0.25:
                    score -= 0.25
                if hazard_strength > 0.2:
                    score -= 0.16
                score -= cls._TARGET_HAZARD_PRIOR_WEIGHT * target_hazard_risk
            scores.append(float(score))
        return tuple(scores)

    @staticmethod
    def _patch_mean(mask: np.ndarray, center: tuple[float, float], tile_step: float) -> float:
        radius = max(1, int(round(tile_step / 2)))
        row = int(round(center[0]))
        col = int(round(center[1]))
        row_start = max(0, row - radius)
        row_end = min(mask.shape[0], row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(mask.shape[1], col + radius + 1)
        if row_start >= row_end or col_start >= col_end:
            return 0.0
        return float(mask[row_start:row_end, col_start:col_end].mean())


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
