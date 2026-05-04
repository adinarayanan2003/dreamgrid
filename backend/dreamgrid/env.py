from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from dreamgrid.types import ACTION_NAMES, ACTION_TO_DELTA, Hazard, Position


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class GridRescueEnv:
    """Seeded 2D rescue grid with simple moving hazards and RGB rendering."""

    palette = {
        "floor": np.array([238, 242, 247], dtype=np.uint8),
        "wall": np.array([36, 42, 54], dtype=np.uint8),
        "agent": np.array([32, 104, 212], dtype=np.uint8),
        "goal": np.array([20, 148, 92], dtype=np.uint8),
        "hazard": np.array([220, 72, 72], dtype=np.uint8),
        "path": np.array([226, 232, 240], dtype=np.uint8),
    }

    def __init__(
        self,
        grid_size: int = 16,
        max_steps: int | None = None,
        hazard_count: int = 3,
        wall_density: float = 0.16,
        tile_pixels: int = 4,
    ) -> None:
        if grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        self.grid_size = grid_size
        self.max_steps = max_steps or grid_size * 4
        self.hazard_count = hazard_count
        self.wall_density = wall_density
        self.tile_pixels = tile_pixels
        self.rng = np.random.default_rng(0)
        self.seed = 0
        self.step_count = 0
        self.walls = np.zeros((grid_size, grid_size), dtype=bool)
        self.agent = Position(1, 1)
        self.goal = Position(grid_size - 2, grid_size - 2)
        self.hazards: list[Hazard] = []

    def clone(self) -> "GridRescueEnv":
        return deepcopy(self)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed = int(seed if seed is not None else self.seed + 1)
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.agent = Position(1, 1)
        self.goal = Position(self.grid_size - 2, self.grid_size - 2)
        self.walls = self._generate_walls()
        self.hazards = self._generate_hazards()
        return self.render()

    def step(self, action: int) -> StepResult:
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"Unknown action id: {action}")

        self.step_count += 1
        delta = ACTION_TO_DELTA[action]
        candidate = self.agent.move(delta.row, delta.col)
        if self._is_open(candidate):
            self.agent = candidate

        self.hazards = [self._move_hazard(hazard) for hazard in self.hazards]

        done = False
        reward = -0.02
        event = "step"

        if self.agent == self.goal:
            done = True
            reward = 1.0
            event = "goal"
        elif self.agent in {hazard.pos for hazard in self.hazards}:
            done = True
            reward = -1.0
            event = "collision"
        elif self.step_count >= self.max_steps:
            done = True
            reward = -0.25
            event = "timeout"
        else:
            old_distance = abs(candidate.row - self.goal.row) + abs(candidate.col - self.goal.col)
            new_distance = abs(self.agent.row - self.goal.row) + abs(self.agent.col - self.goal.col)
            if new_distance < old_distance:
                reward += 0.01

        return StepResult(
            obs=self.render(),
            reward=reward,
            done=done,
            info={
                "event": event,
                "action": ACTION_NAMES[action],
                "step": self.step_count,
                "state": self.symbolic_state(),
            },
        )

    def render(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        grid[:, :] = self.palette["floor"]
        grid[self.walls] = self.palette["wall"]

        for hazard in self.hazards:
            grid[hazard.pos.row, hazard.pos.col] = self.palette["hazard"]
        grid[self.goal.row, self.goal.col] = self.palette["goal"]
        grid[self.agent.row, self.agent.col] = self.palette["agent"]

        if self.tile_pixels > 1:
            grid = np.repeat(np.repeat(grid, self.tile_pixels, axis=0), self.tile_pixels, axis=1)
        return grid

    def symbolic_state(self) -> dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "max_steps": self.max_steps,
            "step": self.step_count,
            "seed": self.seed,
            "agent": {"row": self.agent.row, "col": self.agent.col},
            "goal": {"row": self.goal.row, "col": self.goal.col},
            "walls": self._positions_from_mask(self.walls),
            "hazards": [
                {
                    "row": hazard.pos.row,
                    "col": hazard.pos.col,
                    "dr": hazard.direction.row,
                    "dc": hazard.direction.col,
                }
                for hazard in self.hazards
            ],
        }

    def load_symbolic_state(self, state: dict[str, Any]) -> None:
        self.grid_size = int(state["grid_size"])
        self.max_steps = int(state["max_steps"])
        self.step_count = int(state["step"])
        self.seed = int(state["seed"])
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for wall in state["walls"]:
            self.walls[int(wall["row"]), int(wall["col"])] = True
        self.agent = Position(int(state["agent"]["row"]), int(state["agent"]["col"]))
        self.goal = Position(int(state["goal"]["row"]), int(state["goal"]["col"]))
        self.hazards = [
            Hazard(
                pos=Position(int(h["row"]), int(h["col"])),
                direction=Position(int(h["dr"]), int(h["dc"])),
            )
            for h in state["hazards"]
        ]

    def shortest_path_distance(self, pos: Position | None = None) -> int:
        pos = pos or self.agent
        return abs(pos.row - self.goal.row) + abs(pos.col - self.goal.col)

    def _generate_walls(self) -> np.ndarray:
        walls = self.rng.random((self.grid_size, self.grid_size)) < self.wall_density
        walls[0, :] = True
        walls[-1, :] = True
        walls[:, 0] = True
        walls[:, -1] = True
        walls[self.agent.row, self.agent.col] = False
        walls[self.goal.row, self.goal.col] = False

        # Carve a guaranteed L-shaped route so every seed is solvable.
        walls[self.agent.row, self.agent.col : self.goal.col + 1] = False
        walls[self.agent.row : self.goal.row + 1, self.goal.col] = False
        return walls

    def _generate_hazards(self) -> list[Hazard]:
        hazards: list[Hazard] = []
        directions = [Position(0, 1), Position(0, -1), Position(1, 0), Position(-1, 0)]
        attempts = 0
        while len(hazards) < self.hazard_count and attempts < self.grid_size * self.grid_size * 4:
            attempts += 1
            row = int(self.rng.integers(2, self.grid_size - 2))
            col = int(self.rng.integers(2, self.grid_size - 2))
            pos = Position(row, col)
            if not self._is_open(pos) or pos in {self.agent, self.goal}:
                continue
            if self.shortest_path_distance(pos) < 4:
                continue
            direction = directions[int(self.rng.integers(0, len(directions)))]
            hazards.append(Hazard(pos=pos, direction=direction))
        return hazards

    def _move_hazard(self, hazard: Hazard) -> Hazard:
        candidate = hazard.pos.move(hazard.direction.row, hazard.direction.col)
        if self._is_open(candidate) and candidate != self.goal:
            return Hazard(candidate, hazard.direction)
        reverse = Position(-hazard.direction.row, -hazard.direction.col)
        candidate = hazard.pos.move(reverse.row, reverse.col)
        if self._is_open(candidate) and candidate != self.goal:
            return Hazard(candidate, reverse)
        return hazard

    def _is_open(self, pos: Position) -> bool:
        if pos.row < 0 or pos.col < 0 or pos.row >= self.grid_size or pos.col >= self.grid_size:
            return False
        return not bool(self.walls[pos.row, pos.col])

    @staticmethod
    def _positions_from_mask(mask: np.ndarray) -> list[dict[str, int]]:
        rows, cols = np.where(mask)
        return [{"row": int(row), "col": int(col)} for row, col in zip(rows, cols, strict=False)]

