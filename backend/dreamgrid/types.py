from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActionName = Literal["up", "down", "left", "right", "stay"]


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    def move(self, dr: int, dc: int) -> "Position":
        return Position(self.row + dr, self.col + dc)


@dataclass(frozen=True)
class Hazard:
    pos: Position
    direction: Position


ACTION_TO_DELTA: dict[int, Position] = {
    0: Position(-1, 0),
    1: Position(1, 0),
    2: Position(0, -1),
    3: Position(0, 1),
    4: Position(0, 0),
}

ACTION_NAMES: dict[int, ActionName] = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "stay",
}

ACTION_IDS: dict[ActionName, int] = {name: idx for idx, name in ACTION_NAMES.items()}

