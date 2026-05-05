from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    F = None


class TorchUnavailableError(RuntimeError):
    pass


class ModelCheckpointUnavailableError(RuntimeError):
    pass


MODEL_PATH_ENV = "DREAMGRID_MODEL_PATH"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "experiments" / "world_model_v3_50ep.pt"


def require_torch() -> Any:
    if torch is None or nn is None or F is None:
        raise TorchUnavailableError("Install dreamgrid with the train extra to use world models.")
    return torch


def resolve_model_path(path: str | Path | None = None) -> Path:
    configured_path = path or os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH
    return Path(configured_path).expanduser()


def require_model_path(path: str | Path | None = None) -> Path:
    model_path = resolve_model_path(path)
    if not model_path.exists():
        raise ModelCheckpointUnavailableError(
            f"model checkpoint not found: {model_path}. "
            f"Set {MODEL_PATH_ENV} or pass model_path to use a different checkpoint."
        )
    return model_path


@dataclass
class WorldModelConfig:
    image_size: int = 64
    latent_dim: int = 128
    action_count: int = 5
    action_dim: int = 16


if nn is not None:

    class LatentWorldModel(nn.Module):
        def __init__(self, config: WorldModelConfig | None = None) -> None:
            super().__init__()
            self.config = config or WorldModelConfig()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 96, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(96 * 8 * 8, self.config.latent_dim),
            )
            self.action_embed = nn.Embedding(self.config.action_count, self.config.action_dim)
            self.dynamics = nn.Sequential(
                nn.Linear(self.config.latent_dim + self.config.action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.config.latent_dim),
            )
            self.reward_head = nn.Sequential(nn.Linear(self.config.latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.done_head = nn.Sequential(nn.Linear(self.config.latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.decoder_input = nn.Linear(self.config.latent_dim, 96 * 8 * 8)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, obs: "torch.Tensor", action: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            z = self.encoder(obs)
            action_z = self.action_embed(action)
            next_z = self.dynamics(torch.cat([z, action_z], dim=-1))
            decoded = self.decoder(self.decoder_input(next_z).view(-1, 96, 8, 8))
            return {
                "next_obs": decoded,
                "reward": self.reward_head(next_z).squeeze(-1),
                "done_logit": self.done_head(next_z).squeeze(-1),
                "latent": next_z,
            }


else:

    class LatentWorldModel:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise TorchUnavailableError("Install dreamgrid with the train extra to use world models.")


def load_model(path: Path) -> "LatentWorldModel":
    require_torch()
    path = require_model_path(path)
    payload = torch.load(path, map_location="cpu")
    config = WorldModelConfig(**payload["config"])
    model = LatentWorldModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
