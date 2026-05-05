from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dreamgrid.env import GridRescueEnv
from dreamgrid.model import load_model, require_model_path, require_torch
from dreamgrid.types import ACTION_NAMES

DEFAULT_ROLLOUT_HORIZONS = (1, 3, 5, 10)


def evaluate_learned_rollouts(
    episodes: int = 20,
    horizons: Sequence[int] = DEFAULT_ROLLOUT_HORIZONS,
    grid_size: int = 16,
    seed: int = 0,
    max_steps: int | None = None,
    model_path: str | Path | None = None,
) -> dict:
    normalized_horizons = _normalize_horizons(horizons)
    checkpoint = require_model_path(model_path)
    torch = require_torch()
    model = load_model(checkpoint)
    max_horizon = max(normalized_horizons)
    rows: dict[int, list[dict[str, float]]] = {horizon: [] for horizon in normalized_horizons}

    for episode_idx in range(episodes):
        env = GridRescueEnv(grid_size=grid_size, max_steps=max_steps)
        env.reset(seed=seed + episode_idx)
        current_obs = env.render()
        expected_size = model.config.image_size
        if current_obs.shape[:2] != (expected_size, expected_size):
            raise ValueError(
                f"rollout metrics expect {expected_size}x{expected_size} observations; "
                f"got {current_obs.shape[0]}x{current_obs.shape[1]}"
            )

        rng = np.random.default_rng(seed + episode_idx)
        actions = rng.integers(0, len(ACTION_NAMES), size=max_horizon).tolist()
        actual_env = env.clone()
        predicted_obs_tensor = _obs_tensor(torch, current_obs)
        stopped = False

        with torch.no_grad():
            for step_idx, action in enumerate(actions, start=1):
                if stopped:
                    break

                actual_result = actual_env.step(action)
                action_tensor = torch.tensor([action]).long()
                pred = model(predicted_obs_tensor, action_tensor)
                predicted_next = _tensor_to_uint8(pred["next_obs"][0])
                predicted_reward = float(pred["reward"][0].detach().cpu())
                predicted_done_probability = float(torch.sigmoid(pred["done_logit"][0]).detach().cpu())

                if step_idx in rows:
                    rows[step_idx].append(
                        {
                            "frame_mse": _frame_mse(predicted_next, actual_result.obs),
                            "reward_abs_error": abs(predicted_reward - actual_result.reward),
                            "done_match": float(
                                (predicted_done_probability >= 0.5) == actual_result.done
                            ),
                        }
                    )

                predicted_obs_tensor = pred["next_obs"].detach()
                stopped = actual_result.done

    return {
        "model_id": checkpoint.name,
        "episodes": episodes,
        "grid_size": grid_size,
        "seed": seed,
        "action_policy": "seeded_random",
        "horizons": {
            horizon: {
                "frame_mse": _mean(row["frame_mse"] for row in values),
                "reward_mae": _mean(row["reward_abs_error"] for row in values),
                "done_accuracy": _mean(row["done_match"] for row in values),
                "samples": float(len(values)),
            }
            for horizon, values in rows.items()
        },
    }


def _normalize_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(horizon) for horizon in horizons}))
    if not normalized:
        raise ValueError("at least one horizon is required")
    invalid = [horizon for horizon in normalized if horizon < 1 or horizon > 64]
    if invalid:
        raise ValueError(f"horizons must be between 1 and 64: {invalid}")
    return normalized


def _obs_tensor(torch_module, image: np.ndarray):
    return torch_module.tensor(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _tensor_to_uint8(tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def _frame_mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    diff = predicted.astype(np.float32) / 255.0 - actual.astype(np.float32) / 255.0
    return float(np.mean(diff**2))


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned rollout drift against simulator truth.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_ROLLOUT_HORIZONS))
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    summary = evaluate_learned_rollouts(
        episodes=args.episodes,
        horizons=args.horizons,
        grid_size=args.grid_size,
        seed=args.seed,
        model_path=args.model_path,
    )
    for horizon, metrics in summary["horizons"].items():
        print(f"horizon={horizon}", metrics)


if __name__ == "__main__":
    main()
