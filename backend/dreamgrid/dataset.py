from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import AStarPlanner, RandomPlanner


@dataclass(frozen=True)
class DatasetScenario:
    hazard_count: int
    wall_density: float


DEFAULT_DATASET_SCENARIOS = {
    "nominal": DatasetScenario(hazard_count=3, wall_density=0.16),
    "dense_walls": DatasetScenario(hazard_count=3, wall_density=0.24),
    "moving_hazards": DatasetScenario(hazard_count=6, wall_density=0.16),
}


def generate_dataset(
    episodes: int,
    out: Path,
    grid_size: int = 16,
    max_steps: int | None = None,
    seed: int = 0,
    scenarios: Sequence[str] | None = None,
    astar_probability: float = 0.65,
    random_action_probability: float = 0.15,
) -> Path:
    selected_scenarios = _select_scenarios(scenarios)
    random_planner = RandomPlanner(seed=seed)
    astar_planner = AStarPlanner()
    rng = np.random.default_rng(seed)

    obs_frames = []
    next_obs_frames = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    steps = []
    seeds = []
    scenario_ids = []
    hazard_counts = []
    wall_densities = []

    for episode_id in range(episodes):
        scenario_idx = episode_id % len(selected_scenarios)
        scenario_name = selected_scenarios[scenario_idx]
        scenario = DEFAULT_DATASET_SCENARIOS[scenario_name]
        env = GridRescueEnv(
            grid_size=grid_size,
            max_steps=max_steps,
            hazard_count=scenario.hazard_count,
            wall_density=scenario.wall_density,
        )
        obs = env.reset(seed=seed + episode_id)
        done = False
        while not done:
            planner = astar_planner if rng.random() < astar_probability else random_planner
            plan = planner.plan(env)
            if rng.random() < random_action_probability:
                action = int(rng.integers(0, 5))
            else:
                action = plan.selected_action
            result = env.step(action)

            obs_frames.append(obs)
            next_obs_frames.append(result.obs)
            actions.append(action)
            rewards.append(result.reward)
            dones.append(result.done)
            episode_ids.append(episode_id)
            steps.append(env.step_count)
            seeds.append(seed + episode_id)
            scenario_ids.append(scenario_idx)
            hazard_counts.append(scenario.hazard_count)
            wall_densities.append(scenario.wall_density)

            obs = result.obs
            done = result.done

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.asarray(obs_frames, dtype=np.uint8),
        next_obs=np.asarray(next_obs_frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        scenario_ids=np.asarray(scenario_ids, dtype=np.int64),
        scenario_names=np.asarray(selected_scenarios, dtype="U32"),
        hazard_counts=np.asarray(hazard_counts, dtype=np.int64),
        wall_densities=np.asarray(wall_densities, dtype=np.float32),
    )
    return out


def _select_scenarios(scenarios: Sequence[str] | None) -> list[str]:
    if not scenarios:
        return ["nominal"]
    unknown = [name for name in scenarios if name not in DEFAULT_DATASET_SCENARIOS]
    if unknown:
        raise ValueError(f"unknown dataset scenarios: {unknown}")
    return list(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DreamGrid transition data.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", default=["nominal"], choices=list(DEFAULT_DATASET_SCENARIOS))
    parser.add_argument("--astar-probability", type=float, default=0.65)
    parser.add_argument("--random-action-probability", type=float, default=0.15)
    args = parser.parse_args()
    path = generate_dataset(
        args.episodes,
        args.out,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        seed=args.seed,
        scenarios=args.scenarios,
        astar_probability=args.astar_probability,
        random_action_probability=args.random_action_probability,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
