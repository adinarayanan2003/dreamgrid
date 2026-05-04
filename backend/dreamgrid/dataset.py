from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import AStarPlanner, RandomPlanner


def generate_dataset(
    episodes: int,
    out: Path,
    grid_size: int = 16,
    max_steps: int | None = None,
    seed: int = 0,
) -> Path:
    env = GridRescueEnv(grid_size=grid_size, max_steps=max_steps)
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

    for episode_id in range(episodes):
        obs = env.reset(seed=seed + episode_id)
        done = False
        while not done:
            planner = astar_planner if rng.random() < 0.65 else random_planner
            plan = planner.plan(env)
            if rng.random() < 0.15:
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
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DreamGrid transition data.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = generate_dataset(args.episodes, args.out, grid_size=args.grid_size, seed=args.seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

