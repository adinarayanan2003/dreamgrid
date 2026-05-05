from __future__ import annotations

import argparse
from collections import defaultdict

from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import make_planner
from dreamgrid.rollout_metrics import DEFAULT_ROLLOUT_HORIZONS, evaluate_learned_rollouts


def evaluate(
    planners: list[str],
    episodes: int = 50,
    grid_size: int = 16,
    seed: int = 0,
    max_steps: int | None = None,
    horizon: int = 6,
    num_candidates: int = 48,
) -> dict[str, dict[str, float]]:
    results: dict[str, list[dict[str, float | str]]] = defaultdict(list)

    for planner_name in planners:
        for episode in range(episodes):
            env = GridRescueEnv(grid_size=grid_size, max_steps=max_steps)
            env.reset(seed=seed + episode)
            planner = make_planner(
                planner_name,
                seed=seed + episode,
                horizon=horizon,
                num_candidates=num_candidates,
            )
            done = False
            event = "timeout"
            reward_sum = 0.0
            while not done:
                plan = planner.plan(env)
                step = env.step(plan.selected_action)
                reward_sum += step.reward
                done = step.done
                event = step.info["event"]
            results[planner_name].append(
                {
                    "success": 1.0 if event == "goal" else 0.0,
                    "collision": 1.0 if event == "collision" else 0.0,
                    "steps": float(env.step_count),
                    "reward": reward_sum,
                }
            )

    summary = {}
    for planner_name, rows in results.items():
        summary[planner_name] = {
            "success_rate": sum(float(row["success"]) for row in rows) / len(rows),
            "collision_rate": sum(float(row["collision"]) for row in rows) / len(rows),
            "avg_steps": sum(float(row["steps"]) for row in rows) / len(rows),
            "avg_reward": sum(float(row["reward"]) for row in rows) / len(rows),
            "episodes": float(len(rows)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DreamGrid planners.")
    parser.add_argument(
        "--learned-rollouts",
        action="store_true",
        help="evaluate learned rollout drift against simulator truth instead of planner scores",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["random", "astar", "random_shooting", "cem", "learned_mpc"],
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--rollout-horizons", nargs="+", type=int, default=list(DEFAULT_ROLLOUT_HORIZONS))
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    if args.learned_rollouts:
        summary = evaluate_learned_rollouts(
            episodes=args.episodes,
            horizons=args.rollout_horizons,
            grid_size=args.grid_size,
            seed=args.seed,
            model_path=args.model_path,
        )
        for horizon, metrics in summary["horizons"].items():
            print(f"horizon={horizon}", metrics)
        return

    summary = evaluate(
        args.planners,
        args.episodes,
        args.grid_size,
        args.seed,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
    )
    for planner, metrics in summary.items():
        print(planner, metrics)


if __name__ == "__main__":
    main()
