from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import make_planner
from dreamgrid.rollout_metrics import DEFAULT_ROLLOUT_HORIZONS, evaluate_learned_rollouts


def evaluate(
    planners: list[str],
    episodes: int = 50,
    grid_size: int = 16,
    seed: int = 0,
    max_steps: int | None = None,
    hazard_count: int = 3,
    wall_density: float = 0.16,
    horizon: int = 6,
    num_candidates: int = 48,
    model_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    results: dict[str, list[dict[str, float | str]]] = defaultdict(list)

    for planner_name in planners:
        for episode in range(episodes):
            env = GridRescueEnv(
                grid_size=grid_size,
                max_steps=max_steps,
                hazard_count=hazard_count,
                wall_density=wall_density,
            )
            env.reset(seed=seed + episode)
            planner = make_planner(
                planner_name,
                seed=seed + episode,
                horizon=horizon,
                num_candidates=num_candidates,
                model_path=model_path,
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
    from dreamgrid.heldout import (
        DEFAULT_HELDOUT_SCENARIOS,
        DEFAULT_HELDOUT_SPLITS,
        evaluate_heldout,
        write_heldout_artifacts,
    )

    parser = argparse.ArgumentParser(description="Evaluate DreamGrid planners.")
    parser.add_argument(
        "--learned-rollouts",
        action="store_true",
        help="evaluate learned rollout drift against simulator truth instead of planner scores",
    )
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="evaluate planner and optional learned-rollout metrics on held-out splits",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["random", "astar", "random_shooting", "cem", "learned_mpc"],
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hazard-count", type=int, default=3)
    parser.add_argument("--wall-density", type=float, default=0.16)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--rollout-horizons", nargs="+", type=int, default=list(DEFAULT_ROLLOUT_HORIZONS))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes-per-split", type=int, default=6)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_HELDOUT_SPLITS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_HELDOUT_SCENARIOS))
    parser.add_argument("--skip-learned-rollouts", action="store_true")
    parser.add_argument("--max-failure-cases", type=int, default=8)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()
    if args.heldout:
        summary = evaluate_heldout(
            planners=args.planners,
            episodes_per_split=args.episodes_per_split,
            grid_size=args.grid_size,
            splits=args.splits,
            scenarios=args.scenarios,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
            rollout_horizons=args.rollout_horizons,
            include_learned_rollouts=not args.skip_learned_rollouts,
            model_path=args.model_path,
            max_failure_cases=args.max_failure_cases,
        )
        write_heldout_artifacts(summary, json_path=args.out_json, csv_path=args.out_csv)
        for split_name, split in summary["splits"].items():
            print(f"split={split_name} seed_start={split['seed_start']}")
            for scenario_name, scenario in split["scenarios"].items():
                print(f"  scenario={scenario_name} config={scenario['config']}")
                for planner, metrics in scenario["planners"].items():
                    print(f"    planner={planner} {metrics}")
                    failures = scenario["failure_cases"].get(planner, [])
                    if failures:
                        print(f"      failures={len(failures)} first_seed={failures[0]['seed']}")
                if scenario["learned_rollouts"]:
                    for horizon, metrics in scenario["learned_rollouts"]["horizons"].items():
                        print(f"    rollout_horizon={horizon} {metrics}")
        return

    if args.learned_rollouts:
        summary = evaluate_learned_rollouts(
            episodes=args.episodes,
            horizons=args.rollout_horizons,
            grid_size=args.grid_size,
            seed=args.seed,
            hazard_count=args.hazard_count,
            wall_density=args.wall_density,
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
        hazard_count=args.hazard_count,
        wall_density=args.wall_density,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
            model_path=args.model_path,
        )
    for planner, metrics in summary.items():
        print(planner, metrics)


if __name__ == "__main__":
    main()
