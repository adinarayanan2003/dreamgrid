from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import make_planner
from dreamgrid.rollout_metrics import DEFAULT_ROLLOUT_HORIZONS, evaluate_learned_rollouts

DEFAULT_HELDOUT_SPLITS = {
    "validation": 10_000,
    "test": 20_000,
}


@dataclass(frozen=True)
class ScenarioConfig:
    hazard_count: int
    wall_density: float


DEFAULT_HELDOUT_SCENARIOS = {
    "nominal": ScenarioConfig(hazard_count=3, wall_density=0.16),
    "dense_walls": ScenarioConfig(hazard_count=3, wall_density=0.24),
    "moving_hazards": ScenarioConfig(hazard_count=6, wall_density=0.16),
}


def replay_heldout_case(
    split: str,
    scenario: str,
    seed: int,
    planner: str,
    grid_size: int = 16,
    horizon: int = 6,
    num_candidates: int = 48,
) -> dict:
    _select("splits", [split], DEFAULT_HELDOUT_SPLITS)
    _select("scenarios", [scenario], DEFAULT_HELDOUT_SCENARIOS)
    scenario_config = DEFAULT_HELDOUT_SCENARIOS[scenario]
    env = GridRescueEnv(
        grid_size=grid_size,
        hazard_count=scenario_config.hazard_count,
        wall_density=scenario_config.wall_density,
    )
    env.reset(seed=seed)
    planner_instance = make_planner(
        planner,
        seed=seed,
        horizon=horizon,
        num_candidates=num_candidates,
    )
    initial_state = env.symbolic_state()
    steps = []
    done = False
    final_event = "timeout"
    total_reward = 0.0

    while not done:
        plan = planner_instance.plan(env)
        result = env.step(plan.selected_action)
        total_reward += result.reward
        done = result.done
        final_event = result.info["event"]
        steps.append(
            {
                "step": env.step_count,
                "action": plan.selected_action,
                "action_name": plan.selected_action_name,
                "planner_score": plan.score,
                "candidate_count": len(plan.candidates),
                "reward": result.reward,
                "total_reward": total_reward,
                "event": final_event,
                "done": done,
                "state": env.symbolic_state(),
            }
        )

    return {
        "split": split,
        "scenario": scenario,
        "seed": seed,
        "planner": planner,
        "grid_size": grid_size,
        "config": asdict(scenario_config),
        "initial_state": initial_state,
        "steps": steps,
        "final_event": final_event,
        "total_reward": total_reward,
        "step_count": len(steps),
        "success": final_event == "goal",
    }


def evaluate_heldout(
    planners: list[str],
    episodes_per_split: int = 6,
    grid_size: int = 16,
    splits: list[str] | None = None,
    scenarios: list[str] | None = None,
    horizon: int = 6,
    num_candidates: int = 48,
    rollout_horizons: list[int] | None = None,
    include_learned_rollouts: bool = True,
    model_path: str | Path | None = None,
    max_failure_cases: int = 8,
) -> dict:
    selected_splits = _select("splits", splits, DEFAULT_HELDOUT_SPLITS)
    selected_scenarios = _select("scenarios", scenarios, DEFAULT_HELDOUT_SCENARIOS)
    normalized_rollout_horizons = rollout_horizons or list(DEFAULT_ROLLOUT_HORIZONS)

    payload = {
        "episodes_per_split": episodes_per_split,
        "grid_size": grid_size,
        "planners": planners,
        "rollout_horizons": normalized_rollout_horizons,
        "splits": {},
    }

    for split_name in selected_splits:
        split_seed = DEFAULT_HELDOUT_SPLITS[split_name]
        split_payload = {"seed_start": split_seed, "scenarios": {}}
        for scenario_name in selected_scenarios:
            scenario = DEFAULT_HELDOUT_SCENARIOS[scenario_name]
            planner_metrics, failure_cases = _evaluate_planners_with_failures(
                planners=planners,
                episodes=episodes_per_split,
                grid_size=grid_size,
                seed=split_seed,
                hazard_count=scenario.hazard_count,
                wall_density=scenario.wall_density,
                horizon=horizon,
                num_candidates=num_candidates,
                max_failure_cases=max_failure_cases,
            )
            learned_rollouts = None
            if include_learned_rollouts:
                learned_rollouts = evaluate_learned_rollouts(
                    episodes=episodes_per_split,
                    horizons=normalized_rollout_horizons,
                    grid_size=grid_size,
                    seed=split_seed,
                    hazard_count=scenario.hazard_count,
                    wall_density=scenario.wall_density,
                    model_path=model_path,
                )

            split_payload["scenarios"][scenario_name] = {
                "config": asdict(scenario),
                "planners": planner_metrics,
                "failure_cases": failure_cases,
                "learned_rollouts": learned_rollouts,
            }
        payload["splits"][split_name] = split_payload

    return payload


def write_heldout_artifacts(
    summary: dict,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if csv_path:
        _write_summary_csv(summary, Path(csv_path))


def _evaluate_planners_with_failures(
    planners: list[str],
    episodes: int,
    grid_size: int,
    seed: int,
    hazard_count: int,
    wall_density: float,
    horizon: int,
    num_candidates: int,
    max_failure_cases: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict]]]:
    rows_by_planner: dict[str, list[dict[str, float]]] = {}
    failures_by_planner: dict[str, list[dict]] = {}

    for planner_name in planners:
        rows = []
        failures = []
        for episode_idx in range(episodes):
            episode_seed = seed + episode_idx
            env = GridRescueEnv(
                grid_size=grid_size,
                hazard_count=hazard_count,
                wall_density=wall_density,
            )
            env.reset(seed=episode_seed)
            planner = make_planner(
                planner_name,
                seed=episode_seed,
                horizon=horizon,
                num_candidates=num_candidates,
            )
            done = False
            event = "timeout"
            reward_sum = 0.0
            actions = []

            while not done:
                plan = planner.plan(env)
                step = env.step(plan.selected_action)
                reward_sum += step.reward
                done = step.done
                event = step.info["event"]
                actions.append(
                    {
                        "action": plan.selected_action,
                        "action_name": plan.selected_action_name,
                        "event": event,
                    }
                )

            rows.append(
                {
                    "success": 1.0 if event == "goal" else 0.0,
                    "collision": 1.0 if event == "collision" else 0.0,
                    "steps": float(env.step_count),
                    "reward": reward_sum,
                }
            )
            if event != "goal" and len(failures) < max_failure_cases:
                failures.append(
                    {
                        "seed": episode_seed,
                        "event": event,
                        "steps": env.step_count,
                        "reward": reward_sum,
                        "actions": actions,
                    }
                )

        rows_by_planner[planner_name] = rows
        failures_by_planner[planner_name] = failures

    return _summarize_planner_rows(rows_by_planner), failures_by_planner


def _summarize_planner_rows(rows_by_planner: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    summary = {}
    for planner_name, rows in rows_by_planner.items():
        summary[planner_name] = {
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "collision_rate": sum(row["collision"] for row in rows) / len(rows),
            "avg_steps": sum(row["steps"] for row in rows) / len(rows),
            "avg_reward": sum(row["reward"] for row in rows) / len(rows),
            "episodes": float(len(rows)),
        }
    return summary


def _write_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "kind",
        "split",
        "scenario",
        "planner",
        "horizon",
        "success_rate",
        "collision_rate",
        "avg_steps",
        "avg_reward",
        "frame_mse",
        "reward_mae",
        "done_accuracy",
        "samples",
        "episodes",
        "hazard_count",
        "wall_density",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for split_name, split in summary["splits"].items():
            for scenario_name, scenario in split["scenarios"].items():
                config = scenario["config"]
                for planner_name, metrics in scenario["planners"].items():
                    writer.writerow(
                        {
                            "kind": "planner",
                            "split": split_name,
                            "scenario": scenario_name,
                            "planner": planner_name,
                            "success_rate": metrics["success_rate"],
                            "collision_rate": metrics["collision_rate"],
                            "avg_steps": metrics["avg_steps"],
                            "avg_reward": metrics["avg_reward"],
                            "episodes": metrics["episodes"],
                            "hazard_count": config["hazard_count"],
                            "wall_density": config["wall_density"],
                        }
                    )
                if scenario["learned_rollouts"]:
                    for horizon, metrics in scenario["learned_rollouts"]["horizons"].items():
                        writer.writerow(
                            {
                                "kind": "learned_rollout",
                                "split": split_name,
                                "scenario": scenario_name,
                                "horizon": horizon,
                                "frame_mse": metrics["frame_mse"],
                                "reward_mae": metrics["reward_mae"],
                                "done_accuracy": metrics["done_accuracy"],
                                "samples": metrics["samples"],
                                "hazard_count": config["hazard_count"],
                                "wall_density": config["wall_density"],
                            }
                        )


def _select(
    label: Literal["splits", "scenarios"],
    selected: list[str] | None,
    available: dict,
) -> list[str]:
    if not selected:
        return list(available)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DreamGrid on held-out map splits.")
    parser.add_argument("--replay", action="store_true", help="replay one held-out seed/planner case")
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["random", "astar", "random_shooting", "cem", "learned_mpc"],
    )
    parser.add_argument("--planner", default="astar", help="planner to use with --replay")
    parser.add_argument("--split", default="validation", help="held-out split to use with --replay")
    parser.add_argument("--scenario", default="nominal", help="scenario to use with --replay")
    parser.add_argument("--seed", type=int, default=10_000, help="exact held-out seed to use with --replay")
    parser.add_argument("--episodes-per-split", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_HELDOUT_SPLITS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_HELDOUT_SCENARIOS))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--rollout-horizons", nargs="+", type=int, default=list(DEFAULT_ROLLOUT_HORIZONS))
    parser.add_argument("--skip-learned-rollouts", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-failure-cases", type=int, default=8)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    if args.replay:
        replay = replay_heldout_case(
            split=args.split,
            scenario=args.scenario,
            seed=args.seed,
            planner=args.planner,
            grid_size=args.grid_size,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
        )
        print(json.dumps(replay, indent=2, sort_keys=True))
        return

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
            for planner_name, metrics in scenario["planners"].items():
                print(f"    planner={planner_name} {metrics}")
                failures = scenario["failure_cases"].get(planner_name, [])
                if failures:
                    print(f"      failures={len(failures)} first_seed={failures[0]['seed']}")
            if scenario["learned_rollouts"]:
                for horizon, metrics in scenario["learned_rollouts"]["horizons"].items():
                    print(f"    rollout_horizon={horizon} {metrics}")


if __name__ == "__main__":
    main()
