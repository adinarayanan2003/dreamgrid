from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from dreamgrid.evaluate import evaluate
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
            planner_metrics = evaluate(
                planners=planners,
                episodes=episodes_per_split,
                grid_size=grid_size,
                seed=split_seed,
                hazard_count=scenario.hazard_count,
                wall_density=scenario.wall_density,
                horizon=horizon,
                num_candidates=num_candidates,
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
                "learned_rollouts": learned_rollouts,
            }
        payload["splits"][split_name] = split_payload

    return payload


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
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["random", "astar", "random_shooting", "cem", "learned_mpc"],
    )
    parser.add_argument("--episodes-per-split", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_HELDOUT_SPLITS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_HELDOUT_SCENARIOS))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--rollout-horizons", nargs="+", type=int, default=list(DEFAULT_ROLLOUT_HORIZONS))
    parser.add_argument("--skip-learned-rollouts", action="store_true")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

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
    )
    for split_name, split in summary["splits"].items():
        print(f"split={split_name} seed_start={split['seed_start']}")
        for scenario_name, scenario in split["scenarios"].items():
            print(f"  scenario={scenario_name} config={scenario['config']}")
            for planner_name, metrics in scenario["planners"].items():
                print(f"    planner={planner_name} {metrics}")
            if scenario["learned_rollouts"]:
                for horizon, metrics in scenario["learned_rollouts"]["horizons"].items():
                    print(f"    rollout_horizon={horizon} {metrics}")


if __name__ == "__main__":
    main()
