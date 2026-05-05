from dreamgrid.heldout import evaluate_heldout


def test_evaluate_heldout_without_learned_rollouts() -> None:
    summary = evaluate_heldout(
        planners=["astar"],
        episodes_per_split=1,
        splits=["validation"],
        scenarios=["nominal"],
        include_learned_rollouts=False,
    )

    scenario = summary["splits"]["validation"]["scenarios"]["nominal"]
    assert scenario["config"] == {"hazard_count": 3, "wall_density": 0.16}
    assert "astar" in scenario["planners"]
    assert scenario["learned_rollouts"] is None
