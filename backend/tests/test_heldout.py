import csv
import json

from dreamgrid.heldout import evaluate_heldout, write_heldout_artifacts


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
    assert "astar" in scenario["failure_cases"]
    assert scenario["learned_rollouts"] is None


def test_write_heldout_artifacts(tmp_path) -> None:
    summary = evaluate_heldout(
        planners=["astar"],
        episodes_per_split=1,
        splits=["validation"],
        scenarios=["nominal"],
        include_learned_rollouts=False,
    )
    json_path = tmp_path / "heldout.json"
    csv_path = tmp_path / "heldout.csv"

    write_heldout_artifacts(summary, json_path=json_path, csv_path=csv_path)

    loaded = json.loads(json_path.read_text())
    assert "validation" in loaded["splits"]

    with csv_path.open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["kind"] == "planner"
    assert rows[0]["split"] == "validation"
