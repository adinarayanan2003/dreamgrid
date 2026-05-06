import csv
import json

from PIL import Image

from dreamgrid.heldout import (
    evaluate_heldout,
    replay_heldout_case,
    write_heldout_artifacts,
    write_replay_gif,
)
from dreamgrid.planners import PlanResult


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


def test_evaluate_heldout_passes_model_path_to_planners(monkeypatch) -> None:
    captured = []

    class FakePlanner:
        def plan(self, _env) -> PlanResult:
            return PlanResult(
                selected_action=4,
                selected_action_name="stay",
                score=0.0,
                candidates=[],
            )

    def fake_make_planner(name: str, **kwargs):
        captured.append({"name": name, **kwargs})
        return FakePlanner()

    monkeypatch.setattr("dreamgrid.heldout.make_planner", fake_make_planner)

    evaluate_heldout(
        planners=["learned_mpc"],
        episodes_per_split=1,
        grid_size=8,
        splits=["validation"],
        scenarios=["nominal"],
        include_learned_rollouts=False,
        model_path="/tmp/candidate.pt",
        max_failure_cases=0,
    )

    assert captured[0]["name"] == "learned_mpc"
    assert captured[0]["model_path"] == "/tmp/candidate.pt"


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


def test_replay_heldout_case_returns_step_trace() -> None:
    replay = replay_heldout_case(
        split="validation",
        scenario="moving_hazards",
        seed=10_000,
        planner="astar",
    )

    assert replay["split"] == "validation"
    assert replay["scenario"] == "moving_hazards"
    assert replay["seed"] == 10_000
    assert replay["planner"] == "astar"
    assert replay["steps"]
    assert replay["final_event"] in {"goal", "collision", "timeout"}
    assert replay["steps"][-1]["done"] is True


def test_write_replay_gif(tmp_path) -> None:
    gif_path = tmp_path / "replay.gif"

    write_replay_gif(
        split="validation",
        scenario="moving_hazards",
        seed=10_000,
        planner="astar",
        path=gif_path,
    )

    with Image.open(gif_path) as image:
        assert image.format == "GIF"
        assert getattr(image, "is_animated", False)
        assert image.n_frames >= 2
