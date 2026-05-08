import numpy as np
import pytest

from dreamgrid.env import GridRescueEnv
from dreamgrid.model import DEFAULT_MODEL_PATH, TorchUnavailableError, require_torch
from dreamgrid.planners import AStarPlanner, CEMPlanner, LearnedMPCPlanner, RandomShootingPlanner, make_planner


def test_astar_returns_valid_action() -> None:
    env = GridRescueEnv(grid_size=12, hazard_count=0)
    env.reset(seed=5)

    plan = AStarPlanner().plan(env)

    assert 0 <= plan.selected_action <= 4
    assert plan.candidates[0].path


def test_random_shooting_returns_candidates() -> None:
    env = GridRescueEnv(grid_size=12)
    env.reset(seed=5)

    plan = RandomShootingPlanner(horizon=4, num_candidates=16, seed=1).plan(env)

    assert 0 <= plan.selected_action <= 4
    assert len(plan.candidates) > 0


def test_cem_returns_candidates() -> None:
    env = GridRescueEnv(grid_size=12)
    env.reset(seed=5)

    plan = CEMPlanner(horizon=4, num_candidates=16, iterations=2, seed=1).plan(env)

    assert 0 <= plan.selected_action <= 4
    assert len(plan.candidates) > 0


def test_learned_mpc_returns_valid_action() -> None:
    try:
        require_torch()
    except TorchUnavailableError:
        pytest.skip("train extra is required for learned MPC")
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("default learned-model checkpoint is not available")

    env = GridRescueEnv(grid_size=16)
    env.reset(seed=5)

    plan = LearnedMPCPlanner(horizon=2, num_candidates=4, seed=1).plan(env)

    assert 0 <= plan.selected_action <= 4
    assert len(plan.candidates) > 0


def test_make_planner_passes_model_path_to_learned_mpc(monkeypatch) -> None:
    captured = {}

    class FakeLearnedMPCPlanner:
        def __init__(self, *, horizon: int, num_candidates: int, seed: int, model_path: str) -> None:
            captured["horizon"] = horizon
            captured["num_candidates"] = num_candidates
            captured["seed"] = seed
            captured["model_path"] = model_path

    monkeypatch.setattr("dreamgrid.planners.LearnedMPCPlanner", FakeLearnedMPCPlanner)

    planner = make_planner(
        "learned_mpc",
        horizon=3,
        num_candidates=16,
        seed=7,
        model_path="/tmp/candidate.pt",
    )

    assert isinstance(planner, FakeLearnedMPCPlanner)
    assert captured == {
        "horizon": 3,
        "num_candidates": 16,
        "seed": 7,
        "model_path": "/tmp/candidate.pt",
    }


def test_learned_mpc_visual_features_find_agent_and_goal() -> None:
    env = GridRescueEnv(grid_size=16, hazard_count=0)
    env.reset(seed=5)

    features = LearnedMPCPlanner._visual_features(env.render())

    assert features.agent_center is not None
    assert features.goal_center is not None
    assert features.agent_confidence > 0.9
    assert features.goal_confidence > 0.9
    assert features.goal_distance is not None
    assert features.agent_center[0] < features.goal_center[0]
    assert features.agent_center[1] < features.goal_center[1]


def test_learned_mpc_visual_progress_scores_closer_prediction_higher() -> None:
    previous = LearnedMPCPlanner._visual_features(_synthetic_frame(agent=(3, 3), goal=(12, 12)))
    closer = LearnedMPCPlanner._visual_features(_synthetic_frame(agent=(4, 3), goal=(12, 12)))
    farther = LearnedMPCPlanner._visual_features(_synthetic_frame(agent=(2, 3), goal=(12, 12)))

    closer_score = LearnedMPCPlanner._score_prediction_step(
        reward=-0.02,
        done_probability=0.1,
        action=1,
        previous_features=previous,
        next_features=closer,
    )
    farther_score = LearnedMPCPlanner._score_prediction_step(
        reward=-0.02,
        done_probability=0.1,
        action=0,
        previous_features=previous,
        next_features=farther,
    )

    assert closer_score > farther_score


def test_learned_mpc_visual_hazard_risk_penalizes_near_hazards() -> None:
    previous = LearnedMPCPlanner._visual_features(_synthetic_frame(agent=(3, 3), goal=(12, 12)))
    near_hazard = LearnedMPCPlanner._visual_features(
        _synthetic_frame(agent=(3, 3), goal=(12, 12), hazards=[(3, 4)])
    )
    far_hazard = LearnedMPCPlanner._visual_features(
        _synthetic_frame(agent=(3, 3), goal=(12, 12), hazards=[(12, 3)])
    )

    near_score = LearnedMPCPlanner._score_prediction_step(
        reward=-0.02,
        done_probability=0.1,
        action=3,
        previous_features=previous,
        next_features=near_hazard,
    )
    far_score = LearnedMPCPlanner._score_prediction_step(
        reward=-0.02,
        done_probability=0.1,
        action=3,
        previous_features=previous,
        next_features=far_hazard,
    )

    assert near_hazard.hazard_risk > far_hazard.hazard_risk
    assert near_score < far_score


def test_learned_mpc_action_prior_penalizes_hazard_near_target() -> None:
    clear = LearnedMPCPlanner._visual_features(_synthetic_frame(agent=(3, 3), goal=(12, 12)))
    risky = LearnedMPCPlanner._visual_features(
        _synthetic_frame(agent=(3, 3), goal=(12, 12), hazards=[(4, 3)])
    )

    assert clear.action_scores is not None
    assert risky.action_scores is not None
    assert risky.action_scores[1] < clear.action_scores[1]


def test_learned_mpc_first_action_gate_blocks_walls_and_prefers_progress() -> None:
    adjustments = LearnedMPCPlanner._first_action_adjustments(
        _synthetic_frame(
            agent=(1, 1),
            goal=(14, 14),
            walls=[(0, 1), (1, 0)],
        )
    )

    assert adjustments is not None
    assert adjustments[0] <= -LearnedMPCPlanner._FIRST_ACTION_INVALID_PENALTY
    assert adjustments[2] <= -LearnedMPCPlanner._FIRST_ACTION_INVALID_PENALTY
    assert adjustments[1] > adjustments[4]
    assert adjustments[3] > adjustments[4]


def test_learned_mpc_first_action_gate_penalizes_away_moves_when_progress_exists() -> None:
    adjustments = LearnedMPCPlanner._first_action_adjustments(
        _synthetic_frame(agent=(1, 2), goal=(14, 14), walls=[(0, 2)])
    )

    assert adjustments is not None
    assert adjustments[2] < adjustments[1]
    assert adjustments[2] < adjustments[3]


def test_learned_mpc_first_action_gate_does_not_prefer_direct_hazard() -> None:
    gate = LearnedMPCPlanner._first_action_gate(
        _synthetic_frame(
            agent=(3, 2),
            goal=(14, 14),
            hazards=[(3, 3)],
            walls=[(4, 2)],
        )
    )

    assert gate is not None
    assert 3 not in gate.preferred_actions
    assert 0 in gate.preferred_actions
    assert 2 in gate.preferred_actions


@pytest.mark.parametrize(
    ("seed", "setup_actions", "expected_actions"),
    [
        (8, [], {1, 3}),
        (15, [3], {1, 3}),
        (37, [], {3}),
    ],
)
def test_learned_mpc_avoids_known_bad_first_actions(
    seed: int,
    setup_actions: list[int],
    expected_actions: set[int],
) -> None:
    try:
        require_torch()
    except TorchUnavailableError:
        pytest.skip("train extra is required for learned MPC")
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("default learned-model checkpoint is not available")

    env = GridRescueEnv(grid_size=16)
    env.reset(seed=seed)
    for action in setup_actions:
        env.step(action)

    plan = LearnedMPCPlanner(
        horizon=6,
        num_candidates=128,
        seed=env.seed + env.step_count,
    ).plan(env)

    assert plan.selected_action in expected_actions


def _synthetic_frame(
    *,
    agent: tuple[int, int],
    goal: tuple[int, int],
    hazards: list[tuple[int, int]] | None = None,
    walls: list[tuple[int, int]] | None = None,
    grid_size: int = 16,
    tile_pixels: int = 4,
) -> np.ndarray:
    frame = np.zeros((grid_size * tile_pixels, grid_size * tile_pixels, 3), dtype=np.uint8)
    frame[:, :] = GridRescueEnv.palette["floor"]
    for wall in walls or []:
        _paint_tile(frame, wall, GridRescueEnv.palette["wall"], tile_pixels)
    for hazard in hazards or []:
        _paint_tile(frame, hazard, GridRescueEnv.palette["hazard"], tile_pixels)
    _paint_tile(frame, goal, GridRescueEnv.palette["goal"], tile_pixels)
    _paint_tile(frame, agent, GridRescueEnv.palette["agent"], tile_pixels)
    return frame


def _paint_tile(
    frame: np.ndarray,
    tile: tuple[int, int],
    color: np.ndarray,
    tile_pixels: int,
) -> None:
    row, col = tile
    row_start = row * tile_pixels
    col_start = col * tile_pixels
    frame[row_start : row_start + tile_pixels, col_start : col_start + tile_pixels] = color
