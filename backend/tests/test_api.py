from fastapi.testclient import TestClient

from dreamgrid.api import app


def test_episode_generate_and_plan() -> None:
    client = TestClient(app)
    response = client.post("/api/episodes/generate", json={"grid_size": 12, "seed": 3})

    assert response.status_code == 200
    episode_id = response.json()["episode_id"]

    plan_response = client.post(
        "/api/planners/plan",
        json={"episode_id": episode_id, "planner": "astar", "horizon": 4, "num_candidates": 16},
    )

    assert plan_response.status_code == 200
    assert 0 <= plan_response.json()["selected_action"] <= 4


def test_step_with_planner() -> None:
    client = TestClient(app)
    episode = client.post("/api/episodes/generate", json={"grid_size": 12, "seed": 4}).json()

    response = client.post(
        f"/api/episodes/{episode['episode_id']}/step",
        json={"planner": "astar"},
    )

    assert response.status_code == 200
    assert "state" in response.json()


def test_predict_next_returns_404_for_missing_checkpoint() -> None:
    client = TestClient(app)
    episode = client.post("/api/episodes/generate", json={"grid_size": 12, "seed": 4}).json()

    response = client.post(
        "/api/models/predict-next",
        json={
            "episode_id": episode["episode_id"],
            "action": 3,
            "model_path": "/tmp/does-not-exist-dreamgrid.pt",
        },
    )

    assert response.status_code == 404


def test_predict_next_uses_configured_checkpoint_path(monkeypatch) -> None:
    monkeypatch.setenv("DREAMGRID_MODEL_PATH", "/tmp/does-not-exist-dreamgrid-env.pt")
    client = TestClient(app)
    episode = client.post("/api/episodes/generate", json={"grid_size": 12, "seed": 4}).json()

    response = client.post(
        "/api/models/predict-next",
        json={
            "episode_id": episode["episode_id"],
            "action": 3,
        },
    )

    assert response.status_code == 404
    assert "DREAMGRID_MODEL_PATH" in response.json()["detail"]


def test_learned_planner_returns_404_for_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("DREAMGRID_MODEL_PATH", "/tmp/does-not-exist-dreamgrid-env.pt")
    client = TestClient(app)
    episode = client.post("/api/episodes/generate", json={"grid_size": 16, "seed": 4}).json()

    response = client.post(
        "/api/planners/plan",
        json={
            "episode_id": episode["episode_id"],
            "planner": "learned_mpc",
            "horizon": 2,
            "num_candidates": 8,
        },
    )

    assert response.status_code == 404


def test_predict_rollout_rejects_invalid_action() -> None:
    client = TestClient(app)
    episode = client.post("/api/episodes/generate", json={"grid_size": 12, "seed": 4}).json()

    response = client.post(
        "/api/models/predict-rollout",
        json={
            "episode_id": episode["episode_id"],
            "actions": [3, 99],
        },
    )

    assert response.status_code == 400


def test_learned_rollout_eval_returns_404_for_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("DREAMGRID_MODEL_PATH", "/tmp/does-not-exist-dreamgrid-env.pt")
    client = TestClient(app)

    response = client.post(
        "/api/eval/learned-rollouts",
        json={"episodes": 1, "horizons": [1, 3], "grid_size": 16},
    )

    assert response.status_code == 404


def test_learned_rollout_eval_rejects_invalid_horizon() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/eval/learned-rollouts",
        json={"episodes": 1, "horizons": [0], "grid_size": 16},
    )

    assert response.status_code == 400


def test_heldout_eval_runs_without_learned_rollouts() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/eval/heldout",
        json={
            "planners": ["astar"],
            "episodes_per_split": 1,
            "splits": ["validation"],
            "scenarios": ["moving_hazards"],
            "include_learned_rollouts": False,
        },
    )

    assert response.status_code == 200
    scenario = response.json()["metrics"]["splits"]["validation"]["scenarios"]["moving_hazards"]
    assert scenario["config"]["hazard_count"] == 6
    assert scenario["learned_rollouts"] is None


def test_heldout_eval_returns_404_for_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("DREAMGRID_MODEL_PATH", "/tmp/does-not-exist-dreamgrid-env.pt")
    client = TestClient(app)

    response = client.post(
        "/api/eval/heldout",
        json={
            "planners": ["astar"],
            "episodes_per_split": 1,
            "splits": ["validation"],
            "scenarios": ["nominal"],
            "include_learned_rollouts": True,
        },
    )

    assert response.status_code == 404
