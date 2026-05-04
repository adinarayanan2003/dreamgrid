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

