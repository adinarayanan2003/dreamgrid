import numpy as np
import pytest

from dreamgrid.dataset import generate_dataset


def test_generate_dataset_supports_mixed_scenarios(tmp_path) -> None:
    dataset_path = tmp_path / "mixed_dataset.npz"

    generate_dataset(
        episodes=2,
        out=dataset_path,
        grid_size=16,
        max_steps=3,
        seed=12,
        scenarios=["nominal", "moving_hazards"],
        random_action_probability=0.0,
    )

    data = np.load(dataset_path)

    assert data["obs"].shape[0] > 0
    assert data["next_obs"].shape == data["obs"].shape
    assert data["scenario_names"].tolist() == ["nominal", "moving_hazards"]
    assert set(data["scenario_ids"].tolist()) == {0, 1}
    assert set(data["hazard_counts"].tolist()) == {3, 6}


def test_generate_dataset_rejects_unknown_scenario(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown dataset scenarios"):
        generate_dataset(
            episodes=1,
            out=tmp_path / "bad.npz",
            scenarios=["not-a-scenario"],
        )
