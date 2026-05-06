import json

import pytest

from dreamgrid.dataset import generate_dataset
from dreamgrid.model import TorchUnavailableError, require_torch
from dreamgrid.train import train


def test_train_writes_best_checkpoint_and_metrics(tmp_path) -> None:
    try:
        torch = require_torch()
    except TorchUnavailableError:
        pytest.skip("train extra is required for checkpoint training")

    dataset_path = tmp_path / "dataset.npz"
    final_path = tmp_path / "world_model.pt"
    best_path = tmp_path / "world_model_best.pt"
    metrics_path = tmp_path / "metrics.json"
    generate_dataset(
        episodes=1,
        out=dataset_path,
        grid_size=16,
        max_steps=4,
        seed=21,
        random_action_probability=0.0,
    )

    train(
        dataset=dataset_path,
        out=final_path,
        best_out=best_path,
        metrics_out=metrics_path,
        epochs=1,
        batch_size=2,
        seed=3,
        device_name="cpu",
    )

    final_payload = torch.load(final_path, map_location="cpu")
    best_payload = torch.load(best_path, map_location="cpu")
    metrics_payload = json.loads(metrics_path.read_text())

    assert final_payload["checkpoint_kind"] == "final"
    assert best_payload["checkpoint_kind"] == "best_validation"
    assert final_payload["metadata"]["best_checkpoint"] == str(best_path)
    assert final_payload["metadata"]["loss_weights"] == {"frame": 1.0, "reward": 0.1, "done": 0.1}
    assert final_payload["best_metrics"]["epoch"] == 1
    assert metrics_payload["metadata"]["best_metrics"]["epoch"] == 1
