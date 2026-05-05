import pytest

from dreamgrid.model import DEFAULT_MODEL_PATH, TorchUnavailableError, require_torch
from dreamgrid.rollout_metrics import evaluate_learned_rollouts


def test_evaluate_learned_rollouts_smoke() -> None:
    try:
        require_torch()
    except TorchUnavailableError:
        pytest.skip("train extra is required for learned rollout metrics")
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("default learned-model checkpoint is not available")

    summary = evaluate_learned_rollouts(episodes=1, horizons=[1], grid_size=16, seed=9)

    assert summary["model_id"] == DEFAULT_MODEL_PATH.name
    assert summary["horizons"][1]["samples"] == 1.0
    assert summary["horizons"][1]["frame_mse"] >= 0.0
    assert summary["horizons"][1]["reward_mae"] >= 0.0
    assert 0.0 <= summary["horizons"][1]["done_accuracy"] <= 1.0
