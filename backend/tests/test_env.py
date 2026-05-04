from dreamgrid.env import GridRescueEnv


def test_reset_is_deterministic_for_seed() -> None:
    env_a = GridRescueEnv(grid_size=12)
    env_b = GridRescueEnv(grid_size=12)

    env_a.reset(seed=42)
    env_b.reset(seed=42)

    assert env_a.symbolic_state() == env_b.symbolic_state()


def test_step_keeps_agent_inside_grid() -> None:
    env = GridRescueEnv(grid_size=12)
    env.reset(seed=1)

    for _ in range(8):
        env.step(2)

    assert env.agent.row >= 0
    assert env.agent.col >= 0


def test_render_shape_matches_grid_and_tile_size() -> None:
    env = GridRescueEnv(grid_size=12, tile_pixels=4)
    obs = env.reset(seed=1)

    assert obs.shape == (48, 48, 3)
    assert obs.dtype.name == "uint8"

