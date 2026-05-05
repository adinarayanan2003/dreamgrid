from dreamgrid.env import GridRescueEnv
from dreamgrid.planners import AStarPlanner, CEMPlanner, LearnedMPCPlanner, RandomShootingPlanner


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
    env = GridRescueEnv(grid_size=16)
    env.reset(seed=5)

    plan = LearnedMPCPlanner(horizon=2, num_candidates=4, seed=1).plan(env)

    assert 0 <= plan.selected_action <= 4
    assert len(plan.candidates) > 0
