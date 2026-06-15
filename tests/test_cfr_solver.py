from __future__ import annotations

from pokergpu.cfr.solver import ToyCfrState, make_toy_public_tree, run_toy_cfr_iteration
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.tree.public_tree import NodeType


def test_make_toy_public_tree_is_small_and_solved_like() -> None:
    tree = make_toy_public_tree()

    assert tree.node_count == 3
    assert tree.node_types[0] is NodeType.PLAYER0
    assert tree.child_count[0] == 2


def test_toy_solver_iteration_runs_on_tiny_tree_shape() -> None:
    tree = make_toy_public_tree()
    forward = propagate_forward(tree, infoset_strategies={})

    state = ToyCfrState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))
    result = run_toy_cfr_iteration(state, (1.0, -1.0), reach_weight=forward.node_reach[0])

    assert result.strategy == (0.5, 0.5)
    assert result.state.strategy_sums == (0.5, 0.5)
