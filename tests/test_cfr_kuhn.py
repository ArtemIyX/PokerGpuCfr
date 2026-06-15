from __future__ import annotations

from pokergpu.cfr.solver import (
    SolverState,
    aggregate_action_values,
    make_kuhn_public_tree,
    propagate_reach,
    run_kuhn_root_iteration,
)
from pokergpu.tree.public_tree import NodeId, NodeType


def test_make_kuhn_public_tree_has_chance_root() -> None:
    tree = make_kuhn_public_tree()

    assert tree.node_types[0] is NodeType.CHANCE
    assert tree.child_count[0] == 6


def test_kuhn_reach_and_action_values_are_wired() -> None:
    tree = make_kuhn_public_tree()

    reach = propagate_reach(tree)
    values = aggregate_action_values(tree, NodeId(1))

    assert reach.node_reach[0] == 1.0
    assert reach.node_reach[1] == 1 / 6
    assert values == (1.0, -1.0)


def test_kuhn_root_iteration_runs() -> None:
    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    result = run_kuhn_root_iteration(state)

    assert result.strategy == (0.5, 0.5)
