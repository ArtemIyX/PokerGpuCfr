from __future__ import annotations

from pokergpu.cfr.solver import (
    DenseCfrState,
    SolverState,
    aggregate_action_values,
    build_dense_infoset_table,
    make_kuhn_public_tree,
    propagate_reach,
    run_kuhn_dense_iteration,
    run_kuhn_dense_iterations,
    run_kuhn_root_iteration,
)
from pokergpu.tree.public_tree import NodeId, NodeType


def test_make_kuhn_public_tree_has_chance_root() -> None:
    tree = make_kuhn_public_tree()

    assert tree.node_types[0] is NodeType.CHANCE
    assert tree.child_count[0] == 6


def test_kuhn_reach_and_action_values_are_wired() -> None:
    tree = make_kuhn_public_tree()

    table = build_dense_infoset_table(tree)
    reach = propagate_reach(tree, infoset_table=table)
    values = aggregate_action_values(tree, NodeId(1))

    assert reach.node_reach[0] == 1.0
    assert reach.node_reach[1] == 1 / 6
    assert reach.cumulative_strategy[0] == (1 / 12, 1 / 12)
    assert values == (1.0, -1.0)


def test_kuhn_root_iteration_runs() -> None:
    state = SolverState(regret_sums=(0.0, 0.0), strategy_sums=(0.0, 0.0))

    result = run_kuhn_root_iteration(state)

    assert result.strategy == (0.5, 0.5)


def test_kuhn_dense_iteration_uses_stage1_output_to_update_stage7() -> None:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )

    next_state = run_kuhn_dense_iteration(state)

    assert len(next_state.regret_sums) == table.infoset_count
    assert next_state.strategy_sums[0] == (1 / 12, 1 / 12)


def test_kuhn_dense_iterations_accumulate_changes() -> None:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )

    next_state = run_kuhn_dense_iterations(state, 3)

    assert next_state != state
    assert next_state.regret_sums[0] != (0.0, 0.0)
