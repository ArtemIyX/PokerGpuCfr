from __future__ import annotations

from pokergpu.tree.public_tree import NodeId

from .aggregation import aggregate_action_values
from .infosets import build_dense_infoset_table
from .kuhn import make_kuhn_public_tree
from .reach import propagate_reach
from .state import DenseCfrState, SolverIterationResult, SolverState
from .strategy_update import apply_dense_solver_strategy_update, apply_solver_strategy_update


def run_kuhn_root_iteration(
    state: SolverState,
    *,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    tree = make_kuhn_public_tree()
    reach = propagate_reach(tree)
    action_values = aggregate_action_values(tree, NodeId(1))
    assert len(action_values) == len(state.regret_sums), "kuhn action values must match state width"
    return apply_solver_strategy_update(
        state,
        action_values,
        reach_weight=reach_weight * reach.node_reach[1],
    )


def run_kuhn_dense_iteration(
    state: DenseCfrState,
    *,
    reach_weight: float = 1.0,
    max_workers: int | None = None,
) -> DenseCfrState:
    tree = make_kuhn_public_tree()
    table = build_dense_infoset_table(tree)
    reach = propagate_reach(tree, infoset_table=table, max_workers=max_workers)
    action_values = tuple(
        aggregate_action_values(tree, NodeId(node_index))
        for node_index in table.infoset_to_node
        if node_index >= 0
    )
    assert len(action_values) == table.infoset_count, "dense action values must match infosets"
    return apply_dense_solver_strategy_update(
        state,
        action_values,
        infoset_table=table,
        reach_weights=reach.infoset_reach if reach.infoset_reach else None,
        max_workers=max_workers,
    )


def run_kuhn_dense_iterations(
    state: DenseCfrState,
    iterations: int,
    *,
    reach_weight: float = 1.0,
    max_workers: int | None = None,
) -> DenseCfrState:
    assert iterations > 0, "iterations must be positive"
    current = state
    for _ in range(iterations):
        current = run_kuhn_dense_iteration(
            current,
            reach_weight=reach_weight,
            max_workers=max_workers,
        )
    return current
