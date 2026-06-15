from __future__ import annotations

from pokergpu.tree.public_tree import NodeId

from .aggregation import aggregate_action_values
from .kuhn import make_kuhn_public_tree
from .reach import propagate_reach
from .state import SolverIterationResult, SolverState
from .strategy_update import apply_solver_strategy_update


def run_kuhn_root_iteration(
    state: SolverState,
    *,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    tree = make_kuhn_public_tree()
    reach = propagate_reach(tree)
    action_values = aggregate_action_values(tree, NodeId(1))
    return apply_solver_strategy_update(
        state,
        action_values,
        reach_weight=reach_weight * reach.node_reach[1],
    )
