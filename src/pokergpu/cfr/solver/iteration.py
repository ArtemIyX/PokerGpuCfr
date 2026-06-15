from __future__ import annotations

from pokergpu.tree.public_tree import NodeId, PublicTree

from .state import ToyCfrResult, ToyCfrState
from ..stage1 import normalize_strategy
from ..stage7 import regret_matching, update_average_strategy, update_regret


def run_toy_cfr_iteration(
    state: ToyCfrState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    if not action_values:
        raise ValueError("action values cannot be empty")
    if len(state.regret_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")
    if len(state.strategy_sums) != len(action_values):
        raise ValueError("state and action values must have the same length")

    strategy = normalize_strategy(regret_matching(state.regret_sums))
    node_value = sum(prob * value for prob, value in zip(strategy, action_values, strict=True))
    regret_sums = update_regret(state.regret_sums, action_values, node_value)
    strategy_sums = update_average_strategy(state.strategy_sums, strategy, reach_weight)

    return ToyCfrResult(
        state=ToyCfrState(regret_sums=regret_sums, strategy_sums=strategy_sums),
        strategy=strategy,
        node_value=node_value,
        action_values=action_values,
    )


def run_tree_root_cfr_iteration(
    tree: PublicTree,
    state: ToyCfrState,
    *,
    terminal_values: tuple[float, ...],
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    root_children = tree.child_links(NodeId(0))
    if len(root_children) != len(terminal_values):
        raise ValueError("terminal values must match the root branching factor")

    return run_toy_cfr_iteration(
        state,
        terminal_values,
        reach_weight=reach_weight,
    )
