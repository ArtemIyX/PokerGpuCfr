from __future__ import annotations

from pokergpu.tree.public_tree import PublicTree

from .evaluation import evaluate_root_action_values
from .state import ToyCfrResult, ToyCfrState
from .strategy_update import apply_toy_strategy_update


def run_toy_cfr_iteration(
    state: ToyCfrState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    return apply_toy_strategy_update(
        state,
        action_values,
        reach_weight=reach_weight,
    )


def run_tree_root_cfr_iteration(
    tree: PublicTree,
    state: ToyCfrState,
    reach_weight: float = 1.0,
) -> ToyCfrResult:
    return apply_toy_strategy_update(
        state,
        evaluate_root_action_values(tree),
        reach_weight=reach_weight,
    )
