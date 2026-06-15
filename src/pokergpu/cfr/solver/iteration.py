from __future__ import annotations

from pokergpu.tree.public_tree import PublicTree

from .evaluation import evaluate_root_action_values
from .state import SolverIterationResult, SolverState
from .strategy_update import apply_solver_strategy_update


def run_solver_iteration(
    state: SolverState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    return apply_solver_strategy_update(
        state,
        action_values,
        reach_weight=reach_weight,
    )


def run_tree_root_iteration(
    tree: PublicTree,
    state: SolverState,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    return apply_solver_strategy_update(
        state,
        evaluate_root_action_values(tree),
        reach_weight=reach_weight,
    )
