from __future__ import annotations

from collections.abc import Mapping

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.infosets import build_dense_infoset_table
from pokergpu.cfr.stage6 import BackwardCFVResult
from pokergpu.cfr.stage7 import apply_dense_backward_cfv_update
from pokergpu.cfr.solver.state import DenseCfrState
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import InfosetId
from pokergpu.tree.public_tree import PublicTree

from .evaluation import evaluate_backward_cfv
from .evaluation import evaluate_root_action_values
from .state import SolverIterationResult, SolverState
from .strategy_update import apply_solver_strategy_update


def run_solver_iteration(
    state: SolverState,
    action_values: tuple[float, ...],
    *,
    reach_weight: float = 1.0,
) -> SolverIterationResult:
    assert action_values, "action values cannot be empty"
    return apply_solver_strategy_update(
        state,
        action_values,
        reach_weight=reach_weight,
    )


def run_tree_root_iteration(
    tree: PublicTree,
    state: SolverState,
    reach_weight: float = 1.0,
    max_workers: int | None = None,
) -> SolverIterationResult:
    assert tree.node_count > 0, "public tree cannot be empty"
    return apply_solver_strategy_update(
        state,
        evaluate_root_action_values(tree, max_workers=max_workers),
        reach_weight=reach_weight,
    )


def run_tree_backward_cfv_iteration(
    tree: PublicTree,
    *,
    board: Board | None = None,
    backend: GpuLeafBackend | None = None,
    infoset_strategies: Mapping[InfosetId, tuple[float, ...]] | None = None,
    max_workers: int | None = None,
    ) -> BackwardCFVResult:
    forward = propagate_forward(tree, infoset_strategies=infoset_strategies)
    return evaluate_backward_cfv(
        tree,
        forward,
        board=board,
        backend=backend,
        max_workers=max_workers,
    )


def run_dense_backward_cfv_iteration(
    tree: PublicTree,
    state: DenseCfrState,
    *,
    board: Board | None = None,
    backend: GpuLeafBackend | None = None,
    infoset_strategies: Mapping[InfosetId, tuple[float, ...]] | None = None,
    max_workers: int | None = None,
) -> DenseCfrState:
    table = build_dense_infoset_table(tree)
    forward = propagate_forward(tree, infoset_strategies=infoset_strategies)
    backward = evaluate_backward_cfv(
        tree,
        forward,
        board=board,
        backend=backend,
        max_workers=max_workers,
    )
    return apply_dense_backward_cfv_update(
        state,
        backward,
        infoset_table=table,
    )


def run_dense_backward_cfv_iterations(
    tree: PublicTree,
    state: DenseCfrState,
    iterations: int,
    *,
    board: Board | None = None,
    backend: GpuLeafBackend | None = None,
    infoset_strategies: Mapping[InfosetId, tuple[float, ...]] | None = None,
    max_workers: int | None = None,
) -> DenseCfrState:
    assert iterations > 0, "iterations must be positive"
    current = state
    for _ in range(iterations):
        current = run_dense_backward_cfv_iteration(
            tree,
            current,
            board=board,
            backend=backend,
            infoset_strategies=infoset_strategies,
            max_workers=max_workers,
        )
    return current
