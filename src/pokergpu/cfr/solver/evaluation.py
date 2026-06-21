from __future__ import annotations

from concurrent.futures import Executor
import numpy as np

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.leaf_eval import LeafEvalResult
from pokergpu.cfr.leaf_backend_factory import create_leaf_backend
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityBatchInput
from pokergpu.cfr.stage4 import ShowdownEquityResult
from pokergpu.cfr.stage4 import build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import compute_showdown_equity
from pokergpu.cfr.leaf_eval import evaluate_leaf_batch
from pokergpu.cfr.stage6 import BackwardCFVInput, BackwardCFVResult, backward_cfv
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import PublicTree

from .aggregation import aggregate_root_action_values


def evaluate_showdown_node_values(
    tree: PublicTree,
    forward: ForwardProfileResult,
    *,
    board: Board,
    max_workers: int | None = None,
) -> ShowdownEquityResult:
    aggregate = aggregate_prob_sum(tree, forward, board, max_workers=max_workers)
    opponent_reach = compute_opponent_reach(tree, aggregate, max_workers=max_workers)
    cache = build_showdown_equity_board_cache(board)
    return compute_showdown_equity(
        tree,
        aggregate,
        opponent_reach,
        board=board,
        cache=cache,
        max_workers=max_workers,
    )


def evaluate_leaf_node_values(
    tree: PublicTree,
    forward: ForwardProfileResult,
    *,
    board: Board | None = None,
    backend: LeafEvalBackend | None = None,
    max_workers: int | None = None,
) -> LeafEvalResult:
    aggregate = aggregate_prob_sum(tree, forward, board, max_workers=max_workers)
    batch = build_leaf_eval_batch(aggregate.leaf_batch)
    return evaluate_leaf_batch(batch, backend or create_leaf_backend())


def evaluate_backward_cfv(
    tree: PublicTree,
    forward: ForwardProfileResult,
    *,
    board: Board | None = None,
    backend: GpuLeafBackend | None = None,
    max_workers: int | None = None,
    executor: Executor | None = None,
) -> BackwardCFVResult:
    aggregate = aggregate_prob_sum(tree, forward, board, max_workers=max_workers)
    opponent_reach = compute_opponent_reach(tree, aggregate, max_workers=max_workers)
    showdown = (
        compute_showdown_equity(
            tree,
            aggregate,
            opponent_reach,
            board=board,
            cache=build_showdown_equity_board_cache(board) if board is not None else None,
            max_workers=max_workers,
        )
        if board is not None
        else ShowdownEquityResult(
            node_showdown_equity=tuple(0.0 for _ in range(tree.node_count)),
            node_showdown_equity_bb=tuple(0.0 for _ in range(tree.node_count)),
            input_rows=ShowdownEquityBatchInput(rows=()),
            output_rows=(),
        )
    )
    leaf_result = evaluate_leaf_node_values(
        tree,
        forward,
        board=board,
        backend=backend,
        max_workers=max_workers,
    )
    leaf_values = np.asarray(leaf_result.node_values, dtype=np.float64)
    backward_input = BackwardCFVInput(
        tree=tree,
        forward=forward,
        aggregate=aggregate,
        opponent_reach=opponent_reach,
        showdown=showdown,
        leaf_values=leaf_values,
    )
    return backward_cfv(backward_input, max_workers=max_workers, executor=executor)


def evaluate_root_action_values(
    tree: PublicTree,
    *,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    return aggregate_root_action_values(tree, max_workers=max_workers)
