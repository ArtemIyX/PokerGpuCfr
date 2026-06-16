from __future__ import annotations

from pokergpu.cfr.leaf_eval import LeafEvalBackend
from pokergpu.cfr.leaf_eval import LeafEvalResult
from pokergpu.cfr.gpu_leaf_backend import create_default_leaf_backend
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityResult, build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import compute_showdown_equity
from pokergpu.cfr.leaf_eval import evaluate_leaf_batch
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
    return evaluate_leaf_batch(batch, backend or create_default_leaf_backend())


def evaluate_root_action_values(
    tree: PublicTree,
    *,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    return aggregate_root_action_values(tree, max_workers=max_workers)
