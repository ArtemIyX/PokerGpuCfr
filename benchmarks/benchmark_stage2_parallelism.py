from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from numba import set_num_threads

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import NodeType, PublicTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    os.environ["POKERGPU_STAGE2_NUMBA"] = "1"
    stage2 = importlib.import_module("pokergpu.cfr.stage2")
    stage2 = importlib.reload(stage2)

    tree = _make_leaf_tree(args.nodes)
    forward = _make_forward(tree)
    board = Board.from_str("AhKdTc")
    board_card_mask, board_card_vector, leaf_card_reach_vector = stage2._board_card_features(board)
    live_hand_mask = stage2.private_hand_mask(board.cards)
    leaf_node_ids = tuple(range(tree.node_count))
    leaf_indices = np.asarray(leaf_node_ids, dtype=np.int64)
    node_reach = np.asarray(forward.node_reach, dtype=np.float64)
    leaf_reach_sum = node_reach[leaf_indices]
    leaf_shares = np.zeros_like(leaf_reach_sum, dtype=np.float32)
    total_leaf_reach = float(np.sum(leaf_reach_sum, dtype=np.float64))
    if total_leaf_reach > 0.0:
        np.divide(leaf_reach_sum, np.float32(total_leaf_reach), out=leaf_shares)

    node_card_reach = np.empty((tree.node_count, 52), dtype=np.float64)
    node_hand_reach = np.empty((tree.node_count, stage2.private_hand_count()), dtype=np.float64)
    leaf_features = np.empty((tree.node_count, stage2.LEAF_EVAL_FEATURE_WIDTH), dtype=np.float32)

    print("Stage 2 Numba benchmark")
    print(f"nodes={args.nodes} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("step                 | threads | mean_ms")
    print("-" * 46)

    steps = (
        ("node_aggregates", _bench_node_aggregates),
        ("leaf_features", _bench_leaf_features),
        ("full_stage2", _bench_full_stage2),
    )
    for step_name, bench in steps:
        baseline: float | None = None
        for threads in (1, 2, 4, 8, 16):
            set_num_threads(threads)
            for _ in range(args.warmup):
                bench(
                    stage2,
                    tree,
                    forward,
                    board,
                    board_card_mask,
                    board_card_vector,
                    leaf_card_reach_vector,
                    live_hand_mask,
                    leaf_node_ids,
                    leaf_indices,
                    leaf_reach_sum,
                    leaf_shares,
                    node_card_reach,
                    node_hand_reach,
                    leaf_features,
                    iterations=args.iterations,
                )
            samples = [
                _time_call(
                    bench,
                    stage2,
                    tree,
                    forward,
                    board,
                    board_card_mask,
                    board_card_vector,
                    leaf_card_reach_vector,
                    live_hand_mask,
                    leaf_node_ids,
                    leaf_indices,
                    leaf_reach_sum,
                    leaf_shares,
                    node_card_reach,
                    node_hand_reach,
                    leaf_features,
                    iterations=args.iterations,
                )
                for _ in range(args.runs)
            ]
            mean_ms = sum(samples) / len(samples) * 1000.0
            if baseline is None:
                baseline = mean_ms
            speedup = baseline / mean_ms if mean_ms > 0.0 else 0.0
            print(f"{step_name:<20} | {threads:>7} | {mean_ms:>7.3f}  ({speedup:>5.2f}x)")


def _bench_node_aggregates(
    stage2,
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    board_card_mask,
    board_card_vector,
    leaf_card_reach_vector,
    live_hand_mask,
    leaf_node_ids,
    leaf_indices,
    leaf_reach_sum,
    leaf_shares,
    node_card_reach,
    node_hand_reach,
    leaf_features,
    *,
    iterations: int,
) -> None:
    node_reach = np.asarray(forward.node_reach, dtype=np.float64)
    board_mask = np.asarray(board_card_mask, dtype=np.bool_)
    for _ in range(iterations):
        stage2._fill_node_aggregates_numba(
            node_card_reach,
            node_hand_reach,
            node_reach,
            board_mask,
            live_hand_mask,
        )


def _bench_leaf_features(
    stage2,
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    board_card_mask,
    board_card_vector,
    leaf_card_reach_vector,
    live_hand_mask,
    leaf_node_ids,
    leaf_indices,
    leaf_reach_sum,
    leaf_shares,
    node_card_reach,
    node_hand_reach,
    leaf_features,
    *,
    iterations: int,
) -> None:
    node_reach = np.asarray(forward.node_reach, dtype=np.float64)
    board_mask = np.where(np.asarray(board_card_mask, dtype=np.bool_), np.float32(1.0), np.float32(0.0))
    board_vector = np.asarray(board_card_vector, dtype=np.float32)
    leaf_vector = np.asarray(leaf_card_reach_vector, dtype=np.float32)
    street = np.float32(stage2._street_code(board.street))
    board_size = np.float32(len(board.cards))
    board_signature = np.float32(stage2._board_signature(board))
    for _ in range(iterations):
        stage2._fill_leaf_features_numba(
            leaf_features,
            node_reach,
            leaf_shares,
            board_mask,
            board_vector,
            leaf_vector,
            street,
            board_size,
            board_signature,
            np.asarray(leaf_indices, dtype=np.int64),
        )


def _bench_full_stage2(
    stage2,
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    board_card_mask,
    board_card_vector,
    leaf_card_reach_vector,
    live_hand_mask,
    leaf_node_ids,
    leaf_indices,
    leaf_reach_sum,
    leaf_shares,
    node_card_reach,
    node_hand_reach,
    leaf_features,
    *,
    iterations: int,
) -> None:
    for _ in range(iterations):
        stage2.aggregate_prob_sum(tree, forward, board, max_workers=16)


def _time_call(func, *args, **kwargs) -> float:
    start = perf_counter()
    func(*args, **kwargs)
    return perf_counter() - start


def _make_leaf_tree(node_count: int) -> PublicTree:
    node_types = tuple(NodeType.LEAF for _ in range(node_count))
    first_child = tuple(0 for _ in range(node_count))
    child_count = tuple(0 for _ in range(node_count))
    infoset_ids = tuple(None for _ in range(node_count))
    terminal_payoffs = tuple(None for _ in range(node_count))
    return PublicTree(
        node_types=node_types,
        first_child=first_child,
        child_count=child_count,
        children=(),
        infoset_ids=infoset_ids,
        terminal_payoffs=terminal_payoffs,
    )


def _make_forward(tree: PublicTree) -> ForwardProfileResult:
    node_reach = tuple(1.0 + (index % 7) * 0.125 for index in range(tree.node_count))
    return ForwardProfileResult(
        node_reach=node_reach,
        infoset_reach=(),
        action_reach=tuple(() for _ in range(tree.node_count)),
    )


if __name__ == "__main__":
    main()
