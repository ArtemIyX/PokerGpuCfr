from __future__ import annotations

import argparse
import cProfile
import importlib
import os
import pstats
import sys
from io import StringIO
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import build_showdown_equity_board_cache
from pokergpu.cfr.stage4 import compute_showdown_equity
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import PublicTree


WORKER_COUNTS = (1, 2, 4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--board", type=str, default="AhKdTc9s2c")
    parser.add_argument(
        "--workers",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated worker counts to benchmark; use '1' for single-thread only.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-workers", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("POKERGPU_STAGE2_NUMBA", "1")
    importlib.reload(importlib.import_module("pokergpu.cfr.stage4"))

    base_tree = make_leduc_public_tree()
    tree = _make_repeated_tree(base_tree, args.replicas)
    board = Board.from_str(args.board)
    forward = _make_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward, board)
    opponent = compute_opponent_reach(tree, aggregate)
    cache = build_showdown_equity_board_cache(board)
    worker_counts = _parse_worker_counts(args.workers)

    print("Stage 4 showdown equity benchmark")
    print(f"replicas={args.replicas} tree_nodes={tree.node_count} iterations={args.iterations}")
    print(f"board={board} warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms | speedup")
    print("-" * 30)

    baseline_ms: float | None = None
    for workers in worker_counts:
        max_workers = None if workers == 1 else workers
        _bench(
            tree,
            aggregate,
            opponent,
            board=board,
            cache=cache,
            iterations=args.warmup,
            max_workers=max_workers,
        )
        samples = [
            _time_call(
                _bench,
                tree,
                aggregate,
                opponent,
                board=board,
                cache=cache,
                iterations=args.iterations,
                max_workers=max_workers,
            )
            for _ in range(args.runs)
        ]
        mean_ms = sum(samples) / len(samples) * 1000.0
        if baseline_ms is None:
            baseline_ms = mean_ms
        speedup = baseline_ms / mean_ms if mean_ms > 0.0 else 0.0
        print(f"{workers:>7} | {mean_ms:>7.3f} | {speedup:>7.2f}x")

    if args.profile:
        _run_profile(
            tree,
            aggregate,
            opponent,
            board=board,
            cache=cache,
            iterations=max(1, args.iterations),
            workers=args.profile_workers,
        )


def _parse_worker_counts(value: str) -> tuple[int, ...]:
    worker_counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not worker_counts:
        raise ValueError("workers must contain at least one positive integer")
    if any(worker <= 0 for worker in worker_counts):
        raise ValueError("workers must be positive integers")
    return worker_counts


def _bench(
    tree: PublicTree,
    aggregate,
    opponent,
    *,
    board: Board,
    cache,
    iterations: int,
    max_workers: int | None,
) -> None:
    for _ in range(iterations):
        compute_showdown_equity(
            tree,
            aggregate,
            opponent,
            board=board,
            cache=cache,
            max_workers=max_workers,
        )


def _run_profile(
    tree: PublicTree,
    aggregate,
    opponent,
    *,
    board: Board,
    cache,
    iterations: int,
    workers: int,
) -> None:
    max_workers = None if workers <= 1 else workers
    _bench(tree, aggregate, opponent, board=board, cache=cache, iterations=1, max_workers=max_workers)
    profiler = cProfile.Profile()
    profiler.enable()
    _bench(tree, aggregate, opponent, board=board, cache=cache, iterations=iterations, max_workers=max_workers)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(50)
    print(stream.getvalue())

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("tottime").print_stats(50)
    print(stream.getvalue())

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.print_callees("pokergpu.cfr.stage4")
    print(stream.getvalue())


def _time_call(func, *args, **kwargs) -> float:
    start = perf_counter()
    func(*args, **kwargs)
    return perf_counter() - start


def _make_repeated_tree(base: PublicTree, replicas: int) -> PublicTree:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    node_types = []
    first_child = []
    child_count = []
    children = []
    infoset_ids = []
    terminal_payoffs = []
    base_node_count = base.node_count
    base_infoset_count = max((int(infoset_id) for infoset_id in base.infoset_ids if infoset_id is not None), default=-1) + 1
    for replica_index in range(replicas):
        node_offset = replica_index * base_node_count
        infoset_offset = replica_index * base_infoset_count
        child_offset = len(children)
        for node_index in range(base_node_count):
            node_types.append(base.node_types[node_index])
            first_child.append(child_offset + base.first_child[node_index])
            child_count.append(base.child_count[node_index])
            infoset_id = base.infoset_ids[node_index]
            infoset_ids.append(None if infoset_id is None else type(infoset_id)(int(infoset_id) + infoset_offset))
            terminal_payoffs.append(base.terminal_payoffs[node_index])
        for link in base.children:
            children.append(type(link)(child=type(link.child)(int(link.child) + node_offset), chance_prob=link.chance_prob))
    return PublicTree(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=tuple(children),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
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
