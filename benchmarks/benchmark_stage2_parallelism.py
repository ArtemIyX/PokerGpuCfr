from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True)
class BenchmarkSample:
    workers: int
    mean_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    tree = _make_leaf_tree(args.nodes)
    forward = _make_forward(tree)
    board = Board.from_str("AhKdTc")

    print("Stage 2 parallelism benchmark")
    print(f"nodes={args.nodes} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms | speedup")
    print("-" * 36)

    baseline = None
    for workers in (1, 2, 4, 8, 16):
        for _ in range(args.warmup):
            _run_once(tree, forward, board, workers=workers, iterations=args.iterations)
        samples = [
            _run_once(tree, forward, board, workers=workers, iterations=args.iterations)
            for _ in range(args.runs)
        ]
        mean_ms = sum(samples) / len(samples) * 1000.0
        if baseline is None:
            baseline = mean_ms
        speedup = baseline / mean_ms if mean_ms > 0.0 else 0.0
        print(f"{workers:>7} | {mean_ms:>7.3f} | {speedup:>7.3f}x")


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


def _run_once(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    *,
    workers: int,
    iterations: int,
) -> float:
    start = perf_counter()
    for _ in range(iterations):
        aggregate_prob_sum(tree, forward, board, max_workers=workers)
    return perf_counter() - start


if __name__ == "__main__":
    main()
