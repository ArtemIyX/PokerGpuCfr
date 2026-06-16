from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.core.board import Board
from pokergpu.tree.public_tree import NodeType, PublicTree


@dataclass(slots=True)
class ResultRow:
    mode: str
    workers: int
    mean_ms: float
    speedup: float


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
    print("mode    | workers | mean_ms | speedup")
    print("-" * 44)

    rows: list[ResultRow] = []
    for mode in ("serial", "threaded", "numba"):
        baseline = None
        for workers in (1, 2, 4, 8, 16):
            if mode == "serial":
                result = _run_mode(
                    tree,
                    forward,
                    board,
                    mode=mode,
                    workers=1,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    runs=args.runs,
                )
            else:
                result = _run_mode(
                    tree,
                    forward,
                    board,
                    mode=mode,
                    workers=workers,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    runs=args.runs,
                )
            if baseline is None:
                baseline = result.mean_ms
            rows.append(
                ResultRow(
                    mode=mode,
                    workers=workers,
                    mean_ms=result.mean_ms,
                    speedup=baseline / result.mean_ms if result.mean_ms > 0.0 else 0.0,
                )
            )

    for row in rows:
        print(f"{row.mode:<7} | {row.workers:>7} | {row.mean_ms:>7.3f} | {row.speedup:>7.3f}x")


def _run_mode(
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    *,
    mode: str,
    workers: int,
    iterations: int,
    warmup: int,
    runs: int,
) -> ResultRow:
    stage2 = _load_stage2(mode)
    for _ in range(warmup):
        _run_once(stage2, tree, forward, board, workers=workers, iterations=iterations)
    samples = [
        _run_once(stage2, tree, forward, board, workers=workers, iterations=iterations)
        for _ in range(runs)
    ]
    mean_ms = sum(samples) / len(samples) * 1000.0
    return ResultRow(mode=mode, workers=workers, mean_ms=mean_ms, speedup=0.0)


def _run_once(
    stage2,
    tree: PublicTree,
    forward: ForwardProfileResult,
    board: Board,
    *,
    workers: int,
    iterations: int,
) -> float:
    start = perf_counter()
    for _ in range(iterations):
        stage2.aggregate_prob_sum(tree, forward, board, max_workers=workers)
    return perf_counter() - start


def _load_stage2(mode: str):
    if mode == "numba":
        os.environ["POKERGPU_STAGE2_NUMBA"] = "1"
    else:
        os.environ.pop("POKERGPU_STAGE2_NUMBA", None)
    module = importlib.import_module("pokergpu.cfr.stage2")
    return importlib.reload(module)


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
