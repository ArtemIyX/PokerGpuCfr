from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.stage1 import ForwardProfileResult  # noqa: E402
from pokergpu.cfr.stage2 import aggregate_prob_sum  # noqa: E402
from pokergpu.cfr.stage3 import compute_opponent_reach  # noqa: E402
from pokergpu.tree.public_tree import InfosetId, NodeType, PublicTree  # noqa: E402


@dataclass(slots=True)
class BenchSample:
    workers: int
    mean_ms: float


def make_benchmark_tree(infoset_count: int) -> PublicTree:
    node_types = [NodeType.PLAYER0 for _ in range(infoset_count)]
    first_child = [0 for _ in range(infoset_count)]
    child_count = [0 for _ in range(infoset_count)]
    infoset_ids = [InfosetId(index) for index in range(infoset_count)]
    terminal_payoffs = [None for _ in range(infoset_count)]
    return PublicTree(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=(),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
    )


def make_forward(tree: PublicTree) -> ForwardProfileResult:
    node_reach = tuple(1.0 + (index % 7) * 0.125 for index in range(tree.node_count))
    infoset_reach = tuple(node_reach)
    action_reach = tuple(() for _ in range(tree.node_count))
    return ForwardProfileResult(
        node_reach=node_reach,
        infoset_reach=infoset_reach,
        action_reach=action_reach,
    )


def measure_once(tree: PublicTree, *, workers: int | None, iterations: int) -> float:
    forward = make_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)
    start = time.perf_counter()
    for _ in range(iterations):
        compute_opponent_reach(tree, aggregate, max_workers=workers)
    return time.perf_counter() - start


def summarize(samples: list[float]) -> float:
    return sum(samples) / float(len(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infosets", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    tree = make_benchmark_tree(args.infosets)

    print("Stage 3 opponent reach benchmark")
    print(f"infosets={args.infosets} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms")
    print("-" * 24)

    for workers in (1, 2, 4, 8, 16):
        max_workers = None if workers == 1 else workers
        for _ in range(args.warmup):
            measure_once(tree, workers=max_workers, iterations=args.iterations)
        samples = [
            measure_once(tree, workers=max_workers, iterations=args.iterations)
            for _ in range(args.runs)
        ]
        mean_ms = summarize(samples) * 1000.0
        print(f"{workers:>7} | {mean_ms:>7.3f}")


if __name__ == "__main__":
    main()
