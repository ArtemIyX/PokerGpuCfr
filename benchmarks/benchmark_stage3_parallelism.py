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

import numpy as np

from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("POKERGPU_STAGE2_NUMBA", "1")
    importlib.reload(importlib.import_module("pokergpu.cfr.stage3"))

    tree = _make_repeated_tree(make_leduc_public_tree(), args.replicas)
    forward = _make_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)

    print("Stage 3 benchmark")
    print(f"replicas={args.replicas} tree_nodes={tree.node_count} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | steady_ms")
    print("-" * 20)

    for workers in (1, 2, 4, 8, 16):
        max_workers = None if workers == 1 else workers
        _bench(tree, aggregate, iterations=args.warmup, max_workers=max_workers)
        times = [
            _time_call(
                _bench,
                tree,
                aggregate,
                iterations=args.iterations,
                max_workers=max_workers,
            )
            for _ in range(args.runs)
        ]
        mean_ms = sum(times) / len(times) * 1000.0
        print(f"{workers:>7} | {mean_ms:>9.3f}")

    if args.profile:
        _run_profile(tree, aggregate, iterations=max(1, args.iterations))


def _bench(tree, aggregate, *, iterations: int, max_workers: int | None) -> None:
    for _ in range(iterations):
        compute_opponent_reach(tree, aggregate, max_workers=max_workers)


def _run_profile(tree, aggregate, *, iterations: int) -> None:
    _bench(tree, aggregate, iterations=1, max_workers=16)
    profiler = cProfile.Profile()
    profiler.enable()
    _bench(tree, aggregate, iterations=iterations, max_workers=16)
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
    stats.print_callees("pokergpu.cfr.stage3")
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


def _make_forward(tree):
    node_reach = tuple(1.0 + (index % 7) * 0.125 for index in range(tree.node_count))
    return ForwardProfileResult(
        node_reach=node_reach,
        infoset_reach=(),
        action_reach=tuple(() for _ in range(tree.node_count)),
    )


if __name__ == "__main__":
    main()
