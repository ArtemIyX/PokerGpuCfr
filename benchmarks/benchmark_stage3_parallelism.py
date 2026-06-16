from __future__ import annotations

import argparse
import cProfile
import importlib
import os
import pstats
import sys
from dataclasses import dataclass
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
from pokergpu.cfr.infosets import build_dense_infoset_table
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


@dataclass(slots=True)
class BenchSample:
    mode: str
    workers: int
    mean_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("POKERGPU_STAGE2_NUMBA", "1")
    stage3 = importlib.import_module("pokergpu.cfr.stage3")
    stage3 = importlib.reload(stage3)

    tree = _make_repeated_tree(make_leduc_public_tree(), args.replicas)
    forward = _make_forward(tree)
    aggregate = aggregate_prob_sum(tree, forward)

    print("Stage 3 benchmark")
    print(f"replicas={args.replicas} tree_nodes={tree.node_count} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("mode   | workers | mean_ms | speedup_vs_python")
    print("-" * 52)

    reference = None
    samples: list[BenchSample] = []
    for mode in ("python", "numba"):
        for workers in (1, 2, 4, 8, 16):
            max_workers = None if workers == 1 else workers
            for _ in range(args.warmup):
                _bench(stage3, tree, aggregate, iterations=args.iterations, max_workers=max_workers, mode=mode)
            times = [
                _time_call(
                    _bench,
                    stage3,
                    tree,
                    aggregate,
                    iterations=args.iterations,
                    max_workers=max_workers,
                    mode=mode,
                )
                for _ in range(args.runs)
            ]
            mean_ms = sum(times) / len(times) * 1000.0
            if mode == "python" and workers == 1:
                reference = mean_ms
            speedup = reference / mean_ms if reference and mean_ms > 0.0 else 1.0
            samples.append(BenchSample(mode=mode, workers=workers, mean_ms=mean_ms))
            print(f"{mode:<6} | {workers:>7} | {mean_ms:>7.3f} | {speedup:>16.2f}x")

    if args.profile:
        _run_profile(stage3, tree, aggregate, iterations=args.iterations)


def _bench(stage3, tree, aggregate, *, iterations: int, max_workers: int | None, mode: str) -> None:
    for _ in range(iterations):
        if mode == "python":
            _compute_opponent_reach_python_reference(stage3, tree, aggregate, max_workers=max_workers)
        else:
            stage3.compute_opponent_reach(tree, aggregate, max_workers=max_workers)


def _run_profile(stage3, tree, aggregate, *, iterations: int) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    _bench(stage3, tree, aggregate, iterations=iterations, max_workers=16, mode="numba")
    profiler.disable()
    stream = StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(80)
    print(stream.getvalue())


def _time_call(func, *args, **kwargs) -> float:
    start = perf_counter()
    func(*args, **kwargs)
    return perf_counter() - start


def _compute_opponent_reach_python_reference(stage3, tree, aggregate, *, max_workers: int | None) -> None:
    hand_reach_rows = aggregate.node_aggregate.hand_reach
    if len(hand_reach_rows) == 0:
        return
    infoset_nodes = build_dense_infoset_table(tree).infoset_nodes
    hand_width = int(np.asarray(hand_reach_rows[0]).shape[0])
    node_hand_rows = [tuple(0.0 for _ in range(hand_width)) for _ in range(tree.node_count)]
    node_share = [0.0 for _ in range(tree.node_count)]
    for nodes in infoset_nodes:
        reach_total = 0.0
        card_total = [0.0 for _ in range(52)]
        hand_total = [0.0 for _ in range(hand_width)]
        node_card_rows: list[tuple[float, ...]] = []
        node_hand_rows_local: list[tuple[float, ...]] = []
        for node_index in nodes:
            reach_total += aggregate.node_aggregate.reach[node_index]
            node_card_reach = aggregate.node_aggregate.card_reach[node_index]
            node_hand_reach = aggregate.node_aggregate.hand_reach[node_index]
            for card_index, value in enumerate(node_card_reach):
                card_total[card_index] += value
            for hand_index, value in enumerate(node_hand_reach):
                hand_total[hand_index] += value
            node_card_rows.append(node_card_reach)
            node_hand_rows_local.append(node_hand_reach)
        hand_total_tuple = tuple(hand_total)
        if reach_total > 0.0:
            for node_index in nodes:
                node_share[node_index] = aggregate.node_aggregate.reach[node_index] / reach_total
        else:
            for node_index in nodes:
                node_share[node_index] = 1.0 / len(nodes)
        for node_index, row in zip(nodes, node_hand_rows_local, strict=True):
            node_hand_rows[node_index] = row


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
