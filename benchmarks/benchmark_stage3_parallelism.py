from __future__ import annotations

import argparse
import cProfile
import importlib
import os
import pstats
import sys
from concurrent.futures import ThreadPoolExecutor
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
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.infosets import build_dense_infoset_table
from pokergpu.cfr.solver.leduc import make_leduc_public_tree
from pokergpu.abstraction.hands import private_hand_count
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
    print("mode   | workers | compile_ms | steady_ms | speedup_vs_python")
    print("-" * 66)

    reference = None
    for mode in ("python", "numba"):
        for workers in (1, 2, 4, 8, 16):
            max_workers = None if workers == 1 else workers
            compile_ms = 0.0
            if mode == "numba":
                compile_ms = _time_call(
                    _bench,
                    stage3,
                    tree,
                    aggregate,
                    iterations=1,
                    max_workers=max_workers,
                    mode=mode,
                ) * 1000.0
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
            print(
                f"{mode:<6} | {workers:>7} | {compile_ms:>10.3f} | {mean_ms:>9.3f} | {speedup:>16.2f}x"
            )

    if args.profile:
        _run_profile(stage3, tree, aggregate, iterations=args.iterations)


def _bench(stage3, tree, aggregate, *, iterations: int, max_workers: int | None, mode: str) -> None:
    for _ in range(iterations):
        if mode == "python":
            _compute_opponent_reach_python_reference(tree, aggregate, max_workers=max_workers)
        else:
            compute_opponent_reach(tree, aggregate, max_workers=max_workers)


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


def _compute_opponent_reach_python_reference(tree: PublicTree, aggregate, *, max_workers: int | None) -> None:
    hand_reach_rows = aggregate.node_aggregate.hand_reach
    if len(hand_reach_rows) == 0:
        return
    infoset_nodes = build_dense_infoset_table(tree).infoset_nodes
    hand_width = int(np.asarray(hand_reach_rows[0]).shape[0])
    node_opponent_share = [0.0 for _ in range(tree.node_count)]
    node_hand_opponent_reach = [tuple(0.0 for _ in range(hand_width)) for _ in range(tree.node_count)]
    infoset_opponent_reach = [0.0 for _ in range(len(infoset_nodes))]
    infoset_card_opponent_reach = [tuple(0.0 for _ in range(52)) for _ in range(len(infoset_nodes))]
    infoset_hand_opponent_reach = [tuple(0.0 for _ in range(hand_width)) for _ in range(len(infoset_nodes))]
    infoset_node_hand_ratio: list[tuple[tuple[float, ...], ...]] = [tuple() for _ in range(len(infoset_nodes))]
    infoset_node_card_ratio: list[tuple[tuple[float, ...], ...]] = [tuple() for _ in range(len(infoset_nodes))]

    def process_infoset(infoset_id: int):
        nodes = infoset_nodes[infoset_id]
        if not nodes:
            return infoset_id, 0.0, (), (), (), (), (), []

        reach_total = 0.0
        card_reach = [0.0 for _ in range(52)]
        hand_reach = [0.0 for _ in range(hand_width)]
        node_card_reach_rows: list[tuple[float, ...]] = []
        node_hand_reach_rows: list[tuple[float, ...]] = []
        for node_index in nodes:
            reach_total += aggregate.node_aggregate.reach[node_index]
            node_card_reach = aggregate.node_aggregate.card_reach[node_index]
            node_hand_reach = aggregate.node_aggregate.hand_reach[node_index]
            for card_index, value in enumerate(node_card_reach):
                card_reach[card_index] += value
            for hand_index, value in enumerate(node_hand_reach):
                hand_reach[hand_index] += value
            node_card_reach_rows.append(node_card_reach)
            node_hand_reach_rows.append(node_hand_reach)

        card_total = tuple(card_reach)
        hand_total = tuple(hand_reach)
        node_card_ratio_rows = []
        for row in node_card_reach_rows:
            ratios = []
            for card_index, value in enumerate(row):
                total = card_total[card_index]
                ratios.append(0.0 if total <= 0.0 else value / total)
            node_card_ratio_rows.append(tuple(ratios))
        node_hand_ratio_rows = []
        for row in node_hand_reach_rows:
            ratios = []
            for hand_index, value in enumerate(row):
                total = hand_total[hand_index]
                ratios.append(0.0 if total <= 0.0 else value / total)
            node_hand_ratio_rows.append(tuple(ratios))
        node_hand_rows = []
        for row in node_hand_reach_rows:
            exact = []
            for hand_index, value in enumerate(row):
                total = hand_total[hand_index]
                exact.append(0.0 if total <= 0.0 else value / total)
            node_hand_rows.append(tuple(exact))
        if reach_total > 0.0:
            node_shares = [(node_index, aggregate.node_aggregate.reach[node_index] / reach_total) for node_index in nodes]
        else:
            uniform_share = 1.0 / len(nodes)
            node_shares = [(node_index, uniform_share) for node_index in nodes]
        return (
            infoset_id,
            reach_total,
            tuple(card_reach),
            tuple(hand_reach),
            tuple(node_hand_ratio_rows),
            tuple(node_hand_rows),
            tuple(node_card_ratio_rows),
            node_shares,
        )

    infoset_ids = list(range(len(infoset_nodes)))
    if max_workers is None or max_workers <= 1 or len(infoset_ids) <= 1:
        results = [process_infoset(infoset_id) for infoset_id in infoset_ids]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_infoset, infoset_ids))

    for infoset_id, reach_total, card_reach, hand_reach, node_hand_ratios, node_hand_rows, node_card_ratios, node_shares in results:
        infoset_opponent_reach[infoset_id] = reach_total
        infoset_card_opponent_reach[infoset_id] = card_reach
        infoset_hand_opponent_reach[infoset_id] = hand_reach
        infoset_node_hand_ratio[infoset_id] = node_hand_ratios
        infoset_node_card_ratio[infoset_id] = node_card_ratios
        for node_index, node_hand_row in zip(infoset_nodes[infoset_id], node_hand_rows, strict=True):
            node_hand_opponent_reach[node_index] = node_hand_row
        for node_index, share in node_shares:
            node_opponent_share[node_index] = share


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
