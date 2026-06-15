from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.solver import DenseCfrState, aggregate_action_values, build_dense_infoset_table, make_leduc_public_tree, propagate_reach  # noqa: E402
from pokergpu.cfr.solver.strategy_update import apply_dense_solver_strategy_update  # noqa: E402
from pokergpu.core.betting import Chips  # noqa: E402
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree  # noqa: E402

_WORKER_TREE: PublicTree | None = None
_WORKER_TABLE = None
@dataclass(slots=True)
class BenchSample:
    build_table_s: float
    state_s: float
    reach_s: float
    action_values_s: float
    stage7_s: float
    total_s: float


def make_state(table) -> DenseCfrState:
    return DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )


def make_repeated_tree(base: PublicTree, replicas: int) -> PublicTree:
    if replicas <= 0:
        raise ValueError("replicas must be positive")

    node_types: list[NodeType] = []
    first_child: list[int] = []
    child_count: list[int] = []
    children: list[ChildLink] = []
    infoset_ids: list[InfosetId | None] = []
    terminal_payoffs: list[Chips | None] = []

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
            infoset_ids.append(None if infoset_id is None else InfosetId(int(infoset_id) + infoset_offset))
            terminal_payoffs.append(base.terminal_payoffs[node_index])

        for link in base.children:
            children.append(
                ChildLink(
                    child=NodeId(int(link.child) + node_offset),
                    chance_prob=link.chance_prob,
                )
            )

    return PublicTree(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=tuple(children),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
    )


def _init_worker(replicas: int) -> None:
    global _WORKER_TREE, _WORKER_TABLE
    tree = make_repeated_tree(make_leduc_public_tree(), replicas)
    _WORKER_TREE = tree
    _WORKER_TABLE = build_dense_infoset_table(tree)


def _process_iteration_batch(iteration_count: int) -> tuple[float, float, float]:
    assert _WORKER_TREE is not None, "worker tree not initialized"
    assert _WORKER_TABLE is not None, "worker table not initialized"

    tree = _WORKER_TREE
    table = _WORKER_TABLE
    reach_total = 0.0
    action_total = 0.0
    stage7_total = 0.0

    state = make_state(table)
    action_values = tuple(
        aggregate_action_values(tree, NodeId(node_index))
        for node_index in table.infoset_to_node
        if node_index >= 0
    )
    for _ in range(iteration_count):
        step0 = time.perf_counter()
        reach = propagate_reach(tree, infoset_table=table)
        step1 = time.perf_counter()
        step2 = time.perf_counter()
        apply_dense_solver_strategy_update(
            state,
            action_values,
            infoset_table=table,
            reach_weights=reach.infoset_reach if reach.infoset_reach else None,
        )
        step3 = time.perf_counter()
        reach_total += step1 - step0
        action_total += step2 - step1
        stage7_total += step3 - step2
    return reach_total, action_total, stage7_total


def measure_once(executor: ProcessPoolExecutor | None, iterations: int, replicas: int) -> BenchSample:
    tree = make_repeated_tree(make_leduc_public_tree(), replicas)
    t0 = time.perf_counter()
    table = build_dense_infoset_table(tree)
    t1 = time.perf_counter()
    state_t0 = time.perf_counter()
    _ = make_state(table)
    state_t1 = time.perf_counter()

    if executor is None:
        reach_total = 0.0
        action_total = 0.0
        stage7_total = 0.0
        for _ in range(iterations):
            step0 = time.perf_counter()
            reach = propagate_reach(tree, infoset_table=table)
            step1 = time.perf_counter()
            action_values = tuple(
                aggregate_action_values(tree, NodeId(node_index))
                for node_index in table.infoset_to_node
                if node_index >= 0
            )
            step2 = time.perf_counter()
            state = make_state(table)
            apply_dense_solver_strategy_update(
                state,
                action_values,
                infoset_table=table,
                reach_weights=reach.infoset_reach if reach.infoset_reach else None,
            )
            step3 = time.perf_counter()
            reach_total += step1 - step0
            action_total += step2 - step1
            stage7_total += step3 - step2
    else:
        worker_count = executor._max_workers  # type: ignore[attr-defined]
        batch_size = max(1, (iterations + worker_count - 1) // worker_count)
        batches = tuple(min(batch_size, iterations - index) for index in range(0, iterations, batch_size))
        futures = [executor.submit(_process_iteration_batch, batch) for batch in batches]
        reach_total = 0.0
        action_total = 0.0
        stage7_total = 0.0
        for future in futures:
            reach, action, stage7 = future.result()
            reach_total += reach
            action_total += action
            stage7_total += stage7

    t4 = time.perf_counter()
    return BenchSample(
        build_table_s=t1 - t0,
        state_s=state_t1 - state_t0,
        reach_s=reach_total / max(1, iterations),
        action_values_s=action_total / max(1, iterations),
        stage7_s=stage7_total / max(1, iterations),
        total_s=t4 - t0,
    )


def summarize(samples: list[BenchSample]) -> BenchSample:
    count = float(len(samples))
    return BenchSample(
        build_table_s=sum(sample.build_table_s for sample in samples) / count,
        state_s=sum(sample.state_s for sample in samples) / count,
        reach_s=sum(sample.reach_s for sample in samples) / count,
        action_values_s=sum(sample.action_values_s for sample in samples) / count,
        stage7_s=sum(sample.stage7_s for sample in samples) / count,
        total_s=sum(sample.total_s for sample in samples) / count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1, help="number of dense CFR iterations per sample")
    parser.add_argument("--replicas", type=int, default=256, help="how many copies of the Leduc tree to concatenate")
    parser.add_argument("--profile", action="store_true", help="print detailed per-category timings")
    args = parser.parse_args()

    worker_counts = (1, 2, 4, 8, 16)
    warmup_runs = 3
    timed_runs = 10

    tree = make_repeated_tree(make_leduc_public_tree(), args.replicas)
    table = build_dense_infoset_table(tree)
    print("Leduc dense benchmark")
    print(f"replicas={args.replicas} tree_nodes={tree.node_count} infosets={table.infoset_count}")
    print(f"iterations={args.iterations}")
    print(f"warmup_runs={warmup_runs} timed_runs={timed_runs}")
    print()
    if args.profile:
        print("workers | build_table_ms | state_ms | reach_ms | action_values_ms | stage7_ms | total_ms")
        print("-" * 104)
    else:
        print("workers | build_table_ms | per_iter_ms | total_ms")
        print("-" * 78)

    for workers in worker_counts:
        max_workers = None if workers == 1 else workers
        if max_workers is None:
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(args.replicas,))
        try:
            if executor is not None:
                _init_worker(args.replicas)
            for _ in range(warmup_runs):
                measure_once(executor, args.iterations, args.replicas)
            samples = [measure_once(executor, args.iterations, args.replicas) for _ in range(timed_runs)]
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        summary = summarize(samples)
        if args.profile:
            print(
                f"{workers:>7} | "
                f"{summary.build_table_s * 1000:>13.3f} | "
                f"{summary.state_s * 1000:>8.3f} | "
                f"{summary.reach_s * 1000:>8.3f} | "
                f"{summary.action_values_s * 1000:>16.3f} | "
                f"{summary.stage7_s * 1000:>9.3f} | "
                f"{summary.total_s * 1000:>8.3f}"
            )
        else:
            print(
                f"{workers:>7} | "
                f"{summary.build_table_s * 1000:>13.3f} | "
                f"{summary.reach_s * 1000:>11.3f} | "
                f"{summary.total_s * 1000:>8.3f}"
            )


if __name__ == "__main__":
    main()
