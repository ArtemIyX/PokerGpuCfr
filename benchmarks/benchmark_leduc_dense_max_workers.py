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

from pokergpu.cfr.solver import (  # noqa: E402
    DenseCfrState,
    aggregate_action_values,
    build_dense_infoset_table,
    make_leduc_public_tree,
    propagate_reach,
)
from pokergpu.cfr.solver.strategy_update import apply_dense_solver_strategy_update  # noqa: E402
from pokergpu.tree.public_tree import NodeId  # noqa: E402


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


def measure_once(max_workers: int | None, iterations: int) -> BenchSample:
    tree = make_leduc_public_tree()
    t0 = time.perf_counter()
    table = build_dense_infoset_table(tree)
    t1 = time.perf_counter()
    state_t0 = time.perf_counter()
    state = make_state(table)
    state_t1 = time.perf_counter()
    reach_total = 0.0
    action_total = 0.0
    stage7_total = 0.0
    for _ in range(iterations):
        step0 = time.perf_counter()
        reach = propagate_reach(tree, infoset_table=table, max_workers=max_workers)
        step1 = time.perf_counter()
        action_values = tuple(
            aggregate_action_values(tree, NodeId(node_index))
            for node_index in table.infoset_to_node
            if node_index >= 0
        )
        step2 = time.perf_counter()
        state = apply_dense_solver_strategy_update(
            state,
            action_values,
            infoset_table=table,
            reach_weights=reach.infoset_reach if reach.infoset_reach else None,
            max_workers=max_workers,
        )
        step3 = time.perf_counter()
        reach_total += step1 - step0
        action_total += step2 - step1
        stage7_total += step3 - step2
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="number of dense CFR iterations to run per benchmark sample",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print detailed per-category timings",
    )
    args = parser.parse_args()

    worker_counts = (1, 2, 4, 8, 16)
    warmup_runs = 3
    timed_runs = 10

    print("Leduc dense benchmark")
    print(f"iterations={args.iterations}")
    print(f"warmup_runs={warmup_runs} timed_runs={timed_runs}")
    print()
    if args.profile:
        print(
            "workers | build_table_ms | state_ms | reach_ms | action_values_ms | stage7_ms | total_ms"
        )
        print("-" * 104)
    else:
        print("workers | build_table_ms | per_iter_ms | total_ms")
        print("-" * 78)

    for workers in worker_counts:
        max_workers = None if workers == 1 else workers
        for _ in range(warmup_runs):
            measure_once(max_workers, args.iterations)
        samples = [measure_once(max_workers, args.iterations) for _ in range(timed_runs)]
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
