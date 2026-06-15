from __future__ import annotations

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
    make_kuhn_public_tree,
    propagate_reach,
)
from pokergpu.cfr.solver.strategy_update import apply_dense_solver_strategy_update  # noqa: E402
from pokergpu.tree.public_tree import NodeId  # noqa: E402


@dataclass(slots=True)
class BenchSample:
    build_table_s: float
    reach_s: float
    action_values_s: float
    stage7_s: float
    total_s: float


def measure_once(max_workers: int | None) -> BenchSample:
    tree = make_kuhn_public_tree()
    t0 = time.perf_counter()
    table = build_dense_infoset_table(tree)
    t1 = time.perf_counter()
    reach = propagate_reach(tree, infoset_table=table, max_workers=max_workers)
    t2 = time.perf_counter()
    action_values = tuple(
        aggregate_action_values(tree, NodeId(node_index))
        for node_index in table.infoset_to_node
        if node_index >= 0
    )
    t3 = time.perf_counter()
    state = DenseCfrState(
        regret_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
        strategy_sums=tuple((0.0, 0.0) for _ in range(table.infoset_count)),
    )
    _ = apply_dense_solver_strategy_update(
        state,
        action_values,
        infoset_table=table,
        reach_weights=reach.infoset_reach if reach.infoset_reach else None,
        max_workers=max_workers,
    )
    t4 = time.perf_counter()
    return BenchSample(
        build_table_s=t1 - t0,
        reach_s=t2 - t1,
        action_values_s=t3 - t2,
        stage7_s=t4 - t3,
        total_s=t4 - t0,
    )


def summarize(samples: list[BenchSample]) -> BenchSample:
    count = float(len(samples))
    return BenchSample(
        build_table_s=sum(sample.build_table_s for sample in samples) / count,
        reach_s=sum(sample.reach_s for sample in samples) / count,
        action_values_s=sum(sample.action_values_s for sample in samples) / count,
        stage7_s=sum(sample.stage7_s for sample in samples) / count,
        total_s=sum(sample.total_s for sample in samples) / count,
    )


def main() -> None:
    worker_counts = (1, 2, 4, 8, 16)
    warmup_runs = 3
    timed_runs = 10

    print("Kuhn dense benchmark")
    print(f"warmup_runs={warmup_runs} timed_runs={timed_runs}")
    print()
    print(
        "workers | build_table_ms | reach_ms | action_values_ms | stage7_ms | total_ms"
    )
    print("-" * 78)

    for workers in worker_counts:
        max_workers = None if workers == 1 else workers
        for _ in range(warmup_runs):
            measure_once(max_workers)
        samples = [measure_once(max_workers) for _ in range(timed_runs)]
        summary = summarize(samples)
        print(
            f"{workers:>7} | "
            f"{summary.build_table_s * 1000:>13.3f} | "
            f"{summary.reach_s * 1000:>8.3f} | "
            f"{summary.action_values_s * 1000:>16.3f} | "
            f"{summary.stage7_s * 1000:>9.3f} | "
            f"{summary.total_s * 1000:>8.3f}"
        )


if __name__ == "__main__":
    main()
