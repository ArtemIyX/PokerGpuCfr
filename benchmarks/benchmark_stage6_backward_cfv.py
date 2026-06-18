from __future__ import annotations

import argparse
import cProfile
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

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.solver import make_toy_public_tree
from pokergpu.cfr.solver import run_dense_backward_cfv_iterations
from pokergpu.tree.public_tree import InfosetId
from pokergpu.tree.public_tree import PublicTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--board", type=str, default="")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("POKERGPU_STAGE2_NUMBA", "1")
    tree = make_toy_public_tree()
    table = build_dense_infoset_table(tree)
    backend = GpuLeafBackend(kernel=_FakeLeafKernel())
    state = _make_state(table.action_counts)
    strategies = _make_uniform_strategies(tree)

    print("Stage 6 backward CFV benchmark")
    print(f"tree_nodes={tree.node_count} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms")
    print("-" * 20)

    _bench(
        tree,
        state,
        backend=backend,
        infoset_strategies=strategies,
        iterations=args.warmup,
        max_workers=args.workers,
    )
    samples = [
        _time_call(
            _bench,
            tree,
            state,
            backend=backend,
            infoset_strategies=strategies,
            iterations=args.iterations,
            max_workers=args.workers,
        )
        for _ in range(args.runs)
    ]
    mean_ms = sum(samples) / len(samples) * 1000.0
    print(f"{args.workers:>7} | {mean_ms:>7.3f}")

    if args.profile:
        _run_profile(
            tree,
            state,
            backend=backend,
            infoset_strategies=strategies,
            iterations=max(1, args.profile_iterations),
            max_workers=args.workers,
        )


def _bench(
    tree: PublicTree,
    state: DenseCfrState,
    *,
    backend: GpuLeafBackend,
    infoset_strategies: dict[InfosetId, tuple[float, ...]],
    iterations: int,
    max_workers: int,
) -> None:
    current = state
    for _ in range(iterations):
        current = run_dense_backward_cfv_iterations(
            tree,
            current,
            1,
            backend=backend,
            infoset_strategies=infoset_strategies,
            max_workers=max_workers,
        )


def _run_profile(
    tree: PublicTree,
    state: DenseCfrState,
    *,
    backend: GpuLeafBackend,
    infoset_strategies: dict[InfosetId, tuple[float, ...]],
    iterations: int,
    max_workers: int,
) -> None:
    _bench(
        tree,
        state,
        backend=backend,
        infoset_strategies=infoset_strategies,
        iterations=1,
        max_workers=max_workers,
    )
    profiler = cProfile.Profile()
    profiler.enable()
    _bench(
        tree,
        state,
        backend=backend,
        infoset_strategies=infoset_strategies,
        iterations=iterations,
        max_workers=max_workers,
    )
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
    stats.print_callees("pokergpu.cfr.stage6")
    print(stream.getvalue())


def _time_call(func, *args, **kwargs) -> float:
    start = perf_counter()
    func(*args, **kwargs)
    return perf_counter() - start


def _make_state(action_counts: tuple[int, ...]) -> DenseCfrState:
    return DenseCfrState(
        regret_sums=tuple(tuple(0.0 for _ in range(action_count)) for action_count in action_counts),
        strategy_sums=tuple(tuple(0.0 for _ in range(action_count)) for action_count in action_counts),
    )


def _make_uniform_strategies(tree: PublicTree) -> dict[InfosetId, tuple[float, ...]]:
    strategies: dict[InfosetId, tuple[float, ...]] = {}
    for node_index, infoset_id in enumerate(tree.infoset_ids):
        if infoset_id is None:
            continue
        child_count = tree.child_count[node_index]
        if child_count > 0:
            strategies[infoset_id] = tuple(1.0 / child_count for _ in range(child_count))
    return strategies
class _FakeLeafKernel:
    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)


if __name__ == "__main__":
    main()
