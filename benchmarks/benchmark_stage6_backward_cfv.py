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
from pokergpu.cfr.infosets import DenseInfosetTable
from pokergpu.cfr.leaf_eval import LEAF_EVAL_OUTPUT_WIDTH
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput
from pokergpu.cfr.leaf_eval import evaluate_leaf_batch
from pokergpu.cfr.stage1 import ForwardProfileResult
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.solver import DenseCfrState
from pokergpu.cfr.solver import build_dense_infoset_table
from pokergpu.cfr.stage3 import compute_opponent_reach
from pokergpu.cfr.stage4 import ShowdownEquityBatchInput
from pokergpu.cfr.stage4 import ShowdownEquityResult
from pokergpu.cfr.stage6 import BackwardCFVInput
from pokergpu.cfr.stage6 import backward_cfv
from pokergpu.cfr.stage7 import apply_dense_backward_cfv_update
from pokergpu.core.betting import Chips
from pokergpu.tree.public_tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=str,
        default="1",
        help="Compatibility flag; Stage 6 benchmark is currently serial-only.",
    )
    args = parser.parse_args()

    os.environ.setdefault("POKERGPU_STAGE2_NUMBA", "1")
    tree = _make_stage6_benchmark_tree()
    table = build_dense_infoset_table(tree)
    backend = GpuLeafBackend(kernel=_FakeLeafKernel())
    state = _make_state(table.action_counts)
    strategies = _make_uniform_strategies(tree)
    forward = ForwardProfileResult(
        node_reach=tuple(1.0 for _ in range(tree.node_count)),
        infoset_reach=tuple(1.0 for _ in range(table.infoset_count)),
        action_reach=tuple(
            strategies.get(infoset_id, ())
            if infoset_id is not None
            else ()
            for infoset_id in tree.infoset_ids
        ),
    )
    aggregate = aggregate_prob_sum(tree, forward)
    opponent = compute_opponent_reach(tree, aggregate)
    leaf_result = evaluate_leaf_batch(build_leaf_eval_batch(aggregate.leaf_batch), backend)
    backward_input = BackwardCFVInput(
        tree=tree,
        forward=forward,
        aggregate=aggregate,
        opponent_reach=opponent,
        showdown=ShowdownEquityResult(
            node_showdown_equity=tuple(0.0 for _ in range(tree.node_count)),
            node_showdown_equity_bb=tuple(0.0 for _ in range(tree.node_count)),
            input_rows=ShowdownEquityBatchInput(rows=()),
            output_rows=(),
        ),
        leaf_values=np.asarray(leaf_result.node_values, dtype=np.float64),
    )
    print("Stage 6 backward CFV benchmark")
    print(f"tree_nodes={tree.node_count} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms | speedup")
    print("-" * 28)

    _bench(backward_input, table, state, iterations=args.warmup, max_workers=None)
    samples = [
        _time_call(
            _bench,
            backward_input,
            table,
            state,
            iterations=args.iterations,
            max_workers=None,
        )
        for _ in range(args.runs)
    ]
    mean_ms = sum(samples) / len(samples) * 1000.0
    print(f"{1:>7} | {mean_ms:>7.3f} | {1.00:>7.2f}x")

    if args.profile:
        _run_profile(
            backward_input,
            table,
            state,
            iterations=max(1, args.profile_iterations),
            max_workers=None,
        )


def _bench(
    backward_input: BackwardCFVInput,
    table: DenseInfosetTable,
    state: DenseCfrState,
    *,
    iterations: int,
    max_workers: int | None,
) -> None:
    current = state
    for _ in range(iterations):
        backward = backward_cfv(
            backward_input,
            max_workers=max_workers,
        )
        current = apply_dense_backward_cfv_update(
            current,
            backward,
            infoset_table=table,
            max_workers=max_workers,
        )


def _run_profile(
    backward_input: BackwardCFVInput,
    table: DenseInfosetTable,
    state: DenseCfrState,
    *,
    iterations: int,
    max_workers: int | None,
) -> None:
    _bench(backward_input, table, state, iterations=1, max_workers=max_workers)
    profiler = cProfile.Profile()
    profiler.enable()
    _bench(backward_input, table, state, iterations=iterations, max_workers=max_workers)
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


def _time_call(func: object, *args: object, **kwargs: object) -> float:
    start = perf_counter()
    assert callable(func)
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


def _make_stage6_benchmark_tree() -> PublicTree:
    depth = 9
    node_types: list[NodeType] = []
    first_child: list[int] = []
    child_count: list[int] = []
    children: list[ChildLink] = []
    infoset_ids: list[InfosetId | None] = []
    terminal_payoffs: list[Chips | None] = []
    leaf_start = 2**depth - 1
    node_count = 2 ** (depth + 1) - 1

    for node_index in range(node_count):
        if node_index < leaf_start:
            node_types.append(NodeType.PLAYER0 if node_index % 2 == 0 else NodeType.PLAYER1)
            infoset_ids.append(InfosetId(node_index))
            terminal_payoffs.append(None)
            first_child.append(2 * node_index)
            child_count.append(2)
        else:
            node_types.append(NodeType.TERMINAL)
            infoset_ids.append(None)
            terminal_payoffs.append(Chips((node_index % 11) - 5))
            first_child.append(len(children))
            child_count.append(0)

    for node_index in range(leaf_start):
        left = 2 * node_index + 1
        right = left + 1
        children.append(ChildLink(child=NodeId(left)))
        children.append(ChildLink(child=NodeId(right)))

    return PublicTree(
        node_types=tuple(node_types),
        first_child=tuple(first_child),
        child_count=tuple(child_count),
        children=tuple(children),
        infoset_ids=tuple(infoset_ids),
        terminal_payoffs=tuple(terminal_payoffs),
    )


class _FakeLeafKernel:
    def __call__(self, batch: LeafEvalBatchInput) -> LeafEvalBatchOutput:
        values = np.full((len(batch.node_ids), LEAF_EVAL_OUTPUT_WIDTH), 0.5, dtype=np.float32)
        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)


if __name__ == "__main__":
    main()
