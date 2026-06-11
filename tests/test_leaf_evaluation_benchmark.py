from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import pytest
import torch

from pokergpu.cfr.traversal import build_leaf_feature_batch
from pokergpu.eval import (
    AsyncLeafEvaluator,
    EvalDeviceConfig,
    CpuStubLeafEvaluator,
    make_leaf_evaluator,
)
from pokergpu.tree import ChildLink, InfosetId, NodeId, NodeType, PublicTree

pytestmark = pytest.mark.benchmark_suite


def test_leaf_evaluation_cpu_single_cpu_multi_gpu_cuda() -> None:
    _run_and_print("base", 65_536, repeats=5, workers=8)


def test_leaf_evaluation_cpu_single_cpu_multi_gpu_cuda_50x() -> None:
    _run_and_print("50x", 65_536 * 50, repeats=5, workers=8)


def _build_batch(batch_size: int):
    tree = PublicTree(
        node_types=(NodeType.PLAYER0, NodeType.LEAF),
        is_frontier=(False, True),
        first_child=(0, 1),
        child_count=(1, 0),
        children=(ChildLink(NodeId(1)),),
        infoset_ids=(InfosetId(0), None),
        terminal_payoffs=(None, None),
    )
    return build_leaf_feature_batch(tree, (1,) * batch_size)


def _run_and_print(name: str, batch_size: int, *, repeats: int, workers: int) -> None:
    batch = _build_batch(batch_size)
    single_seconds = _time_single_thread(batch, repeats=repeats)
    multi_seconds = _time_multi_thread(batch, workers=workers, repeats=repeats)
    gpu_seconds = _time_gpu_cuda(batch, repeats=repeats)

    print(f"\n{name} batches={batch_size} repeats={repeats}")
    print("| mode       | time_ms |")
    print("|------------|---------|")
    print(f"| cpu_single | {single_seconds * 1000:.3f} |")
    print(f"| cpu_multi  | {multi_seconds * 1000:.3f} |")
    print(f"| gpu_cuda   | {gpu_seconds * 1000:.3f} |")

    assert single_seconds >= 0.0
    assert multi_seconds >= 0.0
    assert gpu_seconds >= 0.0


def _time_single_thread(batch, *, repeats: int) -> float:
    evaluator = CpuStubLeafEvaluator()
    start = perf_counter()
    for _ in range(repeats):
        for sub_batch in _split_batch(batch, 4):
            evaluator.evaluate(sub_batch)
    return perf_counter() - start


def _time_multi_thread(batch, *, workers: int, repeats: int) -> float:
    evaluator = AsyncLeafEvaluator(CpuStubLeafEvaluator(), max_workers=workers)
    start = perf_counter()
    sub_batches = _split_batch(batch, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(repeats):
            futures = [
                pool.submit(evaluator.evaluate, sub_batch)
                for sub_batch in sub_batches
            ]
            for future in futures:
                future.result()
    elapsed = perf_counter() - start
    evaluator.close()
    return elapsed


def _time_gpu_cuda(batch, *, repeats: int) -> float:
    if not torch.cuda.is_available():
        return 0.0
    evaluator = make_leaf_evaluator(EvalDeviceConfig(mode="cuda"))
    evaluator.evaluate(batch)
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(repeats):
        evaluator.evaluate(batch)
    torch.cuda.synchronize()
    return perf_counter() - start


def _split_batch(batch, chunks: int):
    size = batch.size
    chunk_size = max(1, (size + chunks - 1) // chunks)
    for start in range(0, size, chunk_size):
        end = min(size, start + chunk_size)
        yield type(batch)(
            node_indices=batch.node_indices[start:end],
            player_to_act=batch.player_to_act[start:end],
            street=batch.street[start:end],
            pot=batch.pot[start:end],
            stack_p0=batch.stack_p0[start:end],
            stack_p1=batch.stack_p1[start:end],
            board_size=batch.board_size[start:end],
            reach_p0=batch.reach_p0[start:end],
            reach_p1=batch.reach_p1[start:end],
            is_terminal=batch.is_terminal[start:end],
            is_frontier=batch.is_frontier[start:end],
            infoset_id=batch.infoset_id[start:end],
        )
