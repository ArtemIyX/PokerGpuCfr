from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .interface import LeafEvaluator
from .types import LeafFeatureBatch


@dataclass(slots=True, frozen=True)
class LeafBatchBenchmarkResult:
    batch_size: int
    elapsed_seconds: float
    leaves_per_second: float


def measure_leaf_batch_throughput(
    evaluator: LeafEvaluator,
    batch: LeafFeatureBatch,
    *,
    repeats: int = 10,
) -> LeafBatchBenchmarkResult:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    start = perf_counter()
    for _ in range(repeats):
        evaluator.evaluate(batch)
    elapsed_seconds = perf_counter() - start
    leaves_per_second = float(batch.size * repeats / elapsed_seconds)
    return LeafBatchBenchmarkResult(
        batch_size=batch.size,
        elapsed_seconds=elapsed_seconds,
        leaves_per_second=leaves_per_second,
    )
