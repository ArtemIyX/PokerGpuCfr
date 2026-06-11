from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from .caching_warm_start import main as caching_warm_start_main
from .caching_warm_start import run_caching_benchmark as run_caching_benchmark


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    total_seconds: float

    @property
    def seconds_per_iteration(self) -> float:
        return self.total_seconds / self.iterations


def run_benchmark(
    name: str,
    func: Callable[[], object],
    *,
    iterations: int = 1_000,
) -> BenchmarkResult:
    start = perf_counter()
    for _ in range(iterations):
        func()
    total_seconds = perf_counter() - start
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_seconds=total_seconds,
    )




__all__ = [
    "BenchmarkResult",
    "caching_warm_start_main",
    "run_benchmark",
    "run_caching_benchmark",
]
