from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter


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
    "run_benchmark",
]
