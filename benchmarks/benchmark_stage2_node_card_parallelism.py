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

from pokergpu.cfr.stage1 import ForwardProfileResult, propagate_forward  # noqa: E402
from pokergpu.cfr.stage2 import aggregate_prob_sum  # noqa: E402
from pokergpu.cfr.solver import make_kuhn_public_tree, make_leduc_public_tree  # noqa: E402


@dataclass(slots=True)
class Stage2Sample:
    sequential_s: float
    parallel_s: float
    speedup: float


def make_forward(tree) -> ForwardProfileResult:
    return propagate_forward(tree)


def measure_once(tree, *, iterations: int, workers: int | None) -> float:
    forward = make_forward(tree)
    start = time.perf_counter()
    for _ in range(iterations):
        aggregate_prob_sum(tree, forward, max_workers=workers)
    return time.perf_counter() - start


def summarize(samples: list[Stage2Sample]) -> Stage2Sample:
    count = float(len(samples))
    return Stage2Sample(
        sequential_s=sum(sample.sequential_s for sample in samples) / count,
        parallel_s=sum(sample.parallel_s for sample in samples) / count,
        speedup=sum(sample.speedup for sample in samples) / count,
    )


def benchmark_game(name: str, tree, *, iterations: int, parallel_workers: int) -> Stage2Sample:
    warmup_runs = 3
    timed_runs = 10

    for _ in range(warmup_runs):
        measure_once(tree, iterations=iterations, workers=None)
        measure_once(tree, iterations=iterations, workers=parallel_workers)

    samples = [
        Stage2Sample(
            sequential_s=measure_once(tree, iterations=iterations, workers=None),
            parallel_s=measure_once(tree, iterations=iterations, workers=parallel_workers),
            speedup=0.0,
        )
        for _ in range(timed_runs)
    ]
    for sample in samples:
        sample.speedup = sample.sequential_s / sample.parallel_s if sample.parallel_s > 0.0 else 0.0
    summary = summarize(samples)
    print(f"{name} stage2 benchmark")
    print(f"iterations={iterations} parallel_workers={parallel_workers}")
    print(f"sequential_ms={summary.sequential_s * 1000:.3f}")
    print(f"parallel_ms={summary.parallel_s * 1000:.3f}")
    print(f"speedup={summary.speedup:.3f}x")
    print()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    benchmark_game("Kuhn", make_kuhn_public_tree(), iterations=args.iterations, parallel_workers=args.workers)
    benchmark_game("Leduc", make_leduc_public_tree(), iterations=args.iterations, parallel_workers=args.workers)


if __name__ == "__main__":
    main()
