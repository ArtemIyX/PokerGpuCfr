from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokergpu.cfr.gpu_leaf_backend import GpuLeafBackend  # noqa: E402
from pokergpu.cfr.gpu_leaf_backend import GpuLeafModelSpec  # noqa: E402
from pokergpu.cfr.gpu_leaf_backend import TorchLeafKernel  # noqa: E402
from pokergpu.cfr.leaf_eval import LEAF_EVAL_FEATURE_WIDTH  # noqa: E402
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput  # noqa: E402
from pokergpu.cfr.triton_leaf_backend import TritonLeafKernel  # noqa: E402


@dataclass(slots=True)
class BenchmarkSample:
    backend: str
    mean_ms: float
    runs: int


class TorchCpuLeafKernel:
    def __init__(self, spec: GpuLeafModelSpec) -> None:
        import torch

        self.spec = spec
        layers: list[torch.nn.Module] = []
        in_width = spec.input_width
        for hidden_width in spec.hidden_widths:
            layers.append(torch.nn.Linear(in_width, hidden_width))
            layers.append(torch.nn.ReLU())
            in_width = hidden_width
        layers.append(torch.nn.Linear(in_width, spec.output_width))
        self.network = torch.nn.Sequential(*layers)

    def __call__(self, batch: LeafEvalBatchInput):
        import torch

        features = torch.as_tensor(batch.features, device="cpu", dtype=torch.float32)
        outputs = self.network(features)
        values = outputs.detach().to(device="cpu", dtype=torch.float32).numpy()
        from pokergpu.cfr.leaf_eval import LeafEvalBatchOutput

        return LeafEvalBatchOutput(node_ids=batch.node_ids, values=values)


def _parse_hidden_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(part) for part in value.split(",") if part.strip())
    if not widths:
        raise argparse.ArgumentTypeError("hidden widths must not be empty")
    if any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("hidden widths must be positive")
    return widths


def make_batch(batch_size: int) -> LeafEvalBatchInput:
    features = np.arange(batch_size * LEAF_EVAL_FEATURE_WIDTH, dtype=np.float32).reshape(
        batch_size,
        LEAF_EVAL_FEATURE_WIDTH,
    )
    features = features / 1000.0
    node_ids = tuple(range(batch_size))
    return LeafEvalBatchInput(node_ids=node_ids, features=features)


def _run_backend(backend: GpuLeafBackend, batch: LeafEvalBatchInput, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        backend.evaluate(batch)
    return time.perf_counter() - start


def benchmark_backend(
    backend_name: str,
    backend: GpuLeafBackend,
    batch: LeafEvalBatchInput,
    *,
    warmup_runs: int,
    timed_runs: int,
    iterations: int,
) -> BenchmarkSample:
    for _ in range(warmup_runs):
        _run_backend(backend, batch, iterations)
    samples = [_run_backend(backend, batch, iterations) for _ in range(timed_runs)]
    mean_seconds = sum(samples) / float(len(samples))
    return BenchmarkSample(
        backend=backend_name,
        mean_ms=(mean_seconds / max(1, iterations)) * 1000.0,
        runs=timed_runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument(
        "--hidden-widths",
        type=_parse_hidden_widths,
        default=(256,),
        help="comma-separated hidden widths, for example 256 or 512,512",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="number of hidden layers; used when hidden-widths is not set explicitly",
    )
    parser.add_argument(
        "--hidden-width",
        type=int,
        default=256,
        help="hidden width repeated depth times when hidden-widths is not set explicitly",
    )
    args = parser.parse_args()

    hidden_widths = args.hidden_widths
    if args.depth is not None:
        if args.depth <= 0:
            raise ValueError("depth must be positive")
        hidden_widths = tuple(args.hidden_width for _ in range(args.depth))

    spec = GpuLeafModelSpec(hidden_widths=hidden_widths)
    batch = make_batch(args.batch_size)
    samples: list[BenchmarkSample] = []

    print("leaf backend benchmark")
    print(f"batch_size={args.batch_size} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup_runs} timed_runs={args.timed_runs}")
    print(f"model=input:{spec.input_width} hidden:{spec.hidden_widths} output:{spec.output_width}")
    print()
    print("backend | mean_ms")
    print("-" * 24)

    cpu_backend = GpuLeafBackend(kernel=TorchCpuLeafKernel(spec), spec=spec)
    samples.append(
        benchmark_backend(
            "torch-cpu",
            cpu_backend,
            batch,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            iterations=args.iterations,
        )
    )

    if TorchLeafKernel is not None:
        try:
            import torch

            if torch.cuda.is_available():
                cuda_backend = GpuLeafBackend(kernel=TorchLeafKernel(spec=spec), spec=spec)
                samples.append(
                    benchmark_backend(
                        "torch-cuda",
                        cuda_backend,
                        batch,
                        warmup_runs=args.warmup_runs,
                        timed_runs=args.timed_runs,
                        iterations=args.iterations,
                    )
                )
                triton_backend = GpuLeafBackend(kernel=TritonLeafKernel(spec=spec), spec=spec)
                samples.append(
                    benchmark_backend(
                        "triton",
                        triton_backend,
                        batch,
                        warmup_runs=args.warmup_runs,
                        timed_runs=args.timed_runs,
                        iterations=args.iterations,
                    )
                )
            else:
                print("torch-cuda: skipped (CUDA not available)")
                print("triton: skipped (CUDA not available)")
        except ModuleNotFoundError:
            print("torch-cuda: skipped (torch not available)")
            print("triton: skipped (torch not available)")

    for sample in samples:
        print(f"{sample.backend:>10} | {sample.mean_ms:>7.3f}")


if __name__ == "__main__":
    main()
