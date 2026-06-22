from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from pokergpu.cfr.leaf_backend_factory import create_heuristic_leaf_backend
from pokergpu.cfr.leaf_backend_factory import create_leaf_backend
from pokergpu.cfr.leaf_eval import LeafEvalBatchInput
from pokergpu.cfr.stage1 import propagate_forward
from pokergpu.cfr.stage2 import aggregate_prob_sum
from pokergpu.cfr.stage2 import build_leaf_eval_batch
from pokergpu.cfr.solver import GameVariant
from pokergpu.cfr.solver import make_game_public_tree
from pokergpu.core.board import Board


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512",
        help="Comma-separated frontier batch sizes to benchmark",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default="preflop,flop,turn,river",
        help="Comma-separated Hold'em street fixtures to benchmark",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    sizes = tuple(int(value) for value in args.sizes.split(",") if value.strip())
    fixtures = tuple(value.strip() for value in args.fixtures.split(",") if value.strip())
    tree = make_game_public_tree(GameVariant.HOLDEM_HU)
    forward = propagate_forward(tree)
    heuristic_backend = create_heuristic_leaf_backend()
    gpu_backend = create_leaf_backend()

    print("Hold'em HU frontier batch benchmark")
    print(f"iterations={args.iterations} warmup={args.warmup}")
    print("fixture | street | base_rows | size | rows | assemble_ms | heuristic_ms | gpu_ms | rows_match")
    print("-" * 100)

    for fixture_name in fixtures:
        board = _board_for_fixture(fixture_name)
        aggregate = aggregate_prob_sum(tree, forward, board)
        base_batch = build_leaf_eval_batch(aggregate.leaf_batch)
        for size in sizes:
            batch = _repeat_batch(base_batch, size)
            _warmup(batch, heuristic_backend, gpu_backend, args.warmup)
            assemble_ms = _time_call(_repeat_batch, base_batch, size, repeat=args.iterations)
            heuristic_ms = _time_backend(heuristic_backend, batch, args.iterations)
            gpu_ms = _time_backend(gpu_backend, batch, args.iterations)
            print(
                f"{fixture_name:<7} | {board.street.value:<6} | {base_batch.features.shape[0]:>9} | "
                f"{size:>4} | {batch.features.shape[0]:>4} | "
                f"{assemble_ms:>11.3f} | {heuristic_ms:>12.3f} | {gpu_ms:>7.3f} | "
                f"{batch.features.shape[0] == len(batch.node_ids)}"
            )


def _warmup(batch: LeafEvalBatchInput, heuristic_backend, gpu_backend, warmup: int) -> None:
    for _ in range(warmup):
        heuristic_backend.evaluate(batch)
        _ = _maybe_evaluate_gpu(gpu_backend, batch)


def _time_backend(backend, batch: LeafEvalBatchInput, iterations: int) -> float:
    start = perf_counter()
    for _ in range(iterations):
        _ = backend.evaluate(batch)
    return (perf_counter() - start) * 1000.0 / float(iterations)


def _time_call(func, *args, repeat: int) -> float:
    start = perf_counter()
    for _ in range(repeat):
        _ = func(*args)
    return (perf_counter() - start) * 1000.0 / float(repeat)


def _maybe_evaluate_gpu(backend, batch: LeafEvalBatchInput):
    try:
        return backend.evaluate(batch)
    except RuntimeError:
        return None


def _repeat_batch(batch: LeafEvalBatchInput, size: int) -> LeafEvalBatchInput:
    if size <= 0:
        raise ValueError("batch size must be positive")
    row_count = batch.features.shape[0]
    if row_count == 0:
        raise ValueError("cannot benchmark an empty frontier batch")
    repeats = (size + row_count - 1) // row_count
    features = np.tile(batch.features, (repeats, 1))[:size].astype(np.float32, copy=False)
    node_ids = tuple(int(index) for index in range(size))
    return LeafEvalBatchInput(node_ids=node_ids, features=features)


def _board_for_fixture(fixture_name: str) -> Board:
    if fixture_name == "preflop":
        return Board(cards=())
    if fixture_name == "flop":
        return Board.from_str("AhKdTc")
    if fixture_name == "turn":
        return Board.from_str("AhKdTc9s")
    if fixture_name == "river":
        return Board.from_str("AhKdTc9s2d")
    raise ValueError(f"unknown fixture: {fixture_name}")


if __name__ == "__main__":
    main()
