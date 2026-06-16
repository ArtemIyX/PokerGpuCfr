from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from pokergpu.abstraction.hands import private_hand_count, private_hand_mask
from pokergpu.cfr.stage2 import _board_card_features
from pokergpu.core.board import Board


@dataclass(slots=True)
class Sample:
    workers: int
    mean_ms: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args()

    board = Board.from_str("AhKdTc")
    node_reach = np.linspace(0.0, 10.0, args.nodes, dtype=np.float64)
    board_card_mask, _, _ = _board_card_features(board)
    live_hand_mask = private_hand_mask(board.cards)

    print("Stage 2 parallelism benchmark")
    print(f"nodes={args.nodes} iterations={args.iterations}")
    print(f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print("workers | mean_ms | speedup")
    print("-" * 36)

    baseline = None
    for workers in (1, 2, 4, 8, 16):
        for _ in range(args.warmup):
            _run_once(node_reach, board_card_mask, live_hand_mask, workers=workers, iterations=args.iterations)
        samples = [
            _run_once(node_reach, board_card_mask, live_hand_mask, workers=workers, iterations=args.iterations)
            for _ in range(args.runs)
        ]
        mean_ms = sum(samples) / len(samples) * 1000.0
        if baseline is None:
            baseline = mean_ms
        speedup = baseline / mean_ms if mean_ms > 0.0 else 0.0
        print(f"{workers:>7} | {mean_ms:>7.3f} | {speedup:>7.3f}x")


def _run_once(
    node_reach: np.ndarray,
    board_card_mask: tuple[bool, ...],
    live_hand_mask: np.ndarray,
    *,
    workers: int,
    iterations: int,
) -> float:
    start = perf_counter()
    for _ in range(iterations):
        if workers <= 1:
            _build_node_card_reach_serial(node_reach, board_card_mask)
            _build_node_hand_reach_serial(node_reach, live_hand_mask)
        else:
            _build_node_card_reach_threaded(node_reach, board_card_mask, workers=workers)
            _build_node_hand_reach_threaded(node_reach, live_hand_mask, workers=workers)
    return perf_counter() - start


def _build_node_card_reach_serial(
    node_reach: np.ndarray,
    board_card_mask: tuple[bool, ...],
) -> np.ndarray:
    live_mask = np.asarray([not blocked for blocked in board_card_mask], dtype=np.bool_)
    live_count = int(live_mask.sum())
    if live_count <= 0:
        return np.zeros((len(node_reach), 52), dtype=np.float64)
    weights = node_reach / np.float64(live_count)
    return np.where(live_mask[None, :], weights[:, None], np.float64(0.0))


def _build_node_hand_reach_serial(
    node_reach: np.ndarray,
    live_hand_mask: np.ndarray,
) -> np.ndarray:
    live_count = int(live_hand_mask.sum())
    if live_count <= 0:
        return np.zeros((len(node_reach), private_hand_count()), dtype=np.float64)
    weights = node_reach / np.float64(live_count)
    return np.where(live_hand_mask[None, :], weights[:, None], np.float64(0.0))


def _build_node_card_reach_threaded(
    node_reach: np.ndarray,
    board_card_mask: tuple[bool, ...],
    *,
    workers: int,
) -> np.ndarray:
    chunks = np.array_split(node_reach, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(lambda chunk: _build_node_card_reach_serial(chunk, board_card_mask), chunks))
    return np.concatenate(parts, axis=0)


def _build_node_hand_reach_threaded(
    node_reach: np.ndarray,
    live_hand_mask: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    chunks = np.array_split(node_reach, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(lambda chunk: _build_node_hand_reach_serial(chunk, live_hand_mask), chunks))
    return np.concatenate(parts, axis=0)


if __name__ == "__main__":
    main()
