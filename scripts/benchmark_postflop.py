from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for benchmarking: {exc}") from exc

from pokergpu.abstraction.hands import RangeVector
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.board import Board
from pokergpu.core.cards import Card, make_deck
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu_many


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    spots: int
    repeats: int
    warmup: int
    seed: int
    max_nodes: tuple[int, ...]
    max_depth: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    time_budget: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spots", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="16,64,256")
    parser.add_argument("--depths", type=str, default="2,3")
    parser.add_argument("--nodes", type=str, default="128,512")
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()

    config = BenchmarkConfig(
        spots=max(1, args.spots),
        repeats=max(1, args.repeats),
        warmup=max(0, args.warmup),
        seed=args.seed,
        batch_sizes=_parse_ints(args.batch_sizes),
        max_depth=_parse_ints(args.depths),
        max_nodes=_parse_ints(args.nodes),
        time_budget=max(0.0, args.time_budget),
    )
    spots = _make_spot_pool(config.spots, seed=config.seed)

    print(f"spots={config.spots}")
    print(f"repeats={config.repeats}")
    print(f"warmup={config.warmup}")
    print(f"time_budget_sec={config.time_budget:.3f}")
    print(f"batch_sizes={list(config.batch_sizes)}")
    print(f"depths={list(config.max_depth)}")
    print(f"nodes={list(config.max_nodes)}")

    if args.device in {"cpu", "both"}:
        for depth in config.max_depth:
            for nodes in config.max_nodes:
                cpu_seconds = _benchmark_cpu(spots, repeats=config.repeats, warmup=config.warmup, max_depth=depth, max_nodes=nodes)
                cpu_spots_per_second = config.spots * config.repeats / cpu_seconds
                print(f"cpu depth={depth} nodes={nodes} seconds={cpu_seconds:.6f} spots_per_second={cpu_spots_per_second:.3f}")

    if args.device in {"cuda", "both"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        for depth in config.max_depth:
            for nodes in config.max_nodes:
                for batch_size in config.batch_sizes:
                    cuda_seconds = _benchmark_cuda(
                        spots,
                        repeats=config.repeats,
                        warmup=config.warmup,
                        max_depth=depth,
                        max_nodes=nodes,
                        batch_size=batch_size,
                    )
                    solved = batch_size * config.repeats
                    cuda_spots_per_second = solved / cuda_seconds
                    print(
                        f"cuda depth={depth} nodes={nodes} batch={batch_size} "
                        f"seconds={cuda_seconds:.6f} spots_per_second={cuda_spots_per_second:.3f}"
                    )

    return 0


def _benchmark_cpu(
    spots: tuple[PostflopResolveSpec, ...],
    *,
    repeats: int,
    warmup: int,
    max_depth: int,
    max_nodes: int,
) -> float:
    specs = _rebuild_specs(spots, max_depth=max_depth, max_nodes=max_nodes)
    for _ in range(warmup):
        for spec in specs:
            resolve_postflop_hu(spec)
    started = time.perf_counter()
    for _ in range(repeats):
        for spec in specs:
            resolve_postflop_hu(spec)
    return time.perf_counter() - started


def _benchmark_cuda(
    spots: tuple[PostflopResolveSpec, ...],
    *,
    repeats: int,
    warmup: int,
    max_depth: int,
    max_nodes: int,
    batch_size: int,
) -> float:
    specs = _rebuild_specs(spots, max_depth=max_depth, max_nodes=max_nodes)
    batches = _chunk_specs(specs, batch_size)
    for _ in range(warmup):
        for batch in batches:
            resolve_postflop_gpu_many(batch)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        for batch in batches:
            resolve_postflop_gpu_many(batch)
        torch.cuda.synchronize()
    return time.perf_counter() - started


def _rebuild_specs(
    spots: tuple[PostflopResolveSpec, ...],
    *,
    max_depth: int,
    max_nodes: int,
) -> tuple[PostflopResolveSpec, ...]:
    return tuple(
        PostflopResolveSpec(
            state=spec.state,
            range_p0=spec.range_p0,
            range_p1=spec.range_p1,
            time_budget_sec=0.0,
            seed=spec.seed,
            max_depth=max_depth,
            max_nodes=max_nodes,
            min_reach_prob=spec.min_reach_prob,
        )
        for spec in spots
    )


def _chunk_specs(
    specs: tuple[PostflopResolveSpec, ...],
    batch_size: int,
) -> tuple[tuple[PostflopResolveSpec, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(specs[index : index + batch_size] for index in range(0, len(specs), batch_size))


def _make_spot_pool(count: int, *, seed: int) -> tuple[PostflopResolveSpec, ...]:
    rng = random.Random(seed)
    return tuple(_make_spec(rng.randint(0, 2**31 - 1)) for _ in range(count))


def _make_spec(seed: int) -> PostflopResolveSpec:
    return PostflopResolveSpec(
        state=_make_spot(seed),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        seed=seed,
        max_depth=2,
        max_nodes=128,
    )


def _make_spot(seed: int) -> GameState:
    rng = random.Random(seed)
    deck = list(make_deck())
    rng.shuffle(deck)
    board = Board(cards=tuple(deck[:3]))
    hero = (deck[3], deck[4])
    villain = (deck[5], deck[6])
    return GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=hero),
            PlayerState(player=PlayerIndex(1), hole_cards=villain),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(300)),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=chips(1700)),
                PlayerStack(player=PlayerIndex(1), stack=chips(1700)),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("expected at least one integer")
    return tuple(int(part) for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
