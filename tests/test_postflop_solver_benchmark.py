from __future__ import annotations

from random import Random
from time import perf_counter

import numpy as np
import pytest
import torch
from tqdm import tqdm

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
from pokergpu.core.board import Board, Street
from pokergpu.core.cards import Card, make_deck
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu, resolve_postflop_gpu_many


pytestmark = pytest.mark.benchmark_suite


def test_postflop_cpu_cuda_average_solve_time() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this benchmark")

    seeds = tuple(range(1, 5))
    batch_size = 16
    max_depth = 2
    max_nodes = 128
    iterations = 3

    cpu_times: list[float] = []
    cuda_times: list[float] = []
    total_runs = len(seeds) * iterations

    for seed in tqdm(seeds, desc="cpu seeds", total=len(seeds)):
        print(f"cpu seed={seed}")
        state = _make_spot(seed)
        spec = PostflopResolveSpec(
            state=state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.0,
            seed=seed,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

        resolve_postflop_hu(spec)
        started = perf_counter()
        for _ in range(iterations):
            resolve_postflop_hu(spec)
        cpu_times.append(perf_counter() - started)

    for seed in tqdm(seeds, desc="cuda seeds", total=len(seeds)):
        print(f"cuda seed={seed}")
        specs = tuple(
            _make_spec(seed + offset, max_depth=max_depth, max_nodes=max_nodes)
            for offset in range(batch_size)
        )
        resolve_postflop_gpu_many(specs, allow_cpu_fallback=False)
        torch.cuda.synchronize()
        started = perf_counter()
        for _ in range(iterations):
            resolve_postflop_gpu_many(specs, allow_cpu_fallback=False)
        torch.cuda.synchronize()
        cuda_times.append((perf_counter() - started) / iterations)

    cpu_avg = float(np.mean(cpu_times, dtype=np.float64))
    cuda_avg = float(np.mean(cuda_times, dtype=np.float64))

    print(f"seeds={list(seeds)}")
    print(f"batch_size={batch_size}")
    print(f"iterations={iterations}")
    print(f"total_runs={total_runs}")
    print("time_budget_sec=0.0")
    print(f"max_depth={max_depth}")
    print(f"max_nodes={max_nodes}")
    print("mode=batch_throughput")
    print(f"cpu_avg_seconds={cpu_avg:.6f}")
    print(f"cuda_avg_seconds={cuda_avg:.6f}")
    print(f"speedup={cpu_avg / cuda_avg:.3f}")
    print(f"cpu_throughput_sps={1.0 / cpu_avg:.3f}")
    print(f"cuda_throughput_sps={1.0 / cuda_avg:.3f}")
    print(f"batch_work={batch_size}x depth={max_depth} nodes={max_nodes}")


def _make_spot(seed: int) -> GameState:
    rng = Random(seed)
    board = _sample_board(rng)
    hole_cards = _sample_hole_cards(rng, board.cards)
    return GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0), hole_cards=hole_cards[0]),
            PlayerState(player=PlayerIndex(1), hole_cards=hole_cards[1]),
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
        phase=HandPhase.IN_PROGRESS,
    )


def _make_spec(seed: int, *, max_depth: int, max_nodes: int) -> PostflopResolveSpec:
    return PostflopResolveSpec(
        state=_make_spot(seed),
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.0,
        seed=seed,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def _sample_board(rng: Random) -> Board:
    deck = list(make_deck())
    rng.shuffle(deck)
    return Board(cards=tuple(deck[:3]))


def _sample_hole_cards(rng: Random, board_cards: tuple[Card, ...]) -> tuple[tuple[Card, Card], tuple[Card, Card]]:
    deck = [card for card in make_deck() if card not in board_cards]
    rng.shuffle(deck)
    hero = (deck.pop(), deck.pop())
    villain = (deck.pop(), deck.pop())
    return hero, villain
