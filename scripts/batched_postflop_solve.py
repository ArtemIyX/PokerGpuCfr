from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"torch is required for GPU batch solving: {exc}") from exc

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
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime import PostflopResolveSpec
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu_many


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spots", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--min-reach-prob", type=float, default=0.0)
    parser.add_argument("--board", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rng = random.Random(args.seed)
    specs = tuple(
        _make_spec(
            rng.randint(0, 2**31 - 1),
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            iterations=args.iterations,
            time_budget=args.time_budget,
            min_reach_prob=args.min_reach_prob,
            board_text=args.board or None,
        )
        for _ in range(max(1, args.spots))
    )

    print(f"spots={len(specs)}")
    print(f"batch_size={max(1, args.batch_size)}")
    print(f"repeats={max(1, args.repeats)}")
    print(f"warmup={max(0, args.warmup)}")
    print(f"max_depth={args.max_depth}")
    print(f"max_nodes={args.max_nodes}")
    print(f"iterations={args.iterations}")
    print(f"time_budget_sec={args.time_budget:.3f}")
    print(f"min_reach_prob={args.min_reach_prob}")

    batches = _chunk_specs(specs, max(1, args.batch_size))

    for _ in range(max(0, args.warmup)):
        for batch in batches:
            resolve_postflop_gpu_many(batch)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    started = time.perf_counter()
    total_results = 0
    for _ in range(max(1, args.repeats)):
        for batch in batches:
            results = resolve_postflop_gpu_many(batch)
            total_results += len(results)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"results={total_results}")
    print(f"spots_per_second={total_results / elapsed:.3f}")
    if batches and batches[0]:
        sample = resolve_postflop_gpu_many((batches[0][0],))[0]
        print(f"sample_root_actions={sample.root_actions}")
        print(f"sample_root_strategy={sample.root_strategy.tolist()}")
        print(f"sample_root_action_ev_p0={sample.root_action_ev_player0.tolist()}")
    return 0


def _chunk_specs(
    specs: tuple[PostflopResolveSpec, ...],
    batch_size: int,
) -> tuple[tuple[PostflopResolveSpec, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(specs[index : index + batch_size] for index in range(0, len(specs), batch_size))


def _make_spec(
    seed: int,
    *,
    max_depth: int,
    max_nodes: int,
    iterations: int,
    time_budget: float,
    min_reach_prob: float,
    board_text: str | None,
) -> PostflopResolveSpec:
    state = _make_state(seed, board_text=board_text)
    return PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=time_budget,
        iterations=iterations,
        seed=seed,
        max_depth=max_depth,
        max_nodes=max_nodes,
        min_reach_prob=min_reach_prob,
    )


def _make_state(seed: int, *, board_text: str | None) -> GameState:
    rng = random.Random(seed)
    deck = list(make_deck())
    rng.shuffle(deck)
    if board_text:
        board_cards = tuple(Card.from_str(board_text[i : i + 2]) for i in range(0, len(board_text), 2))
    else:
        street = rng.choice((Street.FLOP, Street.TURN, Street.RIVER))
        board_size = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
        board_cards = tuple(deck.pop() for _ in range(board_size))
    board = Board(cards=board_cards)
    hero = take_two_unique(deck, board.cards)
    villain = take_two_unique(deck, board.cards, hero)
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


def take_two_unique(
    deck: list[Card],
    board_cards: tuple[Card, ...],
    other_hole_cards: tuple[Card, ...] = (),
) -> tuple[Card, Card]:
    seen = set(board_cards) | set(other_hole_cards)
    chosen: list[Card] = []
    while deck and len(chosen) < 2:
        card = deck.pop()
        if card in seen:
            continue
        chosen.append(card)
        seen.add(card)
    if len(chosen) != 2:
        raise RuntimeError("failed to sample unique hole cards")
    return chosen[0], chosen[1]


if __name__ == "__main__":
    raise SystemExit(main())
