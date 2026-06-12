from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from random import Random

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch is required: {exc}") from exc

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
from pokergpu.core.cards import make_deck
from pokergpu.core.state import GameState, PlayerState
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_multi
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu


@dataclass(frozen=True, slots=True)
class SolveRow:
    label: str
    elapsed_s: float
    iterations: int
    nodes: int
    leaves: int
    depth: int
    root_ev: str
    root_actions: str


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    rng = Random(args.seed)
    _print_phase("build-state", 0, 4, 0.0, args.device)
    state = make_random_state(rng, args.players, args.street)
    _print_phase("build-ranges", 1, 4, 0.0, args.device)
    range_vectors = make_random_ranges(rng, args.players)
    _print_phase("solve-start", 2, 4, 0.0, args.device)
    spec = PostflopResolveSpec(
        state=state,
        range_p0=range_vectors[0],
        range_p1=range_vectors[1],
        time_budget_sec=args.time_budget,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_reach_prob=args.min_reach_prob,
    )

    started = time.perf_counter()
    if args.players == 2:
        _print_phase("gpu-cfr", 3, 4, started, args.device)
        result = resolve_postflop_gpu(spec)
        row = SolveRow(
            label="solve",
            elapsed_s=time.perf_counter() - started,
            iterations=result.iterations,
            nodes=result.node_count,
            leaves=result.leaf_count,
            depth=args.max_depth,
            root_ev=f"{result.root_ev_player0:.6f} / {result.root_ev_player1:.6f}",
            root_actions=",".join(result.root_actions),
        )
    else:
        _print_phase("gpu-cfr-multi", 3, 4, started, args.device)
        result = resolve_postflop_multi(spec, max_player_count=args.players)
        row = SolveRow(
            label="solve",
            elapsed_s=time.perf_counter() - started,
            iterations=result.iterations,
            nodes=result.node_count,
            leaves=result.leaf_count,
            depth=args.max_depth,
            root_ev=",".join(f"{float(value):.6f}" for value in result.root_ev),
            root_actions=",".join(result.root_actions),
        )

    _print_phase("done", 4, 4, started, args.device)
    print_summary(args, state, row)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--street", choices=("flop", "turn", "river"), default="flop")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-nodes", type=int, default=8192)
    parser.add_argument("--min-reach-prob", type=float, default=0.0)
    return parser.parse_args()


def make_random_state(rng: Random, players: int, street_name: str) -> GameState:
    deck = list(make_deck())
    rng.shuffle(deck)
    board_cards = {"flop": 3, "turn": 4, "river": 5}[street_name]
    board = tuple(deck[:board_cards])
    if players < 2:
        raise SystemExit("players must be at least 2")
    hole_cards: list[tuple] = []
    offset = board_cards
    for seat in range(players):
        hole_cards.append((deck[offset], deck[offset + 1]))
        offset += 2
    stacks = tuple(
        PlayerStack(player=PlayerIndex(seat), stack=chips(2000))
        for seat in range(players)
    )
    bets = tuple(
        PlayerBet(player=PlayerIndex(seat), committed=chips(0))
        for seat in range(players)
    )
    state = GameState(
        board=Board(cards=board),
        players=tuple(
            PlayerState(player=PlayerIndex(seat), hole_cards=hole_cards[seat])
            for seat in range(players)
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(200)),
            stacks=stacks,
            bets=bets,
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )
    return state


def make_random_ranges(rng: Random, players: int) -> tuple[RangeVector, ...]:
    ranges = []
    for _ in range(players):
        values = np.asarray([rng.random() for _ in range(1326)], dtype=np.float32)
        ranges.append(RangeVector.from_values(values).normalized())
    return tuple(ranges)


def print_summary(args: argparse.Namespace, state: GameState, row: SolveRow) -> None:
    rows = [
        ("seed", str(args.seed)),
        ("players", str(args.players)),
        ("street", str(args.street)),
        ("device", args.device),
        ("board", str(state.board)),
        ("elapsed_s", f"{row.elapsed_s:.6f}"),
        ("iterations", str(row.iterations)),
        ("nodes", str(row.nodes)),
        ("leaves", str(row.leaves)),
        ("depth", str(row.depth)),
        ("root_ev_pot", row.root_ev),
        ("root_actions", row.root_actions),
    ]
    width = max(len(key) for key, _value in rows)
    for key, value in rows:
        print(f"{key:<{width}} : {value}")


def _print_phase(label: str, current: int, total: int, started_at: float, device: str) -> None:
    gpu_mem = ""
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        gpu_mem = f" gpu_mem={allocated:.0f}/{reserved:.0f}MB"
    elapsed = 0.0 if started_at == 0.0 else time.perf_counter() - started_at
    print(f"[{current}/{total}] {label} elapsed={elapsed:.2f}s{gpu_mem}")


if __name__ == "__main__":
    raise SystemExit(main())
