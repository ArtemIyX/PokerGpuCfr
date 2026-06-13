from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

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
from pokergpu.runtime import (
    PostflopResolveSpec,
    SolveCacheState,
    resolve_postflop_gpu_batch,
    resolve_postflop_hu,
    resolve_postflop_threeway,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, choices=(2, 3), default=2)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-batch-size", type=int, default=1)
    parser.add_argument("--min-reach-prob", type=float, default=0.0)
    parser.add_argument("--board", type=str, default="")
    parser.add_argument("--print-tree", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    state = make_random_postflop_state(rng, player_count=args.players, board_text=args.board or None)
    spec = PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        range_p2=RangeVector.uniform() if args.players == 3 else None,
        time_budget_sec=args.time_budget,
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        min_reach_prob=args.min_reach_prob,
        cache_state=SolveCacheState(),
    )

    if args.print_tree:
        print_tree(state)
    if args.players == 3:
        result = resolve_postflop_threeway(spec)
    elif args.device == "cuda":
        batch_size = max(1, int(args.cuda_batch_size))
        if batch_size == 1:
            from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu

            result = resolve_postflop_gpu(spec)
        else:
            result = resolve_postflop_gpu_batch(tuple(spec for _ in range(batch_size))).results[0]
    else:
        result = resolve_postflop_hu(spec)

    print_state(state)
    print_result(result)
    return 0


def make_random_postflop_state(
    rng: random.Random,
    *,
    player_count: int = 2,
    board_text: str | None = None,
) -> GameState:
    deck = list(make_deck())
    rng.shuffle(deck)

    if board_text:
        board_cards = tuple(Card.from_str(board_text[i : i + 2]) for i in range(0, len(board_text), 2))
    else:
        street = rng.choice((Street.FLOP, Street.TURN, Street.RIVER))
        board_size = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
        board_cards = tuple(deck.pop() for _ in range(board_size))
    board = Board(cards=board_cards)

    hole_0 = take_two_unique(deck, board.cards)
    hole_1 = take_two_unique(deck, board.cards, hole_0)
    hole_2 = take_two_unique(deck, board.cards, hole_0 + hole_1) if player_count == 3 else None

    sb = max(1, int(rng.choice((25, 50, 100))))
    bb = max(sb, int(sb * rng.choice((2, 2, 3, 4))))
    stacks_raw = [int(bb * rng.randint(2, 25)) for _ in range(player_count)]
    commitments = _make_random_commitments(rng, stacks_raw, bb, player_count)
    base_pot = int(bb * rng.randint(1, 10))
    pot = max(0, base_pot + sum(commitments))
    to_act = PlayerIndex(int(np.argmax(stacks_raw)))
    dealer = PlayerIndex(rng.randrange(player_count))

    return GameState(
        board=board,
        players=tuple(
            [
                PlayerState(player=PlayerIndex(0), hole_cards=hole_0),
                PlayerState(player=PlayerIndex(1), hole_cards=hole_1),
            ]
            + ([PlayerState(player=PlayerIndex(2), hole_cards=hole_2)] if player_count == 3 else [])
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=chips(pot)),
            stacks=tuple(
                PlayerStack(player=PlayerIndex(index), stack=chips(stack))
                for index, stack in enumerate(stacks_raw)
            ),
            bets=tuple(
                PlayerBet(player=PlayerIndex(index), committed=chips(commitment))
                for index, commitment in enumerate(commitments)
            ),
            blinds=BlindStructure(small_blind=chips(sb), big_blind=chips(bb)),
            to_act=to_act,
        ),
        dealer=dealer,
    )


def _make_random_commitments(
    rng: random.Random,
    stacks: list[int],
    big_blind: int,
    player_count: int,
) -> list[int]:
    commitments = [0 for _ in range(player_count)]
    highest = int(rng.randint(0, max(1, min(stacks) // 4)))
    for index in range(player_count):
        room = max(0, min(int(stacks[index]), highest))
        commitments[index] = int(rng.randint(0, room)) if room > 0 else 0
    if player_count > 1 and len(set(commitments)) == 1:
        idx = int(rng.randrange(player_count))
        room = max(0, int(stacks[idx]))
        commitments[idx] = min(room, commitments[idx] + max(1, big_blind // 2))
    acting_player = int(np.argmax(stacks))
    commitments[acting_player] = min(commitments[acting_player], max(0, stacks[acting_player] - big_blind))
    return [max(0, int(value)) for value in commitments]


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


def print_state(state: GameState) -> None:
    print(f"board: {state.board}")
    print(f"street: {state.current_street.value}")
    print(f"dealer: {state.dealer}")
    print(f"to_act: {state.betting_round.to_act}")
    print(f"pot: {int(state.betting_round.pot.amount)}")
    print(f"blinds: {int(state.betting_round.blinds.small_blind)}/{int(state.betting_round.blinds.big_blind)}")
    for player in state.players:
        print(
            "player",
            int(player.player),
            {
                "hole_cards": format_cards(player.hole_cards),
                "folded": player.folded,
                "all_in": player.all_in,
                "stack": int(next(s.stack for s in state.betting_round.stacks if s.player == player.player)),
                "committed": int(next(b.committed for b in state.betting_round.bets if b.player == player.player)),
            },
        )


def print_tree(state: GameState) -> None:
    print("tree:")
    print(f"  street: {state.current_street.value}")
    print(f"  player_count: {state.player_count}")
    print(f"  to_act: {state.betting_round.to_act}")


def print_result(result) -> None:
    print(f"root_infoset_id: {result.root_infoset_id}")
    print(f"iterations: {result.iterations}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"node_count: {result.node_count}")
    print(f"leaf_count: {result.leaf_count}")
    print(f"root_actions: {result.root_actions}")
    print(f"root_strategy: {np.asarray(result.root_strategy, dtype=np.float32).tolist()}")
    print(f"root_action_ev_player0: {np.asarray(result.root_action_ev_player0, dtype=np.float32).tolist()}")
    print(f"root_action_ev_player1: {np.asarray(result.root_action_ev_player1, dtype=np.float32).tolist()}")
    print(f"root_ev_player0: {float(result.root_ev_player0):.6f}")
    print(f"root_ev_player1: {float(result.root_ev_player1):.6f}")


def format_cards(cards: tuple[Card, ...] | None) -> str:
    if cards is None:
        return ""
    return "".join(str(card) for card in cards)


if __name__ == "__main__":
    raise SystemExit(main())
