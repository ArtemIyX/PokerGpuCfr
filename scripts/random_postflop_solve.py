from __future__ import annotations

import argparse
from dataclasses import dataclass
from random import Random

import numpy as np

from pokergpu.abstraction.actions import BaselineActionAbstraction, make_runtime_profile
from pokergpu.abstraction.hands import RangeVector
from pokergpu.core.actions import Action
from pokergpu.core.board import Board, Street
from pokergpu.core.cards import Card, make_deck
from pokergpu.core.state import GameState, HandPhase, PlayerState
from pokergpu.core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from pokergpu.core.transitions import apply_action
from pokergpu.runtime import PostflopResolveSpec, resolve_postflop_hu
from pokergpu.runtime.gpu_postflop import resolve_postflop_gpu


@dataclass(frozen=True, slots=True)
class SampledSpot:
    state: GameState
    history: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample and solve a random HU postflop spot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--street", choices=("flop", "turn", "river", "random"), default="random")
    parser.add_argument("--history-steps", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--cpu-fallback", action="store_true", default=False)
    parser.add_argument("--compare-cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.use_gpu and not args.cpu_fallback:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
        except Exception as exc:
            raise SystemExit(f"GPU solver required but unavailable: {exc}") from exc

    rng = Random(args.seed)
    sampled = sample_spot(rng, street=args.street, history_steps=args.history_steps)

    print_state(sampled, seed=args.seed)
    print()
    spec = PostflopResolveSpec(
        state=sampled.state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=args.time_budget,
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )

    cpu_result = None
    if args.compare_cpu:
        cpu_result = resolve_postflop_hu(spec)
        print_result("cpu", cpu_result)
        print_action_mix("cpu", cpu_result)

    if args.use_gpu and not args.cpu_fallback:
        gpu_result = resolve_postflop_gpu(spec)
        print_result("cuda", gpu_result)
        print_action_mix("cuda", gpu_result)
        if cpu_result is not None:
            print("\ncompare")
            print(
                "  root_strategy_delta: "
                f"{format_array(np.asarray(gpu_result.root_strategy - cpu_result.root_strategy, dtype=np.float32))}"
            )
            print(
                "  root_action_ev_p0_delta: "
                f"{format_array(np.asarray(gpu_result.root_action_ev_player0 - cpu_result.root_action_ev_player0, dtype=np.float32))}"
            )
            print(
                "  root_action_ev_p1_delta: "
                f"{format_array(np.asarray(gpu_result.root_action_ev_player1 - cpu_result.root_action_ev_player1, dtype=np.float32))}"
            )
            print(f"  root_ev_p0_delta: {gpu_result.root_ev_player0 - cpu_result.root_ev_player0:.6f}")
            print(f"  root_ev_p1_delta: {gpu_result.root_ev_player1 - cpu_result.root_ev_player1:.6f}")
            print(f"  iterations_delta: {gpu_result.iterations - cpu_result.iterations}")
    return 0


def sample_spot(rng: Random, *, street: str, history_steps: int) -> SampledSpot:
    abstraction = BaselineActionAbstraction(profile=make_runtime_profile())
    for _ in range(128):
        selected_street = _choose_street(rng, street)
        board = _sample_board(rng, selected_street)
        hole_cards = _sample_hole_cards(rng, board.cards)
        state = _make_base_state(board, hole_cards)
        history: list[str] = []
        for _ in range(max(0, history_steps)):
            if state.phase is not HandPhase.IN_PROGRESS:
                break
            legal_actions = tuple(
                action
                for action in abstraction.legal_actions(state)
                if _action_is_safe_before_river(state, action)
            )
            if not legal_actions:
                break
            safe_actions = []
            for action in legal_actions:
                next_state = apply_action(state, action)
                if _is_safe_state(next_state):
                    safe_actions.append(action)
            if not safe_actions:
                break
            action = rng.choice(safe_actions)
            next_state = apply_action(state, action)
            history.append(_format_action(action))
            state = next_state
        if not _is_safe_state(state):
            continue
        return SampledSpot(state=state, history=tuple(history))
    raise RuntimeError("could not sample a valid postflop spot")


def _choose_street(rng: Random, street: str) -> Street:
    if street != "random":
        return Street(street)
    return rng.choice((Street.FLOP, Street.TURN, Street.RIVER))


def _sample_board(rng: Random, street: Street) -> Board:
    deck = list(make_deck())
    rng.shuffle(deck)
    size = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
    return Board(cards=tuple(deck[:size]))


def _sample_hole_cards(rng: Random, board_cards: tuple[Card, ...]) -> tuple[tuple[Card, Card], tuple[Card, Card]]:
    deck = [card for card in make_deck() if card not in board_cards]
    rng.shuffle(deck)
    hero = (deck.pop(), deck.pop())
    villain = (deck.pop(), deck.pop())
    return hero, villain


def _make_base_state(
    board: Board,
    hole_cards: tuple[tuple[Card, Card], tuple[Card, Card]],
) -> GameState:
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


def print_state(sampled: SampledSpot, *, seed: int) -> None:
    state = sampled.state
    print("spot")
    print(f"  seed: {seed}")
    print(f"  street: {state.current_street.value}")
    print(f"  board: {state.board}")
    print(f"  phase: {state.phase.value}")
    print(f"  dealer: {int(state.dealer)}")
    print(f"  to_act: {int(state.betting_round.to_act)}")
    print(f"  pot: {int(state.betting_round.pot.amount)}")
    print(f"  stacks: {[int(stack.stack) for stack in state.betting_round.stacks]}")
    print(f"  bets: {[int(bet.committed) for bet in state.betting_round.bets]}")
    print(f"  hole_cards_p0: {format_cards(state.players[0].hole_cards)}")
    print(f"  hole_cards_p1: {format_cards(state.players[1].hole_cards)}")
    print(f"  history: {', '.join(sampled.history) if sampled.history else '-'}")


def print_result(label: str, result: object) -> None:
    print(f"\nsolver[{label}]")
    print(f"  root_infoset_id: {result.root_infoset_id}")
    print(f"  root_actions: {result.root_actions}")
    print(f"  root_strategy: {format_array(result.root_strategy)}")
    print(f"  root_action_ev_p0: {format_array(result.root_action_ev_player0)}")
    print(f"  root_action_ev_p1: {format_array(result.root_action_ev_player1)}")
    print(f"  root_ev_p0: {result.root_ev_player0:.6f}")
    print(f"  root_ev_p1: {result.root_ev_player1:.6f}")
    print(f"  iterations: {result.iterations}")
    print(f"  elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"  node_count: {result.node_count}")
    print(f"  leaf_count: {result.leaf_count}")


def print_action_mix(label: str, result: object) -> None:
    print(f"  action_mix[{label}]:")
    actions = getattr(result, "root_actions", ())
    strategy = np.asarray(getattr(result, "root_strategy", ()), dtype=np.float32)
    limit = min(len(actions), int(strategy.shape[0]))
    for index in range(limit):
        print(f"    {actions[index]}: {float(strategy[index] * 100.0):.1f}%")


def format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values.tolist()) + "]"


def _format_action(action: Action) -> str:
    if action.amount is None:
        return action.action_type.value
    return f"{action.action_type.value}({int(action.amount)})"


def format_cards(cards: tuple[Card, Card] | None) -> str:
    if cards is None:
        return "-"
    return "".join(str(card) for card in cards)


def _action_is_safe_before_river(state: GameState, action: Action) -> bool:
    if state.current_street is Street.RIVER:
        return True
    if action.action_type.value in {"all_in"}:
        return False
    return True


def _is_safe_state(state: GameState) -> bool:
    if state.phase is HandPhase.SHOWDOWN and len(state.board.cards) < 5:
        return False
    if state.phase is HandPhase.TERMINAL:
        return False
    if len(state.board.cards) < 5 and any(player.all_in for player in state.players):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
