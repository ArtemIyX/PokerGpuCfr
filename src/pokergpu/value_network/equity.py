from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.hands import (
    PlayerRangeVectors,
    PrivateHandIndex,
    private_hand_from_index,
)
from pokergpu.core.board import Board, Street
from pokergpu.core.cards import Card, make_deck
from pokergpu.core.state import GameState
from pokergpu.eval.treys_evaluator import TreysHandEvaluator

from .target import PokerValueLabel, build_value_label, scalar_ev_target


@dataclass(frozen=True, slots=True)
class EquityEvalConfig:
    max_range_combos: int | None = None


def build_postflop_equity_label(
    state: GameState,
    ranges: PlayerRangeVectors,
    *,
    evaluator: TreysHandEvaluator | None = None,
    config: EquityEvalConfig | None = None,
) -> PokerValueLabel:
    if state.player_count != 2:
        raise ValueError("postflop equity labels support heads-up only")
    if state.current_street is Street.PREFLOP:
        raise ValueError("postflop equity labels require a postflop state")

    evaluator_impl = evaluator or TreysHandEvaluator()
    cfg = config or EquityEvalConfig()
    ev0 = _compute_heads_up_ev(state, ranges, evaluator_impl, cfg)
    return build_value_label([ev0, -ev0], scalar_ev_target(2))


def _compute_heads_up_ev(
    state: GameState,
    ranges: PlayerRangeVectors,
    evaluator: TreysHandEvaluator,
    config: EquityEvalConfig,
) -> float:
    dead_cards: tuple[Card, ...] = tuple(
        list(state.board.cards)
        + [
            card
            for player in state.players
            if player.hole_cards is not None
            for card in player.hole_cards
        ]
    )
    masked_ranges = ranges.masked(dead_cards).normalized()
    range0 = masked_ranges.values[0]
    range1 = masked_ranges.values[1]

    hands0 = _weighted_private_hands(range0.values)
    hands1 = _weighted_private_hands(range1.values)
    if config.max_range_combos is not None:
        hands0 = hands0[: config.max_range_combos]
        hands1 = hands1[: config.max_range_combos]

    pot = float(state.betting_round.pot.amount + sum(bet.committed for bet in state.betting_round.bets))
    total_weight = 0.0
    total_payout0 = 0.0

    for hand0, weight0 in hands0:
        for hand1, weight1 in hands1:
            if _hands_conflict(hand0, hand1):
                continue
            used_cards = dead_cards + hand0 + hand1
            runouts = _board_runouts(state.board, used_cards)
            if not runouts:
                continue
            pair_weight = weight0 * weight1
            for final_board in runouts:
                score0 = evaluator.evaluate_seven_card_hand(hand0 + final_board).score
                score1 = evaluator.evaluate_seven_card_hand(hand1 + final_board).score
                if score0 < score1:
                    payoff0 = pot
                elif score1 < score0:
                    payoff0 = 0.0
                else:
                    payoff0 = pot * 0.5
                total_payout0 += pair_weight * payoff0
                total_weight += pair_weight

    if total_weight <= 0.0:
        raise ValueError("range combinations produced no valid equity mass")
    expected_payout0 = total_payout0 / total_weight
    return float(expected_payout0 * 2.0 - pot)


def _weighted_private_hands(range_values: NDArray[np.float32]) -> list[tuple[tuple[Card, Card], float]]:
    hands: list[tuple[tuple[Card, Card], float]] = []
    for index, weight in enumerate(range_values):
        if weight <= 0.0:
            continue
        hand = private_hand_from_index(PrivateHandIndex(index))
        hands.append(((hand.first, hand.second), float(weight)))
    return hands


def _board_runouts(board: Board, dead_cards: tuple[Card, ...]) -> list[tuple[Card, ...]]:
    dead_set = set(dead_cards)
    remaining = [card for card in make_deck() if card not in dead_set]
    need = 5 - len(board.cards)
    if need < 0:
        raise ValueError("board cannot exceed five cards")
    if need == 0:
        return [board.cards]
    return [board.cards + runout for runout in combinations(remaining, need)]


def _hands_conflict(hand0: tuple[Card, Card], hand1: tuple[Card, Card]) -> bool:
    return any(card in hand1 for card in hand0)
