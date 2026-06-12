from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache
from random import Random

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.hands import (
    PlayerRangeVectors,
    PrivateHandIndex,
    all_private_hands,
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
    sampled_pairs: int = 0
    sampled_runouts: int = 0
    random_seed: int = 0


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
    if _should_sample(state, ranges, cfg):
        ev0 = _compute_heads_up_ev_sampled(state, ranges, evaluator_impl, cfg)
        return build_value_label([ev0, -ev0], scalar_ev_target(2))
    ev0 = _compute_heads_up_ev(state, ranges, evaluator_impl, cfg)
    return build_value_label([ev0, -ev0], scalar_ev_target(2))


def _should_sample(
    state: GameState,
    ranges: PlayerRangeVectors,
    config: EquityEvalConfig,
) -> bool:
    dead_cards: tuple[Card, ...] = tuple(
        list(state.board.cards)
        + [
            card
            for player in state.players
            if player.hole_cards is not None
            for card in player.hole_cards
        ]
    )
    masked_ranges = ranges.masked(dead_cards)
    active0 = int(np.count_nonzero(masked_ranges.values[0].values))
    active1 = int(np.count_nonzero(masked_ranges.values[1].values))
    pair_count = active0 * active1
    if config.sampled_pairs > 0 or config.sampled_runouts > 0:
        return True
    return config.max_range_combos is not None and pair_count > config.max_range_combos


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

    hands0 = _weighted_private_hands(range0.values, dead_cards)
    hands1 = _weighted_private_hands(range1.values, dead_cards)
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
            runouts = _cached_runouts(state.board, dead_cards + hand0 + hand1)
            if not runouts:
                continue
            pair_weight = weight0 * weight1
            score0_batch = evaluator.evaluate_seven_card_hands(
                tuple(hand0 + final_board for final_board in runouts)
            )
            score1_batch = evaluator.evaluate_seven_card_hands(
                tuple(hand1 + final_board for final_board in runouts)
            )
            for score0, score1 in zip(score0_batch, score1_batch, strict=True):
                if score0.score < score1.score:
                    payoff0 = pot
                elif score1.score < score0.score:
                    payoff0 = 0.0
                else:
                    payoff0 = pot * 0.5
                total_payout0 += pair_weight * payoff0
                total_weight += pair_weight

    if total_weight <= 0.0:
        raise ValueError("range combinations produced no valid equity mass")
    expected_payout0 = total_payout0 / total_weight
    return float(expected_payout0 * 2.0 - pot)


def _compute_heads_up_ev_sampled(
    state: GameState,
    ranges: PlayerRangeVectors,
    evaluator: TreysHandEvaluator,
    config: EquityEvalConfig,
) -> float:
    rng = Random(config.random_seed)
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
    hands0 = _weighted_private_hands(masked_ranges.values[0].values, dead_cards)
    hands1 = _weighted_private_hands(masked_ranges.values[1].values, dead_cards)
    if config.max_range_combos is not None:
        hands0 = hands0[: config.max_range_combos]
        hands1 = hands1[: config.max_range_combos]
    max_pairs = len(hands0) * len(hands1)
    pair_count = config.sampled_pairs if config.sampled_pairs > 0 else min(max_pairs, 4096)
    runout_count = config.sampled_runouts if config.sampled_runouts > 0 else 64
    if pair_count <= 0:
        raise ValueError("sampled equity requires positive sample count")
    pot = float(state.betting_round.pot.amount + sum(bet.committed for bet in state.betting_round.bets))
    total = 0.0
    for _ in range(pair_count):
        hand0, weight0 = hands0[rng.randrange(len(hands0))]
        hand1, weight1 = hands1[rng.randrange(len(hands1))]
        if _hands_conflict(hand0, hand1):
            continue
        runouts = _sample_runouts(
            state.board,
            dead_cards + hand0 + hand1,
            runout_count,
            rng,
        )
        if not runouts:
            continue
        pair_weight = weight0 * weight1
        score0_batch = evaluator.evaluate_seven_card_hands(
            tuple(hand0 + final_board for final_board in runouts)
        )
        score1_batch = evaluator.evaluate_seven_card_hands(
            tuple(hand1 + final_board for final_board in runouts)
        )
        for score0, score1 in zip(score0_batch, score1_batch, strict=True):
            if score0.score < score1.score:
                payoff0 = pot
            elif score1.score < score0.score:
                payoff0 = 0.0
            else:
                payoff0 = pot * 0.5
            total += pair_weight * payoff0
    expected_payout0 = total / float(max(pair_count, 1))
    return float(expected_payout0 * 2.0 - pot)


def _weighted_private_hands(
    range_values: NDArray[np.float32],
    dead_cards: tuple[Card, ...],
) -> list[tuple[tuple[Card, Card], float]]:
    hand_indices = _legal_hand_indices(dead_cards)
    active_mask = range_values[hand_indices] > 0.0
    if not np.any(active_mask):
        return []
    filtered_indices = hand_indices[active_mask]
    filtered_weights = range_values[filtered_indices].astype(np.float32, copy=False)
    total = float(np.sum(filtered_weights, dtype=np.float64))
    if total <= 0.0:
        return []
    normalized = filtered_weights / np.float32(total)
    hands: list[tuple[tuple[Card, Card], float]] = []
    for index, weight in zip(filtered_indices, normalized, strict=True):
        hand = private_hand_from_index(PrivateHandIndex(int(index)))
        hands.append(((hand.first, hand.second), float(weight)))
    return hands


@lru_cache(maxsize=256)
def _legal_hand_indices(dead_cards: tuple[Card, ...]) -> NDArray[np.int32]:
    dead_set = set(dead_cards)
    indices = [
        index
        for index, hand in enumerate(all_private_hands())
        if hand.first not in dead_set and hand.second not in dead_set
    ]
    return np.asarray(indices, dtype=np.int32)


def _board_runouts(board: Board, dead_cards: tuple[Card, ...]) -> list[tuple[Card, ...]]:
    dead_set = set(dead_cards)
    remaining = [card for card in make_deck() if card not in dead_set]
    need = 5 - len(board.cards)
    if need < 0:
        raise ValueError("board cannot exceed five cards")
    if need == 0:
        return [board.cards]
    return [board.cards + runout for runout in combinations(remaining, need)]


def _sample_runouts(
    board: Board,
    dead_cards: tuple[Card, ...],
    sample_count: int,
    rng: Random,
) -> list[tuple[Card, ...]]:
    all_runouts = _board_runouts(board, dead_cards)
    if sample_count <= 0 or len(all_runouts) <= sample_count:
        return all_runouts
    return [all_runouts[rng.randrange(len(all_runouts))] for _ in range(sample_count)]


@lru_cache(maxsize=512)
def _cached_runouts(board: Board, dead_cards: tuple[Card, ...]) -> tuple[tuple[Card, ...], ...]:
    return tuple(_board_runouts(board, dead_cards))


def _hands_conflict(hand0: tuple[Card, Card], hand1: tuple[Card, Card]) -> bool:
    return any(card in hand1 for card in hand0)
