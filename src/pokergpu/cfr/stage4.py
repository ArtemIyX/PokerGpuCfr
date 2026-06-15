from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from pokergpu.abstraction.hands import all_private_hands
from pokergpu.abstraction.hands import all_private_hand_card_masks
from pokergpu.abstraction.hands import private_hand_count
from pokergpu.abstraction.hands import PrivateHand
from pokergpu.abstraction.hands import private_hand_mask
from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage3 import OpponentReachResult
from pokergpu.core.cards import Card
from pokergpu.core.board import Board
from pokergpu.eval.treys_evaluator import TreysHandEvaluator
from pokergpu.tree.public_tree import PublicTree


@dataclass(slots=True, frozen=True)
class ShowdownEquityNodeInput:
    node_id: int
    board: Board
    opponent_reach: tuple[float, ...]
    live_hand_mask: tuple[bool, ...]
    pot_size: float


@dataclass(slots=True, frozen=True)
class ShowdownEquityBatchInput:
    rows: tuple[ShowdownEquityNodeInput, ...]


@dataclass(slots=True, frozen=True)
class ShowdownEquityNodeOutput:
    node_id: int
    showdown_equity: float
    showdown_equity_bb: float


@dataclass(slots=True, frozen=True)
class ShowdownEquityResult:
    node_showdown_equity: tuple[float, ...]
    node_showdown_equity_bb: tuple[float, ...]
    input_rows: ShowdownEquityBatchInput
    output_rows: tuple[ShowdownEquityNodeOutput, ...]


@dataclass(slots=True, frozen=True)
class ShowdownEquityBoardCache:
    board: Board
    live_hand_mask: tuple[bool, ...]
    hand_scores: tuple[int, ...]
    live_hand_indices: tuple[int, ...]
    feasible_opponent_indices: tuple[tuple[int, ...], ...]
    hand_card_masks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.live_hand_mask) != private_hand_count():
            raise ValueError("live hand mask must match private hand count")
        if len(self.hand_scores) != private_hand_count():
            raise ValueError("hand scores must match private hand count")
        if len(self.live_hand_indices) != sum(1 for is_live in self.live_hand_mask if is_live):
            raise ValueError("live hand indices must match live hand mask")
        if len(self.feasible_opponent_indices) != private_hand_count():
            raise ValueError("feasible opponent indices must match private hand count")
        if len(self.hand_card_masks) != private_hand_count():
            raise ValueError("hand card masks must match private hand count")
        for row in self.feasible_opponent_indices:
            if any(index < 0 or index >= private_hand_count() for index in row):
                raise ValueError("feasible opponent indices must stay within hand bounds")


def build_showdown_equity_input(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
) -> ShowdownEquityBatchInput:
    if tree.node_count != len(aggregate.node_aggregate.reach):
        raise ValueError("tree and aggregate result must cover the same number of nodes")
    if tree.node_count != len(opponent_reach.node_opponent_reach):
        raise ValueError("tree and opponent reach result must cover the same number of nodes")

    live_board = board or Board(())
    cache = build_showdown_equity_board_cache(live_board)
    rows = tuple(
        ShowdownEquityNodeInput(
            node_id=node_id,
            board=live_board,
            opponent_reach=opponent_reach.node_hand_opponent_reach[node_id],
            live_hand_mask=cache.live_hand_mask,
            pot_size=max(1.0, aggregate.node_aggregate.reach[node_id]),
        )
        for node_id in range(tree.node_count)
    )
    return ShowdownEquityBatchInput(rows=rows)


def build_showdown_equity_board_cache(
    board: Board,
    *,
    evaluator: TreysHandEvaluator | None = None,
) -> ShowdownEquityBoardCache:
    if len(board.cards) != 5:
        raise ValueError("showdown equity requires a river board")

    evaluator_instance = evaluator or TreysHandEvaluator()
    live_hand_mask = tuple(private_hand_mask(board.cards))
    hands = all_private_hands()
    hand_card_masks = all_private_hand_card_masks()
    live_hand_indices = tuple(
        index for index, is_live in enumerate(live_hand_mask) if is_live
    )
    hand_scores = tuple(
        _score_hand(evaluator_instance, hand, board) if is_live else 0
        for hand, is_live in zip(hands, live_hand_mask, strict=True)
    )
    feasible_opponent_indices = tuple(
        tuple(
            opponent_index
            for opponent_index in live_hand_indices
            if hero_index != opponent_index
            and (hand_card_masks[hero_index] & hand_card_masks[opponent_index]) == 0
        )
        for hero_index in range(private_hand_count())
    )
    return ShowdownEquityBoardCache(
        board=board,
        live_hand_mask=live_hand_mask,
        hand_scores=hand_scores,
        live_hand_indices=live_hand_indices,
        feasible_opponent_indices=feasible_opponent_indices,
        hand_card_masks=hand_card_masks,
    )


def compute_showdown_equity(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
    max_workers: int | None = None,
    evaluator: TreysHandEvaluator | None = None,
) -> ShowdownEquityResult:
    if board is None:
        raise ValueError("showdown equity requires a river board")
    cache = build_showdown_equity_board_cache(board, evaluator=evaluator)
    showdown_input = build_showdown_equity_input(
        tree,
        aggregate,
        opponent_reach,
        board=cache.board,
    )
    node_values = _compute_node_showdown_equity(
        showdown_input,
        cache=cache,
        max_workers=max_workers,
    )
    return ShowdownEquityResult(
        node_showdown_equity=node_values,
        node_showdown_equity_bb=node_values,
        input_rows=showdown_input,
        output_rows=tuple(
            ShowdownEquityNodeOutput(
                node_id=row.node_id,
                showdown_equity=value,
                showdown_equity_bb=value,
            )
            for row, value in zip(showdown_input.rows, node_values, strict=True)
        ),
    )


def _compute_node_showdown_equity(
    showdown_input: ShowdownEquityBatchInput,
    *,
    cache: ShowdownEquityBoardCache,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    if max_workers is None or max_workers <= 1 or len(showdown_input.rows) <= 1:
        return tuple(
            compute_showdown_equity_node(row, cache=cache)
            for row in showdown_input.rows
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(
            executor.map(lambda row: compute_showdown_equity_node(row, cache=cache), showdown_input.rows)
        )


def compute_showdown_equity_node(
    row: ShowdownEquityNodeInput,
    *,
    cache: ShowdownEquityBoardCache,
) -> float:
    if len(row.opponent_reach) != private_hand_count():
        raise ValueError("opponent reach must match private hand count")
    if len(row.live_hand_mask) != private_hand_count():
        raise ValueError("live hand mask must match private hand count")
    if row.pot_size <= 0.0:
        raise ValueError("pot size must be positive")
    if row.board != cache.board:
        raise ValueError("node board must match the cached board")
    if row.live_hand_mask != cache.live_hand_mask:
        raise ValueError("node live hand mask must match the cached mask")
    if not cache.live_hand_indices:
        return 0.0

    opponent_weights = np.asarray(row.opponent_reach, dtype=np.float64)
    opponent_total = float(np.sum(np.maximum(opponent_weights, 0.0), dtype=np.float64))
    if opponent_total <= 0.0:
        return 0.0

    total_equity = 0.0
    hero_count = 0
    for hero_index in cache.live_hand_indices:
        hero_score = cache.hand_scores[hero_index]
        feasible_opponents = cache.feasible_opponent_indices[hero_index]
        hero_opponent_weight = 0.0
        hero_equity = 0.0
        for opponent_index in feasible_opponents:
            opponent_weight = float(opponent_weights[opponent_index])
            if opponent_weight <= 0.0:
                continue
            hero_equity += opponent_weight * _compare_scores(hero_score, cache.hand_scores[opponent_index])
            hero_opponent_weight += opponent_weight
        if hero_opponent_weight > 0.0:
            total_equity += hero_equity / hero_opponent_weight
            hero_count += 1

    if hero_count <= 0:
        return 0.0
    return total_equity / hero_count


def _score_hand(
    evaluator: TreysHandEvaluator,
    hand: PrivateHand,
    board: Board,
) -> int:
    cards: tuple[Card, Card] = (hand.first, hand.second)
    return evaluator.evaluate_seven_card_hand(cards + board.cards).score


def _compare_scores(hero_score: int, opponent_score: int) -> float:
    if hero_score < opponent_score:
        return 1.0
    if hero_score == opponent_score:
        return 0.5
    return 0.0
