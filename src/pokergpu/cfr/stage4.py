from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pokergpu.abstraction.hands import all_private_hands
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
    live_hand_mask = tuple(private_hand_mask(live_board.cards))
    rows = tuple(
        ShowdownEquityNodeInput(
            node_id=node_id,
            board=live_board,
            opponent_reach=opponent_reach.node_hand_opponent_reach[node_id],
            live_hand_mask=live_hand_mask,
            pot_size=max(1.0, aggregate.node_aggregate.reach[node_id]),
        )
        for node_id in range(tree.node_count)
    )
    return ShowdownEquityBatchInput(rows=rows)


def compute_showdown_equity(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
    max_workers: int | None = None,
    evaluator: TreysHandEvaluator | None = None,
) -> ShowdownEquityResult:
    showdown_input = build_showdown_equity_input(
        tree,
        aggregate,
        opponent_reach,
        board=board,
    )
    node_values = _compute_node_showdown_equity(
        showdown_input,
        evaluator=evaluator,
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
    evaluator: TreysHandEvaluator | None = None,
    max_workers: int | None = None,
) -> tuple[float, ...]:
    if max_workers is None or max_workers <= 1 or len(showdown_input.rows) <= 1:
        return tuple(
            _compute_single_node_showdown_equity(row, evaluator=evaluator)
            for row in showdown_input.rows
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return tuple(
            executor.map(
                lambda row: _compute_single_node_showdown_equity(row, evaluator=evaluator),
                showdown_input.rows,
            )
        )


def _compute_single_node_showdown_equity(
    row: ShowdownEquityNodeInput,
    *,
    evaluator: TreysHandEvaluator | None = None,
) -> float:
    if len(row.opponent_reach) != private_hand_count():
        raise ValueError("opponent reach must match private hand count")
    if len(row.live_hand_mask) != private_hand_count():
        raise ValueError("live hand mask must match private hand count")
    if row.pot_size <= 0.0:
        raise ValueError("pot size must be positive")
    if len(row.board.cards) not in {3, 4, 5}:
        raise ValueError("showdown equity requires a postflop board")

    evaluator_instance = evaluator or TreysHandEvaluator()
    live_hands = tuple(
        hand for hand, is_live in zip(all_private_hands(), row.live_hand_mask, strict=True) if is_live
    )
    if not live_hands:
        return 0.0

    opponent_total = sum(max(0.0, weight) for weight in row.opponent_reach)
    if opponent_total <= 0.0:
        return 0.0

    total_equity = 0.0
    hero_count = 0
    for hero_hand in live_hands:
        hero_cards: tuple[Card, Card] = (hero_hand.first, hero_hand.second)
        hero_score = evaluator_instance.evaluate_seven_card_hand(hero_cards + row.board.cards).score
        hero_equity = 0.0
        hero_opponent_weight = 0.0
        for opponent_hand, opponent_weight in zip(all_private_hands(), row.opponent_reach, strict=True):
            if opponent_weight <= 0.0 or not _hands_are_disjoint(hero_hand, opponent_hand):
                continue
            opponent_cards: tuple[Card, Card] = (opponent_hand.first, opponent_hand.second)
            opponent_score = evaluator_instance.evaluate_seven_card_hand(opponent_cards + row.board.cards).score
            if hero_score < opponent_score:
                outcome = 1.0
            elif hero_score == opponent_score:
                outcome = 0.5
            else:
                outcome = 0.0
            hero_equity += opponent_weight * outcome
            hero_opponent_weight += opponent_weight

        if hero_opponent_weight > 0.0:
            total_equity += hero_equity / hero_opponent_weight
            hero_count += 1

    if hero_count <= 0:
        return 0.0
    return total_equity / hero_count


def _hands_are_disjoint(hero_hand: PrivateHand, opponent_hand: PrivateHand) -> bool:
    return (
        hero_hand.first != opponent_hand.first
        and hero_hand.first != opponent_hand.second
        and hero_hand.second != opponent_hand.first
        and hero_hand.second != opponent_hand.second
    )
