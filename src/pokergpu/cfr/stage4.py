from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pokergpu.abstraction.hands import all_private_hands
from pokergpu.abstraction.hands import all_private_hand_card_masks
from pokergpu.abstraction.hands import private_hand_count
from pokergpu.abstraction.hands import PrivateHand
from pokergpu.abstraction.hands import private_hand_mask
from pokergpu.cfr.stage2 import AggregateProbSumResult
from pokergpu.cfr.stage3 import OpponentReachResult
from pokergpu.core.cards import Card
from pokergpu.core.board import Board, Street
from pokergpu.eval.treys_evaluator import TreysHandEvaluator
from pokergpu.tree.public_tree import PublicTree


@dataclass(slots=True, frozen=True)
class ShowdownEquityNodeInput:
    node_id: int
    board: Board
    opponent_reach: np.ndarray
    positive_opponent_reach: np.ndarray
    positive_opponent_mask: np.ndarray
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
    street: Street
    live_hand_mask: tuple[bool, ...]
    hand_scores: tuple[int, ...]
    hand_scores_array: np.ndarray
    hand_buckets: tuple[int, ...]
    hand_buckets_array: np.ndarray
    live_hand_indices: tuple[int, ...]
    feasible_opponent_indices: tuple[np.ndarray, ...]
    feasible_opponent_values: tuple[np.ndarray, ...]
    feasible_opponent_win_masks: tuple[np.ndarray, ...]
    feasible_opponent_tie_masks: tuple[np.ndarray, ...]
    feasible_opponent_win_mask_matrix: np.ndarray
    feasible_opponent_tie_mask_matrix: np.ndarray
    feasible_opponent_offsets: np.ndarray
    feasible_opponent_flat: np.ndarray
    hand_card_masks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.live_hand_mask) != private_hand_count():
            raise ValueError("live hand mask must match private hand count")
        if self.street is Street.PREFLOP:
            raise ValueError("showdown cache requires postflop street")
        if len(self.hand_scores) != private_hand_count():
            raise ValueError("hand scores must match private hand count")
        if self.hand_scores_array.ndim != 1 or len(self.hand_scores_array) != private_hand_count():
            raise ValueError("hand scores array must match private hand count")
        if len(self.hand_buckets) != private_hand_count():
            raise ValueError("hand buckets must match private hand count")
        if self.hand_buckets_array.ndim != 1 or len(self.hand_buckets_array) != private_hand_count():
            raise ValueError("hand buckets array must match private hand count")
        if len(self.live_hand_indices) != sum(1 for is_live in self.live_hand_mask if is_live):
            raise ValueError("live hand indices must match live hand mask")
        if len(self.feasible_opponent_indices) != private_hand_count():
            raise ValueError("feasible opponent indices must match private hand count")
        if len(self.feasible_opponent_values) != private_hand_count():
            raise ValueError("feasible opponent values must match private hand count")
        if len(self.feasible_opponent_win_masks) != private_hand_count():
            raise ValueError("feasible opponent win masks must match private hand count")
        if len(self.feasible_opponent_tie_masks) != private_hand_count():
            raise ValueError("feasible opponent tie masks must match private hand count")
        if self.feasible_opponent_win_mask_matrix.shape != (private_hand_count(), private_hand_count()):
            raise ValueError("feasible opponent win mask matrix must be square and match hand count")
        if self.feasible_opponent_tie_mask_matrix.shape != (private_hand_count(), private_hand_count()):
            raise ValueError("feasible opponent tie mask matrix must be square and match hand count")
        if self.feasible_opponent_offsets.ndim != 1:
            raise ValueError("feasible opponent offsets must be one-dimensional")
        if self.feasible_opponent_flat.ndim != 1:
            raise ValueError("feasible opponent flat indices must be one-dimensional")
        if len(self.hand_card_masks) != private_hand_count():
            raise ValueError("hand card masks must match private hand count")
        for row in self.feasible_opponent_indices:
            if row.ndim != 1:
                raise ValueError("feasible opponent indices must be one-dimensional")
            if np.any((row < 0) | (row >= private_hand_count())):
                raise ValueError("feasible opponent indices must stay within hand bounds")


def build_showdown_equity_input(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
    cache: ShowdownEquityBoardCache | None = None,
) -> ShowdownEquityBatchInput:
    if tree.node_count != len(aggregate.node_aggregate.reach):
        raise ValueError("tree and aggregate result must cover the same number of nodes")
    if tree.node_count != len(opponent_reach.node_opponent_reach):
        raise ValueError("tree and opponent reach result must cover the same number of nodes")

    live_board = board if board is not None else Board(())
    cache = cache if cache is not None else build_showdown_equity_board_cache(live_board)
    rows = tuple(
        ShowdownEquityNodeInput(
            node_id=node_id,
            board=live_board,
            opponent_reach=np.asarray(opponent_reach.node_hand_opponent_reach[node_id], dtype=np.float64),
            positive_opponent_reach=np.maximum(
                np.asarray(opponent_reach.node_hand_opponent_reach[node_id], dtype=np.float64),
                0.0,
            ),
            positive_opponent_mask=np.asarray(
                np.asarray(opponent_reach.node_hand_opponent_reach[node_id], dtype=np.float64) > 0.0,
                dtype=np.bool_,
            ),
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
    if evaluator is not None:
        return _build_showdown_equity_board_cache_uncached(board, evaluator=evaluator)
    return _build_showdown_equity_board_cache_cached(board)


def _build_showdown_equity_board_cache_uncached(
    board: Board,
    *,
    evaluator: TreysHandEvaluator | None = None,
) -> ShowdownEquityBoardCache:
    if len(board.cards) not in {3, 4, 5}:
        raise ValueError("showdown equity requires a postflop board")

    evaluator_instance = evaluator or TreysHandEvaluator()
    live_hand_mask = tuple(private_hand_mask(board.cards))
    hands = all_private_hands()
    hand_card_masks = all_private_hand_card_masks()
    hand_card_masks_array = np.asarray(hand_card_masks, dtype=np.uint64)
    live_mask_array = np.asarray(live_hand_mask, dtype=np.bool_)
    live_hand_indices = tuple(
        index for index, is_live in enumerate(live_hand_mask) if is_live
    )
    hand_scores_list: list[int] = [0] * private_hand_count()
    hand_buckets_list: list[int] = [-1] * private_hand_count()
    if board.street is Street.RIVER:
        live_hands = tuple(
            hand for hand, is_live in zip(hands, live_hand_mask, strict=True) if is_live
        )
        live_scores = evaluator_instance.evaluate_seven_card_hands(
            tuple((hand.first, hand.second, *board.cards) for hand in live_hands)
        )
        live_score_iter = iter(live_scores)
        for hand_index, is_live in enumerate(live_hand_mask):
            if is_live:
                hand_scores_list[hand_index] = next(live_score_iter).score
    else:
        for hand_index, is_live in enumerate(live_hand_mask):
            if is_live:
                hand_buckets_list[hand_index] = _approximate_hand_strength_bucket(
                    hands[hand_index],
                    board,
                )
    hand_scores = tuple(hand_scores_list)
    hand_scores_array = np.asarray(hand_scores, dtype=np.int32)
    hand_buckets = tuple(hand_buckets_list)
    hand_buckets_array = np.asarray(hand_buckets, dtype=np.int32)

    feasible_opponent_rows: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(private_hand_count())]
    feasible_opponent_values: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(private_hand_count())]
    feasible_opponent_win_masks: list[np.ndarray] = [np.empty(0, dtype=np.bool_) for _ in range(private_hand_count())]
    feasible_opponent_tie_masks: list[np.ndarray] = [np.empty(0, dtype=np.bool_) for _ in range(private_hand_count())]
    feasible_opponent_win_mask_matrix = np.zeros((private_hand_count(), private_hand_count()), dtype=np.float64)
    feasible_opponent_tie_mask_matrix = np.zeros((private_hand_count(), private_hand_count()), dtype=np.float64)
    live_indices = np.asarray(live_hand_indices, dtype=np.int32)
    hand_masks = hand_card_masks_array
    for hero_index in range(private_hand_count()):
        if not live_mask_array[hero_index]:
            continue
        hero_mask = hand_masks[hero_index]
        compatibility = (hand_masks & hero_mask) == 0
        compatible_live = live_mask_array & compatibility
        compatible_live[hero_index] = False
        opponent_indices = np.flatnonzero(compatible_live).astype(np.int32, copy=False)
        feasible_opponent_rows[hero_index] = opponent_indices
        feasible_opponent_values[hero_index] = (
            hand_scores_array[opponent_indices]
            if board.street is Street.RIVER
            else hand_buckets_array[opponent_indices]
        )
        hero_value = hand_scores_array[hero_index] if board.street is Street.RIVER else hand_buckets_array[hero_index]
        win_mask = feasible_opponent_values[hero_index] > hero_value
        tie_mask = feasible_opponent_values[hero_index] == hero_value
        feasible_opponent_win_masks[hero_index] = win_mask
        feasible_opponent_tie_masks[hero_index] = tie_mask
        feasible_opponent_win_mask_matrix[hero_index, opponent_indices] = win_mask.astype(np.float64, copy=False)
        feasible_opponent_tie_mask_matrix[hero_index, opponent_indices] = tie_mask.astype(np.float64, copy=False)
    feasible_opponent_indices = tuple(feasible_opponent_rows)
    feasible_opponent_values_tuple = tuple(feasible_opponent_values)
    feasible_opponent_win_masks_tuple = tuple(feasible_opponent_win_masks)
    feasible_opponent_tie_masks_tuple = tuple(feasible_opponent_tie_masks)
    feasible_opponent_offsets = np.empty(private_hand_count() + 1, dtype=np.int32)
    feasible_opponent_flat_parts: list[np.ndarray] = []
    running_offset = 0
    for hero_index, row in enumerate(feasible_opponent_rows):
        feasible_opponent_offsets[hero_index] = running_offset
        if row.size > 0:
            feasible_opponent_flat_parts.append(row)
            running_offset += row.size
    feasible_opponent_offsets[private_hand_count()] = running_offset
    feasible_opponent_flat = (
        np.concatenate(feasible_opponent_flat_parts).astype(np.int32, copy=False)
        if feasible_opponent_flat_parts
        else np.empty(0, dtype=np.int32)
    )
    return ShowdownEquityBoardCache(
        board=board,
        street=board.street,
        live_hand_mask=live_hand_mask,
        hand_scores=hand_scores,
        hand_scores_array=hand_scores_array,
        hand_buckets=hand_buckets,
        hand_buckets_array=hand_buckets_array,
        live_hand_indices=live_hand_indices,
        feasible_opponent_indices=feasible_opponent_indices,
        feasible_opponent_values=feasible_opponent_values_tuple,
        feasible_opponent_win_masks=feasible_opponent_win_masks_tuple,
        feasible_opponent_tie_masks=feasible_opponent_tie_masks_tuple,
        feasible_opponent_win_mask_matrix=feasible_opponent_win_mask_matrix,
        feasible_opponent_tie_mask_matrix=feasible_opponent_tie_mask_matrix,
        feasible_opponent_offsets=feasible_opponent_offsets,
        feasible_opponent_flat=feasible_opponent_flat,
        hand_card_masks=hand_card_masks,
    )


@lru_cache(maxsize=256)
def _build_showdown_equity_board_cache_cached(board: Board) -> ShowdownEquityBoardCache:
    return _build_showdown_equity_board_cache_uncached(board)


def compute_showdown_equity(
    tree: PublicTree,
    aggregate: AggregateProbSumResult,
    opponent_reach: OpponentReachResult,
    *,
    board: Board | None = None,
    cache: ShowdownEquityBoardCache | None = None,
    max_workers: int | None = None,
    evaluator: TreysHandEvaluator | None = None,
) -> ShowdownEquityResult:
    if board is None:
        raise ValueError("showdown equity requires a postflop board")
    cache = cache or build_showdown_equity_board_cache(board, evaluator=evaluator)
    showdown_input = build_showdown_equity_input(
        tree,
        aggregate,
        opponent_reach,
        board=cache.board,
        cache=cache,
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

    chunk_spans = _chunk_row_spans(showdown_input.rows, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda span: _compute_node_showdown_equity_block(showdown_input.rows, start=span[0], stop=span[1], cache=cache),
            chunk_spans,
        )
        flattened: list[float] = []
        for chunk_values in results:
            flattened.extend(float(value) for value in chunk_values)
        return tuple(flattened)


def _compute_node_showdown_equity_block(
    rows: tuple[ShowdownEquityNodeInput, ...],
    *,
    start: int,
    stop: int,
    cache: ShowdownEquityBoardCache,
) -> np.ndarray:
    block_rows = rows[start:stop]
    block_count = len(block_rows)
    if block_count == 0:
        return np.empty(0, dtype=np.float64)
    positive_reach_block = np.stack([row.positive_opponent_reach for row in block_rows], axis=0)
    total_equity = np.zeros(block_count, dtype=np.float64)
    hero_count = np.zeros(block_count, dtype=np.float64)
    win_weight_block = positive_reach_block @ cache.feasible_opponent_win_mask_matrix.T
    tie_weight_block = positive_reach_block @ cache.feasible_opponent_tie_mask_matrix.T
    for hero_index in cache.live_hand_indices:
        win_weight = win_weight_block[:, hero_index]
        tie_weight = tie_weight_block[:, hero_index]
        hero_opponent_weight = win_weight + tie_weight
        valid = hero_opponent_weight > 0.0
        if not np.any(valid):
            continue
        hero_equity = win_weight + (0.5 * tie_weight)
        total_equity[valid] += hero_equity[valid] / hero_opponent_weight[valid]
        hero_count[valid] += 1.0
    output = np.zeros(block_count, dtype=np.float64)
    valid_rows = hero_count > 0.0
    output[valid_rows] = total_equity[valid_rows] / hero_count[valid_rows]
    return output


def _chunk_row_spans(
    rows: tuple[ShowdownEquityNodeInput, ...],
    worker_count: int,
) -> tuple[tuple[int, int], ...]:
    row_count = len(rows)
    chunk_count = min(4, row_count)
    chunk_size = (row_count + chunk_count - 1) // chunk_count
    return tuple(
        (start, min(start + chunk_size, row_count))
        for start in range(0, row_count, chunk_size)
    )


def compute_showdown_equity_node(
    row: ShowdownEquityNodeInput,
    *,
    cache: ShowdownEquityBoardCache,
) -> float:
    return _compute_showdown_equity_node_fast(row, cache)


def _compute_showdown_equity_node_fast(
    row: ShowdownEquityNodeInput,
    cache: ShowdownEquityBoardCache,
) -> float:
    return _compute_showdown_equity_node_fast_with_positive_reach(row, cache, row.positive_opponent_reach)


def _compute_showdown_equity_node_fast_with_positive_reach(
    row: ShowdownEquityNodeInput,
    cache: ShowdownEquityBoardCache,
    positive_reach: np.ndarray,
) -> float:
    if len(row.opponent_reach) != private_hand_count():
        raise ValueError("opponent reach must match private hand count")
    if len(row.live_hand_mask) != private_hand_count():
        raise ValueError("live hand mask must match private hand count")
    if len(row.positive_opponent_reach) != private_hand_count():
        raise ValueError("positive opponent reach must match private hand count")
    if len(row.positive_opponent_mask) != private_hand_count():
        raise ValueError("positive opponent mask must match private hand count")
    if row.pot_size <= 0.0:
        raise ValueError("pot size must be positive")
    if row.board != cache.board:
        raise ValueError("node board must match the cached board")
    if row.live_hand_mask != cache.live_hand_mask:
        raise ValueError("node live hand mask must match the cached mask")
    if not cache.live_hand_indices:
        return 0.0

    opponent_weights = row.opponent_reach
    if opponent_weights.ndim != 1 or len(opponent_weights) != private_hand_count():
        raise ValueError("opponent reach must be one-dimensional and match private hand count")
    opponent_total = float(np.sum(positive_reach, dtype=np.float64))
    if opponent_total <= 0.0:
        return 0.0

    total_equity = 0.0
    hero_count = 0
    for hero_index in cache.live_hand_indices:
        hero_value = cache.hand_scores[hero_index] if cache.street is Street.RIVER else cache.hand_buckets_array[hero_index]
        feasible_opponents = cache.feasible_opponent_indices[hero_index]
        if feasible_opponents.size == 0:
            continue
        feasible_weights = positive_reach[feasible_opponents]
        if not np.any(feasible_weights):
            continue
        win_mask = cache.feasible_opponent_win_masks[hero_index]
        tie_mask = cache.feasible_opponent_tie_masks[hero_index]
        win_weight = float(np.sum(feasible_weights * win_mask, dtype=np.float64))
        tie_weight = float(np.sum(feasible_weights * tie_mask, dtype=np.float64))
        hero_equity = win_weight + (0.5 * tie_weight)
        hero_opponent_weight = win_weight + tie_weight
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


def _approximate_hand_strength_bucket(hand: PrivateHand, board: Board) -> int:
    board_values = [card.rank.order_value for card in board.cards]
    hand_values = [hand.first.rank.order_value, hand.second.rank.order_value]
    all_values = sorted(board_values + hand_values, reverse=True)
    pair_bonus = 0
    ranks = [card.rank.order_value for card in (hand.first, hand.second, *board.cards)]
    for rank_value in set(ranks):
        count = ranks.count(rank_value)
        if count == 4:
            pair_bonus = 8
            break
        if count == 3:
            pair_bonus = max(pair_bonus, 6)
        elif count == 2:
            pair_bonus = max(pair_bonus, 3)

    board_high = max(board_values) if board_values else 0
    hand_high = max(hand_values)
    raw = hand_high + board_high + sum(all_values[:3]) // 3 + pair_bonus
    return max(0, min(8, raw // 6))
