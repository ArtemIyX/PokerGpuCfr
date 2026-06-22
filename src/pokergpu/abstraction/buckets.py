from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.hands import PrivateHand, RangeVector, all_private_hands
from pokergpu.core.board import Board
from pokergpu.core.cards import Card
from pokergpu.eval.treys_evaluator import TreysHandEvaluator

BucketId = NewType("BucketId", int)

_RANK_CLASS_TO_BUCKET: dict[int, BucketId] = {
    0: BucketId(8),
    1: BucketId(8),
    2: BucketId(7),
    3: BucketId(6),
    4: BucketId(5),
    5: BucketId(4),
    6: BucketId(3),
    7: BucketId(2),
    8: BucketId(1),
    9: BucketId(0),
}


class PostflopBucketer(Protocol):
    @property
    def bucket_count(self) -> int: ...

    def bucket_for_hand(self, hand: PrivateHand, board: Board) -> BucketId: ...


class PreflopBucketer(Protocol):
    @property
    def bucket_count(self) -> int: ...

    def bucket_for_hand(self, hand: PrivateHand) -> BucketId: ...


@dataclass(frozen=True, slots=True)
class StrengthTierBucketer:
    evaluator: TreysHandEvaluator

    def __init__(self, evaluator: TreysHandEvaluator | None = None) -> None:
        object.__setattr__(self, "evaluator", evaluator or TreysHandEvaluator())

    @property
    def bucket_count(self) -> int:
        return 9

    def bucket_for_hand(self, hand: PrivateHand, board: Board) -> BucketId:
        if board.is_preflop:
            raise ValueError("postflop bucketing requires a non-preflop board")
        if hand.contains_any(board.cards):
            raise ValueError("private hand cannot overlap the board")
        evaluated = self.evaluator.evaluate_seven_card_hand(
            (hand.first, hand.second, *board.cards)
        )
        return _RANK_CLASS_TO_BUCKET[evaluated.rank_class]

    def bucket_mask(self, board: Board) -> NDArray[np.int32]:
        if board.is_preflop:
            raise ValueError("postflop bucketing requires a non-preflop board")
        assignments = []
        for hand in all_private_hands():
            if hand.contains_any(board.cards):
                assignments.append(-1)
                continue
            assignments.append(int(self.bucket_for_hand(hand, board)))
        return np.asarray(assignments, dtype=np.int32)

    def bucketed_range(
        self,
        hand_range: RangeVector,
        board: Board,
    ) -> NDArray[np.float32]:
        assignments = self.bucket_mask(board)
        result = np.zeros(self.bucket_count, dtype=np.float32)
        for hand_index, bucket_index in enumerate(assignments):
            if bucket_index < 0:
                continue
            result[bucket_index] += hand_range.values[hand_index]
        return result


@dataclass(frozen=True, slots=True)
class PreflopClassBucketer:
    @property
    def bucket_count(self) -> int:
        return 8

    def bucket_for_hand(self, hand: PrivateHand) -> BucketId:
        first_rank = hand.first.rank.order_value
        second_rank = hand.second.rank.order_value
        high = max(first_rank, second_rank)
        low = min(first_rank, second_rank)
        suited = hand.first.suit == hand.second.suit
        gap = high - low

        if first_rank == second_rank:
            return BucketId(_PAIR_BUCKETS[high])
        if high >= 14 and low >= 13:
            return BucketId(6 if suited else 5)
        if high >= 14 and low >= 10:
            return BucketId(5 if suited else 4)
        if high >= 13 and low >= 10:
            return BucketId(4 if suited else 3)
        if gap <= 1:
            return BucketId(3 if suited else 2)
        if suited:
            return BucketId(2 if high >= 10 else 1)
        if high >= 10:
            return BucketId(1)
        return BucketId(0)

    def bucket_mask(self, dead_cards: tuple[Card, ...] | list[Card]) -> NDArray[np.int32]:
        from pokergpu.abstraction.hands import all_private_hands

        dead_card_set = set(dead_cards)
        if len(dead_card_set) != len(dead_cards):
            raise ValueError("dead cards must be unique")
        assignments = []
        for hand in all_private_hands():
            if hand.contains_any(dead_cards):
                assignments.append(-1)
            else:
                assignments.append(int(self.bucket_for_hand(hand)))
        return np.asarray(assignments, dtype=np.int32)

    def bucketed_range(self, hand_range: RangeVector) -> NDArray[np.float32]:
        result = np.zeros(self.bucket_count, dtype=np.float32)
        from pokergpu.abstraction.hands import all_private_hands

        for hand_index, hand in enumerate(all_private_hands()):
            result[int(self.bucket_for_hand(hand))] += hand_range.values[hand_index]
        return result


_PAIR_BUCKETS: dict[int, int] = {
    14: 7,
    13: 6,
    12: 6,
    11: 5,
    10: 5,
    9: 4,
    8: 4,
    7: 3,
    6: 3,
    5: 2,
    4: 2,
    3: 1,
    2: 0,
}
