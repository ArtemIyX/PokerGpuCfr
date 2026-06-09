from __future__ import annotations

from dataclasses import dataclass
from typing import NewType, Protocol

import numpy as np
from numpy.typing import NDArray

from pokergpu.abstraction.hands import PrivateHand, RangeVector, all_private_hands
from pokergpu.core.board import Board
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
