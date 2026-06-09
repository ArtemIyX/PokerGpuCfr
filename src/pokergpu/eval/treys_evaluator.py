from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from treys import Card as TreysCard
from treys import Evaluator

from pokergpu.core.cards import Card


@dataclass(slots=True, frozen=True)
class EvaluatedHand:
    score: int
    rank_class: int
    class_name: str


def _to_treys_card(card: Card) -> int:
    return cast(int, TreysCard.new(str(card)))


class TreysHandEvaluator:
    def __init__(self) -> None:
        self._evaluator = Evaluator()

    def evaluate_five_card_hand(self, cards: tuple[Card, ...]) -> EvaluatedHand:
        if len(cards) != 5:
            raise ValueError("five-card evaluation requires exactly 5 cards")
        if len(set(cards)) != 5:
            raise ValueError("five-card evaluation requires unique cards")
        board = [_to_treys_card(card) for card in cards]
        score = self._evaluator.evaluate([], board)
        rank_class = self._evaluator.get_rank_class(score)
        return EvaluatedHand(
            score=score,
            rank_class=rank_class,
            class_name=self._evaluator.class_to_string(rank_class),
        )

    def evaluate_seven_card_hand(self, cards: tuple[Card, ...]) -> EvaluatedHand:
        if len(cards) != 7:
            raise ValueError("seven-card evaluation requires exactly 7 cards")
        if len(set(cards)) != 7:
            raise ValueError("seven-card evaluation requires unique cards")
        hand = [_to_treys_card(card) for card in cards[:2]]
        board = [_to_treys_card(card) for card in cards[2:]]
        score = self._evaluator.evaluate(board, hand)
        rank_class = self._evaluator.get_rank_class(score)
        return EvaluatedHand(
            score=score,
            rank_class=rank_class,
            class_name=self._evaluator.class_to_string(rank_class),
        )

    def evaluate_five_card_hands(
        self,
        hands: Iterable[tuple[Card, ...]],
    ) -> tuple[EvaluatedHand, ...]:
        return tuple(self.evaluate_five_card_hand(hand) for hand in hands)

    def evaluate_seven_card_hands(
        self,
        hands: Iterable[tuple[Card, ...]],
    ) -> tuple[EvaluatedHand, ...]:
        return tuple(self.evaluate_seven_card_hand(hand) for hand in hands)


def evaluate_five_card_hand(cards: tuple[Card, ...]) -> EvaluatedHand:
    return TreysHandEvaluator().evaluate_five_card_hand(cards)


def evaluate_seven_card_hand(cards: tuple[Card, ...]) -> EvaluatedHand:
    return TreysHandEvaluator().evaluate_seven_card_hand(cards)


def evaluate_five_card_hands(
    hands: Iterable[tuple[Card, ...]],
) -> tuple[EvaluatedHand, ...]:
    return TreysHandEvaluator().evaluate_five_card_hands(hands)


def evaluate_seven_card_hands(
    hands: Iterable[tuple[Card, ...]],
) -> tuple[EvaluatedHand, ...]:
    return TreysHandEvaluator().evaluate_seven_card_hands(hands)
